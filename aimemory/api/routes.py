import json
import logging
import hashlib
import time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse, Response

from aimemory.api.deps import get_current_user
from aimemory.core.config import get_settings
from aimemory.db.session import get_db
from aimemory.models.user import User
from aimemory.repositories.memory_categories import UNCATEGORIZED_CATEGORY, get_active_category, list_category_summaries
from aimemory.repositories.memory_agents import ResolvedMemoryIdentity, resolve_memory_identity
from aimemory.repositories.memories import (
    SearchAttachment,
    filter_high_frequency_terms,
    get_attachment_for_user,
    search_memories,
    soft_delete_memory,
    upsert_memory,
)
from aimemory.repositories.search_stopwords import active_search_stopword_terms
from aimemory.schemas.memory import (
    MemoryContextItem,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryDeleteRequest,
    MemoryDeleteResponse,
    MemoryExtractItem,
    MemoryExtractRequest,
    MemoryExtractResponse,
    MemoryAttachmentMeta,
    MemoryCategoriesResponse,
    MemoryCategoryItem,
    MemorySearchItem,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryUpsertRequest,
    MemoryUpsertResponse,
    MemoryWritePolicyResponse,
    ScoreParts,
)
from aimemory.services.attachments import AttachmentValidationError
from aimemory.services.ai_crypto import decrypt_secret
from aimemory.services.ai_memory_review import (
    build_auto_merge_candidate_memory_ids,
    get_llm_config,
    run_auto_merge_review_for_extracted_memories,
)
from aimemory.services.category_analysis import CategoryAnalysis, analyze_memory_category
from aimemory.services.openai_compatible import chat_completion, token_usage_summary
from aimemory.services.query_analysis import (
    MemoryRequestAnalysis,
    QueryAnalysis,
    analyze_memory_query,
    analyze_memory_request,
    effective_terms_from_ai_keywords,
)
from aimemory.services.text import filter_query_terms, normalize_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/memories", tags=["memories"])

CONTEXT_PROMPT_HEADER = (
    "以下是与当前请求可能相关的长期记忆。请只在相关时自然参考，不要告诉用户你读取了记忆，"
    "不要逐字复述；长期记忆不是强制规则，只有明确标为规则或长期指令时才按规则执行，"
    "普通偏好只作参考；如果记忆与用户当前消息冲突，以当前消息为准。"
)

REQUEST_LOG_QUERY_PREVIEW_CHARS = 200
REQUEST_LOG_QUERY_TERM_LIMIT = 24
REQUEST_LOG_MATCHED_TERM_LIMIT = 12
REQUEST_LOG_CONTENT_PREVIEW_CHARS = 80
REQUEST_LOG_CONTEXT_PREVIEW_CHARS = 300

CONTEXT_USAGE_HINT: dict[str, Any] = {
    "recommended_position": "system_or_developer_context",
    "current_user_message_priority": "higher_than_memory",
    "notes": "把 context_text 放进大模型请求上下文；AIMemory 不会代替客户端请求主模型。",
}

WRITE_POLICY_RULES = [
    "只保存未来有复用价值的信息。",
    "一条事实、偏好、规则或工作流保存成一条记忆。",
    "相似内容合并；如果新信息覆盖旧规则，复用同一个 external_id 更新旧记忆。",
    "每条记忆必须选择一个 category；优先使用已有分类，只有确实不合适时才创建新分类。",
    "metadata 至少建议包含 category、scope、priority、tags、source。",
    "保存前去重，避免把同一偏好或规则反复写入。",
    "一次性出图要求、单次生成参数、临时姿势/服装/风格要求默认不保存；除非用户明确说记住、以后都这样、长期固定。",
]

WRITE_POLICY_FORBIDDEN = [
    "密码",
    "密钥",
    "sudo 密码",
    "访问令牌",
    "一次性闲聊",
    "一次性出图要求",
    "单次生成参数",
    "临时姿势、服装或风格要求",
    "明显很快过期的临时信息",
    "未经允许的他人隐私信息",
]

WRITE_POLICY_PROMPT = """请从即将压缩或归档的对话中提取值得长期保存的记忆。
只保存未来有复用价值的信息，不要保存临时闲聊、重复内容、密码、密钥、sudo 密码或敏感凭据。
不要保存一次性出图要求、单次生成参数、临时姿势/服装/风格要求，除非用户明确要求“记住”“以后都这样”“长期固定”。
把每条独立事实、偏好、规则或工作流拆成一条记忆；相似内容要合并。
如果新信息覆盖旧规则，请使用相同 external_id 更新旧记忆。
每条记忆必须包含 category。优先从已有分类列表选择；如果没有合适分类，可以创建简短明确的新分类。
请只输出 JSON 数组，每条包含 external_id、category、title、content、metadata、occurred_at。"""

WRITE_POLICY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["external_id", "category", "title", "content", "metadata"],
        "properties": {
            "external_id": "稳定唯一 ID，用 kebab-case 或带日期的 slug。",
            "category": "事务分类名称，例如 爱吃的水果、喜欢的人、性取向、脾气；优先使用已有分类。",
            "title": "简短可检索标题。",
            "content": "完整但精炼的记忆内容。",
            "metadata": {
                "category": "回答风格|群聊规则|视觉身份|语音流程|自动化|技术资料|安全规则|其他",
                "scope": "私聊|群聊|全局|图片|语音|工作流",
                "priority": "高|普通|低",
                "tags": ["关键词"],
                "source": "来源，例如 conversation_compression",
            },
            "occurred_at": "可选 ISO 8601 时间；不确定时可省略或设为 null。",
        },
    },
}

EXTRACT_MEMORY_SYSTEM_PROMPT = (
    "你是 AIMemory 的长期记忆提取器。请从对话 transcript 中提取值得未来复用的长期记忆。"
    "只保存稳定事实、偏好、约束、关系、项目背景、工作流和长期指令；不要保存临时闲聊、重复表达、"
    "密码、密钥、访问令牌、sudo 密码或敏感凭据。"
    "不要保存一次性出图要求、单次生成参数、临时姿势/服装/风格要求；"
    "只有当用户明确说“记住”“以后都这样”“长期固定”“以后默认”时，才可保存为长期出图偏好。"
    "每条记忆必须使用第三方视角，例如“用户偏好……”“助手应……”。"
    "优先使用已有分类；没有合适分类时可以创建简短明确的新分类。"
    "只输出 JSON 对象，不要输出解释。"
    "\n\nJSON 输出格式："
    "{\"memories\":[{\"external_id\":\"稳定 slug，可省略\","
    "\"category\":\"分类\",\"title\":\"简短标题\",\"content\":\"完整但精炼的正文\","
    "\"metadata\":{\"scope\":\"全局|私聊|群聊|工作流\",\"priority\":\"高|普通|低\",\"tags\":[\"关键词\"]},"
    "\"occurred_at\":null}]}"
)

RECENT_EXTRACT_REQUESTS: dict[str, float] = {}
EXTRACT_DEDUPE_TTL_SECONDS = 120


def resolved_payload(
    request: Request,
    db: Session,
    current_user: User,
    payload: MemoryUpsertRequest | MemorySearchRequest | MemoryContextRequest | MemoryExtractRequest | MemoryDeleteRequest,
):
    try:
        identity = resolve_memory_identity(
            db,
            current_user.id,
            agent_id=payload.agent_id,
            device_id=getattr(payload, "device_id", None),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    mark_request_identity(request, identity)
    return payload.model_copy(update={"agent_id": identity.agent_id, "device_id": identity.device_id}), identity


def mark_request_identity(request: Request, identity: ResolvedMemoryIdentity) -> None:
    request.state.request_log_agent_id = identity.agent_id
    request.state.request_log_agent_name = identity.agent_name
    request.state.request_log_device_id = identity.device_id
    request.state.request_log_device_name = identity.device_name


@router.post("", response_model=MemoryUpsertResponse)
def write_memory(
    request: Request,
    payload: MemoryUpsertRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemoryUpsertResponse:
    start = time.perf_counter()
    retried_after_integrity_error = False
    payload, identity = resolved_payload(request, db, current_user, payload)
    category_name, category_meta = resolve_category_for_write(db, current_user, payload)
    payload_for_write = payload.model_copy(update={"category": category_name})
    try:
        memory, action = upsert_memory(db, current_user.id, payload_for_write)
        db.commit()
    except AttachmentValidationError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except IntegrityError:
        retried_after_integrity_error = True
        db.rollback()
        try:
            memory, action = upsert_memory(db, current_user.id, payload_for_write)
            db.commit()
        except AttachmentValidationError as exc:
            db.rollback()
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    request.state.request_log_response_summary = build_write_response_summary(
        payload_for_write,
        memory.id,
        action,
        category_meta,
        identity,
    )
    logger.info(
        "memory.write",
        extra={
            "event": "memory.write",
            "user_id": current_user.id,
            "agent_id": payload.agent_id,
            "agent_name": identity.agent_name,
            "device_id": identity.device_id,
            "device_name": identity.device_name,
            "external_id": payload.external_id,
            "category": category_name,
            "category_source": category_meta.get("category_source"),
            "category_confidence": category_meta.get("category_confidence"),
            "category_total_tokens": category_meta.get("category_total_tokens"),
            "action": action,
            "memory_id": memory.id,
            "indexing_mode": "text",
            "attachment_count": len(payload_for_write.attachments or []),
            "attachments_replaced": payload_for_write.attachments is not None,
            "retried_after_integrity_error": retried_after_integrity_error,
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        },
    )

    return MemoryUpsertResponse(
        memory_id=memory.id,
        external_id=memory.external_id,
        action=action,
        embedding_status=memory.embedding_status,
    )


@router.post("/search", response_model=MemorySearchResponse)
def search_memory(
    request: Request,
    payload: MemorySearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemorySearchResponse:
    payload, identity = resolved_payload(request, db, current_user, payload)
    results, used_vector, duration_ms, query_terms, ignored_terms, category_found, query_analysis = _search_results(
        db,
        current_user,
        payload,
    )
    logger.info(
        "memory.search",
        extra={
            "event": "memory.search",
            "user_id": current_user.id,
            "agent_id": payload.agent_id,
            "agent_name": identity.agent_name,
            "device_id": identity.device_id,
            "device_name": identity.device_name,
            "category": payload.category,
            "category_found": category_found,
            "category_source": query_analysis.get("category_source"),
            "category_confidence": query_analysis.get("category_confidence"),
            "top_k": payload.top_k,
            "result_count": len(results),
            "used_vector": used_vector,
            "query_term_count": len(query_terms),
            "ignored_term_count": len(ignored_terms),
            "keyword_source": query_analysis.get("keyword_source"),
            "ai_duration_ms": query_analysis.get("ai_duration_ms"),
            "category_total_tokens": query_analysis.get("category_total_tokens"),
            "ai_total_tokens": query_analysis.get("ai_total_tokens"),
            "ai_request_total_tokens": query_analysis.get("ai_request_total_tokens"),
            "duration_ms": duration_ms,
        },
    )
    return MemorySearchResponse(
        items=[
            MemorySearchItem(
                memory_id=result.memory_id,
                external_id=result.external_id,
                category=result.category,
                title=result.title,
                content=result.content,
                metadata=result.metadata,
                created_at=result.created_at,
                updated_at=result.updated_at,
                score=result.score,
                score_parts=ScoreParts(**result.score_parts),
                embedding_status=result.embedding_status,
                attachments=[attachment_meta(attachment) for attachment in result.attachments],
            )
            for result in results
        ]
    )


@router.post("/context", response_model=MemoryContextResponse)
def build_memory_context(
    request: Request,
    payload: MemoryContextRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemoryContextResponse:
    payload, identity = resolved_payload(request, db, current_user, payload)
    results, used_vector, search_duration_ms, query_terms, ignored_terms, category_found, query_analysis = _search_results(
        db,
        current_user,
        payload,
    )
    context_text = build_context_text(results, payload.max_chars)
    request.state.request_log_response_summary = build_context_response_summary(
        identity,
        payload.category or "",
        category_found,
        payload.query,
        payload.top_k,
        query_terms,
        ignored_terms,
        query_analysis,
        results,
        context_text,
        payload.max_chars,
    )
    logger.info(
        "memory.context",
        extra={
            "event": "memory.context",
            "user_id": current_user.id,
            "agent_id": payload.agent_id,
            "agent_name": identity.agent_name,
            "device_id": identity.device_id,
            "device_name": identity.device_name,
            "category": payload.category,
            "category_found": category_found,
            "category_source": query_analysis.get("category_source"),
            "category_confidence": query_analysis.get("category_confidence"),
            "top_k": payload.top_k,
            "result_count": len(results),
            "used_vector": used_vector,
            "context_chars": len(context_text),
            "truncated": bool(context_text) and len(context_text) >= payload.max_chars,
            "query_term_count": len(query_terms),
            "ignored_term_count": len(ignored_terms),
            "keyword_source": query_analysis.get("keyword_source"),
            "ai_duration_ms": query_analysis.get("ai_duration_ms"),
            "category_total_tokens": query_analysis.get("category_total_tokens"),
            "ai_total_tokens": query_analysis.get("ai_total_tokens"),
            "ai_request_total_tokens": query_analysis.get("ai_request_total_tokens"),
            "duration_ms": search_duration_ms,
        },
    )
    return MemoryContextResponse(
        context_text=context_text,
        items=[
            MemoryContextItem(
                memory_id=result.memory_id,
                external_id=result.external_id,
                category=result.category,
                title=result.title,
                score=result.score,
                embedding_status=result.embedding_status,
            )
            for result in results
        ],
        usage_hint=CONTEXT_USAGE_HINT,
    )


@router.get("/categories", response_model=MemoryCategoriesResponse)
def list_memory_categories(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemoryCategoriesResponse:
    categories = list_category_summaries(db, current_user.id)
    return MemoryCategoriesResponse(items=[category_item(category) for category in categories])


@router.get("/write-policy", response_model=MemoryWritePolicyResponse)
def get_write_policy(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemoryWritePolicyResponse:
    logger.info(
        "memory.write_policy",
        extra={
            "event": "memory.write_policy",
            "user_id": current_user.id,
        },
    )
    return MemoryWritePolicyResponse(
        prompt=WRITE_POLICY_PROMPT,
        output_schema=WRITE_POLICY_OUTPUT_SCHEMA,
        required_fields=["external_id", "category", "title", "content", "metadata"],
        rules=WRITE_POLICY_RULES,
        forbidden=WRITE_POLICY_FORBIDDEN,
        categories=[category_item(category) for category in list_category_summaries(db, current_user.id)],
    )


@router.post("/extract", response_model=MemoryExtractResponse)
def extract_memory_from_transcript(
    request: Request,
    payload: MemoryExtractRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemoryExtractResponse:
    payload, identity = resolved_payload(request, db, current_user, payload)
    dedupe_key = extract_request_dedupe_key(current_user.id, payload)
    now = time.time()
    prune_recent_extract_requests(now)
    if now - RECENT_EXTRACT_REQUESTS.get(dedupe_key, 0) < EXTRACT_DEDUPE_TTL_SECONDS:
        response = MemoryExtractResponse(extracted=0, written=0, items=[])
        request.state.request_log_response_summary = {
            "type": "extract",
            "agent_id": payload.agent_id,
            "agent_name": identity.agent_name,
            "device_id": identity.device_id,
            "device_name": identity.device_name,
            "reason": payload.reason,
            "transcript_chars": len(payload.transcript),
            "deduplicated": True,
            "extracted": 0,
            "written": 0,
        }
        return response
    RECENT_EXTRACT_REQUESTS[dedupe_key] = now

    config = get_llm_config(db)
    if not config or not getattr(config, "enabled", False):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="AI 配置未启用。")
    if not getattr(config, "encrypted_api_key", None):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="AI API Key 未配置。")

    api_key = decrypt_secret(config.encrypted_api_key, get_settings().ai_config_encryption_secret)
    categories = [category.name for category in list_category_summaries(db, current_user.id)]
    result = chat_completion(
        config,
        api_key,
        build_extract_memory_messages(payload.transcript, categories, payload.reason),
        response_format={"type": "json_object"},
        max_tokens=1400,
        temperature=0.1,
        timeout_ms=max(int(getattr(config, "timeout_ms", 30000) or 30000), 30000),
    )
    usage = token_usage_summary(result.usage)
    skipped_candidates: list[dict[str, Any]] = []
    candidates = normalize_extracted_memory_candidates(
        result.content,
        payload.reason,
        payload.metadata,
        transcript=payload.transcript,
        skipped=skipped_candidates,
    )
    written_items: list[MemoryExtractItem] = []
    written_memory_ids: list[UUID] = []
    for candidate in candidates:
        try:
            memory_payload = MemoryUpsertRequest(agent_id=payload.agent_id, **candidate)
            _memory, action = upsert_memory(db, current_user.id, memory_payload)
            db.commit()
            written_memory_ids.append(_memory.id)
            written_items.append(
                MemoryExtractItem(
                    external_id=memory_payload.external_id,
                    category=memory_payload.category,
                    title=memory_payload.title,
                    action=action,
                )
            )
        except Exception as exc:
            db.rollback()
            logger.warning(
                "memory.extract.item_skipped",
                extra={
                    "event": "memory.extract.item_skipped",
                    "user_id": current_user.id,
                    "agent_id": payload.agent_id,
                    "external_id": candidate.get("external_id"),
                    "reason": str(exc)[:300],
                },
            )

    auto_merge_summary = build_auto_merge_extract_summary(
        db,
        current_user.id,
        payload.agent_id,
        written_memory_ids,
    )
    if auto_merge_summary["auto_merge_scheduled"]:
        background_tasks.add_task(
            run_auto_merge_review_for_extracted_memories,
            str(current_user.id),
            payload.agent_id,
            [str(memory_id) for memory_id in written_memory_ids],
        )

    response = MemoryExtractResponse(extracted=len(candidates), written=len(written_items), items=written_items)
    request.state.request_log_response_summary = {
        "type": "extract",
        "agent_id": payload.agent_id,
        "agent_name": identity.agent_name,
        "device_id": identity.device_id,
        "device_name": identity.device_name,
        "reason": payload.reason,
        "transcript_chars": len(payload.transcript),
        "extracted": response.extracted,
        "written": response.written,
        "skipped": len(skipped_candidates),
        "skipped_items": skipped_candidates[:20],
        "ai_usage": usage,
        "ai_total_tokens": token_total(usage),
        "ai_request_total_tokens": token_total(usage),
        **auto_merge_summary,
        "items": [item.model_dump() for item in response.items[:20]],
    }
    logger.info(
        "memory.extract",
        extra={
            "event": "memory.extract",
            "user_id": current_user.id,
            "agent_id": payload.agent_id,
            "agent_name": identity.agent_name,
            "device_id": identity.device_id,
            "device_name": identity.device_name,
            "reason": payload.reason,
            "transcript_chars": len(payload.transcript),
            "extracted": response.extracted,
            "written": response.written,
            "skipped": len(skipped_candidates),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    )
    return response


@router.get("/attachments/{attachment_id}")
def download_attachment(
    attachment_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    attachment = get_attachment_for_user(db, current_user.id, attachment_id)
    if attachment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="附件不存在。")
    logger.info(
        "memory.attachment_download",
        extra={
            "event": "memory.attachment_download",
            "user_id": current_user.id,
            "attachment_id": attachment.id,
            "mime_type": attachment.mime_type,
            "size_bytes": attachment.size_bytes,
            "sha256": attachment.sha256,
        },
    )
    return Response(content=attachment.image_bytes, media_type=attachment.mime_type)


@router.delete("", response_model=MemoryDeleteResponse)
def delete_memory(
    request: Request,
    payload: MemoryDeleteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemoryDeleteResponse:
    start = time.perf_counter()
    payload, identity = resolved_payload(request, db, current_user, payload)
    deleted = soft_delete_memory(db, current_user.id, payload.agent_id, payload.external_id)
    db.commit()
    logger.info(
        "memory.delete",
        extra={
            "event": "memory.delete",
            "user_id": current_user.id,
            "agent_id": payload.agent_id,
            "agent_name": identity.agent_name,
            "device_id": identity.device_id,
            "device_name": identity.device_name,
            "external_id": payload.external_id,
            "deleted": deleted,
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        },
    )
    return MemoryDeleteResponse(deleted=deleted)


def resolve_category_for_write(
    db: Session,
    current_user: User,
    payload: MemoryUpsertRequest,
) -> tuple[str, dict[str, Any]]:
    client_category = str(payload.category or "").strip()
    categories = list_category_summaries(db, current_user.id)
    config = get_llm_config(db)
    if category_ai_enabled(config):
        try:
            analysis = analyze_category_with_config(
                config,
                "write",
                write_category_text(payload),
                categories,
            )
            if analysis.category:
                return analysis.category, category_meta("ai", analysis)
            if client_category:
                meta = category_meta("client", analysis, reason=analysis.reason or "ai_empty_selection")
                meta["selected_category"] = client_category
                return client_category, meta
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="AI 分类失败且未提供分类。",
            )
        except HTTPException:
            raise
        except Exception as exc:
            if client_category:
                meta = category_meta("client", error=str(exc), reason="ai_failed")
                meta["selected_category"] = client_category
                return client_category, meta
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="AI 分类失败且未提供分类。",
            ) from exc

    if client_category:
        meta = category_meta("client")
        meta["selected_category"] = client_category
        return client_category, meta
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="未提供分类，且后台 AI 分类未启用。",
    )


def resolve_category_for_query(
    db: Session,
    current_user: User,
    payload: MemorySearchRequest | MemoryContextRequest,
) -> tuple[Any | None, dict[str, Any]]:
    client_category = str(payload.category or "").strip()
    categories = list_category_summaries(db, current_user.id)
    config = get_llm_config(db)

    if category_ai_enabled(config):
        try:
            analysis = analyze_category_with_config(
                config,
                "context" if isinstance(payload, MemoryContextRequest) else "search",
                payload.query,
                categories,
            )
            meta = category_meta("ai", analysis)
            if not analysis.category:
                meta["ignored_terms"] = ["分类未选择"]
                return None, meta
            category = get_active_category(db, current_user.id, analysis.category)
            if category is None:
                meta["ignored_terms"] = [f"{analysis.category}:分类不存在"]
                return None, meta
            meta["selected_category"] = category.name
            meta["category_found"] = True
            return category, meta
        except Exception as exc:
            if client_category:
                category = get_active_category(db, current_user.id, client_category)
                meta = category_meta("client", error=str(exc), reason="ai_failed")
                meta["selected_category"] = client_category
                meta["category_found"] = category is not None
                if category is None:
                    meta["ignored_terms"] = [f"{client_category}:分类不存在"]
                return category, meta
            fallback = get_active_category(db, current_user.id, UNCATEGORIZED_CATEGORY)
            meta = category_meta("fallback", error=str(exc), reason="ai_failed")
            meta["selected_category"] = UNCATEGORIZED_CATEGORY
            meta["category_found"] = fallback is not None
            if fallback is None:
                meta["ignored_terms"] = [f"{UNCATEGORIZED_CATEGORY}:分类不存在"]
            return fallback, meta

    if client_category:
        category = get_active_category(db, current_user.id, client_category)
        meta = category_meta("client")
        meta["selected_category"] = client_category
        meta["category_found"] = category is not None
        if category is None:
            meta["ignored_terms"] = [f"{client_category}:分类不存在"]
        return category, meta

    fallback = get_active_category(db, current_user.id, UNCATEGORIZED_CATEGORY)
    meta = category_meta("fallback", reason="ai_disabled")
    meta["selected_category"] = UNCATEGORIZED_CATEGORY
    meta["category_found"] = fallback is not None
    if fallback is None:
        meta["ignored_terms"] = [f"{UNCATEGORIZED_CATEGORY}:分类不存在"]
    return fallback, meta


def analyze_category_with_config(
    config: Any,
    operation: str,
    text_value: str,
    categories: list[Any],
) -> CategoryAnalysis:
    api_key = decrypt_secret(config.encrypted_api_key, get_settings().ai_config_encryption_secret)
    return analyze_memory_category(
        config,
        api_key,
        text=text_value,
        operation=operation,
        categories=categories,
    )


def category_ai_enabled(config: Any) -> bool:
    return bool(config is not None and getattr(config, "enabled", False) and getattr(config, "encrypted_api_key", None))


def category_meta(
    source: str,
    analysis: CategoryAnalysis | None = None,
    *,
    error: str = "",
    reason: str = "",
) -> dict[str, Any]:
    selected = analysis.category if analysis else ""
    return {
        "category_source": source,
        "selected_category": selected,
        "category_found": bool(analysis.matched_existing) if analysis else False,
        "category_matched_existing": bool(analysis.matched_existing) if analysis else False,
        "category_confidence": analysis.confidence if analysis else 0.0,
        "category_reason": reason or (analysis.reason if analysis else ""),
        "category_error": str(error or "")[:300],
        "category_duration_ms": analysis.duration_ms if analysis else 0.0,
        "category_usage": analysis.usage if analysis else {},
        "category_total_tokens": token_total(analysis.usage if analysis else {}),
        "ignored_terms": [],
    }


def with_category_meta(query_analysis: dict[str, Any], category_info: dict[str, Any]) -> dict[str, Any]:
    return {
        **query_analysis,
        "category_source": category_info.get("category_source", ""),
        "selected_category": category_info.get("selected_category", ""),
        "category_reason": category_info.get("category_reason", ""),
        "category_confidence": category_info.get("category_confidence", 0.0),
        "category_error": category_info.get("category_error", ""),
        "category_duration_ms": category_info.get("category_duration_ms", 0.0),
        "category_usage": category_info.get("category_usage", {}),
        "category_total_tokens": category_info.get("category_total_tokens", 0),
        "ai_request_total_tokens": token_total(category_info.get("category_usage", {}))
        + token_total(query_analysis.get("ai_usage", {})),
    }


def write_category_text(payload: MemoryUpsertRequest) -> str:
    metadata_text = json.dumps(payload.metadata or {}, ensure_ascii=False, sort_keys=True)
    return "\n".join(
        part
        for part in [
            f"标题: {payload.title}",
            f"正文: {payload.content}",
            f"metadata: {metadata_text}" if metadata_text != "{}" else "",
        ]
        if part
    )


def _search_results(
    db: Session,
    current_user: User,
    payload: MemorySearchRequest | MemoryContextRequest,
):
    start = time.perf_counter()
    stopwords = active_search_stopword_terms(db, current_user.id)
    config = get_llm_config(db)
    if query_analysis_enabled(config):
        category, query_terms, ignored_terms, query_analysis = combined_ai_query_context(
            db,
            current_user,
            payload,
            stopwords,
            config,
        )
        payload.category = query_analysis.get("selected_category") or payload.category
    else:
        category, category_meta = resolve_category_for_query(db, current_user, payload)
        payload.category = category_meta.get("selected_category") or payload.category
        if category is None:
            return (
                [],
                False,
                round((time.perf_counter() - start) * 1000, 2),
                [],
                category_meta.get("ignored_terms") or [f"{payload.category or ''}:分类不存在"],
                False,
                with_category_meta(default_query_analysis_meta("disabled"), category_meta),
            )
        query_terms, ignored_terms, query_analysis = query_terms_for_search(db, payload, stopwords)
        query_analysis = with_category_meta(query_analysis, category_meta)

    if category is None:
        return (
            [],
            False,
            round((time.perf_counter() - start) * 1000, 2),
            [],
            ignored_terms or [f"{payload.category or ''}:分类不存在"],
            False,
            query_analysis,
        )

    query_terms, high_frequency_ignored_terms = filter_high_frequency_terms(
        db,
        current_user.id,
        category.id,
        payload.agent_id,
        query_terms,
    )
    ignored_terms.extend(high_frequency_ignored_terms)
    if query_analysis.get("keyword_source") == "ai" and high_frequency_ignored_terms:
        query_analysis["ai_ignored_terms"] = [
            *query_analysis.get("ai_ignored_terms", []),
            *high_frequency_ignored_terms,
        ]
    if not query_terms:
        query_analysis["no_result_reason"] = "无有效关键词"
        return [], False, round((time.perf_counter() - start) * 1000, 2), query_terms, ignored_terms, True, query_analysis

    normalized_query = normalize_query(" ".join(query_terms))
    min_matched_terms = minimum_matched_terms_for_query(query_terms, payload.query)
    query_analysis["min_matched_terms"] = min_matched_terms
    default_min_matched_terms = 1 if len(query_terms) == 1 else 2
    search_kwargs = {"min_matched_terms": min_matched_terms} if min_matched_terms != default_min_matched_terms else {}
    results = search_memories(db, current_user.id, category.id, payload, normalized_query, query_terms, **search_kwargs)
    if not results:
        query_analysis["no_result_reason"] = "分类内没有匹配记忆或分数低于阈值"
    return results, False, round((time.perf_counter() - start) * 1000, 2), query_terms, ignored_terms, True, query_analysis


def minimum_matched_terms_for_query(query_terms: list[str], query: str) -> int:
    required_terms = [term for term in query_terms if term_appears_in_query(term, query)]
    count = len(required_terms) if required_terms else len(query_terms)
    return 1 if count <= 1 else 2


def term_appears_in_query(term: str, query: str) -> bool:
    return normalize_query(term) in normalize_query(query)


def combined_ai_query_context(
    db: Session,
    current_user: User,
    payload: MemorySearchRequest | MemoryContextRequest,
    stopwords: set[str],
    config: Any,
) -> tuple[Any | None, list[str], list[str], dict[str, Any]]:
    try:
        categories = list_category_summaries(db, current_user.id)
        api_key = decrypt_secret(config.encrypted_api_key, get_settings().ai_config_encryption_secret)
        analysis = analyze_memory_request(
            config,
            api_key,
            query=payload.query,
            operation="context" if isinstance(payload, MemoryContextRequest) else "search",
            categories=categories,
            agent_id=payload.agent_id,
        )
        meta = combined_query_analysis_meta("ai", analysis)
        if not analysis.category:
            meta["ignored_terms"] = ["分类未选择"]
            return None, [], ["分类未选择"], meta
        category = get_active_category(db, current_user.id, analysis.category)
        if category is None:
            ignored = [f"{analysis.category}:分类不存在"]
            meta["ignored_terms"] = ignored
            meta["category_found"] = False
            return None, [], ignored, meta
        meta["selected_category"] = category.name
        meta["category_found"] = True
        query_terms, ignored_terms = effective_terms_from_ai_keywords(analysis.keywords, stopwords)
        meta["ai_ignored_terms"] = ignored_terms[:REQUEST_LOG_QUERY_TERM_LIMIT]
        return category, query_terms, ignored_terms, meta
    except Exception as exc:
        category, category_meta = fallback_category_for_query(db, current_user, payload, error=str(exc))
        query_terms, ignored_terms = filter_query_terms(payload.query, stopwords)
        meta = with_category_meta(default_query_analysis_meta("failed", str(exc)), category_meta)
        return category, query_terms, ignored_terms, meta


def fallback_category_for_query(
    db: Session,
    current_user: User,
    payload: MemorySearchRequest | MemoryContextRequest,
    *,
    error: str = "",
) -> tuple[Any | None, dict[str, Any]]:
    client_category = str(payload.category or "").strip()
    if client_category:
        category = get_active_category(db, current_user.id, client_category)
        meta = category_meta("client", error=error, reason="combined_ai_failed" if error else "")
        meta["selected_category"] = client_category
        meta["category_found"] = category is not None
        if category is None:
            meta["ignored_terms"] = [f"{client_category}:分类不存在"]
        return category, meta

    fallback = get_active_category(db, current_user.id, UNCATEGORIZED_CATEGORY)
    meta = category_meta("fallback", error=error, reason="combined_ai_failed" if error else "ai_disabled")
    meta["selected_category"] = UNCATEGORIZED_CATEGORY
    meta["category_found"] = fallback is not None
    if fallback is None:
        meta["ignored_terms"] = [f"{UNCATEGORIZED_CATEGORY}:分类不存在"]
    return fallback, meta


def query_terms_for_search(
    db: Session,
    payload: MemorySearchRequest | MemoryContextRequest,
    stopwords: set[str],
) -> tuple[list[str], list[str], dict[str, Any]]:
    config = get_llm_config(db)
    if not query_analysis_enabled(config):
        query_terms, ignored_terms = filter_query_terms(payload.query, stopwords)
        return query_terms, ignored_terms, default_query_analysis_meta("disabled")

    try:
        api_key = decrypt_secret(config.encrypted_api_key, get_settings().ai_config_encryption_secret)
        analysis = analyze_memory_query(
            config,
            api_key,
            query=payload.query,
            category=payload.category or "",
            agent_id=payload.agent_id,
        )
        query_terms, ignored_terms = effective_terms_from_ai_keywords(analysis.keywords, stopwords)
        meta = query_analysis_meta("ai", analysis)
        meta["ai_ignored_terms"] = ignored_terms[:REQUEST_LOG_QUERY_TERM_LIMIT]
        return query_terms, ignored_terms, meta
    except Exception as exc:
        query_terms, ignored_terms = filter_query_terms(payload.query, stopwords)
        return query_terms, ignored_terms, default_query_analysis_meta("failed", str(exc))


def query_analysis_enabled(config: Any) -> bool:
    if config is None:
        return False
    return bool(
        getattr(config, "enabled", False)
        and getattr(config, "query_analysis_enabled", True)
        and getattr(config, "encrypted_api_key", None)
    )


def default_query_analysis_meta(keyword_source: str, error: str = "") -> dict[str, Any]:
    return {
        "keyword_source": keyword_source,
        "intent_summary": "",
        "ai_keywords": [],
        "negative_keywords": [],
        "ai_ignored_terms": [],
        "ai_error": str(error or "")[:300],
        "ai_duration_ms": 0.0,
        "ai_usage": {},
        "ai_total_tokens": 0,
        "ai_request_total_tokens": 0,
    }


def query_analysis_meta(keyword_source: str, analysis: QueryAnalysis) -> dict[str, Any]:
    return {
        "keyword_source": keyword_source,
        "intent_summary": analysis.intent_summary,
        "ai_keywords": analysis.keywords,
        "negative_keywords": analysis.negative_keywords,
        "ai_ignored_terms": [],
        "ai_error": "",
        "ai_duration_ms": analysis.duration_ms,
        "ai_usage": analysis.usage,
        "ai_total_tokens": token_total(analysis.usage),
        "ai_request_total_tokens": token_total(analysis.usage),
    }


def combined_query_analysis_meta(keyword_source: str, analysis: MemoryRequestAnalysis) -> dict[str, Any]:
    total_tokens = token_total(analysis.usage)
    return {
        "keyword_source": keyword_source,
        "analysis_mode": "combined",
        "intent_summary": analysis.intent_summary,
        "ai_keywords": analysis.keywords,
        "negative_keywords": analysis.negative_keywords,
        "ai_ignored_terms": [],
        "ai_error": "",
        "ai_duration_ms": analysis.duration_ms,
        "ai_usage": analysis.usage,
        "ai_total_tokens": total_tokens,
        "ai_request_total_tokens": total_tokens,
        "category_source": "ai",
        "selected_category": analysis.category,
        "category_found": bool(analysis.matched_existing),
        "category_matched_existing": bool(analysis.matched_existing),
        "category_reason": analysis.reason,
        "category_confidence": analysis.confidence,
        "category_error": "",
        "category_duration_ms": analysis.duration_ms,
        "category_usage": {},
        "category_total_tokens": 0,
        "ignored_terms": [],
    }


def token_total(usage: dict[str, Any] | None) -> int:
    if not isinstance(usage, dict):
        return 0
    try:
        return max(0, int(usage.get("total_tokens") or 0))
    except (TypeError, ValueError):
        return 0


def build_context_text(results: list[Any], max_chars: int) -> str:
    if not results:
        return ""

    text = f"{CONTEXT_PROMPT_HEADER}\n\n[长期记忆]"
    if len(text) >= max_chars:
        return text[:max_chars].rstrip()

    for index, result in enumerate(results, start=1):
        attachment_text = attachment_context_text(result.attachments)
        entry = f"\n\n{index}. {result.title}\n{result.content}{attachment_text}"
        if len(text) + len(entry) <= max_chars:
            text += entry
            continue

        remaining = max_chars - len(text)
        if remaining > 20:
            text += entry[:remaining].rstrip()
        break

    return text[:max_chars].rstrip()


def build_context_response_summary(
    identity: ResolvedMemoryIdentity,
    category: str,
    category_found: bool,
    query: str,
    top_k: int,
    query_terms: list[str],
    ignored_terms: list[str],
    query_analysis: dict[str, Any],
    results: list[Any],
    context_text: str,
    max_chars: int,
) -> dict[str, Any]:
    logged_terms = query_terms[:REQUEST_LOG_QUERY_TERM_LIMIT]
    logged_ignored_terms = ignored_terms[:REQUEST_LOG_QUERY_TERM_LIMIT]
    return {
        "type": "context",
        "agent_id": identity.agent_id,
        "agent_name": identity.agent_name,
        "device_id": identity.device_id,
        "device_name": identity.device_name,
        "category": category,
        "category_found": category_found,
        "category_not_found": not category_found,
        "query": preview_text(query, REQUEST_LOG_QUERY_PREVIEW_CHARS),
        "query_preview": preview_text(query, REQUEST_LOG_QUERY_PREVIEW_CHARS),
        "top_k": top_k,
        "max_chars": max_chars,
        "keyword_source": query_analysis.get("keyword_source", "disabled"),
        "analysis_mode": query_analysis.get("analysis_mode", ""),
        "category_source": query_analysis.get("category_source", ""),
        "selected_category": query_analysis.get("selected_category", category),
        "category_reason": query_analysis.get("category_reason", ""),
        "category_confidence": query_analysis.get("category_confidence", 0.0),
        "category_error": query_analysis.get("category_error", ""),
        "category_duration_ms": query_analysis.get("category_duration_ms", 0.0),
        "category_usage": query_analysis.get("category_usage", {}),
        "category_total_tokens": query_analysis.get("category_total_tokens", 0),
        "intent_summary": query_analysis.get("intent_summary", ""),
        "ai_keywords": query_analysis.get("ai_keywords", [])[:REQUEST_LOG_QUERY_TERM_LIMIT],
        "negative_keywords": query_analysis.get("negative_keywords", [])[:REQUEST_LOG_QUERY_TERM_LIMIT],
        "ai_ignored_terms": query_analysis.get("ai_ignored_terms", [])[:REQUEST_LOG_QUERY_TERM_LIMIT],
        "ai_error": query_analysis.get("ai_error", ""),
        "ai_duration_ms": query_analysis.get("ai_duration_ms", 0.0),
        "ai_usage": query_analysis.get("ai_usage", {}),
        "ai_total_tokens": query_analysis.get("ai_total_tokens", 0),
        "ai_request_total_tokens": query_analysis.get("ai_request_total_tokens", 0),
        "query_terms": logged_terms,
        "ignored_terms": logged_ignored_terms,
        "min_matched_terms": query_analysis.get("min_matched_terms", 0),
        "no_result_reason": query_analysis.get("no_result_reason", ""),
        "result_count": len(results),
        "context_chars": len(context_text),
        "truncated": bool(context_text) and len(context_text) >= max_chars,
        "context_text_preview": preview_multiline_text(context_text, REQUEST_LOG_CONTEXT_PREVIEW_CHARS),
        "context_text_preview_truncated": len(str(context_text or "").strip()) > REQUEST_LOG_CONTEXT_PREVIEW_CHARS,
        "items": [
            {
                "memory_id": str(result.memory_id),
                "external_id": result.external_id,
                "title": result.title,
                "score": result.score,
                "score_parts": getattr(result, "score_parts", {}),
                "embedding_status": result.embedding_status,
                "matched_terms": matched_terms_for_log(result, logged_terms),
                "matched_fields": matched_fields_for_log(result, logged_terms),
                "content_preview": preview_text(result.content, REQUEST_LOG_CONTENT_PREVIEW_CHARS),
            }
            for result in results
        ],
    }


def build_write_response_summary(
    payload: MemoryUpsertRequest,
    memory_id: UUID,
    action: str,
    category_info: dict[str, Any],
    identity: ResolvedMemoryIdentity,
) -> dict[str, Any]:
    return {
        "type": "write",
        "agent_id": payload.agent_id,
        "agent_name": identity.agent_name,
        "device_id": identity.device_id,
        "device_name": identity.device_name,
        "external_id": payload.external_id,
        "memory_id": str(memory_id),
        "action": action,
        "category": payload.category,
        "category_source": category_info.get("category_source", ""),
        "selected_category": category_info.get("selected_category", payload.category),
        "category_reason": category_info.get("category_reason", ""),
        "category_confidence": category_info.get("category_confidence", 0.0),
        "category_error": category_info.get("category_error", ""),
        "category_duration_ms": category_info.get("category_duration_ms", 0.0),
        "category_usage": category_info.get("category_usage", {}),
        "category_total_tokens": category_info.get("category_total_tokens", 0),
        "ai_request_total_tokens": category_info.get("category_total_tokens", 0),
        "title_preview": preview_text(payload.title, REQUEST_LOG_QUERY_PREVIEW_CHARS),
        "content_preview": preview_text(payload.content, REQUEST_LOG_CONTENT_PREVIEW_CHARS),
        "attachment_count": len(payload.attachments or []),
    }


def category_item(category: Any) -> MemoryCategoryItem:
    return MemoryCategoryItem(
        category_id=category.id,
        name=category.name,
        description=category.description,
        memory_count=category.memory_count,
    )


def matched_terms_for_log(result: Any, query_terms: list[str]) -> list[str]:
    terms = list(getattr(result, "matched_terms", []) or [])
    if terms:
        return terms[:REQUEST_LOG_MATCHED_TERM_LIMIT]
    return matched_query_terms(result, query_terms)


def matched_fields_for_log(result: Any, query_terms: list[str]) -> list[str]:
    fields = list(getattr(result, "matched_fields", []) or [])
    if fields:
        return fields

    matched_fields: list[str] = []
    field_values = {
        "标题": result.title,
        "正文": result.content,
        "外部ID": result.external_id,
        "元数据": json.dumps(result.metadata or {}, ensure_ascii=False, sort_keys=True),
    }
    attachment_parts: list[str] = []
    for attachment in getattr(result, "attachments", []):
        attachment_parts.extend([attachment.filename, attachment.description, attachment.ocr_text])
    field_values["附件"] = " ".join(str(part or "") for part in attachment_parts)

    for field_name, value in field_values.items():
        normalized_value = normalize_query(str(value or ""))
        if any(normalize_query(term) in normalized_value for term in query_terms):
            matched_fields.append(field_name)
    return matched_fields


def matched_query_terms(result: Any, query_terms: list[str]) -> list[str]:
    if not query_terms:
        return []

    parts = [
        result.title,
        result.content,
        result.external_id,
        json.dumps(result.metadata or {}, ensure_ascii=False, sort_keys=True),
    ]
    for attachment in getattr(result, "attachments", []):
        parts.extend(
            [
                attachment.filename,
                attachment.description,
                attachment.ocr_text,
            ]
        )
    searchable_text = normalize_query(" ".join(str(part or "") for part in parts))

    matched: list[str] = []
    seen: set[str] = set()
    for term in query_terms:
        normalized_term = normalize_query(term)
        if normalized_term and normalized_term in searchable_text and normalized_term not in seen:
            matched.append(term)
            seen.add(normalized_term)
        if len(matched) >= REQUEST_LOG_MATCHED_TERM_LIMIT:
            break
    return matched


def preview_text(value: object, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def preview_multiline_text(value: object, max_chars: int) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip()


def build_extract_memory_messages(transcript: str, categories: list[str], reason: str) -> list[dict[str, str]]:
    category_text = "\n".join(f"- {category}" for category in categories) or "- 其它"
    return [
        {
            "role": "system",
            "content": f"{EXTRACT_MEMORY_SYSTEM_PROMPT}\n\n已有分类：\n{category_text}",
        },
        {
            "role": "user",
            "content": (
                f"提取原因：{reason}\n\n"
                "请从下面 transcript 提取长期记忆，输出 JSON：\n"
                f"{transcript[:120000]}"
            ),
        },
    ]


def build_auto_merge_extract_summary(
    db: Session,
    user_id: UUID,
    agent_id: str,
    written_memory_ids: list[UUID],
) -> dict[str, Any]:
    summary = {
        "auto_merge_scheduled": False,
        "auto_merge_candidate_count": 0,
        "auto_merge_reason": "no_written_memories" if not written_memory_ids else "not_checked",
    }
    if not written_memory_ids:
        return summary
    try:
        review_memory_ids, candidate_count = build_auto_merge_candidate_memory_ids(
            db,
            user_id=user_id,
            agent_id=agent_id,
            seed_memory_ids=written_memory_ids,
        )
    except Exception as exc:
        logger.warning(
            "memory.extract.auto_merge_candidate_check_failed",
            extra={
                "event": "memory.extract.auto_merge_candidate_check_failed",
                "user_id": user_id,
                "agent_id": agent_id,
                "reason": str(exc)[:300],
            },
        )
        summary["auto_merge_reason"] = "candidate_check_failed"
        return summary

    summary["auto_merge_candidate_count"] = candidate_count
    if candidate_count <= 0 or len(review_memory_ids) < 2:
        summary["auto_merge_reason"] = "no_similar_candidates"
        return summary
    summary["auto_merge_scheduled"] = True
    summary["auto_merge_reason"] = "similar_candidates"
    summary["auto_merge_review_memory_count"] = len(review_memory_ids)
    return summary


def extract_request_dedupe_key(user_id: UUID, payload: MemoryExtractRequest) -> str:
    digest = hashlib.sha256(payload.transcript.encode("utf-8")).hexdigest()
    return f"{user_id}:{payload.agent_id}:{payload.reason}:{digest}"


def prune_recent_extract_requests(now: float) -> None:
    expired = [
        key
        for key, created_at in RECENT_EXTRACT_REQUESTS.items()
        if now - created_at >= EXTRACT_DEDUPE_TTL_SECONDS
    ]
    for key in expired:
        RECENT_EXTRACT_REQUESTS.pop(key, None)


def normalize_extracted_memory_candidates(
    content: str,
    reason: str,
    source_metadata: dict[str, Any],
    transcript: str = "",
    skipped: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    parsed = parse_extracted_memory_json(content)
    candidates: list[dict[str, Any]] = []
    for raw in parsed:
        if not isinstance(raw, dict):
            continue
        category = clean_extracted_text(raw.get("category"), 128)
        title = clean_extracted_text(raw.get("title"), 512)
        memory_content = clean_extracted_text(raw.get("content"), 20000)
        if not category or not title or not memory_content:
            continue
        if contains_forbidden_memory_text(title, memory_content, category):
            record_skipped_candidate(skipped, raw, "forbidden_sensitive_content")
            continue
        temporary_reason = temporary_image_memory_skip_reason(
            title=title,
            content=memory_content,
            category=category,
            metadata=raw.get("metadata"),
            transcript=transcript,
            reason=reason,
        )
        if temporary_reason:
            record_skipped_candidate(skipped, raw, temporary_reason)
            continue
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        metadata = {
            **metadata,
            "source": metadata.get("source") or "conversation_compaction",
            "extract_reason": reason,
            "source_metadata": source_metadata,
        }
        external_id = clean_external_id(raw.get("external_id")) or stable_memory_external_id(
            category,
            title,
            memory_content,
        )
        candidates.append(
            {
                "external_id": external_id,
                "category": category,
                "title": title,
                "content": memory_content,
                "metadata": metadata,
                "occurred_at": raw.get("occurred_at") or raw.get("occurredAt"),
            }
        )
    return candidates


def parse_extracted_memory_json(content: str) -> list[Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    candidates = [text]
    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start >= 0 and array_end > array_start:
        candidates.append(text[array_start : array_end + 1])
    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start >= 0 and object_end > object_start:
        candidates.append(text[object_start : object_end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("memories", "items", "results"):
                items = parsed.get(key)
                if isinstance(items, list):
                    return items
    if last_error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="AI 返回不是合法 JSON。") from last_error
    return []


def clean_extracted_text(value: object, max_chars: int) -> str:
    return " ".join(str(value or "").split()).strip()[:max_chars]


def clean_external_id(value: object) -> str:
    text = str(value or "").strip().lower()
    cleaned = []
    prev_dash = False
    for char in text:
        allowed = char.isascii() and (char.isalnum() or char in "_.:-")
        if allowed:
            cleaned.append(char)
            prev_dash = False
            continue
        if not prev_dash:
            cleaned.append("-")
            prev_dash = True
    return "".join(cleaned).strip("-")[:180]


def stable_memory_external_id(category: str, title: str, content: str) -> str:
    category_slug = clean_external_id(category) or "memory"
    digest = hashlib.sha256(f"{title}\n{content}".encode("utf-8")).hexdigest()[:12]
    return f"auto-{category_slug}-{digest}"


TEMPORARY_IMAGE_TASK_TERMS = [
    "出图",
    "生图",
    "绘图",
    "画图",
    "画一张",
    "生成图片",
    "生成图",
    "图片生成",
    "做图",
    "图像生成",
    "image generation",
    "generate image",
]

TEMPORARY_IMAGE_DETAIL_TERMS = [
    "一张",
    "这张",
    "单次",
    "临时",
    "换成",
    "改成",
    "姿势",
    "服装",
    "衣服",
    "风格",
    "表情",
    "构图",
    "镜头",
    "背景",
    "光影",
    "黑丝",
    "白丝",
    "兔女郎",
    "女仆",
    "腿",
    "身材",
    "胸",
    "裙",
    "丝袜",
    "银白发",
]

DURABLE_MEMORY_MARKERS = [
    "记住",
    "记下来",
    "帮我记",
    "以后都",
    "以后默认",
    "以后固定",
    "长期",
    "永久",
    "固定偏好",
    "作为偏好",
    "下次也",
    "每次都",
    "默认这样",
    "remember",
    "always",
    "default",
]


def record_skipped_candidate(skipped: list[dict[str, Any]] | None, raw: dict[str, Any], reason: str) -> None:
    if skipped is None:
        return
    skipped.append(
        {
            "title": clean_extracted_text(raw.get("title"), 120),
            "category": clean_extracted_text(raw.get("category"), 80),
            "reason": reason,
        }
    )


def temporary_image_memory_skip_reason(
    *,
    title: str,
    content: str,
    category: str,
    metadata: object,
    transcript: str,
    reason: str,
) -> str:
    if reason == "explicit_remember":
        return ""
    transcript_text = str(transcript or "").lower()
    if any(marker.lower() in transcript_text for marker in DURABLE_MEMORY_MARKERS):
        return ""

    metadata_text = ""
    if isinstance(metadata, dict):
        metadata_text = json.dumps(metadata, ensure_ascii=False)
    candidate_text = f"{category}\n{title}\n{content}\n{metadata_text}".lower()
    has_image_task = any(term.lower() in candidate_text for term in TEMPORARY_IMAGE_TASK_TERMS)
    has_image_detail = any(term.lower() in candidate_text for term in TEMPORARY_IMAGE_DETAIL_TERMS)
    if has_image_task and has_image_detail:
        return "temporary_image_request_without_durable_marker"
    return ""


def contains_forbidden_memory_text(*values: object) -> bool:
    text = "\n".join(str(value or "") for value in values).lower()
    forbidden = [
        "password",
        "passwd",
        "api key",
        "apikey",
        "secret",
        "token",
        "private key",
        "sudo",
        "密码",
        "密钥",
        "私钥",
        "令牌",
        "口令",
    ]
    return any(term in text for term in forbidden)


def attachment_meta(attachment: SearchAttachment) -> MemoryAttachmentMeta:
    return MemoryAttachmentMeta(
        attachment_id=attachment.attachment_id,
        filename=attachment.filename,
        mime_type=attachment.mime_type,
        size_bytes=attachment.size_bytes,
        sha256=attachment.sha256,
        description=attachment.description,
        download_url=f"/v1/memories/attachments/{attachment.attachment_id}",
    )


def attachment_context_text(attachments: list[SearchAttachment]) -> str:
    lines = []
    for attachment in attachments:
        label = attachment.filename
        if attachment.description:
            label = f"{label}：{attachment.description}"
        elif attachment.ocr_text:
            label = f"{label}：{attachment.ocr_text}"
        lines.append(f"- {label}")
    if not lines:
        return ""
    return "\n图片附件：\n" + "\n".join(lines)


health_router = APIRouter(tags=["health"])


@health_router.get("/", include_in_schema=False)
@health_router.head("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)


@health_router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
