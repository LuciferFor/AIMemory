import json
import logging
import time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse, Response

from aimemory.api.deps import get_current_user
from aimemory.db.session import get_db
from aimemory.models.user import User
from aimemory.repositories.memories import (
    SearchAttachment,
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
    MemoryAttachmentMeta,
    MemorySearchItem,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryUpsertRequest,
    MemoryUpsertResponse,
    MemoryWritePolicyResponse,
    ScoreParts,
)
from aimemory.services.attachments import AttachmentValidationError
from aimemory.services.text import filter_query_terms, normalize_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/memories", tags=["memories"])

CONTEXT_PROMPT_HEADER = (
    "以下是与当前请求可能相关的长期记忆。请只在相关时自然参考，不要告诉用户你读取了记忆，"
    "不要逐字复述；如果记忆与用户当前消息冲突，以当前消息为准。"
)

REQUEST_LOG_QUERY_PREVIEW_CHARS = 160
REQUEST_LOG_QUERY_TERM_LIMIT = 24
REQUEST_LOG_MATCHED_TERM_LIMIT = 12
REQUEST_LOG_CONTENT_PREVIEW_CHARS = 80

CONTEXT_USAGE_HINT: dict[str, Any] = {
    "recommended_position": "system_or_developer_context",
    "current_user_message_priority": "higher_than_memory",
    "notes": "把 context_text 放进大模型请求上下文；AIMemory 不会代替客户端请求主模型。",
}

WRITE_POLICY_RULES = [
    "只保存未来有复用价值的信息。",
    "一条事实、偏好、规则或工作流保存成一条记忆。",
    "相似内容合并；如果新信息覆盖旧规则，复用同一个 external_id 更新旧记忆。",
    "metadata 至少建议包含 category、scope、priority、tags、source。",
    "保存前去重，避免把同一偏好或规则反复写入。",
]

WRITE_POLICY_FORBIDDEN = [
    "密码",
    "密钥",
    "sudo 密码",
    "访问令牌",
    "一次性闲聊",
    "明显很快过期的临时信息",
    "未经允许的他人隐私信息",
]

WRITE_POLICY_PROMPT = """请从即将压缩或归档的对话中提取值得长期保存的记忆。
只保存未来有复用价值的信息，不要保存临时闲聊、重复内容、密码、密钥、sudo 密码或敏感凭据。
把每条独立事实、偏好、规则或工作流拆成一条记忆；相似内容要合并。
如果新信息覆盖旧规则，请使用相同 external_id 更新旧记忆。
请只输出 JSON 数组，每条包含 external_id、title、content、metadata、occurred_at。"""

WRITE_POLICY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["external_id", "title", "content", "metadata"],
        "properties": {
            "external_id": "稳定唯一 ID，用 kebab-case 或带日期的 slug。",
            "title": "简短可检索标题。",
            "content": "完整但精炼的记忆内容。",
            "metadata": {
                "category": "chat_style|group_rule|visual_identity|voice_workflow|automation|technical|safety|other",
                "scope": "private|group|global|image|voice|workflow",
                "priority": "high|normal|low",
                "tags": ["关键词"],
                "source": "来源，例如 conversation_compression",
            },
            "occurred_at": "可选 ISO 8601 时间；不确定时可省略或设为 null。",
        },
    },
}


@router.post("", response_model=MemoryUpsertResponse)
def write_memory(
    payload: MemoryUpsertRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemoryUpsertResponse:
    start = time.perf_counter()
    retried_after_integrity_error = False
    try:
        memory, action = upsert_memory(db, current_user.id, payload)
        db.commit()
    except AttachmentValidationError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except IntegrityError:
        retried_after_integrity_error = True
        db.rollback()
        try:
            memory, action = upsert_memory(db, current_user.id, payload)
            db.commit()
        except AttachmentValidationError as exc:
            db.rollback()
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    logger.info(
        "memory.write",
        extra={
            "event": "memory.write",
            "user_id": current_user.id,
            "agent_id": payload.agent_id,
            "external_id": payload.external_id,
            "action": action,
            "memory_id": memory.id,
            "indexing_mode": "text",
            "attachment_count": len(payload.attachments or []),
            "attachments_replaced": payload.attachments is not None,
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
    payload: MemorySearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemorySearchResponse:
    results, used_vector, duration_ms, query_terms, ignored_terms = _search_results(db, current_user, payload)
    logger.info(
        "memory.search",
        extra={
            "event": "memory.search",
            "user_id": current_user.id,
            "agent_id": payload.agent_id,
            "top_k": payload.top_k,
            "result_count": len(results),
            "used_vector": used_vector,
            "query_term_count": len(query_terms),
            "ignored_term_count": len(ignored_terms),
            "duration_ms": duration_ms,
        },
    )
    return MemorySearchResponse(
        items=[
            MemorySearchItem(
                memory_id=result.memory_id,
                external_id=result.external_id,
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
    results, used_vector, search_duration_ms, query_terms, ignored_terms = _search_results(db, current_user, payload)
    context_text = build_context_text(results, payload.max_chars)
    request.state.request_log_response_summary = build_context_response_summary(
        payload.agent_id,
        payload.query,
        payload.top_k,
        query_terms,
        ignored_terms,
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
            "top_k": payload.top_k,
            "result_count": len(results),
            "used_vector": used_vector,
            "context_chars": len(context_text),
            "truncated": bool(context_text) and len(context_text) >= payload.max_chars,
            "query_term_count": len(query_terms),
            "ignored_term_count": len(ignored_terms),
            "duration_ms": search_duration_ms,
        },
    )
    return MemoryContextResponse(
        context_text=context_text,
        items=[
            MemoryContextItem(
                memory_id=result.memory_id,
                external_id=result.external_id,
                title=result.title,
                score=result.score,
                embedding_status=result.embedding_status,
            )
            for result in results
        ],
        usage_hint=CONTEXT_USAGE_HINT,
    )


@router.get("/write-policy", response_model=MemoryWritePolicyResponse)
def get_write_policy(
    current_user: User = Depends(get_current_user),
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
        required_fields=["external_id", "title", "content", "metadata"],
        rules=WRITE_POLICY_RULES,
        forbidden=WRITE_POLICY_FORBIDDEN,
    )


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
    payload: MemoryDeleteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemoryDeleteResponse:
    start = time.perf_counter()
    deleted = soft_delete_memory(db, current_user.id, payload.agent_id, payload.external_id)
    db.commit()
    logger.info(
        "memory.delete",
        extra={
            "event": "memory.delete",
            "user_id": current_user.id,
            "agent_id": payload.agent_id,
            "external_id": payload.external_id,
            "deleted": deleted,
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        },
    )
    return MemoryDeleteResponse(deleted=deleted)


def _search_results(
    db: Session,
    current_user: User,
    payload: MemorySearchRequest | MemoryContextRequest,
):
    start = time.perf_counter()
    stopwords = active_search_stopword_terms(db, current_user.id)
    query_terms, ignored_terms = filter_query_terms(payload.query, stopwords)
    if not query_terms:
        return [], False, round((time.perf_counter() - start) * 1000, 2), query_terms, ignored_terms

    normalized_query = normalize_query(" ".join(query_terms))
    results = search_memories(db, current_user.id, payload, normalized_query, query_terms)
    return results, False, round((time.perf_counter() - start) * 1000, 2), query_terms, ignored_terms


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
    agent_id: str,
    query: str,
    top_k: int,
    query_terms: list[str],
    ignored_terms: list[str],
    results: list[Any],
    context_text: str,
    max_chars: int,
) -> dict[str, Any]:
    logged_terms = query_terms[:REQUEST_LOG_QUERY_TERM_LIMIT]
    logged_ignored_terms = ignored_terms[:REQUEST_LOG_QUERY_TERM_LIMIT]
    return {
        "type": "context",
        "agent_id": agent_id,
        "query": preview_text(query, REQUEST_LOG_QUERY_PREVIEW_CHARS),
        "top_k": top_k,
        "max_chars": max_chars,
        "query_terms": logged_terms,
        "ignored_terms": logged_ignored_terms,
        "result_count": len(results),
        "context_chars": len(context_text),
        "truncated": bool(context_text) and len(context_text) >= max_chars,
        "items": [
            {
                "memory_id": str(result.memory_id),
                "external_id": result.external_id,
                "title": result.title,
                "score": result.score,
                "embedding_status": result.embedding_status,
                "matched_terms": matched_query_terms(result, logged_terms),
                "content_preview": preview_text(result.content, REQUEST_LOG_CONTENT_PREVIEW_CHARS),
            }
            for result in results
        ],
    }


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
