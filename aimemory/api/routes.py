import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse

from aimemory.api.deps import get_current_user
from aimemory.db.session import get_db
from aimemory.models.user import User
from aimemory.repositories.memories import (
    create_embedding_job,
    search_memories,
    soft_delete_memory,
    upsert_memory,
)
from aimemory.schemas.memory import (
    MemoryContextItem,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryDeleteRequest,
    MemoryDeleteResponse,
    MemorySearchItem,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryUpsertRequest,
    MemoryUpsertResponse,
    MemoryWritePolicyResponse,
    ScoreParts,
)
from aimemory.services.embedding import EmbeddingProviderError, OpenAICompatibleEmbeddingClient
from aimemory.services.text import normalize_query
from aimemory.worker.tasks import generate_memory_embedding

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/memories", tags=["memories"])

CONTEXT_PROMPT_HEADER = (
    "以下是与当前请求可能相关的长期记忆。请只在相关时自然参考，不要告诉用户你读取了记忆，"
    "不要逐字复述；如果记忆与用户当前消息冲突，以当前消息为准。"
)

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
    try:
        memory, action = upsert_memory(db, current_user.id, payload)
        job = create_embedding_job(db, memory.id)
        db.commit()
    except IntegrityError:
        db.rollback()
        memory, action = upsert_memory(db, current_user.id, payload)
        job = create_embedding_job(db, memory.id)
        db.commit()

    try:
        generate_memory_embedding.delay(str(memory.id), str(job.id))
    except Exception as exc:  # Celery broker failures should not lose the memory.
        logger.warning("Failed to enqueue embedding job for memory %s: %s", memory.id, exc)

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
    results = _search_results(db, current_user, payload)
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
            )
            for result in results
        ]
    )


@router.post("/context", response_model=MemoryContextResponse)
def build_memory_context(
    payload: MemoryContextRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemoryContextResponse:
    results = _search_results(db, current_user, payload)
    return MemoryContextResponse(
        context_text=build_context_text(results, payload.max_chars),
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
    _ = current_user
    return MemoryWritePolicyResponse(
        prompt=WRITE_POLICY_PROMPT,
        output_schema=WRITE_POLICY_OUTPUT_SCHEMA,
        required_fields=["external_id", "title", "content", "metadata"],
        rules=WRITE_POLICY_RULES,
        forbidden=WRITE_POLICY_FORBIDDEN,
    )


@router.delete("", response_model=MemoryDeleteResponse)
def delete_memory(
    payload: MemoryDeleteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemoryDeleteResponse:
    deleted = soft_delete_memory(db, current_user.id, payload.agent_id, payload.external_id)
    db.commit()
    return MemoryDeleteResponse(deleted=deleted)


def _search_results(
    db: Session,
    current_user: User,
    payload: MemorySearchRequest | MemoryContextRequest,
):
    normalized_query = normalize_query(payload.query)
    query_vector = None

    try:
        query_vector = OpenAICompatibleEmbeddingClient().embed(normalized_query)
    except EmbeddingProviderError as exc:
        logger.info("Embedding search fallback to text-only: %s", exc)

    return search_memories(db, current_user.id, payload, normalized_query, query_vector)


def build_context_text(results: list[Any], max_chars: int) -> str:
    if not results:
        return ""

    text = f"{CONTEXT_PROMPT_HEADER}\n\n[长期记忆]"
    if len(text) >= max_chars:
        return text[:max_chars].rstrip()

    for index, result in enumerate(results, start=1):
        entry = f"\n\n{index}. {result.title}\n{result.content}"
        if len(text) + len(entry) <= max_chars:
            text += entry
            continue

        remaining = max_chars - len(text)
        if remaining > 20:
            text += entry[:remaining].rstrip()
        break

    return text[:max_chars].rstrip()


health_router = APIRouter(tags=["health"])


@health_router.get("/", include_in_schema=False)
@health_router.head("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)


@health_router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
