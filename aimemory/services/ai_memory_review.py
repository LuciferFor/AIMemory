import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from aimemory.models.ai_memory_review import AiMemoryReviewRun, AiMemoryReviewSuggestion
from aimemory.models.llm_provider_config import LlmProviderConfig
from aimemory.models.memory import Memory
from aimemory.models.memory_attachment import MemoryAttachment
from aimemory.models.memory_category import MemoryCategory
from aimemory.repositories.memories import utcnow
from aimemory.repositories.memory_categories import get_or_create_category
from aimemory.services.attachments import attachment_search_text
from aimemory.services.text import build_search_text
from aimemory.services.openai_compatible import chat_completion

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_EXTRA_BODY = {"thinking": {"type": "disabled"}}
DEFAULT_REVIEW_PROMPT_INJECTION = (
    "整理长期记忆时请遵循：\n"
    "1. 先判断每条记忆的长期价值，只提出必要、低风险的整理建议。\n"
    "2. 保留具体事实、偏好、约束、关系、项目背景和可执行指令；删除口水话、重复表达、临时状态和过期上下文。\n"
    "3. rewrite 要让标题简短明确，正文自然、可检索，并保持第三方视角。\n"
    "4. merge 只用于事实高度重复或互补的记忆，目标记忆选择信息最完整的一条。\n"
    "5. move_category 优先使用已有分类；只有明显不合适时才提出新分类。\n"
    "6. soft_delete 仅用于明显无价值、错误、过期或已被其他记忆完整覆盖的内容。"
)
ALLOWED_SUGGESTION_TYPES = {"rewrite", "merge", "move_category", "soft_delete"}
MAX_REVIEW_MEMORIES = 50


class AiMemoryReviewError(ValueError):
    pass


def default_config_values() -> dict[str, Any]:
    return {
        "name": "default",
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "timeout_ms": 30000,
        "max_output_tokens": 4096,
        "temperature": 0.0,
        "extra_body_json": DEFAULT_EXTRA_BODY,
        "review_prompt_injection": DEFAULT_REVIEW_PROMPT_INJECTION,
        "enabled": True,
        "query_analysis_enabled": True,
        "query_analysis_max_output_tokens": 256,
        "query_analysis_timeout_ms": 3000,
    }


def get_llm_config(db: Session) -> LlmProviderConfig | None:
    return db.scalar(select(LlmProviderConfig).where(LlmProviderConfig.name == "default"))


def create_default_llm_config() -> LlmProviderConfig:
    return LlmProviderConfig(**default_config_values())


def memory_summary(memory: Memory, category_name: str, user_name: str = "") -> dict[str, Any]:
    return {
        "memory_id": str(memory.id),
        "user_id": str(memory.user_id),
        "user_name": user_name,
        "agent_id": memory.agent_id,
        "external_id": memory.external_id,
        "category": category_name,
        "title": memory.title,
        "content": memory.content,
        "metadata": memory.metadata_json or {},
        "created_at": str(memory.created_at),
        "updated_at": str(memory.updated_at),
    }


def build_review_messages(
    memories: list[dict[str, Any]],
    categories: list[str],
    prompt_injection: str = "",
) -> list[dict[str, str]]:
    category_text = "\n".join(f"- {category}" for category in categories) or "- 其它"
    payload = {
        "categories": categories,
        "memories": memories,
    }
    fixed_system_prompt = (
        "你是 AIMemory 后台的长期记忆整理助手。请阅读管理员选中的记忆，输出 json。"
        "任务是生成谨慎的整理建议，不要直接回答用户，不要编造新事实。"
        "支持建议类型：rewrite、merge、move_category、soft_delete。"
        "rewrite 用于压缩/改写单条记忆；merge 用于合并重复或高度相似记忆；"
        "move_category 用于调整分类；soft_delete 用于明显无价值或重复的删除候选。"
        "除非明显重复或无价值，不要建议删除。"
        "标题和正文必须使用第三方视角，例如“用户偏好……”“助手应……”。"
        "优先使用已有分类；确实没有合适分类时可以给出新分类名。"
        "只输出 JSON 对象，不要输出解释。"
        "\n\nJSON 输出格式："
        "{\"suggestions\":[{\"type\":\"rewrite|merge|move_category|soft_delete\","
        "\"confidence\":0.0,\"reason\":\"原因\",\"memory_ids\":[\"uuid\"],"
        "\"target_memory_id\":\"uuid\",\"proposed\":{\"title\":\"\",\"content\":\"\",\"category\":\"\"}}]}"
        f"\n\n已有分类：\n{category_text}"
    )
    system_sections = []
    cleaned_prompt_injection = str(prompt_injection or "").strip()
    if cleaned_prompt_injection:
        system_sections.append(f"管理员整理前置注入提示：\n{cleaned_prompt_injection}")
    system_sections.append(fixed_system_prompt)
    return [
        {
            "role": "system",
            "content": "\n\n".join(system_sections),
        },
        {
            "role": "user",
            "content": "请整理以下 AIMemory 记忆并输出 json：\n" + json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def create_review_run(
    db: Session,
    *,
    config: LlmProviderConfig,
    api_key: str,
    admin_username: str,
    memory_rows: list[tuple[Memory, str, str]],
    source: str,
) -> AiMemoryReviewRun:
    if not memory_rows:
        raise AiMemoryReviewError("请选择至少一条记忆。")
    if len(memory_rows) > MAX_REVIEW_MEMORIES:
        raise AiMemoryReviewError(f"一次最多整理 {MAX_REVIEW_MEMORIES} 条记忆。")

    memory_payload = [memory_summary(memory, category_name, user_name) for memory, user_name, category_name in memory_rows]
    user_ids = {memory.user_id for memory, _, _ in memory_rows}
    categories = [
        category.name
        for category in db.scalars(
            select(MemoryCategory)
            .where(MemoryCategory.deleted_at.is_(None))
            .where(MemoryCategory.user_id.in_(user_ids))
            .order_by(MemoryCategory.name)
        ).all()
    ]
    run = AiMemoryReviewRun(
        provider_config_id=config.id,
        admin_username=admin_username,
        status="running",
        source=source,
        selection_json={
            "memory_ids": [item["memory_id"] for item in memory_payload],
            "source": source,
        },
        request_summary={
            "memory_count": len(memory_payload),
            "titles": [item["title"] for item in memory_payload[:10]],
            "categories": sorted(set(item["category"] for item in memory_payload)),
        },
        prompt_preview=build_prompt_preview(memory_payload),
    )
    db.add(run)
    db.flush()

    try:
        prompt_injection = getattr(config, "review_prompt_injection", "") or ""
        result = chat_completion(
            config,
            api_key,
            build_review_messages(memory_payload, categories, prompt_injection),
            response_format={"type": "json_object"},
        )
        suggestions = normalize_suggestions(result.content, {item["memory_id"] for item in memory_payload})
        for suggestion in suggestions:
            db.add(
                AiMemoryReviewSuggestion(
                    run_id=run.id,
                    suggestion_type=suggestion["type"],
                    confidence=suggestion.get("confidence"),
                    reason=suggestion.get("reason"),
                    memory_ids=suggestion["memory_ids"],
                    target_memory_id=uuid.UUID(suggestion["target_memory_id"]) if suggestion.get("target_memory_id") else None,
                    proposed_json=suggestion.get("proposed") or {},
                    original_json={
                        "memories": [
                            item for item in memory_payload if item["memory_id"] in set(suggestion["memory_ids"])
                        ]
                    },
                )
            )
        usage = result.usage
        run.status = "completed"
        run.response_summary = {"suggestion_count": len(suggestions)}
        run.prompt_tokens = usage.get("prompt_tokens")
        run.completion_tokens = usage.get("completion_tokens")
        run.total_tokens = usage.get("total_tokens")
        run.completed_at = utcnow()
    except Exception as exc:
        run.status = "failed"
        run.error = str(exc)[:2000]
        run.completed_at = utcnow()
    db.add(run)
    db.commit()
    return run


def normalize_suggestions(content: str, allowed_memory_ids: set[str]) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AiMemoryReviewError("AI 返回不是合法 JSON。") from exc
    raw_items = parsed.get("suggestions") if isinstance(parsed, dict) else parsed
    if not isinstance(raw_items, list):
        raise AiMemoryReviewError("AI 返回 JSON 缺少 suggestions 数组。")

    suggestions: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        suggestion_type = str(raw.get("type") or raw.get("suggestion_type") or "").strip()
        if suggestion_type not in ALLOWED_SUGGESTION_TYPES:
            continue
        memory_ids = [str(item) for item in raw.get("memory_ids", []) if str(item) in allowed_memory_ids]
        if not memory_ids:
            target = str(raw.get("target_memory_id") or "").strip()
            if target in allowed_memory_ids:
                memory_ids = [target]
        if not memory_ids:
            continue
        target_memory_id = str(raw.get("target_memory_id") or memory_ids[0])
        if target_memory_id not in memory_ids:
            target_memory_id = memory_ids[0]
        confidence = raw.get("confidence")
        try:
            confidence = max(0.0, min(1.0, float(confidence))) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        proposed = raw.get("proposed") if isinstance(raw.get("proposed"), dict) else {}
        suggestions.append(
            {
                "type": suggestion_type,
                "confidence": confidence,
                "reason": str(raw.get("reason") or "").strip()[:2000],
                "memory_ids": memory_ids,
                "target_memory_id": target_memory_id,
                "proposed": proposed,
            }
        )
    return suggestions


def apply_suggestion(db: Session, suggestion: AiMemoryReviewSuggestion) -> None:
    if suggestion.status != "pending":
        raise AiMemoryReviewError("这个建议已经处理过。")

    ids = parse_memory_ids(suggestion.memory_ids)
    memories = db.scalars(
        select(Memory)
        .options(selectinload(Memory.attachments).defer(MemoryAttachment.image_bytes))
        .where(Memory.id.in_(ids))
    ).all()
    memory_by_id = {memory.id: memory for memory in memories}
    if not memory_by_id:
        raise AiMemoryReviewError("建议关联的记忆不存在。")

    proposed = suggestion.proposed_json or {}
    target_id = suggestion.target_memory_id or ids[0]
    target = memory_by_id.get(target_id) or next(iter(memory_by_id.values()))

    if suggestion.suggestion_type in {"rewrite", "move_category"}:
        update_memory_from_proposal(db, target, proposed)
    elif suggestion.suggestion_type == "merge":
        update_memory_from_proposal(db, target, proposed)
        for memory_id, memory in memory_by_id.items():
            if memory_id != target.id and memory.deleted_at is None:
                memory.deleted_at = utcnow()
                memory.updated_at = utcnow()
                db.add(memory)
    elif suggestion.suggestion_type == "soft_delete":
        for memory in memory_by_id.values():
            if memory.deleted_at is None:
                memory.deleted_at = utcnow()
                memory.updated_at = utcnow()
                db.add(memory)
    else:
        raise AiMemoryReviewError("未知建议类型。")

    suggestion.status = "applied"
    suggestion.applied_at = utcnow()
    suggestion.updated_at = utcnow()
    db.add(suggestion)
    db.commit()


def ignore_suggestion(db: Session, suggestion: AiMemoryReviewSuggestion) -> None:
    if suggestion.status != "pending":
        raise AiMemoryReviewError("这个建议已经处理过。")
    suggestion.status = "ignored"
    suggestion.ignored_at = utcnow()
    suggestion.updated_at = utcnow()
    db.add(suggestion)
    db.commit()


def update_memory_from_proposal(db: Session, memory: Memory, proposed: dict[str, Any]) -> None:
    title = str(proposed.get("title") or "").strip()
    content = str(proposed.get("content") or "").strip()
    category = str(proposed.get("category") or "").strip()
    if title:
        memory.title = title[:512]
    if content:
        memory.content = content
    if category:
        target_category, _ = get_or_create_category(db, memory.user_id, category)
        memory.category_id = target_category.id
    memory.search_text = build_search_text(memory.title, memory.content, attachment_search_text(active_attachments(memory)))
    memory.updated_at = utcnow()
    db.add(memory)


def parse_memory_ids(values: list[Any]) -> list[uuid.UUID]:
    ids: list[uuid.UUID] = []
    for value in values or []:
        try:
            ids.append(uuid.UUID(str(value)))
        except ValueError:
            continue
    return ids


def active_attachments(memory: Memory) -> list[MemoryAttachment]:
    return [attachment for attachment in getattr(memory, "attachments", []) if attachment.deleted_at is None]


def build_prompt_preview(memories: list[dict[str, Any]]) -> str:
    lines = [f"{item['title']} / {item['category']} / {item['memory_id']}" for item in memories[:20]]
    return "\n".join(lines)[:2000]
