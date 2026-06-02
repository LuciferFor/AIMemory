import json
import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, defer, selectinload

from aimemory.core.config import get_settings
from aimemory.db.session import SessionLocal
from aimemory.models.ai_memory_review import AiMemoryReviewRun, AiMemoryReviewSuggestion
from aimemory.models.llm_provider_config import LlmProviderConfig
from aimemory.models.memory import Memory
from aimemory.models.memory_attachment import MemoryAttachment
from aimemory.models.memory_category import MemoryCategory
from aimemory.models.user import User
from aimemory.repositories.memories import duplicate_memory_score, utcnow
from aimemory.repositories.memory_categories import get_or_create_category
from aimemory.services.ai_crypto import decrypt_secret
from aimemory.services.attachments import attachment_search_text
from aimemory.services.text import build_search_text
from aimemory.services.openai_compatible import chat_completion, token_usage_summary
from aimemory.services.query_analysis import strip_json_code_fence

logger = logging.getLogger(__name__)

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
AUTO_EXTRACT_MERGE_SOURCE = "auto_extract_merge"
AUTO_EXTRACT_MERGE_CANDIDATE_THRESHOLD = 0.62
AUTO_EXTRACT_MERGE_PER_MEMORY_LIMIT = 8
AUTO_EXTRACT_MERGE_MAX_MEMORIES = 30
AUTO_EXTRACT_MERGE_AUTO_APPLY_CONFIDENCE = 0.75
AUTO_EXTRACT_MERGE_ALLOWED_TYPES = {"rewrite", "merge", "move_category"}


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


def build_auto_merge_candidate_memory_ids(
    db: Session,
    *,
    user_id: uuid.UUID,
    agent_id: str,
    seed_memory_ids: list[uuid.UUID],
    threshold: float = AUTO_EXTRACT_MERGE_CANDIDATE_THRESHOLD,
    per_memory_limit: int = AUTO_EXTRACT_MERGE_PER_MEMORY_LIMIT,
    max_memories: int = AUTO_EXTRACT_MERGE_MAX_MEMORIES,
) -> tuple[list[uuid.UUID], int]:
    if not seed_memory_ids:
        return [], 0

    seeds = db.scalars(
        select(Memory)
        .options(defer(Memory.embedding))
        .where(
            Memory.id.in_(seed_memory_ids),
            Memory.user_id == user_id,
            Memory.agent_id == agent_id,
            Memory.deleted_at.is_(None),
        )
    ).all()
    if not seeds:
        return [], 0

    selected: list[uuid.UUID] = []
    selected_set: set[uuid.UUID] = set()
    candidate_count = 0

    def add_memory_id(memory_id: uuid.UUID) -> None:
        if memory_id not in selected_set and len(selected) < max_memories:
            selected.append(memory_id)
            selected_set.add(memory_id)

    seed_id_set = {seed.id for seed in seeds}
    for seed in seeds:
        scored_candidates: list[tuple[float, Memory]] = []
        candidates = db.scalars(
            select(Memory)
            .options(defer(Memory.embedding))
            .where(
                Memory.user_id == user_id,
                Memory.agent_id == agent_id,
                Memory.category_id == seed.category_id,
                Memory.deleted_at.is_(None),
                Memory.id.notin_(seed_id_set),
            )
            .order_by(Memory.updated_at.desc())
            .limit(300)
        ).all()
        for candidate in candidates:
            score = duplicate_memory_score(seed.title, seed.content, candidate.title, candidate.content)
            if score >= threshold:
                scored_candidates.append((score, candidate))

        if not scored_candidates:
            continue
        add_memory_id(seed.id)
        for _score, candidate in sorted(scored_candidates, key=lambda item: item[0], reverse=True)[:per_memory_limit]:
            candidate_count += 1
            add_memory_id(candidate.id)
            if len(selected) >= max_memories:
                break
        if len(selected) >= max_memories:
            break

    return selected, candidate_count


def load_review_memory_rows(db: Session, memory_ids: list[uuid.UUID]) -> list[tuple[Memory, str, str]]:
    if not memory_ids:
        return []
    rows = db.execute(
        select(Memory, User.name.label("user_name"), MemoryCategory.name.label("category_name"))
        .select_from(Memory)
        .join(User, User.id == Memory.user_id)
        .join(MemoryCategory, MemoryCategory.id == Memory.category_id)
        .where(Memory.id.in_(memory_ids), Memory.deleted_at.is_(None))
    ).all()
    order = {memory_id: index for index, memory_id in enumerate(memory_ids)}
    return sorted(rows, key=lambda row: order.get(row[0].id, len(order)))


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
        "只输出 JSON 对象，不要输出解释，不要使用 markdown 代码块，不要在 JSON 前后添加任何文字。"
        "如果没有整理建议，也必须输出 {\"suggestions\":[]}。"
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


def build_auto_merge_review_messages(
    memories: list[dict[str, Any]],
    categories: list[str],
    prompt_injection: str = "",
) -> list[dict[str, str]]:
    category_text = "\n".join(f"- {category}" for category in categories) or "- 其它"
    payload = {"categories": categories, "memories": memories}
    system_prompt = (
        "你是 AIMemory 自动相似记忆合并子任务。请只整理输入列表中重复、高度相似或互补的长期记忆。"
        "不要回答用户，不要补充输入中没有的新事实。"
        "只允许输出 rewrite、merge、move_category 三类建议；不要输出 soft_delete。"
        "rewrite 用于压缩单条记忆；merge 用于把重复或互补记忆合并到信息最完整的一条；"
        "move_category 用于明显分类错误。"
        "如果没有明确重复或无需优化，输出 {\"suggestions\":[]}。"
        "标题和正文必须保持第三方视角，例如“用户偏好……”“助手应……”。"
        "优先使用已有分类；确实没有合适分类时可以给出新分类名。"
        "只输出 JSON 对象，不要输出解释，不要使用 markdown 代码块。"
        "\n\nJSON 输出格式："
        "{\"suggestions\":[{\"type\":\"rewrite|merge|move_category\","
        "\"confidence\":0.0,\"reason\":\"原因\",\"memory_ids\":[\"uuid\"],"
        "\"target_memory_id\":\"uuid\",\"proposed\":{\"title\":\"\",\"content\":\"\",\"category\":\"\"}}]}"
        f"\n\n已有分类：\n{category_text}"
    )
    system_sections = []
    cleaned_prompt_injection = str(prompt_injection or "").strip()
    if cleaned_prompt_injection:
        system_sections.append(f"管理员整理前置注入提示：\n{cleaned_prompt_injection}")
    system_sections.append(system_prompt)
    return [
        {"role": "system", "content": "\n\n".join(system_sections)},
        {
            "role": "user",
            "content": "请合并优化以下相似 AIMemory 记忆并输出 json：\n"
            + json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def create_auto_extract_merge_review_run(
    db: Session,
    *,
    config: LlmProviderConfig,
    api_key: str,
    memory_rows: list[tuple[Memory, str, str]],
) -> AiMemoryReviewRun:
    if not memory_rows:
        raise AiMemoryReviewError("没有相似记忆需要自动整理。")
    if len(memory_rows) > AUTO_EXTRACT_MERGE_MAX_MEMORIES:
        memory_rows = memory_rows[:AUTO_EXTRACT_MERGE_MAX_MEMORIES]

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
        admin_username="system",
        status="running",
        source=AUTO_EXTRACT_MERGE_SOURCE,
        selection_json={
            "memory_ids": [item["memory_id"] for item in memory_payload],
            "source": AUTO_EXTRACT_MERGE_SOURCE,
        },
        request_summary={
            "memory_count": len(memory_payload),
            "titles": [item["title"] for item in memory_payload[:10]],
            "categories": sorted(set(item["category"] for item in memory_payload)),
            "auto_apply_confidence": AUTO_EXTRACT_MERGE_AUTO_APPLY_CONFIDENCE,
        },
        prompt_preview=build_prompt_preview(memory_payload),
    )
    db.add(run)
    db.flush()

    result = None
    usage: dict[str, int] = {}
    try:
        prompt_injection = getattr(config, "review_prompt_injection", "") or ""
        result = chat_completion(
            config,
            api_key,
            build_auto_merge_review_messages(memory_payload, categories, prompt_injection),
            response_format={"type": "json_object"},
        )
        usage = token_usage_summary(result.usage)
        run.prompt_tokens = usage.get("prompt_tokens")
        run.completion_tokens = usage.get("completion_tokens")
        run.total_tokens = usage.get("total_tokens")
        suggestions = normalize_suggestions(result.content, {item["memory_id"] for item in memory_payload})
        now = utcnow()
        created_suggestions: list[AiMemoryReviewSuggestion] = []
        ignored_count = 0
        for suggestion in suggestions:
            suggestion_type = suggestion["type"]
            ignored = suggestion_type not in AUTO_EXTRACT_MERGE_ALLOWED_TYPES
            review_suggestion = AiMemoryReviewSuggestion(
                run_id=run.id,
                suggestion_type=suggestion_type,
                status="ignored" if ignored else "pending",
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
                ignored_at=now if ignored else None,
                error="自动相似合并不允许单独 soft_delete。" if ignored and suggestion_type == "soft_delete" else None,
            )
            if ignored:
                ignored_count += 1
            db.add(review_suggestion)
            created_suggestions.append(review_suggestion)

        run.status = "completed"
        run.response_summary = {
            "suggestion_count": len(created_suggestions),
            "ai_total_tokens": usage.get("total_tokens"),
            "ai_usage": dict(usage),
            "auto_applied": 0,
            "auto_pending": sum(1 for item in created_suggestions if item.status == "pending"),
            "auto_ignored": ignored_count,
        }
        run.completed_at = utcnow()
        db.add(run)
        db.commit()
        apply_auto_merge_suggestions(db, run)
    except Exception as exc:
        run.status = "failed"
        run.error = str(exc)[:2000]
        if result is not None:
            run.response_summary = {
                "raw_preview": preview_ai_output(result.content),
                "raw_chars": len(str(result.content or "")),
                "ai_total_tokens": usage.get("total_tokens"),
                "ai_usage": dict(usage),
            }
        run.completed_at = utcnow()
        db.add(run)
        db.commit()
    return run


def apply_auto_merge_suggestions(db: Session, run: AiMemoryReviewRun) -> dict[str, Any]:
    suggestions = db.scalars(
        select(AiMemoryReviewSuggestion)
        .where(AiMemoryReviewSuggestion.run_id == run.id)
        .order_by(AiMemoryReviewSuggestion.created_at)
    ).all()
    applied = 0
    pending = 0
    ignored = 0
    errors: list[str] = []
    for suggestion in suggestions:
        if suggestion.status == "ignored":
            ignored += 1
            continue
        if suggestion.status != "pending":
            continue
        if suggestion.suggestion_type not in AUTO_EXTRACT_MERGE_ALLOWED_TYPES:
            suggestion.status = "ignored"
            suggestion.ignored_at = utcnow()
            suggestion.updated_at = utcnow()
            suggestion.error = "自动相似合并不允许这个建议类型。"
            db.add(suggestion)
            db.commit()
            ignored += 1
            continue
        confidence = float(suggestion.confidence or 0.0)
        if confidence < AUTO_EXTRACT_MERGE_AUTO_APPLY_CONFIDENCE:
            pending += 1
            continue
        try:
            apply_suggestion(db, suggestion)
            applied += 1
        except AiMemoryReviewError as exc:
            db.rollback()
            pending += 1
            errors.append(str(exc)[:300])

    run.response_summary = {
        **(run.response_summary or {}),
        "auto_applied": applied,
        "auto_pending": pending,
        "auto_ignored": ignored,
        "auto_errors": errors[:5],
    }
    run.updated_at = utcnow()
    db.add(run)
    db.commit()
    return run.response_summary


def run_auto_merge_review_for_extracted_memories(
    user_id: str,
    agent_id: str,
    seed_memory_ids: list[str],
) -> None:
    try:
        try:
            parsed_user_id = uuid.UUID(str(user_id))
            parsed_memory_ids = [uuid.UUID(str(memory_id)) for memory_id in seed_memory_ids]
        except ValueError:
            return
        with SessionLocal() as db:
            config = get_llm_config(db)
            if config is None or not getattr(config, "enabled", False) or not getattr(config, "encrypted_api_key", None):
                return
            api_key = decrypt_secret(config.encrypted_api_key, get_settings().ai_config_encryption_secret)
            review_memory_ids, candidate_count = build_auto_merge_candidate_memory_ids(
                db,
                user_id=parsed_user_id,
                agent_id=agent_id,
                seed_memory_ids=parsed_memory_ids,
            )
            if candidate_count <= 0 or len(review_memory_ids) < 2:
                return
            memory_rows = load_review_memory_rows(db, review_memory_ids)
            if len(memory_rows) < 2:
                return
            create_auto_extract_merge_review_run(db, config=config, api_key=api_key, memory_rows=memory_rows)
    except Exception as exc:
        logger.warning(
            "memory.extract.auto_merge_failed",
            extra={
                "event": "memory.extract.auto_merge_failed",
                "user_id": user_id,
                "agent_id": agent_id,
                "reason": str(exc)[:300],
            },
        )


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

    result = None
    usage: dict[str, int] = {}
    try:
        prompt_injection = getattr(config, "review_prompt_injection", "") or ""
        result = chat_completion(
            config,
            api_key,
            build_review_messages(memory_payload, categories, prompt_injection),
            response_format={"type": "json_object"},
        )
        usage = token_usage_summary(result.usage)
        run.prompt_tokens = usage.get("prompt_tokens")
        run.completion_tokens = usage.get("completion_tokens")
        run.total_tokens = usage.get("total_tokens")
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
        run.status = "completed"
        run.response_summary = {
            "suggestion_count": len(suggestions),
            "ai_total_tokens": usage.get("total_tokens"),
            "ai_usage": dict(usage),
        }
        run.completed_at = utcnow()
    except Exception as exc:
        run.status = "failed"
        run.error = str(exc)[:2000]
        if result is not None:
            run.response_summary = {
                "raw_preview": preview_ai_output(result.content),
                "raw_chars": len(str(result.content or "")),
                "ai_total_tokens": usage.get("total_tokens"),
                "ai_usage": dict(usage),
            }
        run.completed_at = utcnow()
    db.add(run)
    db.commit()
    return run


def normalize_suggestions(content: str, allowed_memory_ids: set[str]) -> list[dict[str, Any]]:
    normalized_content = strip_json_code_fence(content)
    try:
        parsed = json.loads(normalized_content)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(extract_json_candidate(normalized_content))
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


def extract_json_candidate(value: str) -> str:
    text_value = str(value or "").strip()
    if not text_value:
        return text_value
    if text_value[0] in "[{":
        return text_value
    candidates: list[str] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text_value.find(opener)
        if start < 0:
            continue
        extracted = balanced_json_slice(text_value, start, opener, closer)
        if extracted:
            candidates.append(extracted)
    return candidates[0] if candidates else text_value


def balanced_json_slice(value: str, start: int, opener: str, closer: str) -> str:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
    return ""


def preview_ai_output(value: str) -> str:
    return " ".join(str(value or "").split())[:500]


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
