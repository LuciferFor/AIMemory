import json
import time
from dataclasses import dataclass
from typing import Any

from aimemory.models.llm_provider_config import LlmProviderConfig
from aimemory.repositories.memory_categories import CategorySummary, normalize_category_name
from aimemory.services.openai_compatible import chat_completion
from aimemory.services.query_analysis import strip_json_code_fence
from aimemory.services.text import normalize_query


MAX_CATEGORY_ANALYSIS_CHARS = 4000
MAX_CATEGORY_REASON_CHARS = 200
DEFAULT_CATEGORY_ANALYSIS_TOKENS = 256
DEFAULT_CATEGORY_ANALYSIS_TIMEOUT_MS = 3000


@dataclass(frozen=True)
class CategoryAnalysis:
    category: str = ""
    matched_existing: bool = False
    confidence: float = 0.0
    reason: str = ""
    duration_ms: float = 0.0


def build_category_analysis_messages(
    *,
    text: str,
    operation: str,
    categories: list[CategorySummary],
) -> list[dict[str, str]]:
    category_text = "\n".join(
        f"- {category.name}{f'：{category.description}' if category.description else ''}"
        for category in categories
    )
    return [
        {
            "role": "system",
            "content": (
                "你是 AIMemory 的事务分类器，只负责把当前内容归入一个长期记忆分类。"
                "不要回答用户，不要提取关键词，不要查询记忆。"
                "优先复用已有分类；只有写入记忆且已有分类明显都不合适时，才允许创建新的简短中文分类。"
                "查询记忆时只能选择已有分类；如果无法从已有分类判断，category 输出 null。"
                "分类应是日常事务类别，例如 技术问题、回答风格、工作流程、自动化、角色偏好、图片偏好、其它；"
                "不要把一次性关键词、错别字、工具名或普通名词直接当分类。"
                "遇到 OneBot、OpenClaw、AIMemory、接口、插件、数据库、部署、日志、报错、连接问题，应优先选择最接近的技术/流程/资料/自动化类已有分类。"
                "只输出 JSON 对象，格式为："
                "{\"category\":\"分类名或 null\",\"matched_existing\":true,"
                "\"confidence\":0.0,\"reason\":\"简短原因\"}。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"操作类型: {operation}\n"
                f"已有分类:\n{category_text or '无'}\n\n"
                f"当前内容:\n{str(text or '')[:MAX_CATEGORY_ANALYSIS_CHARS]}"
            ),
        },
    ]


def analyze_memory_category(
    config: LlmProviderConfig,
    api_key: str,
    *,
    text: str,
    operation: str,
    categories: list[CategorySummary],
) -> CategoryAnalysis:
    start = time.perf_counter()
    result = chat_completion(
        config,
        api_key,
        build_category_analysis_messages(text=text, operation=operation, categories=categories),
        response_format={"type": "json_object"},
        max_tokens=int(getattr(config, "query_analysis_max_output_tokens", DEFAULT_CATEGORY_ANALYSIS_TOKENS) or DEFAULT_CATEGORY_ANALYSIS_TOKENS),
        temperature=0.0,
        timeout_ms=int(getattr(config, "query_analysis_timeout_ms", DEFAULT_CATEGORY_ANALYSIS_TIMEOUT_MS) or DEFAULT_CATEGORY_ANALYSIS_TIMEOUT_MS),
    )
    parsed = parse_category_analysis(result.content, categories)
    return CategoryAnalysis(
        category=parsed.category,
        matched_existing=parsed.matched_existing,
        confidence=parsed.confidence,
        reason=parsed.reason,
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
    )


def parse_category_analysis(content: str, categories: list[CategorySummary]) -> CategoryAnalysis:
    try:
        parsed = json.loads(strip_json_code_fence(content))
    except json.JSONDecodeError as exc:
        raise ValueError("AI 分类返回不是合法 JSON。") from exc
    if not isinstance(parsed, dict):
        raise ValueError("AI 分类返回必须是 JSON 对象。")

    selected = normalize_category_output(parsed.get("category"))
    known_by_normalized = {
        normalize_category_name(category.name): category.name
        for category in categories
        if normalize_category_name(category.name)
    }
    normalized_selected = normalize_category_name(selected)
    matched_name = known_by_normalized.get(normalized_selected, "")
    matched_existing = bool(matched_name)

    confidence = parse_confidence(parsed.get("confidence"))
    reason = " ".join(str(parsed.get("reason") or "").split())[:MAX_CATEGORY_REASON_CHARS]
    return CategoryAnalysis(
        category=matched_name or selected,
        matched_existing=matched_existing,
        confidence=confidence,
        reason=reason,
    )


def normalize_category_output(value: Any) -> str:
    if value is None:
        return ""
    text = normalize_query(str(value or ""))
    if text.lower() in {"null", "none", "无", "未知", "不确定"}:
        return ""
    return text[:128]


def parse_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, number)), 3)
