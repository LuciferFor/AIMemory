import json
import time
from dataclasses import dataclass, field
from typing import Any

from aimemory.models.llm_provider_config import LlmProviderConfig
from aimemory.services.openai_compatible import chat_completion, token_usage_summary
from aimemory.services.text import ignored_term_reason, normalize_query

QUERY_ANALYSIS_WEAK_TERMS = {
    "老婆",
    "老公",
    "宝贝",
    "亲爱的",
    "帮我",
    "给我",
    "请你",
    "请求",
    "生成",
    "图片",
    "画面",
    "换成",
    "然后",
    "一点",
    "一些",
    "更好",
    "稍微",
    "比较",
    "这个",
    "那个",
}
MAX_QUERY_KEYWORDS = 12
MAX_QUERY_ANALYSIS_CHARS = 2000


@dataclass(frozen=True)
class QueryAnalysis:
    intent_summary: str = ""
    keywords: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)


def build_query_analysis_messages(query: str, category: str, agent_id: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "你是 AIMemory 的检索请求提词器，只负责把用户当前请求压缩成检索关键词。"
                "不要查询记忆，不要回答用户，不要扩写。"
                "只输出 JSON 对象，格式为："
                "{\"intent_summary\":\"一句话概括请求目的\","
                "\"keywords\":[\"真正适合检索长期记忆的词语或短语\"],"
                "\"negative_keywords\":[\"用户明确不要的内容\"]}。"
                "分类只是后续检索范围提示，不是判断请求是否有效的条件；不要因为分类看起来不匹配就返回空关键词。"
                "keywords 只保留名词、对象、偏好、属性、目标短语、稳定约束。"
                "绘图、角色、姿势、服装、身体、画面修改等请求也必须提取视觉属性关键词。"
                "不要输出称呼、语气词、连接词、动作泛词、程度泛词，例如：老婆、宝贝、换成、然后、一点、一些、更好、帮我、给我、生成、图片。"
                "只有请求完全是寒暄、纯噪声、或没有任何可复用检索对象/属性时，keywords 才能为空。"
                "示例：'老婆换成黑丝,然后腿粗一点,身材更好一些' 应输出 "
                "{\"intent_summary\":\"生成/修改兔女郎或人物图片偏好\","
                "\"keywords\":[\"黑丝\",\"腿更粗\",\"身材更好\"],\"negative_keywords\":[]}。"
                "中英文都可以；中文优先使用 2 到 8 字短语，英文优先使用 2 到 4 个词短语。"
            ),
        },
        {
            "role": "user",
            "content": f"分类: {category}\n智能体: {agent_id}\n当前请求:\n{str(query or '')[:MAX_QUERY_ANALYSIS_CHARS]}",
        },
    ]


def analyze_memory_query(
    config: LlmProviderConfig,
    api_key: str,
    *,
    query: str,
    category: str,
    agent_id: str,
) -> QueryAnalysis:
    start = time.perf_counter()
    result = chat_completion(
        config,
        api_key,
        build_query_analysis_messages(query, category, agent_id),
        response_format={"type": "json_object"},
        max_tokens=int(getattr(config, "query_analysis_max_output_tokens", 256) or 256),
        temperature=0.0,
        timeout_ms=int(getattr(config, "query_analysis_timeout_ms", 3000) or 3000),
    )
    parsed = parse_query_analysis(result.content)
    return QueryAnalysis(
        intent_summary=parsed.intent_summary,
        keywords=parsed.keywords,
        negative_keywords=parsed.negative_keywords,
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
        usage=token_usage_summary(result.usage),
    )


def parse_query_analysis(content: str) -> QueryAnalysis:
    try:
        parsed = json.loads(strip_json_code_fence(content))
    except json.JSONDecodeError as exc:
        raise ValueError("AI 提词返回不是合法 JSON。") from exc
    if not isinstance(parsed, dict):
        raise ValueError("AI 提词返回必须是 JSON 对象。")
    keywords = normalize_keyword_list(parsed.get("keywords"))
    negative_keywords = normalize_keyword_list(parsed.get("negative_keywords"))
    return QueryAnalysis(
        intent_summary=str(parsed.get("intent_summary") or "").strip()[:200],
        keywords=keywords[:MAX_QUERY_KEYWORDS],
        negative_keywords=negative_keywords[:MAX_QUERY_KEYWORDS],
    )


def strip_json_code_fence(value: str) -> str:
    text_value = str(value or "").strip()
    if text_value.startswith("```"):
        lines = text_value.splitlines()
        if lines and lines[0].strip().lower() in {"```json", "```"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text_value


def normalize_keyword_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    keywords: list[str] = []
    seen: set[str] = set()
    for raw in value:
        keyword = normalize_query(str(raw or ""))
        if not keyword or keyword in seen:
            continue
        keywords.append(keyword)
        seen.add(keyword)
    return keywords


def effective_terms_from_ai_keywords(keywords: list[str], stopwords: set[str]) -> tuple[list[str], list[str]]:
    normalized_stopwords = {normalize_query(term) for term in stopwords if normalize_query(term)}
    terms: list[str] = []
    ignored: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        term = normalize_query(keyword)
        if not term or term in seen:
            continue
        seen.add(term)
        reason = ai_keyword_ignore_reason(term, normalized_stopwords)
        if reason:
            ignored.append(f"{term}:{reason}")
            continue
        terms.append(term)
    return terms, ignored


def ai_keyword_ignore_reason(term: str, stopwords: set[str]) -> str | None:
    if term in stopwords:
        return "停用词"
    if any(word in stopwords for word in term.split()):
        return "停用词"
    if term in QUERY_ANALYSIS_WEAK_TERMS:
        return "弱检索词"
    reason = ignored_term_reason(term)
    if reason:
        return reason
    return None
