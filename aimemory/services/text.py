import logging
import re
import unicodedata

import jieba
from wordfreq import zipf_frequency

jieba.setLogLevel(logging.WARNING)

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_SPACE_RE = re.compile(r"\s+")
_NUMERIC_RE = re.compile(r"\d+")
_ENGLISH_RE = re.compile(r"[a-z]+")
_ALNUM_RE = re.compile(r"[a-z0-9_]+")

MIN_ENGLISH_ZIPF_FREQUENCY = 2.5
TECHNICAL_QUERY_STOPWORDS = {
    "api",
    "arg",
    "args",
    "bool",
    "char",
    "class",
    "const",
    "dict",
    "enum",
    "false",
    "float",
    "func",
    "http",
    "https",
    "int",
    "json",
    "key",
    "list",
    "none",
    "null",
    "obj",
    "param",
    "params",
    "sql",
    "str",
    "string",
    "token",
    "true",
    "tuple",
    "url",
    "uuid",
    "var",
}


def normalize_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower().strip()
    return _SPACE_RE.sub(" ", normalized)


def is_numeric_term(value: str) -> bool:
    return bool(_NUMERIC_RE.fullmatch(normalize_query(value)))


def raw_query_terms(value: str) -> list[str]:
    normalized = normalize_query(value)
    terms: list[str] = []
    seen: set[str] = set()

    def add_term(term: str) -> None:
        if term and term not in seen:
            terms.append(term)
            seen.add(term)

    for cjk_group in _CJK_RE.findall(normalized):
        for segment in jieba.lcut(cjk_group):
            add_term(segment.strip())
    for term in _WORD_RE.findall(normalized):
        add_term(term)
    return terms


def split_query_terms(value: str) -> list[str]:
    terms, _ = filter_query_terms(value, set())
    return terms


def filter_query_terms(value: str, stopwords: set[str]) -> tuple[list[str], list[str]]:
    normalized_stopwords = {normalize_query(term) for term in stopwords if normalize_query(term)}
    effective_terms: list[str] = []
    ignored_terms: list[str] = []
    for term in raw_query_terms(value):
        ignore_reason = ignored_term_reason(term)
        if ignore_reason is not None:
            ignored_terms.append(format_ignored_term(term, ignore_reason))
        elif term in normalized_stopwords:
            ignored_terms.append(format_ignored_term(term, "停用词"))
        else:
            effective_terms.append(term)
    return effective_terms, ignored_terms


def ignored_term_reason(term: str) -> str | None:
    normalized = normalize_query(term)
    if is_numeric_term(normalized):
        return "数字"
    if _CJK_RE.fullmatch(normalized):
        if len(normalized) < 2:
            return "中文单字"
        return None
    if _ALNUM_RE.fullmatch(normalized) and not _ENGLISH_RE.fullmatch(normalized):
        return "英文数字混合"
    if _ENGLISH_RE.fullmatch(normalized):
        if len(normalized) < 3:
            return "短英文"
        if normalized in TECHNICAL_QUERY_STOPWORDS:
            return "技术词"
        if zipf_frequency(normalized, "en") < MIN_ENGLISH_ZIPF_FREQUENCY:
            return "非英文词典"
        return None
    return "无效词"


def format_ignored_term(term: str, reason: str) -> str:
    return f"{term}:{reason}"


def build_search_text(title: str, content: str, *extra_parts: str) -> str:
    return normalize_query("\n".join([title, content, *extra_parts]))


def weighted_score(
    keyword: float,
    fuzzy: float,
    term: float = 0.0,
    title: float = 0.0,
    exact: float = 0.0,
    metadata: float = 0.0,
    recency: float = 0.0,
) -> float:
    capped_keyword = min(max(keyword, 0.0), 1.0)
    capped_fuzzy = min(max(fuzzy, 0.0), 1.0)
    capped_term = min(max(term, 0.0), 1.0)
    capped_title = min(max(title, 0.0), 1.0)
    capped_exact = min(max(exact, 0.0), 1.0)
    capped_metadata = min(max(metadata, 0.0), 1.0)
    capped_recency = min(max(recency, 0.0), 1.0)
    return min(
        1.0,
        (0.30 * capped_keyword)
        + (0.20 * capped_fuzzy)
        + (0.20 * capped_term)
        + (0.15 * capped_title)
        + (0.10 * capped_exact)
        + (0.03 * capped_metadata)
        + (0.02 * capped_recency),
    )
