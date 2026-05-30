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
MIN_ENGLISH_PHRASE_WORDS = 2
MAX_ENGLISH_PHRASE_WORDS = 4
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
ENGLISH_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "with",
}
WEAK_CJK_QUERY_STOPWORDS = {
    "不要",
    "不是",
    "不能",
    "可以",
    "需要",
    "应该",
    "这个",
    "那个",
    "这些",
    "那些",
    "什么",
    "怎么",
    "现在",
    "目前",
    "之前",
    "之后",
    "时候",
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
    for phrase in english_phrases(normalized):
        add_term(phrase)
    return terms


def split_query_terms(value: str) -> list[str]:
    terms, _ = filter_query_terms(value, set())
    return terms


def filter_query_terms(value: str, stopwords: set[str]) -> tuple[list[str], list[str]]:
    normalized_stopwords = {normalize_query(term) for term in stopwords if normalize_query(term)}
    effective_terms: list[str] = []
    ignored_terms: list[str] = []
    for term in raw_query_terms(value):
        if " " in term and any(word in normalized_stopwords for word in term.split()):
            continue
        if term in normalized_stopwords:
            ignored_terms.append(format_ignored_term(term, "停用词"))
            continue
        ignore_reason = ignored_term_reason(term)
        if ignore_reason is not None:
            ignored_terms.append(format_ignored_term(term, ignore_reason))
        else:
            effective_terms.append(term)
    return effective_terms, ignored_terms


def english_phrases(value: str) -> list[str]:
    text = normalize_query(value)
    phrases: list[str] = []
    seen: set[str] = set()
    current_run: list[str] = []
    previous_end = 0

    def flush_run() -> None:
        nonlocal current_run
        for size in range(MIN_ENGLISH_PHRASE_WORDS, MAX_ENGLISH_PHRASE_WORDS + 1):
            for index in range(0, len(current_run) - size + 1):
                phrase_words = current_run[index : index + size]
                if any(english_phrase_word_reason(word) is not None for word in phrase_words):
                    continue
                phrase = " ".join(phrase_words)
                if phrase not in seen:
                    phrases.append(phrase)
                    seen.add(phrase)
        current_run = []

    for match in _WORD_RE.finditer(text):
        token = match.group(0)
        gap = text[previous_end : match.start()]
        if _CJK_RE.search(gap):
            flush_run()
        if _ENGLISH_RE.fullmatch(token):
            current_run.append(token)
        else:
            flush_run()
        previous_end = match.end()
    flush_run()
    return phrases


def ignored_term_reason(term: str) -> str | None:
    normalized = normalize_query(term)
    if " " in normalized:
        return ignored_english_phrase_reason(normalized)
    if is_numeric_term(normalized):
        return "数字"
    if _CJK_RE.fullmatch(normalized):
        if len(normalized) < 2:
            return "中文单字"
        if normalized in WEAK_CJK_QUERY_STOPWORDS:
            return "弱语义词"
        return None
    if _ALNUM_RE.fullmatch(normalized) and not _ENGLISH_RE.fullmatch(normalized):
        return "英文数字混合"
    if _ENGLISH_RE.fullmatch(normalized):
        if len(normalized) < 3:
            return "短英文"
        if normalized in TECHNICAL_QUERY_STOPWORDS:
            return "技术词"
        if normalized in ENGLISH_FUNCTION_WORDS:
            return "功能词"
        if zipf_frequency(normalized, "en") < MIN_ENGLISH_ZIPF_FREQUENCY:
            return "非英文词典"
        return "英文单词"
    return "无效词"


def ignored_english_phrase_reason(term: str) -> str | None:
    words = term.split()
    if len(words) < MIN_ENGLISH_PHRASE_WORDS:
        return "英文单词"
    if len(words) > MAX_ENGLISH_PHRASE_WORDS:
        return "英文短语过长"
    for word in words:
        reason = english_phrase_word_reason(word)
        if reason is not None:
            return reason
    return None


def english_phrase_word_reason(word: str) -> str | None:
    normalized = normalize_query(word)
    if is_numeric_term(normalized):
        return "数字"
    if _ALNUM_RE.fullmatch(normalized) and not _ENGLISH_RE.fullmatch(normalized):
        return "英文数字混合"
    if not _ENGLISH_RE.fullmatch(normalized):
        return "无效词"
    if len(normalized) < 3:
        return "短英文"
    if normalized in TECHNICAL_QUERY_STOPWORDS:
        return "技术词"
    if normalized in ENGLISH_FUNCTION_WORDS:
        return "功能词"
    if zipf_frequency(normalized, "en") < MIN_ENGLISH_ZIPF_FREQUENCY:
        return "非英文词典"
    return None


def format_ignored_term(term: str, reason: str) -> str:
    return f"{term}:{reason}"


def build_search_text(title: str, content: str, *extra_parts: str) -> str:
    return normalize_query("\n".join([title, content, *extra_parts]))


def weighted_score(
    keyword: float,
    fuzzy: float,
    term: float = 0.0,
    title: float = 0.0,
    content: float = 0.0,
    exact: float = 0.0,
    metadata: float = 0.0,
    recency: float = 0.0,
) -> float:
    capped_keyword = min(max(keyword, 0.0), 1.0)
    capped_fuzzy = min(max(fuzzy, 0.0), 1.0)
    capped_term = min(max(term, 0.0), 1.0)
    capped_title = min(max(title, 0.0), 1.0)
    capped_content = min(max(content, 0.0), 1.0)
    capped_exact = min(max(exact, 0.0), 1.0)
    capped_metadata = min(max(metadata, 0.0), 1.0)
    capped_recency = min(max(recency, 0.0), 1.0)
    return min(
        1.0,
        (0.24 * capped_title)
        + (0.24 * capped_term)
        + (0.18 * capped_content)
        + (0.16 * capped_metadata)
        + (0.08 * capped_exact)
        + (0.05 * capped_keyword)
        + (0.03 * capped_fuzzy)
        + (0.02 * capped_recency),
    )
