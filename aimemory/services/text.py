import re
import unicodedata

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_SPACE_RE = re.compile(r"\s+")
_NUMERIC_RE = re.compile(r"\d+")


def normalize_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower().strip()
    return _SPACE_RE.sub(" ", normalized)


def is_numeric_term(value: str) -> bool:
    return bool(_NUMERIC_RE.fullmatch(normalize_query(value)))


def raw_query_terms(value: str) -> list[str]:
    normalized = normalize_query(value)
    terms: set[str] = set(_WORD_RE.findall(normalized))
    for cjk_group in _CJK_RE.findall(normalized):
        terms.add(cjk_group)
        terms.update(cjk_group[index : index + 2] for index in range(max(len(cjk_group) - 1, 0)))
    return sorted(term for term in terms if term)


def split_query_terms(value: str) -> list[str]:
    return [term for term in raw_query_terms(value) if not is_numeric_term(term)]


def filter_query_terms(value: str, stopwords: set[str]) -> tuple[list[str], list[str]]:
    normalized_stopwords = {normalize_query(term) for term in stopwords if normalize_query(term)}
    effective_terms: list[str] = []
    ignored_terms: list[str] = []
    for term in raw_query_terms(value):
        if is_numeric_term(term) or term in normalized_stopwords:
            ignored_terms.append(term)
        else:
            effective_terms.append(term)
    return effective_terms, ignored_terms


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
