import re
import unicodedata

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower().strip()
    return _SPACE_RE.sub(" ", normalized)


def split_query_terms(value: str) -> list[str]:
    normalized = normalize_query(value)
    terms: set[str] = set(_WORD_RE.findall(normalized))
    for cjk_group in _CJK_RE.findall(normalized):
        terms.add(cjk_group)
        terms.update(cjk_group[index : index + 2] for index in range(max(len(cjk_group) - 1, 0)))
    return sorted(term for term in terms if term)


def build_search_text(title: str, content: str) -> str:
    return normalize_query(f"{title}\n{content}")


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
