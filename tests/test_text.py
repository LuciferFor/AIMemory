from aimemory.services.text import filter_query_terms, is_numeric_term, normalize_query, split_query_terms, weighted_score


def test_normalize_query_collapses_space_and_width() -> None:
    assert normalize_query("  Hello\nWORLD  ") == "hello world"
    assert normalize_query("ＡＩ Memory") == "ai memory"


def test_split_query_terms_handles_english_and_cjk() -> None:
    terms = split_query_terms("AI 喜欢短回答 memory")

    assert "ai" in terms
    assert "memory" in terms
    assert "喜欢短回答" in terms
    assert "喜欢" in terms
    assert "回答" in terms


def test_split_query_terms_ignores_pure_numbers() -> None:
    terms = split_query_terms("2026-05-30 lucifer skill OpenClaw key gpt4")

    assert "2026" not in terms
    assert "05" not in terms
    assert "30" not in terms
    assert "gpt4" in terms
    assert "key" in terms
    assert is_numeric_term("２０２６") is True


def test_filter_query_terms_uses_stopwords() -> None:
    terms, ignored = filter_query_terms("2026-05-30 lucifer skill OpenClaw key", {"lucifer", "skill", "openclaw"})

    assert terms == ["key"]
    assert ignored == ["05", "2026", "30", "lucifer", "openclaw", "skill"]


def test_filter_query_terms_can_return_no_effective_terms() -> None:
    terms, ignored = filter_query_terms("2026 lucifer skill", {"lucifer", "skill"})

    assert terms == []
    assert ignored == ["2026", "lucifer", "skill"]


def test_weighted_score_caps_score_parts() -> None:
    assert weighted_score(keyword=2.0, fuzzy=2.0, term=2.0, title=2.0, exact=2.0, metadata=2.0, recency=2.0) == 1.0


def test_weighted_score_prefers_title_and_exact_matches() -> None:
    loose_score = weighted_score(keyword=0.0, fuzzy=0.2, term=0.2)
    title_score = weighted_score(keyword=0.0, fuzzy=0.2, term=0.2, title=1.0, exact=1.0)

    assert title_score > loose_score
