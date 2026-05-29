from aimemory.services.text import normalize_query, split_query_terms, weighted_score


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


def test_weighted_score_caps_score_parts() -> None:
    assert weighted_score(2.0, 2.0, -1.0) == 0.9
