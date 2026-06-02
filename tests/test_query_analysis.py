import json
from types import SimpleNamespace

import pytest

from aimemory.services import query_analysis as qa


def test_parse_query_analysis_accepts_json_output() -> None:
    parsed = qa.parse_query_analysis(
        json.dumps(
            {
                "intent_summary": "生成兔女郎图片",
                "keywords": ["黑丝", "兔女郎", "腿更粗", "身材更好"],
                "negative_keywords": ["白丝"],
            },
            ensure_ascii=False,
        )
    )

    assert parsed.intent_summary == "生成兔女郎图片"
    assert parsed.keywords == ["黑丝", "兔女郎", "腿更粗", "身材更好"]
    assert parsed.negative_keywords == ["白丝"]


def test_effective_terms_from_ai_keywords_filters_weak_words() -> None:
    terms, ignored = qa.effective_terms_from_ai_keywords(
        ["老婆", "换成", "一点", "一些", "黑丝", "兔女郎", "腿更粗", "身材更好"],
        set(),
    )

    assert terms == ["黑丝", "兔女郎", "腿更粗", "身材更好"]
    assert "老婆:弱检索词" in ignored
    assert "换成:弱检索词" in ignored
    assert "一点:弱检索词" in ignored
    assert "一些:弱检索词" in ignored


def test_effective_terms_from_ai_keywords_keeps_english_phrases_only() -> None:
    terms, ignored = qa.effective_terms_from_ai_keywords(["fantasy", "dark armor", "api", "gpt4"], set())

    assert terms == ["dark armor"]
    assert "fantasy:英文单词" in ignored
    assert "api:技术词" in ignored
    assert "gpt4:英文数字混合" in ignored


def test_analyze_memory_query_uses_small_overrides(monkeypatch) -> None:
    captured = {}

    def fake_chat_completion(config, api_key, messages, **kwargs):
        captured["api_key"] = api_key
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            content='{"intent_summary":"生成兔女郎图片","keywords":["黑丝"],"negative_keywords":[]}',
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    monkeypatch.setattr(qa, "chat_completion", fake_chat_completion)
    config = SimpleNamespace(
        model="deepseek-v4-flash",
        query_analysis_max_output_tokens=256,
        query_analysis_timeout_ms=3000,
    )

    result = qa.analyze_memory_query(config, "sk-test", query="老婆换成黑丝", category="图片", agent_id="assistant")

    assert result.intent_summary == "生成兔女郎图片"
    assert result.keywords == ["黑丝"]
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert captured["api_key"] == "sk-test"
    assert captured["kwargs"]["response_format"] == {"type": "json_object"}
    assert captured["kwargs"]["max_tokens"] == 256
    assert captured["kwargs"]["temperature"] == 0.0
    assert captured["kwargs"]["timeout_ms"] == 3000
    assert "老婆换成黑丝" in captured["messages"][1]["content"]
    assert "不要因为分类看起来不匹配就返回空关键词" in captured["messages"][0]["content"]
    assert "绘图、角色、姿势、服装、身体、画面修改" in captured["messages"][0]["content"]


def test_parse_memory_request_analysis_combines_category_and_keywords() -> None:
    categories = [SimpleNamespace(name="娱乐偏好", description="游戏、动漫、影视", memory_count=1)]

    parsed = qa.parse_memory_request_analysis(
        json.dumps(
            {
                "category": "娱乐偏好",
                "matched_existing": True,
                "confidence": 0.93,
                "reason": "命运2是游戏内容。",
                "intent_summary": "查询命运2通行证任务",
                "keywords": ["命运2", "猎杀通行证", "游戏任务"],
                "negative_keywords": [],
            },
            ensure_ascii=False,
        ),
        categories,
    )

    assert parsed.category == "娱乐偏好"
    assert parsed.matched_existing is True
    assert parsed.confidence == 0.93
    assert parsed.intent_summary == "查询命运2通行证任务"
    assert parsed.keywords == ["命运2", "猎杀通行证", "游戏任务"]


def test_analyze_memory_request_uses_one_json_call(monkeypatch) -> None:
    captured = {}

    def fake_chat_completion(config, api_key, messages, **kwargs):
        captured["api_key"] = api_key
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            content=json.dumps(
                {
                    "category": "娱乐偏好",
                    "matched_existing": True,
                    "confidence": 1,
                    "reason": "游戏资料",
                    "intent_summary": "查询命运2任务",
                    "keywords": ["猎杀通行证"],
                    "negative_keywords": [],
                },
                ensure_ascii=False,
            ),
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(qa, "chat_completion", fake_chat_completion)
    config = SimpleNamespace(query_analysis_max_output_tokens=128, query_analysis_timeout_ms=2000)
    result = qa.analyze_memory_request(
        config,
        "sk-test",
        query="命运2猎杀通行证任务",
        operation="context",
        categories=[SimpleNamespace(name="娱乐偏好", description="游戏、动漫、影视", memory_count=1)],
        agent_id="assistant",
    )

    assert result.category == "娱乐偏好"
    assert result.keywords == ["猎杀通行证"]
    assert result.usage == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    assert captured["api_key"] == "sk-test"
    assert captured["kwargs"]["response_format"] == {"type": "json_object"}
    assert captured["kwargs"]["max_tokens"] == 128
    assert "命运2/Destiny 2 是游戏" in captured["messages"][0]["content"]
    assert "已有分类" in captured["messages"][1]["content"]


def test_parse_query_analysis_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="合法 JSON"):
        qa.parse_query_analysis("不是 json")
