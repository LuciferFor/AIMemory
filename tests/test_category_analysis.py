import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from aimemory.services import category_analysis as ca


def _category(name: str, description: str | None = None):
    return SimpleNamespace(id=uuid4(), name=name, description=description, memory_count=1)


def test_parse_category_analysis_reuses_existing_category() -> None:
    parsed = ca.parse_category_analysis(
        json.dumps(
            {
                "category": "技术记忆",
                "matched_existing": True,
                "confidence": 0.91,
                "reason": "OneBot 连接问题属于技术排查。",
            },
            ensure_ascii=False,
        ),
        [_category("技术记忆")],
    )

    assert parsed.category == "技术记忆"
    assert parsed.matched_existing is True
    assert parsed.confidence == 0.91
    assert "OneBot" in parsed.reason


def test_parse_category_analysis_allows_new_category_for_write_layer() -> None:
    parsed = ca.parse_category_analysis(
        '{"category":"图片偏好","matched_existing":false,"confidence":0.72,"reason":"新的图片偏好分类"}',
        [_category("回答风格")],
    )

    assert parsed.category == "图片偏好"
    assert parsed.matched_existing is False


def test_analyze_memory_category_uses_json_output(monkeypatch) -> None:
    captured = {}

    def fake_chat_completion(config, api_key, messages, **kwargs):
        captured["api_key"] = api_key
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            content='{"category":"工作流程","matched_existing":true,"confidence":0.8,"reason":"部署流程"}',
            usage={},
        )

    monkeypatch.setattr(ca, "chat_completion", fake_chat_completion)
    config = SimpleNamespace(
        model="deepseek-v4-flash",
        query_analysis_max_output_tokens=256,
        query_analysis_timeout_ms=3000,
    )

    result = ca.analyze_memory_category(
        config,
        "sk-test",
        text="OneBot 又连不上了",
        operation="context",
        categories=[_category("工作流程")],
    )

    assert result.category == "工作流程"
    assert captured["api_key"] == "sk-test"
    assert captured["kwargs"]["response_format"] == {"type": "json_object"}
    assert captured["kwargs"]["temperature"] == 0.0
    assert "查询记忆时只能选择已有分类" in captured["messages"][0]["content"]


def test_parse_category_analysis_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="合法 JSON"):
        ca.parse_category_analysis("不是 json", [])
