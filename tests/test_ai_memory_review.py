import json
from uuid import uuid4

import pytest

from aimemory.services.ai_memory_review import AiMemoryReviewError, normalize_suggestions


def test_normalize_suggestions_accepts_markdown_json_block() -> None:
    memory_id = str(uuid4())
    content = "```json\n" + json.dumps(
        {
            "suggestions": [
                {
                    "type": "rewrite",
                    "memory_ids": [memory_id],
                    "proposed": {"title": "整理后标题", "content": "整理后正文", "category": "工作流程"},
                }
            ]
        },
        ensure_ascii=False,
    ) + "\n```"

    suggestions = normalize_suggestions(content, {memory_id})

    assert len(suggestions) == 1
    assert suggestions[0]["type"] == "rewrite"
    assert suggestions[0]["proposed"]["title"] == "整理后标题"


def test_normalize_suggestions_extracts_json_from_extra_text() -> None:
    memory_id = str(uuid4())
    content = (
        "下面是建议：\n"
        + json.dumps(
            {
                "suggestions": [
                    {
                        "type": "soft_delete",
                        "memory_ids": [memory_id],
                        "reason": "重复。",
                    }
                ]
            },
            ensure_ascii=False,
        )
        + "\n请审核。"
    )

    suggestions = normalize_suggestions(content, {memory_id})

    assert len(suggestions) == 1
    assert suggestions[0]["type"] == "soft_delete"


def test_normalize_suggestions_rejects_non_json() -> None:
    with pytest.raises(AiMemoryReviewError, match="合法 JSON"):
        normalize_suggestions("我没有建议。", {str(uuid4())})
