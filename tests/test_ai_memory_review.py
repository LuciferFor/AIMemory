import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from aimemory.services import ai_memory_review
from aimemory.services.ai_memory_review import (
    AiMemoryReviewError,
    apply_auto_merge_suggestions,
    build_auto_merge_candidate_memory_ids,
    normalize_suggestions,
)


class _Rows:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values


class _AutoMergeCandidateDb:
    def __init__(self, seeds, candidates) -> None:
        self.calls = 0
        self.seeds = seeds
        self.candidates = candidates

    def scalars(self, _query):
        self.calls += 1
        return _Rows(self.seeds if self.calls == 1 else self.candidates)


class _AutoApplyDb:
    def __init__(self, suggestions) -> None:
        self.suggestions = suggestions
        self.commits = 0

    def scalars(self, _query):
        return _Rows(self.suggestions)

    def add(self, _value) -> None:
        pass

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        pass


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


def test_auto_merge_candidate_selection_keeps_similar_same_scope_memories() -> None:
    user_id = uuid4()
    category_id = uuid4()
    seed = SimpleNamespace(
        id=uuid4(),
        user_id=user_id,
        category_id=category_id,
        agent_id="assistant",
        title="部署诊断直接执行并给简短结论",
        content="当用户问部署或诊断问题时，希望直接执行并给出简短结论。",
    )
    similar = SimpleNamespace(
        id=uuid4(),
        user_id=user_id,
        category_id=category_id,
        agent_id="assistant",
        title="部署诊断直接执行并给出简短结论",
        content="当用户问部署或诊断问题时，助手应直接执行并给出简短结论。",
    )
    unrelated = SimpleNamespace(
        id=uuid4(),
        user_id=user_id,
        category_id=category_id,
        agent_id="assistant",
        title="中文优先",
        content="用户偏好优先使用中文回答。",
    )
    db = _AutoMergeCandidateDb([seed], [similar, unrelated])

    selected, candidate_count = build_auto_merge_candidate_memory_ids(
        db,
        user_id=user_id,
        agent_id="assistant",
        seed_memory_ids=[seed.id],
    )

    assert selected == [seed.id, similar.id]
    assert candidate_count == 1


def test_auto_merge_candidate_selection_skips_low_similarity() -> None:
    user_id = uuid4()
    category_id = uuid4()
    seed = SimpleNamespace(
        id=uuid4(),
        user_id=user_id,
        category_id=category_id,
        agent_id="assistant",
        title="中文优先",
        content="用户偏好优先使用中文回答。",
    )
    unrelated = SimpleNamespace(
        id=uuid4(),
        user_id=user_id,
        category_id=category_id,
        agent_id="assistant",
        title="部署诊断直接执行并给出简短结论",
        content="当用户问部署或诊断问题时，助手应直接执行并给出简短结论。",
    )
    db = _AutoMergeCandidateDb([seed], [unrelated])

    selected, candidate_count = build_auto_merge_candidate_memory_ids(
        db,
        user_id=user_id,
        agent_id="assistant",
        seed_memory_ids=[seed.id],
    )

    assert selected == []
    assert candidate_count == 0


def test_auto_merge_auto_apply_ignores_soft_delete_and_applies_high_confidence(monkeypatch) -> None:
    run_id = uuid4()
    run = SimpleNamespace(id=run_id, response_summary={})
    soft_delete = SimpleNamespace(
        run_id=run_id,
        status="pending",
        suggestion_type="soft_delete",
        confidence=0.99,
        error=None,
        ignored_at=None,
        updated_at=None,
    )
    rewrite = SimpleNamespace(
        run_id=run_id,
        status="pending",
        suggestion_type="rewrite",
        confidence=0.9,
        error=None,
    )
    low_confidence_merge = SimpleNamespace(
        run_id=run_id,
        status="pending",
        suggestion_type="merge",
        confidence=0.5,
        error=None,
    )
    db = _AutoApplyDb([soft_delete, rewrite, low_confidence_merge])

    def fake_apply_suggestion(_db, suggestion) -> None:
        suggestion.status = "applied"

    monkeypatch.setattr(ai_memory_review, "apply_suggestion", fake_apply_suggestion)

    summary = apply_auto_merge_suggestions(db, run)

    assert soft_delete.status == "ignored"
    assert soft_delete.error == "自动相似合并不允许这个建议类型。"
    assert rewrite.status == "applied"
    assert low_confidence_merge.status == "pending"
    assert summary["auto_applied"] == 1
    assert summary["auto_pending"] == 1
    assert summary["auto_ignored"] == 1
