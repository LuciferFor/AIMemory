from aimemory.repositories.memories import (
    duplicate_memory_score,
    is_probable_duplicate_memory,
    should_attempt_auto_memory_dedupe,
)
from aimemory.schemas.memory import MemoryUpsertRequest


def test_duplicate_memory_detects_small_rewrites() -> None:
    assert is_probable_duplicate_memory(
        "部署诊断直接执行并给简短结论",
        "当用户问部署或诊断问题时，希望直接执行并给出简短结论。",
        "部署诊断直接执行并给出简短结论",
        "当用户问部署或诊断问题时，助手应直接执行并给出简短结论。",
    )


def test_duplicate_memory_detects_same_title_short_content_rewrite() -> None:
    assert is_probable_duplicate_memory(
        "中文优先",
        "用户偏好在 OpenClaw 里优先使用中文回答。",
        "中文优先",
        "用户偏好在 OpenClaw 里优先用中文回答。",
    )


def test_duplicate_memory_rejects_unrelated_memories() -> None:
    score = duplicate_memory_score(
        "中文优先",
        "用户偏好优先使用中文回答。",
        "部署诊断直接执行并给出简短结论",
        "当用户问部署或诊断问题时，助手应直接执行并给出简短结论。",
    )

    assert score < 0.88


def test_auto_memory_dedupe_only_for_generated_memories() -> None:
    manual = MemoryUpsertRequest(
        agent_id="assistant",
        external_id="manual-answer-style",
        category="回答风格",
        title="中文优先",
        content="用户偏好优先使用中文回答。",
        metadata={},
    )
    automatic = MemoryUpsertRequest(
        agent_id="assistant",
        external_id="auto-memory-abc123",
        category="回答风格",
        title="中文优先",
        content="用户偏好优先使用中文回答。",
        metadata={"source": "conversation_compaction"},
    )

    assert should_attempt_auto_memory_dedupe(manual) is False
    assert should_attempt_auto_memory_dedupe(automatic) is True
