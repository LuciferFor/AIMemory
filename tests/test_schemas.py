from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from aimemory.schemas.memory import MemoryContextRequest, MemorySearchRequest, MemoryUpsertRequest


def test_memory_upsert_request_accepts_title_and_content() -> None:
    payload = MemoryUpsertRequest(
        agent_id="assistant",
        external_id="mem-1",
        category="偏好",
        title="Preference",
        content="The user likes short answers.",
        metadata={"source": "test"},
    )

    assert payload.metadata["source"] == "test"


def test_memory_upsert_request_accepts_missing_category() -> None:
    payload = MemoryUpsertRequest(
        agent_id="assistant",
        external_id="mem-1",
        title="Preference",
        content="The user likes short answers.",
    )

    assert payload.category is None


def test_search_request_rejects_inverted_time_window() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError):
        MemorySearchRequest(
            agent_id="assistant",
            category="偏好",
            query="preference",
            since=now,
            until=now - timedelta(days=1),
        )


def test_context_request_defaults_and_limits() -> None:
    payload = MemoryContextRequest(agent_id="assistant", query="用户偏好")

    assert payload.top_k == 8
    assert payload.max_chars == 3000
    assert payload.category is None

    with pytest.raises(ValidationError):
        MemoryContextRequest(agent_id="assistant", query="用户偏好", max_chars=12001)


def test_context_request_rejects_inverted_time_window() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError):
        MemoryContextRequest(
            agent_id="assistant",
            category="偏好",
            query="preference",
            since=now,
            until=now - timedelta(days=1),
        )
