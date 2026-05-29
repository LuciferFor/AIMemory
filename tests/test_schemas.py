from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from aimemory.schemas.memory import MemorySearchRequest, MemoryUpsertRequest


def test_memory_upsert_request_accepts_title_and_content() -> None:
    payload = MemoryUpsertRequest(
        agent_id="assistant",
        external_id="mem-1",
        title="Preference",
        content="The user likes short answers.",
        metadata={"source": "test"},
    )

    assert payload.metadata["source"] == "test"


def test_search_request_rejects_inverted_time_window() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError):
        MemorySearchRequest(
            agent_id="assistant",
            query="preference",
            since=now,
            until=now - timedelta(days=1),
        )
