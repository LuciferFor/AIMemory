from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from aimemory.api.deps import get_current_user
from aimemory.api import routes
from aimemory.db.session import get_db
from aimemory.main import create_app
from aimemory.repositories.memories import SearchResult
from aimemory.schemas.memory import MemorySearchRequest


class _FakeDb:
    pass


def _client_with_user() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_db] = lambda: _FakeDb()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=uuid4())
    return TestClient(app)


def test_context_requires_auth() -> None:
    client = TestClient(create_app())

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "query": "偏好"})

    assert response.status_code == 401


def test_write_policy_requires_auth() -> None:
    client = TestClient(create_app())

    response = client.get("/v1/memories/write-policy")

    assert response.status_code == 401


def test_context_returns_empty_text_without_memories(monkeypatch) -> None:
    client = _client_with_user()
    monkeypatch.setattr(routes, "search_memories", lambda *args, **kwargs: [])

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "query": "偏好"})

    assert response.status_code == 200
    body = response.json()
    assert body["context_text"] == ""
    assert body["items"] == []
    assert body["usage_hint"]["recommended_position"] == "system_or_developer_context"


def test_context_returns_standard_prompt_and_items(monkeypatch) -> None:
    now = datetime.now(UTC)
    result = SearchResult(
        memory_id=uuid4(),
        external_id="pref-short-replies",
        title="回复偏好：简短自然",
        content="用户喜欢短一点、自然一点的回答。",
        metadata={},
        created_at=now,
        updated_at=now,
        score=0.91,
        score_parts={"semantic": 0.0, "keyword": 0.1, "fuzzy": 0.01},
        embedding_status="disabled",
    )
    client = _client_with_user()
    monkeypatch.setattr(routes, "search_memories", lambda *args, **kwargs: [result])

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "query": "回答偏好"})

    assert response.status_code == 200
    body = response.json()
    assert "以下是与当前请求可能相关的长期记忆" in body["context_text"]
    assert "回复偏好：简短自然" in body["context_text"]
    assert body["items"][0]["external_id"] == "pref-short-replies"
    assert body["items"][0]["embedding_status"] == "disabled"


def test_context_respects_max_chars(monkeypatch) -> None:
    now = datetime.now(UTC)
    result = SearchResult(
        memory_id=uuid4(),
        external_id="long-memory",
        title="很长的记忆",
        content="内容" * 200,
        metadata={},
        created_at=now,
        updated_at=now,
        score=0.8,
        score_parts={"semantic": 0.0, "keyword": 0.1, "fuzzy": 0.0},
        embedding_status="disabled",
    )
    client = _client_with_user()
    monkeypatch.setattr(routes, "search_memories", lambda *args, **kwargs: [result])

    response = client.post(
        "/v1/memories/context",
        json={"agent_id": "assistant", "query": "很长", "max_chars": 80},
    )

    assert response.status_code == 200
    assert len(response.json()["context_text"]) <= 80


def test_write_policy_returns_standard_fields() -> None:
    client = _client_with_user()

    response = client.get("/v1/memories/write-policy")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prompt", "output_schema", "required_fields", "rules", "forbidden"}
    assert body["required_fields"] == ["external_id", "title", "content", "metadata"]
    assert "密码" in body["forbidden"]


def test_search_results_uses_text_only(monkeypatch) -> None:
    calls = []

    def _fake_search_memories(*args):
        calls.append(args)
        return []

    monkeypatch.setattr(routes, "search_memories", _fake_search_memories)
    payload = MemorySearchRequest(agent_id="assistant", query="自然 回复")

    results, used_vector, duration_ms = routes._search_results(_FakeDb(), SimpleNamespace(id=uuid4()), payload)

    assert results == []
    assert used_vector is False
    assert duration_ms >= 0
    assert len(calls[0]) == 4
