from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from aimemory.api.deps import get_current_user
from aimemory.api import routes
from aimemory.core.config import get_settings
from aimemory.db.session import get_db
from aimemory.main import create_app
from aimemory.repositories.memories import SearchAttachment, SearchResult
from aimemory.schemas.memory import MemorySearchRequest


class _FakeDb:
    def scalars(self, query):
        return SimpleNamespace(all=lambda: [])


def _client_with_user(monkeypatch=None) -> TestClient:
    if monkeypatch is not None:
        monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "false")
        get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_db] = lambda: _FakeDb()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=uuid4())
    return TestClient(app)


def test_context_requires_auth(monkeypatch) -> None:
    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "false")
    get_settings.cache_clear()
    client = TestClient(create_app())

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "query": "偏好"})

    assert response.status_code == 401


def test_write_policy_requires_auth(monkeypatch) -> None:
    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "false")
    get_settings.cache_clear()
    client = TestClient(create_app())

    response = client.get("/v1/memories/write-policy")

    assert response.status_code == 401


def test_context_returns_empty_text_without_memories(monkeypatch) -> None:
    client = _client_with_user(monkeypatch)
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
    client = _client_with_user(monkeypatch)
    monkeypatch.setattr(routes, "search_memories", lambda *args, **kwargs: [result])

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "query": "回答偏好"})

    assert response.status_code == 200
    body = response.json()
    assert "以下是与当前请求可能相关的长期记忆" in body["context_text"]
    assert "回复偏好：简短自然" in body["context_text"]
    assert body["items"][0]["external_id"] == "pref-short-replies"
    assert body["items"][0]["embedding_status"] == "disabled"


def test_context_writes_response_summary_to_request_log(monkeypatch) -> None:
    import aimemory.main as main_module

    records = []
    now = datetime.now(UTC)
    long_content = "用户喜欢短一点、自然一点的回答。" * 20
    result = SearchResult(
        memory_id=uuid4(),
        external_id="pref-short-replies",
        title="回复偏好：简短自然",
        content=long_content,
        metadata={},
        created_at=now,
        updated_at=now,
        score=0.91,
        score_parts={"semantic": 0.0, "keyword": 0.1, "fuzzy": 0.01},
        embedding_status="disabled",
    )
    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "true")
    monkeypatch.setattr(routes, "search_memories", lambda *args, **kwargs: [result])
    monkeypatch.setattr(main_module, "insert_request_log", lambda data: records.append(data.copy()))
    get_settings.cache_clear()
    app = main_module.create_app()
    app.dependency_overrides[get_db] = lambda: _FakeDb()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=uuid4())
    client = TestClient(app)

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "query": "回答偏好"})

    assert response.status_code == 200
    assert len(records) == 1
    summary = records[0]["response_summary"]
    assert summary["type"] == "context"
    assert summary["agent_id"] == "assistant"
    assert summary["query"] == "回答偏好"
    assert summary["top_k"] == 8
    assert summary["max_chars"] == 3000
    assert summary["ignored_terms"] == []
    assert "回答" in summary["query_terms"]
    assert "偏好" in summary["query_terms"]
    assert summary["result_count"] == 1
    assert summary["context_chars"] == len(response.json()["context_text"])
    assert summary["items"][0]["external_id"] == "pref-short-replies"
    assert summary["items"][0]["title"] == "回复偏好：简短自然"
    assert "回答" in summary["items"][0]["matched_terms"]
    assert "偏好" in summary["items"][0]["matched_terms"]
    assert len(summary["items"][0]["content_preview"]) <= 80
    rendered = str(summary)
    assert response.json()["context_text"] not in rendered
    assert long_content not in rendered


def test_context_ignores_numeric_and_user_stopwords(monkeypatch) -> None:
    import aimemory.main as main_module

    records = []

    def _fail_search(*args, **kwargs):
        raise AssertionError("search should not run when all terms are ignored")

    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "true")
    monkeypatch.setattr(routes, "active_search_stopword_terms", lambda *args, **kwargs: {"lucifer", "skill"})
    monkeypatch.setattr(routes, "search_memories", _fail_search)
    monkeypatch.setattr(main_module, "insert_request_log", lambda data: records.append(data.copy()))
    get_settings.cache_clear()
    app = main_module.create_app()
    app.dependency_overrides[get_db] = lambda: _FakeDb()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=uuid4())
    client = TestClient(app)

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "query": "2026-05-30 lucifer skill"})

    assert response.status_code == 200
    assert response.json()["items"] == []
    summary = records[0]["response_summary"]
    assert summary["query_terms"] == []
    assert summary["ignored_terms"] == ["2026:数字", "05:数字", "30:数字", "lucifer:停用词", "skill:停用词"]
    assert summary["result_count"] == 0


def test_context_uses_effective_terms_for_search_and_log(monkeypatch) -> None:
    now = datetime.now(UTC)
    calls = []
    result = SearchResult(
        memory_id=uuid4(),
        external_id="apple-preference",
        title="苹果偏好",
        content="用户喜欢苹果。",
        metadata={},
        created_at=now,
        updated_at=now,
        score=0.91,
        score_parts={"semantic": 0.0, "keyword": 0.1, "fuzzy": 0.01},
        embedding_status="disabled",
    )

    def _fake_search(*args):
        calls.append(args)
        return [result]

    client = _client_with_user(monkeypatch)
    monkeypatch.setattr(routes, "active_search_stopword_terms", lambda *args, **kwargs: {"lucifer"})
    monkeypatch.setattr(routes, "search_memories", _fake_search)

    response = client.post(
        "/v1/memories/context",
        json={"agent_id": "assistant", "query": "OpenClaw AIMemory lucifer 苹果"},
    )

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0][3] == "苹果"
    assert calls[0][4] == ["苹果"]


def test_search_returns_attachment_metadata(monkeypatch) -> None:
    now = datetime.now(UTC)
    attachment_id = uuid4()
    result = SearchResult(
        memory_id=uuid4(),
        external_id="image-memory",
        title="图片记忆",
        content="这条记忆带图片。",
        metadata={},
        created_at=now,
        updated_at=now,
        score=0.8,
        score_parts={"semantic": 0.0, "keyword": 0.2, "fuzzy": 0.3},
        embedding_status="disabled",
        attachments=[
            SearchAttachment(
                attachment_id=attachment_id,
                filename="scene.png",
                mime_type="image/png",
                size_bytes=24,
                sha256="a" * 64,
                description="图片里有一座桥",
            )
        ],
    )
    client = _client_with_user(monkeypatch)
    monkeypatch.setattr(routes, "search_memories", lambda *args, **kwargs: [result])

    response = client.post("/v1/memories/search", json={"agent_id": "assistant", "query": "图片"})

    assert response.status_code == 200
    attachment = response.json()["items"][0]["attachments"][0]
    assert attachment["attachment_id"] == str(attachment_id)
    assert attachment["download_url"] == f"/v1/memories/attachments/{attachment_id}"
    assert "data_base64" not in attachment


def test_context_includes_attachment_description_without_base64(monkeypatch) -> None:
    now = datetime.now(UTC)
    result = SearchResult(
        memory_id=uuid4(),
        external_id="image-memory",
        title="图片记忆",
        content="这条记忆带图片。",
        metadata={},
        created_at=now,
        updated_at=now,
        score=0.8,
        score_parts={"semantic": 0.0, "keyword": 0.2, "fuzzy": 0.3},
        embedding_status="disabled",
        attachments=[
            SearchAttachment(
                attachment_id=uuid4(),
                filename="scene.png",
                mime_type="image/png",
                size_bytes=24,
                sha256="a" * 64,
                description="图片里有一座桥",
            )
        ],
    )
    client = _client_with_user(monkeypatch)
    monkeypatch.setattr(routes, "search_memories", lambda *args, **kwargs: [result])

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "query": "图片"})

    assert response.status_code == 200
    context_text = response.json()["context_text"]
    assert "图片附件" in context_text
    assert "图片里有一座桥" in context_text
    assert "data_base64" not in context_text


def test_attachment_download_requires_auth(monkeypatch) -> None:
    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "false")
    get_settings.cache_clear()
    client = TestClient(create_app())

    response = client.get(f"/v1/memories/attachments/{uuid4()}")

    assert response.status_code == 401


def test_attachment_download_returns_image(monkeypatch) -> None:
    attachment_id = uuid4()
    client = _client_with_user(monkeypatch)
    monkeypatch.setattr(
        routes,
        "get_attachment_for_user",
        lambda *args, **kwargs: SimpleNamespace(
            id=attachment_id,
            image_bytes=b"\x89PNG\r\n\x1a\n",
            mime_type="image/png",
            size_bytes=8,
            sha256="a" * 64,
        ),
    )

    response = client.get(f"/v1/memories/attachments/{attachment_id}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"\x89PNG\r\n\x1a\n"


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
    client = _client_with_user(monkeypatch)
    monkeypatch.setattr(routes, "search_memories", lambda *args, **kwargs: [result])

    response = client.post(
        "/v1/memories/context",
        json={"agent_id": "assistant", "query": "很长", "max_chars": 80},
    )

    assert response.status_code == 200
    assert len(response.json()["context_text"]) <= 80


def test_write_policy_returns_standard_fields(monkeypatch) -> None:
    client = _client_with_user(monkeypatch)

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

    results, used_vector, duration_ms, query_terms, ignored_terms = routes._search_results(
        _FakeDb(),
        SimpleNamespace(id=uuid4()),
        payload,
    )

    assert results == []
    assert used_vector is False
    assert duration_ms >= 0
    assert query_terms == ["自然", "回复"]
    assert ignored_terms == []
    assert len(calls[0]) == 5
    assert calls[0][3] == "自然 回复"
    assert calls[0][4] == ["自然", "回复"]
