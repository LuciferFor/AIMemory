from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from aimemory.api.deps import get_current_user
from aimemory.api import routes
from aimemory.core.config import get_settings
from aimemory.db.session import get_db
from aimemory.main import create_app
from aimemory.repositories.memories import SearchAttachment, SearchResult, filter_high_frequency_terms
from aimemory.schemas.memory import MemorySearchRequest


class _FakeDb:
    def __init__(self) -> None:
        self.category_id = uuid4()

    def scalar(self, query):
        return SimpleNamespace(id=self.category_id, name="偏好", description=None)

    def scalars(self, query):
        return SimpleNamespace(all=lambda: [])

    def execute(self, query, params=None):
        return SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: []))


class _HighFrequencyDb:
    def execute(self, query, params=None):
        rows = [
            {"term": "不要", "match_count": 3, "total_count": 10},
            {"term": "苹果", "match_count": 1, "total_count": 10},
        ]
        return SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: rows))


class _CategoryListDb(_FakeDb):
    def execute(self, query, params=None):
        rows = [
            {
                "id": uuid4(),
                "name": "偏好",
                "description": "偏好类记忆",
                "memory_count": 3,
            }
        ]
        return SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: rows))


class _NoCategoryDb(_FakeDb):
    def scalar(self, query):
        return None


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

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "category": "偏好", "query": "偏好"})

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

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "category": "偏好", "query": "偏好"})

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
        category="偏好",
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

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "category": "偏好", "query": "回答偏好"})

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
        category="偏好",
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

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "category": "偏好", "query": "回答偏好"})

    assert response.status_code == 200
    assert len(records) == 1
    summary = records[0]["response_summary"]
    assert summary["type"] == "context"
    assert summary["agent_id"] == "assistant"
    assert summary["category"] == "偏好"
    assert summary["category_found"] is True
    assert summary["category_not_found"] is False
    assert summary["query"] == "回答偏好"
    assert summary["query_preview"] == "回答偏好"
    assert summary["top_k"] == 8
    assert summary["max_chars"] == 3000
    assert summary["ignored_terms"] == []
    assert "回答" in summary["query_terms"]
    assert "偏好" in summary["query_terms"]
    assert summary["result_count"] == 1
    assert summary["context_chars"] == len(response.json()["context_text"])
    assert summary["context_text_preview"].startswith("以下是与当前请求可能相关的长期记忆")
    assert "\n\n[长期记忆]" in summary["context_text_preview"]
    assert "\n\n1. 回复偏好：简短自然\n" in summary["context_text_preview"]
    assert summary["context_text_preview_truncated"] is True
    assert summary["items"][0]["external_id"] == "pref-short-replies"
    assert summary["items"][0]["title"] == "回复偏好：简短自然"
    assert "回答" in summary["items"][0]["matched_terms"]
    assert "偏好" in summary["items"][0]["matched_terms"]
    assert "标题" in summary["items"][0]["matched_fields"]
    assert "正文" in summary["items"][0]["matched_fields"]
    assert "score_parts" in summary["items"][0]
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

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "category": "偏好", "query": "2026-05-30 lucifer skill"})

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
        category="偏好",
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
        json={"agent_id": "assistant", "category": "偏好", "query": "OpenClaw AIMemory lucifer 苹果"},
    )

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0][4] == "苹果"
    assert calls[0][5] == ["苹果"]


def test_context_ignores_single_english_words(monkeypatch) -> None:
    import aimemory.main as main_module

    records = []
    calls = []

    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "true")
    monkeypatch.setattr(routes, "search_memories", lambda *args, **kwargs: calls.append(args) or [])
    monkeypatch.setattr(main_module, "insert_request_log", lambda data: records.append(data.copy()))
    get_settings.cache_clear()
    app = main_module.create_app()
    app.dependency_overrides[get_db] = lambda: _FakeDb()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=uuid4())
    client = TestClient(app)

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "category": "偏好", "query": "fantasy poster"})

    assert response.status_code == 200
    summary = records[0]["response_summary"]
    assert summary["query_terms"] == ["fantasy poster"]
    assert "fantasy:英文单词" in summary["ignored_terms"]
    assert "poster:英文单词" in summary["ignored_terms"]
    assert len(calls) == 1
    assert calls[0][4] == "fantasy poster"
    assert calls[0][5] == ["fantasy poster"]

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "category": "偏好", "query": "fantasy"})

    assert response.status_code == 200
    assert response.json()["items"] == []
    summary = records[-1]["response_summary"]
    assert summary["query_terms"] == []
    assert summary["ignored_terms"] == ["fantasy:英文单词"]
    assert len(calls) == 1


def test_context_returns_empty_when_category_is_missing(monkeypatch) -> None:
    import aimemory.main as main_module

    records = []
    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "true")
    monkeypatch.setattr(main_module, "insert_request_log", lambda data: records.append(data.copy()))
    get_settings.cache_clear()
    app = main_module.create_app()
    app.dependency_overrides[get_db] = lambda: _NoCategoryDb()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=uuid4())
    client = TestClient(app)

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "category": "不存在", "query": "苹果"})

    assert response.status_code == 200
    assert response.json()["items"] == []
    summary = records[0]["response_summary"]
    assert summary["category"] == "不存在"
    assert summary["category_found"] is False
    assert summary["category_not_found"] is True
    assert summary["ignored_terms"] == ["不存在:分类不存在"]


def test_context_requires_category(monkeypatch) -> None:
    client = _client_with_user(monkeypatch)

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "query": "偏好"})

    assert response.status_code == 422


def test_search_results_filters_high_frequency_terms(monkeypatch) -> None:
    calls = []

    def _fake_search_memories(*args):
        calls.append(args)
        return []

    monkeypatch.setattr(routes, "active_search_stopword_terms", lambda *args, **kwargs: set())
    monkeypatch.setattr(routes, "filter_high_frequency_terms", lambda *args, **kwargs: (["苹果"], ["回答:高频弱词"]))
    monkeypatch.setattr(routes, "search_memories", _fake_search_memories)
    payload = MemorySearchRequest(agent_id="assistant", category="偏好", query="苹果 回答")

    results, used_vector, duration_ms, query_terms, ignored_terms, category_found = routes._search_results(
        _FakeDb(),
        SimpleNamespace(id=uuid4()),
        payload,
    )

    assert results == []
    assert used_vector is False
    assert category_found is True
    assert duration_ms >= 0
    assert query_terms == ["苹果"]
    assert ignored_terms == ["回答:高频弱词"]
    assert len(calls) == 1
    assert calls[0][4] == "苹果"
    assert calls[0][5] == ["苹果"]


def test_filter_high_frequency_terms_uses_user_memory_distribution() -> None:
    terms, ignored = filter_high_frequency_terms(
        _HighFrequencyDb(),
        uuid4(),
        uuid4(),
        "assistant",
        ["不要", "苹果"],
    )

    assert terms == ["苹果"]
    assert ignored == ["不要:高频弱词"]


def test_search_returns_attachment_metadata(monkeypatch) -> None:
    now = datetime.now(UTC)
    attachment_id = uuid4()
    result = SearchResult(
        memory_id=uuid4(),
        external_id="image-memory",
        category="图片",
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

    response = client.post("/v1/memories/search", json={"agent_id": "assistant", "category": "图片", "query": "图片"})

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
        category="图片",
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

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "category": "图片", "query": "图片"})

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
        category="测试",
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
        json={"agent_id": "assistant", "category": "测试", "query": "很长", "max_chars": 80},
    )

    assert response.status_code == 200
    assert len(response.json()["context_text"]) <= 80


def test_write_policy_returns_standard_fields(monkeypatch) -> None:
    client = _client_with_user(monkeypatch)

    response = client.get("/v1/memories/write-policy")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prompt", "output_schema", "required_fields", "rules", "forbidden", "categories"}
    assert body["required_fields"] == ["external_id", "category", "title", "content", "metadata"]
    assert "密码" in body["forbidden"]


def test_categories_endpoint_returns_user_categories(monkeypatch) -> None:
    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "false")
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_db] = lambda: _CategoryListDb()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=uuid4())
    client = TestClient(app)

    response = client.get("/v1/memories/categories")

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["name"] == "偏好"
    assert item["description"] == "偏好类记忆"
    assert item["memory_count"] == 3


def test_search_results_uses_text_only(monkeypatch) -> None:
    calls = []

    def _fake_search_memories(*args):
        calls.append(args)
        return []

    monkeypatch.setattr(routes, "search_memories", _fake_search_memories)
    payload = MemorySearchRequest(agent_id="assistant", category="偏好", query="自然 回复")

    results, used_vector, duration_ms, query_terms, ignored_terms, category_found = routes._search_results(
        _FakeDb(),
        SimpleNamespace(id=uuid4()),
        payload,
    )

    assert results == []
    assert used_vector is False
    assert category_found is True
    assert duration_ms >= 0
    assert query_terms == ["自然", "回复"]
    assert ignored_terms == []
    assert len(calls[0]) == 6
    assert calls[0][4] == "自然 回复"
    assert calls[0][5] == ["自然", "回复"]
