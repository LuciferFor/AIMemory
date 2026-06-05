import json
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
            {"term": "命运2", "match_count": 10, "total_count": 10},
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


def test_extract_schedules_auto_merge_background_task(monkeypatch) -> None:
    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "false")
    get_settings.cache_clear()
    user_id = uuid4()
    memory_id = uuid4()
    scheduled = []

    class ExtractDb(_FakeDb):
        def commit(self) -> None:
            pass

        def rollback(self) -> None:
            pass

    app = create_app()
    app.dependency_overrides[get_db] = lambda: ExtractDb()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    monkeypatch.setattr(routes, "get_llm_config", lambda _db: SimpleNamespace(enabled=True, encrypted_api_key="encrypted"))
    monkeypatch.setattr(routes, "decrypt_secret", lambda *_args, **_kwargs: "sk-test")
    monkeypatch.setattr(routes, "list_category_summaries", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        routes,
        "chat_completion",
        lambda *_args, **_kwargs: SimpleNamespace(
            content=json.dumps(
                [
                    {
                        "category": "回答风格",
                        "title": "回答偏好：直接给结论",
                        "content": "用户偏好助手先直接给出简短结论，再补充必要说明。",
                    }
                ],
                ensure_ascii=False,
            ),
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        ),
    )
    monkeypatch.setattr(
        routes,
        "upsert_memory",
        lambda *_args, **_kwargs: (SimpleNamespace(id=memory_id), "created"),
    )
    monkeypatch.setattr(
        routes,
        "build_auto_merge_candidate_memory_ids",
        lambda *_args, **_kwargs: ([memory_id, uuid4()], 1),
    )
    monkeypatch.setattr(
        routes,
        "run_auto_merge_review_for_extracted_memories",
        lambda user, agent, memory_ids: scheduled.append((user, agent, memory_ids)),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/memories/extract",
        json={
            "agent_id": "assistant",
            "transcript": "user: 以后回答先给结论\nassistant: 好的，我会先给简短结论",
            "reason": "conversation_compaction",
        },
    )

    assert response.status_code == 200
    assert response.json()["written"] == 1
    assert scheduled == [(str(user_id), "assistant", [str(memory_id)])]


def test_extract_skips_temporary_image_request_and_logs_reason(monkeypatch) -> None:
    import aimemory.main as main_module

    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "true")
    get_settings.cache_clear()
    user_id = uuid4()
    records = []

    class ExtractDb(_FakeDb):
        def commit(self) -> None:
            pass

        def rollback(self) -> None:
            pass

    app = main_module.create_app()
    app.dependency_overrides[get_db] = lambda: ExtractDb()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    monkeypatch.setattr(main_module, "insert_request_log", lambda data: records.append(data.copy()))
    monkeypatch.setattr(routes, "get_llm_config", lambda _db: SimpleNamespace(enabled=True, encrypted_api_key="encrypted"))
    monkeypatch.setattr(routes, "decrypt_secret", lambda *_args, **_kwargs: "sk-test")
    monkeypatch.setattr(routes, "list_category_summaries", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        routes,
        "chat_completion",
        lambda *_args, **_kwargs: SimpleNamespace(
            content=json.dumps(
                {
                    "memories": [
                        {
                            "category": "娱乐偏好",
                            "title": "绯夜出图风格要求",
                            "content": "用户偏好绯夜出图：黑丝、兔女郎、腿粗一点、冷淡表情。",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            usage={"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
        ),
    )
    monkeypatch.setattr(routes, "upsert_memory", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should skip")))
    client = TestClient(app)

    response = client.post(
        "/v1/memories/extract",
        json={
            "agent_id": "assistant",
            "transcript": "user: 老婆出一张黑丝兔女郎图，腿粗一点",
            "reason": "conversation_compaction",
        },
    )

    assert response.status_code == 200
    assert response.json()["written"] == 0
    assert records[0]["response_summary"]["skipped"] == 1
    assert records[0]["response_summary"]["skipped_items"][0]["reason"] == "temporary_image_request_without_durable_marker"


def test_extract_keeps_explicit_durable_image_preference() -> None:
    candidates = routes.normalize_extracted_memory_candidates(
        json.dumps(
            {
                "memories": [
                    {
                        "category": "娱乐偏好",
                        "title": "绯夜长期出图偏好",
                        "content": "用户长期偏好绯夜出图时使用黑丝和冷淡表情。",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        "conversation_compaction",
        {},
        transcript="user: 记住，以后绯夜出图默认黑丝和冷淡表情",
        skipped=[],
    )

    assert len(candidates) == 1
    assert candidates[0]["title"] == "绯夜长期出图偏好"


def test_extract_skips_technical_memory_even_when_explicit_remember() -> None:
    skipped = []

    candidates = routes.normalize_extracted_memory_candidates(
        json.dumps(
            {
                "memories": [
                    {
                        "category": "故障排查",
                        "title": "OneBot 连接失败修复流程",
                        "content": "用户要求记住 OneBot 连接失败时先看日志、检查端口并重启服务。",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        "explicit_remember",
        {},
        transcript="user: 记住这个 OneBot 修复方法",
        skipped=skipped,
    )

    assert candidates == []
    assert skipped[0]["reason"] == "technical_or_troubleshooting_memory_not_allowed"


def test_extract_keeps_human_persona_memory_with_rule_word() -> None:
    skipped = []

    candidates = routes.normalize_extracted_memory_candidates(
        json.dumps(
            {
                "memories": [
                    {
                        "category": "回答风格",
                        "title": "绯夜说话风格",
                        "content": "绯夜的角色互动规则是语气冷淡但关心用户，称呼用户为主人。",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        "conversation_compaction",
        {},
        transcript="user: 绯夜说话要冷淡一点但别太凶",
        skipped=skipped,
    )

    assert len(candidates) == 1
    assert skipped == []


def test_extract_ignores_source_metadata_paths_for_human_memory() -> None:
    skipped = []

    candidates = routes.normalize_extracted_memory_candidates(
        json.dumps(
            {
                "memories": [
                    {
                        "category": "角色关系",
                        "title": "用户与绯夜的互动模式",
                        "content": "用户喜欢绯夜用冷淡但关心的方式互动。",
                        "metadata": {
                            "tags": ["角色关系"],
                            "source_metadata": {"session_file": "/home/lucifer/.openclaw/session.jsonl"},
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        "conversation_compaction",
        {},
        transcript="user: 绯夜这样说话挺好",
        skipped=skipped,
    )

    assert len(candidates) == 1
    assert skipped == []


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
    assert summary["keyword_source"] == "disabled"
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


def test_context_response_summary_includes_ai_query_analysis(monkeypatch) -> None:
    import aimemory.main as main_module

    records = []
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
    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "true")
    monkeypatch.setattr(
        routes,
        "get_llm_config",
        lambda db: SimpleNamespace(enabled=True, query_analysis_enabled=True, encrypted_api_key="encrypted"),
    )
    monkeypatch.setattr(routes, "decrypt_secret", lambda *args, **kwargs: "sk-test")
    monkeypatch.setattr(
        routes,
        "analyze_memory_request",
        lambda *args, **kwargs: routes.MemoryRequestAnalysis(
            category="偏好",
            matched_existing=True,
            confidence=0.88,
            reason="回答偏好",
            intent_summary="查找回复偏好",
            keywords=["回答偏好", "老婆"],
            negative_keywords=["长篇"],
            duration_ms=8.5,
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        ),
    )
    monkeypatch.setattr(routes, "search_memories", lambda *args, **kwargs: [result])
    monkeypatch.setattr(main_module, "insert_request_log", lambda data: records.append(data.copy()))
    get_settings.cache_clear()
    app = main_module.create_app()
    app.dependency_overrides[get_db] = lambda: _FakeDb()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=uuid4())
    client = TestClient(app)

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "category": "偏好", "query": "老婆我想要短回答"})

    assert response.status_code == 200
    summary = records[0]["response_summary"]
    assert summary["keyword_source"] == "ai"
    assert summary["analysis_mode"] == "combined"
    assert summary["intent_summary"] == "查找回复偏好"
    assert summary["ai_keywords"] == ["回答偏好", "老婆"]
    assert summary["negative_keywords"] == ["长篇"]
    assert summary["ai_ignored_terms"] == ["老婆:弱检索词"]
    assert summary["ai_error"] == ""
    assert summary["ai_duration_ms"] == 8.5
    assert summary["ai_usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    assert summary["ai_total_tokens"] == 18
    assert summary["ai_request_total_tokens"] == 18
    assert summary["category_total_tokens"] == 0
    assert summary["query_terms"] == ["回答偏好"]


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

    response = client.post(
        "/v1/memories/context",
        json={"agent_id": "am_testagent000000000000000", "category": "不存在", "query": "苹果"},
    )

    assert response.status_code == 200
    assert response.json()["items"] == []
    summary = records[0]["response_summary"]
    assert summary["category"] == "不存在"
    assert summary["category_found"] is False
    assert summary["category_not_found"] is True
    assert summary["ignored_terms"] == ["不存在:分类不存在"]


def test_context_allows_missing_category(monkeypatch) -> None:
    client = _client_with_user(monkeypatch)

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "query": "偏好"})

    assert response.status_code == 200


def test_search_results_filters_high_frequency_terms(monkeypatch) -> None:
    calls = []

    def _fake_search_memories(*args):
        calls.append(args)
        return []

    monkeypatch.setattr(routes, "active_search_stopword_terms", lambda *args, **kwargs: set())
    monkeypatch.setattr(routes, "filter_high_frequency_terms", lambda *args, **kwargs: (["苹果"], ["回答:高频弱词"]))
    monkeypatch.setattr(routes, "search_memories", _fake_search_memories)
    payload = MemorySearchRequest(agent_id="assistant", category="偏好", query="苹果 回答")

    results, used_vector, duration_ms, query_terms, ignored_terms, category_found, query_analysis = routes._search_results(
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
    assert query_analysis["keyword_source"] == "disabled"
    assert len(calls) == 1
    assert calls[0][4] == "苹果"
    assert calls[0][5] == ["苹果"]


def test_search_results_uses_cross_category_title_fallback(monkeypatch) -> None:
    now = datetime.now(UTC)
    fallback_result = SearchResult(
        memory_id=uuid4(),
        external_id="relation-memory",
        category="其它",
        title="用户关系：鸦羽与月见绫音",
        content="月见绫音与绯夜存在角色关系。",
        metadata={},
        created_at=now,
        updated_at=now,
        score=0.49,
        score_parts={"semantic": 0.0, "title": 1.0, "term": 1.0},
        embedding_status="disabled",
        matched_terms=["绯夜"],
        matched_fields=["标题"],
    )
    fallback_calls = []

    monkeypatch.setattr(routes, "query_terms_for_search", lambda *args, **kwargs: (["绯夜"], [], routes.default_query_analysis_meta("disabled")))
    monkeypatch.setattr(routes, "filter_high_frequency_terms", lambda *args, **kwargs: (args[4], []))
    monkeypatch.setattr(routes, "search_memories", lambda *args, **kwargs: [])

    def _fake_fallback(*args, **kwargs):
        fallback_calls.append(kwargs)
        return [fallback_result] if kwargs.get("field") == "title" else []

    monkeypatch.setattr(routes, "search_memories_across_categories", _fake_fallback)
    payload = MemorySearchRequest(agent_id="assistant", category="娱乐偏好", query="老婆你记得绯夜嘛")

    results, used_vector, duration_ms, query_terms, ignored_terms, category_found, query_analysis = routes._search_results(
        _FakeDb(),
        SimpleNamespace(id=uuid4()),
        payload,
    )

    assert results == [fallback_result]
    assert used_vector is False
    assert category_found is True
    assert duration_ms >= 0
    assert query_terms == ["绯夜"]
    assert ignored_terms == []
    assert fallback_calls == [{"field": "title", "min_matched_terms": 1}]
    assert query_analysis["fallback_used"] is True
    assert query_analysis["fallback_stage"] == "跨分类标题"
    assert query_analysis["category_result_count"] == 0
    assert query_analysis["title_fallback_count"] == 1
    assert query_analysis["content_fallback_count"] == 0


def test_search_results_uses_cross_category_content_fallback_after_title_miss(monkeypatch) -> None:
    now = datetime.now(UTC)
    fallback_result = SearchResult(
        memory_id=uuid4(),
        external_id="relation-memory",
        category="其它",
        title="用户关系：鸦羽与月见绫音",
        content="月见绫音与绯夜存在角色关系。",
        metadata={},
        created_at=now,
        updated_at=now,
        score=0.31,
        score_parts={"semantic": 0.0, "content": 1.0, "term": 1.0},
        embedding_status="disabled",
        matched_terms=["绯夜"],
        matched_fields=["正文"],
    )
    fallback_calls = []

    monkeypatch.setattr(routes, "query_terms_for_search", lambda *args, **kwargs: (["绯夜"], [], routes.default_query_analysis_meta("disabled")))
    monkeypatch.setattr(routes, "filter_high_frequency_terms", lambda *args, **kwargs: (args[4], []))
    monkeypatch.setattr(routes, "search_memories", lambda *args, **kwargs: [])

    def _fake_fallback(*args, **kwargs):
        fallback_calls.append(kwargs)
        return [fallback_result] if kwargs.get("field") == "content" else []

    monkeypatch.setattr(routes, "search_memories_across_categories", _fake_fallback)
    payload = MemorySearchRequest(agent_id="assistant", category="娱乐偏好", query="老婆你记得绯夜嘛")

    results, used_vector, duration_ms, query_terms, ignored_terms, category_found, query_analysis = routes._search_results(
        _FakeDb(),
        SimpleNamespace(id=uuid4()),
        payload,
    )

    assert results == [fallback_result]
    assert used_vector is False
    assert category_found is True
    assert duration_ms >= 0
    assert query_terms == ["绯夜"]
    assert ignored_terms == []
    assert fallback_calls == [
        {"field": "title", "min_matched_terms": 1},
        {"field": "content", "min_matched_terms": 1, "min_score": 0.18},
    ]
    assert query_analysis["fallback_used"] is True
    assert query_analysis["fallback_stage"] == "跨分类正文"
    assert query_analysis["category_result_count"] == 0
    assert query_analysis["title_fallback_count"] == 0
    assert query_analysis["content_fallback_count"] == 1


def test_search_results_uses_ai_query_analysis_when_available(monkeypatch) -> None:
    calls = []

    def _fake_search_memories(*args):
        calls.append(args)
        return []

    monkeypatch.setattr(
        routes,
        "get_llm_config",
        lambda db: SimpleNamespace(enabled=True, query_analysis_enabled=True, encrypted_api_key="encrypted"),
    )
    monkeypatch.setattr(routes, "decrypt_secret", lambda *args, **kwargs: "sk-test")
    monkeypatch.setattr(
        routes,
        "analyze_memory_request",
        lambda *args, **kwargs: routes.MemoryRequestAnalysis(
            category="偏好",
            matched_existing=True,
            confidence=0.9,
            reason="图片偏好",
            intent_summary="生成兔女郎图片",
            keywords=["黑丝", "兔女郎", "腿更粗", "身材更好"],
            negative_keywords=[],
            duration_ms=12.5,
            usage={"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
        ),
    )
    monkeypatch.setattr(routes, "search_memories", _fake_search_memories)
    payload = MemorySearchRequest(agent_id="assistant", category="图片", query="老婆换成黑丝，然后腿粗一点，身材更好一些")

    results, used_vector, duration_ms, query_terms, ignored_terms, category_found, query_analysis = routes._search_results(
        _FakeDb(),
        SimpleNamespace(id=uuid4()),
        payload,
    )

    assert results == []
    assert used_vector is False
    assert category_found is True
    assert duration_ms >= 0
    assert query_terms == ["黑丝", "兔女郎", "腿更粗", "身材更好"]
    assert ignored_terms == []
    assert query_analysis["keyword_source"] == "ai"
    assert query_analysis["analysis_mode"] == "combined"
    assert query_analysis["intent_summary"] == "生成兔女郎图片"
    assert query_analysis["ai_keywords"] == ["黑丝", "兔女郎", "腿更粗", "身材更好"]
    assert query_analysis["ai_duration_ms"] == 12.5
    assert query_analysis["ai_request_total_tokens"] == 100
    assert len(calls) == 1
    assert calls[0][4] == "黑丝 兔女郎 腿更粗 身材更好"
    assert calls[0][5] == ["黑丝", "兔女郎", "腿更粗", "身材更好"]


def test_search_results_uses_server_ai_category_when_missing(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        routes,
        "get_llm_config",
        lambda db: SimpleNamespace(enabled=True, query_analysis_enabled=True, encrypted_api_key="encrypted"),
    )
    monkeypatch.setattr(routes, "decrypt_secret", lambda *args, **kwargs: "sk-test")
    monkeypatch.setattr(
        routes,
        "analyze_memory_request",
        lambda *args, **kwargs: routes.MemoryRequestAnalysis(
            category="偏好",
            matched_existing=True,
            confidence=0.86,
            reason="请求回答偏好",
            duration_ms=9.0,
            intent_summary="回答偏好",
            keywords=["苹果"],
            negative_keywords=[],
            usage={"prompt_tokens": 32, "completion_tokens": 9, "total_tokens": 41},
        ),
    )
    monkeypatch.setattr(routes, "search_memories", lambda *args: calls.append(args) or [])
    monkeypatch.setattr(routes, "search_memories_across_categories", lambda *args, **kwargs: [])
    monkeypatch.setattr(routes, "filter_high_frequency_terms", lambda *args, **kwargs: (args[4], []))
    payload = MemorySearchRequest(agent_id="assistant", query="苹果")

    results, used_vector, duration_ms, query_terms, ignored_terms, category_found, query_analysis = routes._search_results(
        _CategoryListDb(),
        SimpleNamespace(id=uuid4()),
        payload,
    )

    assert results == []
    assert used_vector is False
    assert category_found is True
    assert payload.category == "偏好"
    assert query_terms == ["苹果"]
    assert query_analysis["category_source"] == "ai"
    assert query_analysis["selected_category"] == "偏好"
    assert query_analysis["category_confidence"] == 0.86
    assert query_analysis["category_total_tokens"] == 0
    assert query_analysis["ai_total_tokens"] == 41
    assert query_analysis["ai_request_total_tokens"] == 41
    assert query_analysis["analysis_mode"] == "combined"
    assert len(calls) == 1


def test_search_results_does_not_create_unknown_ai_category_for_query(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        routes,
        "get_llm_config",
        lambda db: SimpleNamespace(enabled=True, query_analysis_enabled=True, encrypted_api_key="encrypted"),
    )
    monkeypatch.setattr(routes, "decrypt_secret", lambda *args, **kwargs: "sk-test")
    monkeypatch.setattr(
        routes,
        "analyze_memory_request",
        lambda *args, **kwargs: routes.MemoryRequestAnalysis(
            category="临时问候",
            matched_existing=False,
            confidence=0.4,
            reason="不是已有分类",
            duration_ms=7.0,
            intent_summary="问候",
            keywords=[],
        ),
    )
    monkeypatch.setattr(routes, "search_memories", lambda *args: calls.append(args) or [])
    payload = MemorySearchRequest(agent_id="assistant", query="早")

    results, used_vector, duration_ms, query_terms, ignored_terms, category_found, query_analysis = routes._search_results(
        _NoCategoryDb(),
        SimpleNamespace(id=uuid4()),
        payload,
    )

    assert results == []
    assert used_vector is False
    assert category_found is False
    assert query_terms == []
    assert ignored_terms == ["临时问候:分类不存在"]
    assert query_analysis["category_source"] == "ai"
    assert query_analysis["selected_category"] == "临时问候"
    assert calls == []


def test_search_results_does_not_fallback_when_ai_keywords_are_empty(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        routes,
        "get_llm_config",
        lambda db: SimpleNamespace(enabled=True, query_analysis_enabled=True, encrypted_api_key="encrypted"),
    )
    monkeypatch.setattr(routes, "decrypt_secret", lambda *args, **kwargs: "sk-test")
    monkeypatch.setattr(
        routes,
        "analyze_memory_request",
        lambda *args, **kwargs: routes.MemoryRequestAnalysis(
            category="偏好",
            matched_existing=True,
            confidence=0.8,
            reason="图片偏好",
            intent_summary="弱请求词",
            keywords=["老婆", "换成", "一点", "一些"],
            negative_keywords=[],
            duration_ms=10.0,
        ),
    )
    monkeypatch.setattr(routes, "search_memories", lambda *args: calls.append(args) or [])
    payload = MemorySearchRequest(agent_id="assistant", category="图片", query="老婆换成黑丝")

    results, used_vector, duration_ms, query_terms, ignored_terms, category_found, query_analysis = routes._search_results(
        _FakeDb(),
        SimpleNamespace(id=uuid4()),
        payload,
    )

    assert results == []
    assert used_vector is False
    assert category_found is True
    assert query_terms == []
    assert "老婆:弱检索词" in ignored_terms
    assert query_analysis["keyword_source"] == "ai"
    assert query_analysis["ai_ignored_terms"] == ignored_terms
    assert calls == []


def test_search_results_does_not_require_ai_numeric_alias_to_match(monkeypatch) -> None:
    calls = []
    kwargs_list = []

    monkeypatch.setattr(
        routes,
        "get_llm_config",
        lambda db: SimpleNamespace(enabled=True, query_analysis_enabled=True, encrypted_api_key="encrypted"),
    )
    monkeypatch.setattr(routes, "decrypt_secret", lambda *args, **kwargs: "sk-test")
    monkeypatch.setattr(
        routes,
        "analyze_memory_request",
        lambda *args, **kwargs: routes.MemoryRequestAnalysis(
            category="偏好",
            matched_existing=True,
            confidence=1.0,
            reason="命运2是游戏。",
            intent_summary="询问命运2看法",
            keywords=["命运2", "destiny 2", "游戏评价"],
            negative_keywords=[],
            duration_ms=10.0,
        ),
    )

    def _fake_search_memories(*args, **kwargs):
        calls.append(args)
        kwargs_list.append(kwargs)
        return []

    monkeypatch.setattr(routes, "search_memories", _fake_search_memories)
    payload = MemorySearchRequest(agent_id="assistant", query="老婆你觉得命运2 怎么样")

    results, used_vector, duration_ms, query_terms, ignored_terms, category_found, query_analysis = routes._search_results(
        _FakeDb(),
        SimpleNamespace(id=uuid4()),
        payload,
    )

    assert results == []
    assert used_vector is False
    assert category_found is True
    assert query_terms == ["命运2", "destiny 2", "游戏评价"]
    assert ignored_terms == []
    assert query_analysis["min_matched_terms"] == 1
    assert kwargs_list == [{"min_matched_terms": 1}]
    assert calls[0][4] == "命运2 destiny 2 游戏评价"
    assert calls[0][5] == ["命运2", "destiny 2", "游戏评价"]


def test_search_results_falls_back_when_ai_query_analysis_fails(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        routes,
        "get_llm_config",
        lambda db: SimpleNamespace(enabled=True, query_analysis_enabled=True, encrypted_api_key="encrypted"),
    )
    monkeypatch.setattr(routes, "decrypt_secret", lambda *args, **kwargs: "sk-test")
    monkeypatch.setattr(routes, "analyze_memory_request", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad json")))
    monkeypatch.setattr(routes, "search_memories", lambda *args: calls.append(args) or [])
    payload = MemorySearchRequest(agent_id="assistant", category="偏好", query="苹果")

    results, used_vector, duration_ms, query_terms, ignored_terms, category_found, query_analysis = routes._search_results(
        _FakeDb(),
        SimpleNamespace(id=uuid4()),
        payload,
    )

    assert results == []
    assert used_vector is False
    assert category_found is True
    assert query_terms == ["苹果"]
    assert ignored_terms == []
    assert query_analysis["keyword_source"] == "failed"
    assert "bad json" in query_analysis["ai_error"]
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


def test_filter_high_frequency_terms_keeps_numeric_proper_nouns() -> None:
    terms, ignored = filter_high_frequency_terms(
        _HighFrequencyDb(),
        uuid4(),
        uuid4(),
        "assistant",
        ["不要", "命运2"],
    )

    assert terms == ["命运2"]
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
    assert "故障排查" in body["forbidden"]
    schema_text = json.dumps(body["output_schema"], ensure_ascii=False)
    assert "技术资料" not in schema_text
    assert "自动化任务" not in schema_text
    assert "配置、修复、故障" in body["prompt"]


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

    results, used_vector, duration_ms, query_terms, ignored_terms, category_found, query_analysis = routes._search_results(
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
    assert query_analysis["keyword_source"] == "disabled"
    assert len(calls[0]) == 6
    assert calls[0][4] == "自然 回复"
    assert calls[0][5] == ["自然", "回复"]
