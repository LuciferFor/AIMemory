import json
import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient

from aimemory.admin.auth import COOKIE_NAME, get_serializer
from aimemory.core.config import get_settings
from aimemory.db.session import get_db
from aimemory.main import create_app
from aimemory.models.ai_chat import AiChatMessage, AiChatThread
from aimemory.models.ai_memory_review import AiMemoryReviewRun, AiMemoryReviewSuggestion
from aimemory.models.llm_provider_config import LlmProviderConfig
from aimemory.models.memory_category import MemoryCategory
from aimemory.models.search_stopword import SearchStopword
from aimemory.models.user import User
from aimemory.services.ai_crypto import decrypt_secret, encrypt_secret


class _Rows:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def all(self) -> list:
        return self._rows

    def one_or_none(self):
        return self._rows[0] if self._rows else None


class _FakeDb:
    def scalar(self, query) -> int:
        return 0

    def execute(self, query) -> _Rows:
        return _Rows([("ready", 0), ("pending", 0), ("failed", 0)])

    def scalars(self, query) -> _Rows:
        return _Rows([])


class _EmptyDb(_FakeDb):
    def execute(self, query) -> _Rows:
        return _Rows([])


class _ApiKeyDb(_FakeDb):
    def __init__(self, api_key=None) -> None:
        self.api_key = api_key
        self.added = []
        self.committed = False

    def get(self, model, key_id):
        if self.api_key is not None and self.api_key.id == key_id:
            return self.api_key
        return None

    def add(self, value) -> None:
        self.added.append(value)

    def commit(self) -> None:
        self.committed = True


class _MemoryDb(_FakeDb):
    def __init__(self) -> None:
        self.user_id = uuid.uuid4()
        self.attachment_id = uuid.uuid4()
        self.category_id = uuid.uuid4()
        self.users = [SimpleNamespace(id=self.user_id, name="lucifer")]
        self.categories = [
            SimpleNamespace(
                id=self.category_id,
                user_id=self.user_id,
                name="偏好",
                normalized_name="偏好",
                description="测试分类",
                deleted_at=None,
                merged_into_id=None,
            )
        ]
        self.agents = ["agent-1"]
        attachment = SimpleNamespace(
            id=self.attachment_id,
            filename="screen.png",
            mime_type="image/png",
            size_bytes=1234,
            description="截图描述",
            ocr_text="图片文字",
            deleted_at=None,
        )
        self.memories = [
            SimpleNamespace(
                id=uuid.uuid4(),
                title="测试记忆标题",
                content="第一行正文，第二行正文，第三行正文。",
                user_id=self.user_id,
                category_id=self.category_id,
                agent_id="agent-1",
                external_id="memory-active",
                metadata_json={"category": "test"},
                created_at="2026-05-30 10:00:00+00:00",
                updated_at="2026-05-30 10:01:00+00:00",
                deleted_at=None,
                attachments=[attachment],
            ),
            SimpleNamespace(
                id=uuid.uuid4(),
                title="已删除记忆",
                content="已删除正文",
                user_id=self.user_id,
                category_id=self.category_id,
                agent_id="agent-1",
                external_id="memory-deleted",
                metadata_json={},
                created_at="2026-05-30 09:00:00+00:00",
                updated_at="2026-05-30 09:01:00+00:00",
                deleted_at="2026-05-30 09:02:00+00:00",
                attachments=[],
            ),
        ]
        review_run_id = uuid.uuid4()
        self.review_runs = [
            AiMemoryReviewRun(
                id=review_run_id,
                admin_username="admin",
                status="completed",
                source="selection",
                request_summary={"memory_count": 2},
                response_summary={
                    "suggestion_count": 1,
                    "ai_usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 5,
                        "total_tokens": 17,
                        "cached_tokens": 3,
                    },
                },
                prompt_tokens=12,
                completion_tokens=5,
                total_tokens=17,
                created_at="2026-05-30 11:00:00+00:00",
            )
        ]
        self.applied_suggestions = [
            AiMemoryReviewSuggestion(
                id=uuid.uuid4(),
                run_id=review_run_id,
                suggestion_type="rewrite",
                status="applied",
                memory_ids=[str(self.memories[0].id)],
                target_memory_id=self.memories[0].id,
                proposed_json={},
                original_json={},
                created_at="2026-05-30 11:00:00+00:00",
                updated_at="2026-05-30 11:01:00+00:00",
                applied_at="2026-05-30 11:02:00+00:00",
            ),
        ]
        self.pending_suggestions = [
            AiMemoryReviewSuggestion(
                id=uuid.uuid4(),
                run_id=review_run_id,
                suggestion_type="rewrite",
                status="pending",
                memory_ids=[str(self.memories[1].id)],
                target_memory_id=self.memories[1].id,
                proposed_json={},
                original_json={},
                created_at="2026-05-30 11:00:00+00:00",
                updated_at="2026-05-30 11:01:00+00:00",
            ),
        ]
        self.scalars_calls = 0
        self.added = []
        self.committed = False

    def execute(self, query) -> _Rows:
        return _Rows([(memory, "lucifer", "偏好") for memory in self.memories])

    def scalar(self, query):
        return self.memories[0]

    def scalars(self, query) -> _Rows:
        self.scalars_calls += 1
        if self.scalars_calls == 1:
            return _Rows(self.applied_suggestions)
        if self.scalars_calls == 2:
            return _Rows(self.users)
        if self.scalars_calls == 3:
            return _Rows(self.categories)
        if self.scalars_calls == 4:
            return _Rows(self.agents)
        return _Rows(self.review_runs)

    def add(self, value) -> None:
        self.added.append(value)

    def commit(self) -> None:
        self.committed = True


class _RequestLogDb(_FakeDb):
    def __init__(self) -> None:
        self.user_id = uuid.uuid4()
        self.logs = [
            SimpleNamespace(
                id=uuid.uuid4(),
                request_id="request-log-123",
                source="api",
                method="POST",
                path="/v1/memories/context",
                route_path="/v1/memories/context",
                status_code=200,
                duration_ms=12.34,
                client_ip="192.168.31.9",
                user_agent="node",
                user_id=self.user_id,
                api_key_id=uuid.uuid4(),
                api_key_prefix="aim_test",
                admin_username=None,
                error_type=None,
                response_summary={
                    "type": "context",
                    "agent_id": "5df9cbfb-d31b-46dd-972b-05d466d2257c",
                    "category": "偏好",
                    "category_found": True,
                    "category_not_found": False,
                    "query": "回答偏好",
                    "top_k": 8,
                    "max_chars": 3000,
                    "query_terms": ["回答", "偏好"],
                    "ignored_terms": ["lucifer:停用词", "skill:停用词"],
                    "result_count": 1,
                    "context_chars": 128,
                    "truncated": False,
                    "context_text_preview": "以下是与当前请求可能相关的长期记忆。\n\n[长期记忆]\n\n1. 回复偏好\n用户喜欢短一点、自然一点的回答。",
                    "context_text_preview_truncated": False,
                    "items": [
                        {
                            "memory_id": str(uuid.uuid4()),
                            "external_id": "pref-short-replies",
                            "title": "回复偏好",
                            "score": 0.91,
                            "embedding_status": "disabled",
                            "matched_terms": ["回答", "偏好"],
                            "content_preview": "用户喜欢短一点、自然一点的回答。",
                        }
                    ],
                },
                created_at="2026-05-30 10:00:00+00:00",
            ),
            SimpleNamespace(
                id=uuid.uuid4(),
                request_id="admin-log-456",
                source="admin",
                method="GET",
                path="/admin",
                route_path="/admin",
                status_code=303,
                duration_ms=3.21,
                client_ip="127.0.0.1",
                user_agent="browser",
                user_id=None,
                api_key_id=None,
                api_key_prefix=None,
                admin_username="admin",
                error_type=None,
                response_summary=None,
                created_at="2026-05-30 10:01:00+00:00",
            ),
            SimpleNamespace(
                id=uuid.uuid4(),
                request_id="extract-log-789",
                source="api",
                method="POST",
                path="/v1/memories/extract",
                route_path="/v1/memories/extract",
                status_code=200,
                duration_ms=456.78,
                client_ip="192.168.31.9",
                user_agent="node",
                user_id=self.user_id,
                api_key_id=uuid.uuid4(),
                api_key_prefix="aim_test",
                admin_username=None,
                error_type=None,
                response_summary={
                    "type": "extract",
                    "agent_id": "5df9cbfb-d31b-46dd-972b-05d466d2257c",
                    "reason": "before_compaction",
                    "transcript_chars": 512,
                    "extracted": 2,
                    "written": 2,
                    "items": [
                        {
                            "external_id": "reply-style-cn-first",
                            "category": "回答风格",
                            "title": "中文优先",
                            "action": "updated",
                        }
                    ],
                },
                created_at="2026-05-30 10:02:00+00:00",
            ),
        ]
        self.users = [SimpleNamespace(id=self.user_id, name="lucifer")]
        self.scalars_calls = 0

    def execute(self, query) -> _Rows:
        return _Rows([(self.logs[0], "lucifer"), (self.logs[1], None), (self.logs[2], "lucifer")])

    def scalars(self, query) -> _Rows:
        return _Rows(self.users)


class _AiConfigDb(_FakeDb):
    def __init__(self) -> None:
        self.config = None
        self.added = []
        self.committed = False

    def scalar(self, query):
        return self.config

    def add(self, value) -> None:
        self.added.append(value)
        if isinstance(value, LlmProviderConfig):
            if value.id is None:
                value.id = uuid.uuid4()
            self.config = value

    def commit(self) -> None:
        self.committed = True


class _AiReviewDb(_MemoryDb):
    def __init__(self) -> None:
        super().__init__()
        self.config = LlmProviderConfig(
            id=uuid.uuid4(),
            name="default",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            encrypted_api_key=encrypt_secret("sk-test", "test-ai-secret"),
            api_key_hint="sk...test",
            timeout_ms=30000,
            max_output_tokens=4096,
            temperature=0.0,
            extra_body_json={},
            review_prompt_injection="先合并重复记忆，再谨慎改写。",
            enabled=True,
            query_analysis_enabled=True,
            query_analysis_max_output_tokens=256,
            query_analysis_timeout_ms=3000,
        )
        self.runs = {}
        self.suggestions = {}

    def scalar(self, query):
        return self.config

    def scalars(self, query) -> _Rows:
        return _Rows(self.categories)

    def add(self, value) -> None:
        if getattr(value, "id", None) is None:
            value.id = uuid.uuid4()
        if hasattr(value, "run_id") and isinstance(value, AiMemoryReviewSuggestion):
            self.suggestions[value.id] = value
        elif hasattr(value, "suggestions"):
            self.runs[value.id] = value
        self.added.append(value)

    def flush(self) -> None:
        for value in self.added:
            if getattr(value, "id", None) is None:
                value.id = uuid.uuid4()


class _AiReviewApplyAllDb(_FakeDb):
    def __init__(self) -> None:
        self.run_id = uuid.uuid4()
        self.pending_one = AiMemoryReviewSuggestion(
            id=uuid.uuid4(),
            run_id=self.run_id,
            suggestion_type="rewrite",
            status="pending",
            memory_ids=[str(uuid.uuid4())],
            proposed_json={"title": "标题一"},
            original_json={},
            created_at="2026-05-30 10:00:00+00:00",
        )
        self.pending_two = AiMemoryReviewSuggestion(
            id=uuid.uuid4(),
            run_id=self.run_id,
            suggestion_type="soft_delete",
            status="pending",
            memory_ids=[str(uuid.uuid4())],
            proposed_json={},
            original_json={},
            created_at="2026-05-30 10:01:00+00:00",
        )
        self.ignored = AiMemoryReviewSuggestion(
            id=uuid.uuid4(),
            run_id=self.run_id,
            suggestion_type="rewrite",
            status="ignored",
            memory_ids=[str(uuid.uuid4())],
            proposed_json={"title": "已忽略"},
            original_json={},
            created_at="2026-05-30 10:02:00+00:00",
        )
        self.run = AiMemoryReviewRun(
            id=self.run_id,
            admin_username="admin",
            status="completed",
            source="selection",
            request_summary={"memory_count": 2},
            response_summary={
                "suggestion_count": 3,
                "ai_usage": {"prompt_tokens": 7, "completion_tokens": 8, "total_tokens": 15, "cached_tokens": 2},
            },
            prompt_tokens=7,
            completion_tokens=8,
            total_tokens=15,
            created_at="2026-05-30 10:00:00+00:00",
        )
        self.run.suggestions = [self.pending_two, self.ignored, self.pending_one]
        self.rolled_back = False

    def scalar(self, query):
        return self.run

    def rollback(self) -> None:
        self.rolled_back = True


class _AiChatDb(_FakeDb):
    def __init__(self) -> None:
        self.thread_id = uuid.uuid4()
        self.thread = AiChatThread(id=self.thread_id, admin_username="admin", title="测试对话")
        self.thread.messages = []
        self.threads = [self.thread]
        self.config = LlmProviderConfig(
            id=uuid.uuid4(),
            name="default",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            encrypted_api_key=encrypt_secret("sk-test", "test-ai-secret"),
            api_key_hint="sk...test",
            timeout_ms=30000,
            max_output_tokens=4096,
            temperature=0.0,
            extra_body_json={},
            enabled=True,
            query_analysis_enabled=True,
            query_analysis_max_output_tokens=256,
            query_analysis_timeout_ms=3000,
        )
        self.added = []
        self.committed = False

    def scalar(self, query):
        text = str(query)
        if "ai_chat_threads" in text:
            return self.thread
        if "llm_provider_configs" in text:
            return self.config
        return self.config

    def scalars(self, query) -> _Rows:
        return _Rows(self.threads)

    def add(self, value) -> None:
        if getattr(value, "id", None) is None:
            value.id = uuid.uuid4()
        if isinstance(value, AiChatThread):
            value.messages = getattr(value, "messages", [])
            if value not in self.threads:
                self.threads.insert(0, value)
            self.thread = value
        if isinstance(value, AiChatMessage):
            target = next((item for item in self.threads if item.id == value.thread_id), self.thread)
            if value not in target.messages:
                target.messages.append(value)
        self.added.append(value)

    def commit(self) -> None:
        self.committed = True


class _StopwordDb(_FakeDb):
    def __init__(self) -> None:
        self.user_id = uuid.uuid4()
        self.stopword_id = uuid.uuid4()
        self.user = SimpleNamespace(id=self.user_id, name="lucifer")
        self.stopword = SimpleNamespace(
            id=self.stopword_id,
            user_id=self.user_id,
            term="lucifer",
            note="名字",
            created_at="2026-05-30 10:00:00+00:00",
            deleted_at=None,
        )
        self.added = []
        self.committed = False

    def execute(self, query) -> _Rows:
        return _Rows([(self.stopword, "lucifer")])

    def scalars(self, query) -> _Rows:
        return _Rows([self.user])

    def get(self, model, key):
        if model is User and key == self.user_id:
            return self.user
        if model is SearchStopword and key == self.stopword_id:
            return self.stopword
        return None

    def add(self, value) -> None:
        self.added.append(value)

    def commit(self) -> None:
        self.committed = True


class _CategoryDb(_FakeDb):
    def __init__(self, for_merge: bool = False) -> None:
        self.for_merge = for_merge
        self.user_id = uuid.uuid4()
        self.source_id = uuid.uuid4()
        self.target_id = uuid.uuid4()
        self.user = SimpleNamespace(id=self.user_id, name="lucifer")
        self.source = SimpleNamespace(
            id=self.source_id,
            user_id=self.user_id,
            name="爱吃水果",
            normalized_name="爱吃水果",
            description="水果偏好",
            deleted_at=None,
            merged_into_id=None,
            updated_at=None,
        )
        self.target = SimpleNamespace(
            id=self.target_id,
            user_id=self.user_id,
            name="爱吃的水果",
            normalized_name="爱吃的水果",
            description=None,
            deleted_at=None,
            merged_into_id=None,
            updated_at=None,
        )
        self.memory = SimpleNamespace(id=uuid.uuid4(), category_id=self.source_id, deleted_at=None, updated_at=None)
        self.added = []
        self.committed = False
        self.scalar_values: list[int] = []

    def execute(self, query) -> _Rows:
        return _Rows([(self.source, "lucifer", 1), (self.target, "lucifer", 0)])

    def scalars(self, query) -> _Rows:
        if self.for_merge:
            return _Rows([self.memory])
        return _Rows([self.user])

    def scalar(self, query):
        return self.scalar_values.pop(0) if self.scalar_values else 0

    def get(self, model, key):
        if model is User and key == self.user_id:
            return self.user
        if model is MemoryCategory and key == self.source_id:
            return self.source
        if model is MemoryCategory and key == self.target_id:
            return self.target
        return None

    def add(self, value) -> None:
        self.added.append(value)

    def commit(self) -> None:
        self.committed = True


def _client(monkeypatch, db=None) -> TestClient:
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-secret")
    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "false")
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db or _FakeDb()
    return TestClient(app)


def _login_and_csrf(client: TestClient) -> str:
    client.post("/admin/login", data={"username": "admin", "password": "secret", "next": "/admin"})
    token = client.cookies.get(COOKIE_NAME)
    assert token is not None
    return get_serializer().loads(token)["csrf"]


def test_admin_requires_login(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.get("/admin", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/login")


def test_root_redirects_to_admin_login(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/admin/login"


def test_root_head_redirects_to_admin_login(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.head("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/admin/login"


def test_admin_login_success(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.post(
        "/admin/login",
        data={"username": "admin", "password": "secret", "next": "/admin"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/admin"
    assert "aimemory_admin" in response.headers["set-cookie"]


def test_admin_login_failure(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.post(
        "/admin/login",
        data={"username": "admin", "password": "wrong", "next": "/admin"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/login")


def test_admin_dashboard_for_logged_in_user(monkeypatch) -> None:
    client = _client(monkeypatch)
    client.post("/admin/login", data={"username": "admin", "password": "secret", "next": "/admin"})

    response = client.get("/admin")

    assert response.status_code == 200
    assert "仪表盘" in response.text
    assert "待处理向量" not in response.text
    assert "失败向量" not in response.text


def test_admin_jobs_page_explains_disabled_embedding(monkeypatch) -> None:
    client = _client(monkeypatch)
    client.post("/admin/login", data={"username": "admin", "password": "secret", "next": "/admin"})

    response = client.get("/admin/jobs")

    assert response.status_code == 200
    assert "不再创建向量任务" in response.text


def test_admin_request_logs_page_lists_request_metadata(monkeypatch) -> None:
    db = _RequestLogDb()
    client = _client(monkeypatch, db=db)
    _login_and_csrf(client)

    response = client.get("/admin/request-logs?source=api&method=POST&status_code=200&q=context&user_id=&limit=100")

    assert response.status_code == 200
    assert "请求日志" in response.text
    assert "请求记忆" in response.text
    assert "/v1/memories/context" in response.text
    assert "request-log-123" in response.text
    assert "192.168.31.9" in response.text
    assert "aim_test" in response.text
    assert "POST" in response.text
    assert "12.34ms" in response.text
    assert "请求内容" in response.text
    assert "回答偏好" in response.text
    assert "请求参数" in response.text
    assert "分类 偏好" in response.text
    assert "top_k 8" in response.text
    assert "max_chars 3000" in response.text
    assert "有效关键词" in response.text
    assert "忽略关键词" in response.text
    assert "lucifer" in response.text
    assert "实际返回" in response.text
    assert "[长期记忆]" in response.text
    assert "命中关键词" in response.text
    assert "命中字段" in response.text
    assert "回复偏好" in response.text
    assert "pref-short-replies" in response.text
    assert "用户喜欢短一点、自然一点的回答。" in response.text
    assert "总结保存" in response.text
    assert "/v1/memories/extract" in response.text
    assert "提取 2" in response.text
    assert "写入 2" in response.text
    assert "对话 512 字" in response.text
    assert "中文优先" in response.text
    assert "reply-style-cn-first" in response.text
    assert "updated" in response.text
    assert "Authorization" not in response.text


def test_admin_ai_settings_requires_login(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.get("/admin/ai-settings", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/login")


def test_admin_ai_settings_page_shows_review_prompt_injection(monkeypatch) -> None:
    db = _AiConfigDb()
    db.config = LlmProviderConfig(
        id=uuid.uuid4(),
        name="default",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        encrypted_api_key=None,
        api_key_hint=None,
        timeout_ms=30000,
        max_output_tokens=4096,
        temperature=0.0,
        extra_body_json={},
        review_prompt_injection="按长期价值和重复度整理。",
        enabled=True,
        query_analysis_enabled=True,
        query_analysis_max_output_tokens=256,
        query_analysis_timeout_ms=3000,
    )
    client = _client(monkeypatch, db=db)
    _login_and_csrf(client)

    response = client.get("/admin/ai-settings")

    assert response.status_code == 200
    assert "整理前置注入提示" in response.text
    assert "按长期价值和重复度整理。" in response.text


def test_admin_can_save_encrypted_ai_settings(monkeypatch) -> None:
    monkeypatch.setenv("AI_CONFIG_ENCRYPTION_SECRET", "test-ai-secret")
    db = _AiConfigDb()
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        "/admin/ai-settings",
        data={
            "csrf_token": csrf,
            "base_url": "https://api.deepseek.com/",
            "model": "deepseek-v4-flash",
            "api_key": "sk-secret-value",
            "timeout_ms": "45000",
            "max_output_tokens": "2048",
            "temperature": "0",
            "extra_body_json": '{"thinking":{"type":"disabled"}}',
            "review_prompt_injection": "先压缩重复表达，再给出谨慎建议。",
            "enabled": "on",
            "query_analysis_enabled": "on",
            "query_analysis_max_output_tokens": "256",
            "query_analysis_timeout_ms": "3000",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/ai-settings")
    assert db.config is not None
    assert db.config.base_url == "https://api.deepseek.com"
    assert db.config.model == "deepseek-v4-flash"
    assert db.config.encrypted_api_key != "sk-secret-value"
    assert decrypt_secret(db.config.encrypted_api_key, "test-ai-secret") == "sk-secret-value"
    assert db.config.api_key_hint.startswith("sk")
    assert db.config.timeout_ms == 45000
    assert db.config.max_output_tokens == 2048
    assert db.config.extra_body_json == {"thinking": {"type": "disabled"}}
    assert db.config.review_prompt_injection == "先压缩重复表达，再给出谨慎建议。"
    assert db.config.enabled is True
    assert db.config.query_analysis_enabled is True
    assert db.config.query_analysis_max_output_tokens == 256
    assert db.config.query_analysis_timeout_ms == 3000
    assert db.committed is True


def test_admin_ai_settings_test_shows_token_usage(monkeypatch) -> None:
    monkeypatch.setenv("AI_CONFIG_ENCRYPTION_SECRET", "test-ai-secret")
    db = _AiReviewDb()

    def fake_chat_completion(config, api_key, messages, **kwargs):
        return SimpleNamespace(
            content='{"ok": true}',
            usage={"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
        )

    monkeypatch.setattr("aimemory.admin.routes.chat_completion", fake_chat_completion)
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        "/admin/ai-settings/test",
        data={"csrf_token": csrf},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "AI+%E8%BF%9E%E6%8E%A5%E6%B5%8B%E8%AF%95%E6%88%90%E5%8A%9F" in response.headers["location"]
    assert "%E8%BE%93%E5%85%A5+6" in response.headers["location"]
    assert "%E8%BE%93%E5%87%BA+2" in response.headers["location"]
    assert "%E6%80%BB+8+tokens" in response.headers["location"]


def test_admin_ai_chat_requires_login(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.get("/admin/ai-chat", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/login")


def test_admin_ai_chat_home_lists_threads(monkeypatch) -> None:
    monkeypatch.setenv("AI_CONFIG_ENCRYPTION_SECRET", "test-ai-secret")
    db = _AiChatDb()
    client = _client(monkeypatch, db=db)
    _login_and_csrf(client)

    response = client.get("/admin/ai-chat")

    assert response.status_code == 200
    assert "AI 对话" in response.text
    assert "新对话" in response.text
    assert "测试对话" in response.text


def test_admin_can_create_ai_chat_thread(monkeypatch) -> None:
    monkeypatch.setenv("AI_CONFIG_ENCRYPTION_SECRET", "test-ai-secret")
    db = _AiChatDb()
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post("/admin/ai-chat", data={"csrf_token": csrf}, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/ai-chat/")
    assert any(isinstance(item, AiChatThread) and item.title == "新对话" for item in db.added)
    assert db.committed is True


def test_admin_can_create_ai_chat_thread_with_first_message(monkeypatch) -> None:
    monkeypatch.setenv("AI_CONFIG_ENCRYPTION_SECRET", "test-ai-secret")
    db = _AiChatDb()
    captured = {}

    def fake_generate(db_arg, *, config, api_key, history):
        captured["api_key"] = api_key
        captured["history"] = history
        return {
            "content": "自动创建后的回复。",
            "metadata": {},
            "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
        }

    monkeypatch.setattr("aimemory.admin.routes.generate_ai_chat_reply", fake_generate)
    monkeypatch.setattr(
        "aimemory.admin.routes.generate_ai_chat_title_result",
        lambda config, api_key, content: {"title": "自动标题", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
    )
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        "/admin/ai-chat",
        data={"csrf_token": csrf, "content": "直接查一下请求日志"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/ai-chat/")
    assert captured["api_key"] == "sk-test"
    assert captured["history"][-1].content == "直接查一下请求日志"
    assert db.thread.title == "自动标题"
    assert db.thread.messages[-2].role == "user"
    assert db.thread.messages[-1].content == "自动创建后的回复。"
    assert db.thread.messages[-1].metadata_json["ai_usage_breakdown"]["title"]["total_tokens"] == 2
    assert db.thread.messages[-1].total_tokens == 11


def test_admin_ai_chat_json_create_saves_user_before_reply(monkeypatch) -> None:
    monkeypatch.setenv("AI_CONFIG_ENCRYPTION_SECRET", "test-ai-secret")
    db = _AiChatDb()
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        "/admin/ai-chat",
        data={"csrf_token": csrf, "content": "先显示这条消息"},
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["reply_url"].endswith("/reply")
    assert db.thread.messages[-1].role == "user"
    assert db.thread.messages[-1].content == "先显示这条消息"


def test_admin_ai_chat_reply_endpoint_generates_assistant(monkeypatch) -> None:
    monkeypatch.setenv("AI_CONFIG_ENCRYPTION_SECRET", "test-ai-secret")
    db = _AiChatDb()

    def fake_generate(db_arg, *, config, api_key, history):
        return {"content": "异步 AI 回复。", "metadata": {}, "usage": {"total_tokens": 8}}

    monkeypatch.setattr("aimemory.admin.routes.generate_ai_chat_reply", fake_generate)
    monkeypatch.setattr(
        "aimemory.admin.routes.generate_ai_chat_title_result",
        lambda config, api_key, content: {"title": "异步标题", "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
    )
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)
    client.post(
        "/admin/ai-chat",
        data={"csrf_token": csrf, "content": "异步问一句"},
        headers={"Accept": "application/json"},
    )

    response = client.post(
        f"/admin/ai-chat/{db.thread.id}/reply",
        data={"csrf_token": csrf},
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["assistant"]["content"] == "异步 AI 回复。"
    assert payload["assistant"]["total_tokens"] == 11
    assert payload["assistant"]["usage"]["total_tokens"] == 11
    assert payload["assistant"]["usage_breakdown"]["title"]["total_tokens"] == 3
    assert db.thread.messages[-1].role == "assistant"
    assert db.thread.title == "异步标题"


def test_admin_ai_chat_sends_message_and_saves_reply(monkeypatch) -> None:
    monkeypatch.setenv("AI_CONFIG_ENCRYPTION_SECRET", "test-ai-secret")
    db = _AiChatDb()
    captured = {}

    def fake_generate(db_arg, *, config, api_key, history):
        captured["api_key"] = api_key
        captured["history"] = history
        return {
            "content": "这里是 AI 回复。",
            "metadata": {
                "sql_results": [
                    {
                        "title": "分类数量",
                        "purpose": "查看分类",
                        "sql": "select count(*) from memory_categories",
                        "status": "ok",
                        "columns": ["count"],
                        "rows": [{"count": 3}],
                        "row_count": 1,
                        "truncated": False,
                        "error": None,
                    }
                ]
            },
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    monkeypatch.setattr("aimemory.admin.routes.generate_ai_chat_reply", fake_generate)
    monkeypatch.setattr(
        "aimemory.admin.routes.generate_ai_chat_title_result",
        lambda config, api_key, content: {"title": "分类查询", "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}},
    )
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        f"/admin/ai-chat/{db.thread_id}/messages",
        data={"csrf_token": csrf, "content": "查一下分类数量"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/admin/ai-chat/{db.thread_id}"
    assert captured["api_key"] == "sk-test"
    assert captured["history"][-1].content == "查一下分类数量"
    assert db.thread.messages[-2].role == "user"
    assert db.thread.messages[-1].role == "assistant"
    assert db.thread.messages[-1].content == "这里是 AI 回复。"
    assert db.thread.messages[-1].metadata_json["sql_results"][0]["row_count"] == 1
    assert db.thread.messages[-1].metadata_json["ai_usage_breakdown"]["title"]["total_tokens"] == 4
    assert db.thread.messages[-1].total_tokens == 7
    assert db.thread.title == "分类查询"


def test_admin_can_delete_ai_chat_thread(monkeypatch) -> None:
    monkeypatch.setenv("AI_CONFIG_ENCRYPTION_SECRET", "test-ai-secret")
    db = _AiChatDb()
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        f"/admin/ai-chat/{db.thread_id}/delete",
        data={"csrf_token": csrf},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/ai-chat")
    assert db.thread.deleted_at is not None
    assert db.committed is True


def test_admin_ai_review_creates_suggestions(monkeypatch) -> None:
    monkeypatch.setenv("AI_CONFIG_ENCRYPTION_SECRET", "test-ai-secret")
    db = _AiReviewDb()
    captured = {}

    def fake_chat_completion(config, api_key, messages, **kwargs):
        captured["api_key"] = api_key
        captured["messages"] = messages
        captured["response_format"] = kwargs.get("response_format")
        return SimpleNamespace(
            content=json.dumps(
                {
                    "suggestions": [
                        {
                            "type": "rewrite",
                            "memory_ids": [str(db.memories[0].id)],
                            "target_memory_id": str(db.memories[0].id),
                            "proposed": {"title": "压缩标题", "content": "压缩正文", "category": "偏好"},
                            "reason": "去掉重复表达。",
                            "confidence": 0.9,
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    monkeypatch.setattr("aimemory.services.ai_memory_review.chat_completion", fake_chat_completion)
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        "/admin/memories/ai-review",
        data={"csrf_token": csrf, "memory_ids": str(db.memories[0].id)},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/ai-reviews/")
    assert captured["api_key"] == "sk-test"
    assert captured["messages"][0]["content"].startswith("管理员整理前置注入提示：\n先合并重复记忆，再谨慎改写。")
    assert "只输出 JSON" in captured["messages"][0]["content"]
    assert captured["response_format"] == {"type": "json_object"}
    assert len(db.suggestions) == 1
    suggestion = next(iter(db.suggestions.values()))
    assert suggestion.suggestion_type == "rewrite"
    assert suggestion.proposed_json["title"] == "压缩标题"
    assert suggestion.proposed_json["content"] == "压缩正文"
    assert suggestion.proposed_json["category"] == "偏好"
    assert suggestion.reason == "去掉重复表达。"
    run = next(iter(db.runs.values()))
    assert run.prompt_tokens == 10
    assert run.completion_tokens == 20
    assert run.total_tokens == 30
    assert run.response_summary["ai_usage"] == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    assert db.committed is True


def test_admin_ai_review_page_shows_apply_all(monkeypatch) -> None:
    db = _AiReviewApplyAllDb()
    client = _client(monkeypatch, db=db)
    _login_and_csrf(client)

    response = client.get(f"/admin/ai-reviews/{db.run_id}")

    assert response.status_code == 200
    assert "全部应用 2 条" in response.text
    assert f"/admin/ai-reviews/{db.run_id}/apply-all" in response.text
    assert "输入 7 tokens" in response.text
    assert "输出 8 tokens" in response.text
    assert "总 15 tokens" in response.text
    assert "缓存 2 tokens" in response.text


def test_admin_can_apply_all_pending_ai_suggestions(monkeypatch) -> None:
    db = _AiReviewApplyAllDb()
    applied_ids = []

    def fake_apply_suggestion(_db, suggestion) -> None:
        applied_ids.append(suggestion.id)
        suggestion.status = "applied"

    monkeypatch.setattr("aimemory.admin.routes.apply_suggestion", fake_apply_suggestion)
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        f"/admin/ai-reviews/{db.run_id}/apply-all",
        data={"csrf_token": csrf},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith(f"/admin/ai-reviews/{db.run_id}")
    assert applied_ids == [db.pending_one.id, db.pending_two.id]
    assert db.ignored.status == "ignored"


def test_admin_search_stopwords_page_lists_terms(monkeypatch) -> None:
    db = _StopwordDb()
    client = _client(monkeypatch, db=db)
    _login_and_csrf(client)

    response = client.get(f"/admin/search-stopwords?user_id={db.user_id}")

    assert response.status_code == 200
    assert "停用词" in response.text
    assert "lucifer" in response.text
    assert "名字" in response.text
    assert str(db.stopword_id) in response.text


def test_admin_can_add_search_stopword(monkeypatch) -> None:
    db = _StopwordDb()
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        "/admin/search-stopwords",
        data={
            "csrf_token": csrf,
            "selected_user_id": str(db.user_id),
            "user_id": str(db.user_id),
            "term": " Skill ",
            "note": " 高频词 ",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/search-stopwords")
    assert db.added[-1].term == "skill"
    assert db.added[-1].note == "高频词"
    assert db.committed is True


def test_admin_rejects_numeric_search_stopword(monkeypatch) -> None:
    db = _StopwordDb()
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        "/admin/search-stopwords",
        data={
            "csrf_token": csrf,
            "selected_user_id": str(db.user_id),
            "user_id": str(db.user_id),
            "term": "2026",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    assert db.added == []
    assert db.committed is False


def test_admin_can_delete_search_stopword(monkeypatch) -> None:
    db = _StopwordDb()
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        f"/admin/search-stopwords/{db.stopword_id}/delete",
        data={"csrf_token": csrf, "selected_user_id": str(db.user_id)},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert db.stopword.deleted_at is not None
    assert db.committed is True


def test_admin_categories_page_lists_categories(monkeypatch) -> None:
    db = _CategoryDb()
    client = _client(monkeypatch, db=db)
    _login_and_csrf(client)

    response = client.get(f"/admin/categories?user_id={db.user_id}")

    assert response.status_code == 200
    assert "分类" in response.text
    assert "爱吃水果" in response.text
    assert "水果偏好" in response.text
    assert str(db.source_id) in response.text


def test_admin_can_merge_category(monkeypatch) -> None:
    db = _CategoryDb(for_merge=True)
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        f"/admin/categories/{db.source_id}/merge",
        data={"csrf_token": csrf, "target_category_id": str(db.target_id)},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert db.memory.category_id == db.target_id
    assert db.source.merged_into_id == db.target_id
    assert db.source.deleted_at is not None
    assert db.committed is True


def test_admin_memories_page_uses_compact_table(monkeypatch) -> None:
    db = _MemoryDb()
    client = _client(monkeypatch, db=db)
    _login_and_csrf(client)

    response = client.get("/admin/memories")

    assert response.status_code == 200
    assert "memory-table" in response.text
    assert "最近 AI 整理记录" in response.text
    assert "查看结果" in response.text
    assert "输入 12" in response.text
    assert "输出 5" in response.text
    assert "总 17" in response.text
    assert "缓存 3" in response.text
    assert "is-ai-applied" in response.text
    assert response.text.count("AI 已应用") == 1
    assert "改写压缩 / 2026-05-30 19:02:00 北京时间" in response.text
    assert "col-actions" in response.text
    assert "全选" in response.text
    assert 'data-select-all="memory_ids"' in response.text
    assert "action-buttons" in response.text
    assert "action-button detail" in response.text
    assert "action-button danger" in response.text
    assert "测试记忆标题" in response.text
    assert "agent-1" in response.text
    assert "第一行正文" not in response.text
    assert "memory-active" not in response.text
    assert f"/admin/memories/{db.memories[0].id}" in response.text
    assert "05-30 18:00" in response.text
    assert f"/admin/attachments/{db.attachment_id}" in response.text
    assert "已删除" in response.text
    assert response.text.count('data-confirm="确认删除这条记忆？"') == 1


def test_admin_memories_search_accepts_empty_user_id(monkeypatch) -> None:
    db = _MemoryDb()
    client = _client(monkeypatch, db=db)
    _login_and_csrf(client)

    response = client.get("/admin/memories?user_id=&agent_id=&q=&deleted=active&since=&until=&limit=50")

    assert response.status_code == 200
    assert "测试记忆标题" in response.text


def test_admin_api_keys_filter_accepts_empty_user_id(monkeypatch) -> None:
    client = _client(monkeypatch, db=_EmptyDb())
    _login_and_csrf(client)

    response = client.get("/admin/api-keys?user_id=")

    assert response.status_code == 200
    assert "暂无接口密钥" in response.text


def test_admin_memory_detail_page_shows_full_memory(monkeypatch) -> None:
    db = _MemoryDb()
    client = _client(monkeypatch, db=db)
    _login_and_csrf(client)

    response = client.get(f"/admin/memories/{db.memories[0].id}")

    assert response.status_code == 200
    assert "测试记忆标题" in response.text
    assert "第一行正文，第二行正文，第三行正文。" in response.text
    assert "memory_id" in response.text
    assert str(db.memories[0].id) in response.text
    assert "agent-1" in response.text
    assert "memory-active" in response.text
    assert "偏好" in response.text
    assert "2026-05-30 18:00:00 北京时间" in response.text
    assert f"/admin/attachments/{db.attachment_id}" in response.text
    assert "screen.png" in response.text


def test_admin_can_update_memory_title_and_content(monkeypatch) -> None:
    db = _MemoryDb()
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        f"/admin/memories/{db.memories[0].id}/update",
        data={"csrf_token": csrf, "title": " 新标题 ", "content": " 新正文 "},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith(f"/admin/memories/{db.memories[0].id}")
    assert db.memories[0].title == "新标题"
    assert db.memories[0].content == "新正文"
    assert "新标题" in db.memories[0].search_text
    assert "新正文" in db.memories[0].search_text
    assert "screen.png" in db.memories[0].search_text
    assert db.committed is True


def test_admin_rejects_invalid_memory_update(monkeypatch) -> None:
    db = _MemoryDb()
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        f"/admin/memories/{db.memories[0].id}/update",
        data={"csrf_token": csrf, "title": "", "content": "正文"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    assert db.memories[0].title == "测试记忆标题"
    assert db.committed is False


def test_admin_can_update_api_key_label(monkeypatch) -> None:
    key_id = uuid.uuid4()
    api_key = SimpleNamespace(id=key_id, label="旧标签")
    db = _ApiKeyDb(api_key)
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        f"/admin/api-keys/{key_id}/label",
        data={"csrf_token": csrf, "label": "  新标签  ", "selected_user_id": "lucifer"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/api-keys")
    assert "user_id=lucifer" in response.headers["location"]
    assert api_key.label == "新标签"
    assert db.committed is True


def test_admin_can_clear_api_key_label(monkeypatch) -> None:
    key_id = uuid.uuid4()
    api_key = SimpleNamespace(id=key_id, label="旧标签")
    db = _ApiKeyDb(api_key)
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        f"/admin/api-keys/{key_id}/label",
        data={"csrf_token": csrf, "label": "   "},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert api_key.label is None
    assert db.committed is True


def test_admin_rejects_overlong_api_key_label(monkeypatch) -> None:
    key_id = uuid.uuid4()
    api_key = SimpleNamespace(id=key_id, label="旧标签")
    db = _ApiKeyDb(api_key)
    client = _client(monkeypatch, db=db)
    csrf = _login_and_csrf(client)

    response = client.post(
        f"/admin/api-keys/{key_id}/label",
        data={"csrf_token": csrf, "label": "x" * 129},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/api-keys")
    assert api_key.label == "旧标签"
    assert db.committed is False


def test_admin_update_api_key_label_requires_login(monkeypatch) -> None:
    key_id = uuid.uuid4()
    db = _ApiKeyDb(SimpleNamespace(id=key_id, label="旧标签"))
    client = _client(monkeypatch, db=db)

    response = client.post(f"/admin/api-keys/{key_id}/label", data={"label": "新标签"}, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/admin/login")
    assert db.committed is False


def test_admin_update_api_key_label_missing_key_returns_404(monkeypatch) -> None:
    key_id = uuid.uuid4()
    client = _client(monkeypatch, db=_ApiKeyDb())
    csrf = _login_and_csrf(client)

    response = client.post(
        f"/admin/api-keys/{key_id}/label",
        data={"csrf_token": csrf, "label": "新标签"},
    )

    assert response.status_code == 404
