import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient

from aimemory.admin.auth import COOKIE_NAME, get_serializer
from aimemory.core.config import get_settings
from aimemory.db.session import get_db
from aimemory.main import create_app
from aimemory.models.search_stopword import SearchStopword
from aimemory.models.user import User


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
        self.users = [SimpleNamespace(id=self.user_id, name="lucifer")]
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
                agent_id="agent-1",
                external_id="memory-deleted",
                metadata_json={},
                created_at="2026-05-30 09:00:00+00:00",
                updated_at="2026-05-30 09:01:00+00:00",
                deleted_at="2026-05-30 09:02:00+00:00",
                attachments=[],
            ),
        ]
        self.scalars_calls = 0
        self.added = []
        self.committed = False

    def execute(self, query) -> _Rows:
        return _Rows([(memory, "lucifer") for memory in self.memories])

    def scalar(self, query):
        return self.memories[0]

    def scalars(self, query) -> _Rows:
        self.scalars_calls += 1
        if self.scalars_calls == 1:
            return _Rows(self.users)
        return _Rows(self.agents)

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
                    "query": "回答偏好",
                    "top_k": 8,
                    "max_chars": 3000,
                    "query_terms": ["回答", "偏好"],
                    "ignored_terms": ["lucifer:停用词", "skill:停用词"],
                    "result_count": 1,
                    "context_chars": 128,
                    "truncated": False,
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
        ]
        self.users = [SimpleNamespace(id=self.user_id, name="lucifer")]
        self.scalars_calls = 0

    def execute(self, query) -> _Rows:
        return _Rows([(self.logs[0], "lucifer"), (self.logs[1], None)])

    def scalars(self, query) -> _Rows:
        return _Rows(self.users)


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
    assert "/v1/memories/context" in response.text
    assert "request-log-123" in response.text
    assert "192.168.31.9" in response.text
    assert "aim_test" in response.text
    assert "POST" in response.text
    assert "12.34ms" in response.text
    assert "请求内容" in response.text
    assert "回答偏好" in response.text
    assert "请求参数" in response.text
    assert "top_k 8" in response.text
    assert "max_chars 3000" in response.text
    assert "有效关键词" in response.text
    assert "忽略关键词" in response.text
    assert "lucifer" in response.text
    assert "命中关键词" in response.text
    assert "命中字段" in response.text
    assert "回复偏好" in response.text
    assert "pref-short-replies" in response.text
    assert "用户喜欢短一点、自然一点的回答。" in response.text
    assert "Authorization" not in response.text


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


def test_admin_memories_page_uses_compact_table(monkeypatch) -> None:
    db = _MemoryDb()
    client = _client(monkeypatch, db=db)
    _login_and_csrf(client)

    response = client.get("/admin/memories")

    assert response.status_code == 200
    assert "memory-table" in response.text
    assert "col-actions" in response.text
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
    assert "category" in response.text
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
