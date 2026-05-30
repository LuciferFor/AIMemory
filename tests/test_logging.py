import json
import logging
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from aimemory.core.config import Settings, get_settings
from aimemory.core.logging import JsonFormatter, sanitize_for_log
from aimemory.core.security import hash_api_key
from aimemory.db.session import get_db
from aimemory.main import create_app


class _FakeRequestLogger:
    def __init__(self) -> None:
        self.records = []

    def log(self, level, message, extra=None) -> None:
        self.records.append((level, message, extra or {}))

    def exception(self, message, extra=None) -> None:
        self.records.append((logging.ERROR, message, extra or {}))


def test_json_formatter_outputs_valid_json() -> None:
    record = logging.LogRecord(
        name="aimemory.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.event = "test.event"
    record.api_key = "aim_abcdefghijklmnopqrstuvwxyz"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["event"] == "test.event"
    assert payload["api_key"] == "[REDACTED]"


def test_sanitize_for_log_redacts_sensitive_values() -> None:
    payload = sanitize_for_log(
        {
            "Authorization": "Bearer abc123",
            "password": "secret",
            "message": "token=abc123 api_key=def456 aim_abcdefghijklmnopqrstuvwxyz",
        }
    )

    rendered = json.dumps(payload, ensure_ascii=False)
    assert "abc123" not in rendered
    assert "secret" not in rendered
    assert "def456" not in rendered
    assert "abcdefghijklmnopqrstuvwxyz" not in rendered


def test_request_id_header_is_generated(monkeypatch) -> None:
    monkeypatch.setenv("LOG_FORMAT", "json")
    get_settings.cache_clear()
    app = create_app()
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]


def test_request_id_header_is_propagated(monkeypatch) -> None:
    get_settings.cache_clear()
    app = create_app()
    client = TestClient(app)

    response = client.get("/healthz", headers={"X-Request-ID": "request-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-123"


def test_healthz_does_not_emit_normal_request_log(monkeypatch) -> None:
    import aimemory.main as main_module

    fake_logger = _FakeRequestLogger()
    monkeypatch.setattr(main_module, "request_logger", fake_logger)
    get_settings.cache_clear()
    app = main_module.create_app()
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert fake_logger.records == []


def test_settings_include_logging_defaults() -> None:
    settings = Settings()

    assert settings.log_level == "INFO"
    assert settings.log_format == "json"
    assert settings.slow_request_ms == 1000
    assert settings.request_log_db_enabled is True


def test_unauthorized_api_request_is_written_to_request_log(monkeypatch) -> None:
    import aimemory.main as main_module

    records = []
    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "true")
    monkeypatch.setattr(main_module, "insert_request_log", lambda data: records.append(data.copy()))
    get_settings.cache_clear()
    client = TestClient(main_module.create_app())

    response = client.post("/v1/memories/context", json={"agent_id": "assistant", "category": "偏好", "query": "secret memory text"})

    assert response.status_code == 401
    assert len(records) == 1
    record = records[0]
    assert record["source"] == "api"
    assert record["method"] == "POST"
    assert record["path"] == "/v1/memories/context"
    assert record["status_code"] == 401
    assert record["user_id"] is None
    rendered = json.dumps(record, ensure_ascii=False, default=str)
    assert "secret memory text" not in rendered
    assert "Authorization" not in rendered


def test_authorized_api_request_log_includes_api_identity(monkeypatch) -> None:
    import aimemory.main as main_module

    raw_key = "aim_test_request_log_key"
    user_id = uuid4()
    api_key_id = uuid4()
    api_key = SimpleNamespace(
        id=api_key_id,
        user_id=user_id,
        key_hash=hash_api_key(raw_key),
        key_prefix="aim_test",
        revoked_at=None,
        user=SimpleNamespace(id=user_id, is_active=True),
        last_used_at=None,
    )

    class FakeDb:
        def scalar(self, query):
            return api_key

        def execute(self, query):
            return SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: []))

        def add(self, value) -> None:
            pass

        def commit(self) -> None:
            pass

    records = []
    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "true")
    monkeypatch.setattr(main_module, "insert_request_log", lambda data: records.append(data.copy()))
    get_settings.cache_clear()
    app = main_module.create_app()
    app.dependency_overrides[get_db] = lambda: FakeDb()
    client = TestClient(app)

    response = client.get("/v1/memories/write-policy", headers={"Authorization": f"Bearer {raw_key}"})

    assert response.status_code == 200
    assert len(records) == 1
    record = records[0]
    assert record["user_id"] == user_id
    assert record["api_key_id"] == api_key_id
    assert record["api_key_prefix"] == "aim_test"
    rendered = json.dumps(record, ensure_ascii=False, default=str)
    assert raw_key not in rendered


def test_admin_request_log_includes_admin_username(monkeypatch) -> None:
    import aimemory.main as main_module

    records = []
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-secret")
    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "true")
    monkeypatch.setattr(main_module, "insert_request_log", lambda data: records.append(data.copy()))
    get_settings.cache_clear()
    client = TestClient(main_module.create_app())

    response = client.post("/admin/login", data={"username": "admin", "password": "secret", "next": "/admin"}, follow_redirects=False)

    assert response.status_code == 303
    assert len(records) == 1
    assert records[0]["source"] == "admin"
    assert records[0]["admin_username"] == "admin"


def test_request_log_write_failure_does_not_block_response(monkeypatch) -> None:
    import aimemory.main as main_module

    def fail_insert(data) -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "true")
    monkeypatch.setattr(main_module, "insert_request_log", fail_insert)
    get_settings.cache_clear()
    client = TestClient(main_module.create_app())

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/admin/login"


def test_skipped_paths_are_not_written_to_request_log(monkeypatch) -> None:
    import aimemory.main as main_module

    records = []
    monkeypatch.setenv("REQUEST_LOG_DB_ENABLED", "true")
    monkeypatch.setattr(main_module, "insert_request_log", lambda data: records.append(data.copy()))
    get_settings.cache_clear()
    client = TestClient(main_module.create_app())

    health_response = client.get("/healthz")
    static_response = client.get("/admin/static/admin.css")

    assert health_response.status_code == 200
    assert static_response.status_code == 200
    assert records == []
