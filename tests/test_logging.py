import json
import logging

from fastapi.testclient import TestClient

from aimemory.core.config import Settings, get_settings
from aimemory.core.logging import JsonFormatter, sanitize_for_log
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
    assert settings.slow_embedding_ms == 3000
