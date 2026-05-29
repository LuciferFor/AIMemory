from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

REQUEST_ID_HEADER = "X-Request-ID"

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

_SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "admin_password",
    "password",
    "passwd",
    "token",
    "access_token",
    "refresh_token",
    "secret",
    "session_secret",
    "embedding_api_key",
    "credentials",
    "credential",
    "content",
    "metadata",
    "input",
    "embedding_input",
    "embedding",
    "vector",
    "embedding_vector",
}

_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_AIM_KEY_RE = re.compile(r"\baim_[A-Za-z0-9._~+/=-]{12,}\b")
_QUERY_SECRET_RE = re.compile(
    r"([?&](?:api[_-]?key|token|access_token|refresh_token|password|secret)=)[^&\s]+",
    re.IGNORECASE,
)
_ASSIGNMENT_SECRET_RE = re.compile(
    r"\b(api[_-]?key|token|access_token|refresh_token|password|secret|authorization)=([^\s,;]+)",
    re.IGNORECASE,
)
_SAFE_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

_RESERVED_LOG_RECORD_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


def new_request_id() -> str:
    return uuid.uuid4().hex


def normalize_request_id(value: str | None) -> str:
    if value:
        candidate = value.strip()
        if _SAFE_REQUEST_ID_RE.fullmatch(candidate):
            return candidate
    return new_request_id()


def set_request_id(value: str) -> contextvars.Token[str]:
    return _request_id_var.set(value)


def reset_request_id(token: contextvars.Token[str]) -> None:
    _request_id_var.reset(token)


def get_request_id() -> str:
    return _request_id_var.get()


def sanitize_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _SENSITIVE_KEYS:
                sanitized[key_text] = "[REDACTED]"
            else:
                sanitized[key_text] = sanitize_for_log(item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_log(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value))


def redact_text(value: str) -> str:
    redacted = _BEARER_RE.sub("Bearer [REDACTED]", value)
    redacted = _AIM_KEY_RE.sub("aim_[REDACTED]", redacted)
    redacted = _QUERY_SECRET_RE.sub(r"\1[REDACTED]", redacted)
    redacted = _ASSIGNMENT_SECRET_RE.sub(r"\1=[REDACTED]", redacted)
    return redacted


def url_host(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    return parsed.netloc or parsed.path


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = get_request_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_for_log(record.getMessage()),
            "request_id": getattr(record, "request_id", get_request_id() or "-"),
        }

        if record.exc_info:
            payload["exception"] = sanitize_for_log(self.formatException(record.exc_info))

        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_ATTRS or key.startswith("_"):
                continue
            if key in payload:
                continue
            payload[key] = "[REDACTED]" if key.lower() in _SENSITIVE_KEYS else sanitize_for_log(value)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(settings: Any) -> None:
    level_name = str(getattr(settings, "log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = str(getattr(settings, "log_format", "json")).lower()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    if log_format == "text":
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
        )
    else:
        handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
