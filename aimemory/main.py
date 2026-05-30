from pathlib import Path
import logging
import time

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from aimemory.admin.routes import router as admin_router
from aimemory.api.routes import health_router, router
from aimemory.core.config import get_settings
from aimemory.core.logging import (
    REQUEST_ID_HEADER,
    configure_logging,
    normalize_request_id,
    reset_request_id,
    set_request_id,
)
from aimemory.repositories.request_logs import insert_request_log, request_log_source, should_record_request_log

request_logger = logging.getLogger("aimemory.request")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)
    app = FastAPI(title=settings.app_name, version="0.1.0")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = normalize_request_id(request.headers.get(REQUEST_ID_HEADER))
        token = set_request_id(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            if _should_log_request(request.url.path, 500, duration_ms, settings.slow_request_ms):
                request_logger.exception(
                    "http.request_failed",
                    extra=_request_log_extra(request, 500, duration_ms, "http.request_failed"),
                )
            await _write_request_log(request, settings, request_id, 500, duration_ms, type(exc).__name__)
            reset_request_id(token)
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers[REQUEST_ID_HEADER] = request_id
        if request.url.path.startswith("/admin/static/"):
            response.headers["Cache-Control"] = "no-cache, max-age=0"
        if _should_log_request(request.url.path, response.status_code, duration_ms, settings.slow_request_ms):
            level = logging.WARNING if response.status_code >= 500 or duration_ms >= settings.slow_request_ms else logging.INFO
            request_logger.log(
                level,
                "http.request",
                extra=_request_log_extra(request, response.status_code, duration_ms, "http.request"),
            )
        await _write_request_log(request, settings, request_id, response.status_code, duration_ms)
        reset_request_id(token)
        return response

    admin_static = Path(__file__).parent / "admin" / "static"
    app.mount("/admin/static", StaticFiles(directory=admin_static), name="admin_static")
    app.include_router(health_router)
    app.include_router(router)
    app.include_router(admin_router)
    return app


def _should_log_request(path: str, status_code: int, duration_ms: float, slow_request_ms: int) -> bool:
    if status_code >= 400 or duration_ms >= slow_request_ms:
        return True
    return path != "/healthz" and not path.startswith("/admin/static/")


def _request_log_extra(request: Request, status_code: int, duration_ms: float, event: str) -> dict[str, object]:
    return {
        "event": event,
        "method": request.method,
        "path": request.url.path,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "client_ip": request.client.host if request.client else "",
        "user_agent": request.headers.get("user-agent", ""),
    }


async def _write_request_log(
    request: Request,
    settings: object,
    request_id: str,
    status_code: int,
    duration_ms: float,
    error_type: str | None = None,
) -> None:
    path = request.url.path
    if not getattr(settings, "request_log_db_enabled", True) or not should_record_request_log(path):
        return

    route = request.scope.get("route")
    data = {
        "request_id": request_id,
        "source": request_log_source(path),
        "method": request.method,
        "path": path,
        "route_path": getattr(route, "path", None),
        "status_code": status_code,
        "duration_ms": duration_ms,
        "client_ip": request.client.host if request.client else None,
        "user_agent": _truncate(request.headers.get("user-agent"), 512),
        "user_id": getattr(request.state, "request_log_user_id", None),
        "api_key_id": getattr(request.state, "request_log_api_key_id", None),
        "api_key_prefix": getattr(request.state, "request_log_api_key_prefix", None),
        "admin_username": getattr(request.state, "request_log_admin_username", None),
        "error_type": _truncate(error_type, 128),
    }
    try:
        await run_in_threadpool(insert_request_log, data)
    except Exception:
        request_logger.warning(
            "request_log.write_failed",
            extra={
                "event": "request_log.write_failed",
                "method": request.method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
            exc_info=True,
        )


def _truncate(value: object | None, max_chars: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= max_chars else text[:max_chars]


app = create_app()
