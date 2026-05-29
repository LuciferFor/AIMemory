from pathlib import Path
import logging
import time

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

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
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            if _should_log_request(request.url.path, 500, duration_ms, settings.slow_request_ms):
                request_logger.exception(
                    "http.request_failed",
                    extra=_request_log_extra(request, 500, duration_ms, "http.request_failed"),
                )
            reset_request_id(token)
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers[REQUEST_ID_HEADER] = request_id
        if _should_log_request(request.url.path, response.status_code, duration_ms, settings.slow_request_ms):
            level = logging.WARNING if response.status_code >= 500 or duration_ms >= settings.slow_request_ms else logging.INFO
            request_logger.log(
                level,
                "http.request",
                extra=_request_log_extra(request, response.status_code, duration_ms, "http.request"),
            )
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


app = create_app()
