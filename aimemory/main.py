from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from aimemory.admin.routes import router as admin_router
from aimemory.api.routes import health_router, router
from aimemory.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0")
    admin_static = Path(__file__).parent / "admin" / "static"
    app.mount("/admin/static", StaticFiles(directory=admin_static), name="admin_static")
    app.include_router(health_router)
    app.include_router(router)
    app.include_router(admin_router)
    return app


app = create_app()
