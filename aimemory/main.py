from fastapi import FastAPI

from aimemory.api.routes import health_router, router
from aimemory.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.include_router(health_router)
    app.include_router(router)
    return app


app = create_app()
