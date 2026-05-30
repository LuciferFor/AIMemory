import logging
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from aimemory.db.session import SessionLocal
from aimemory.models.request_log import RequestLog

logger = logging.getLogger(__name__)


def should_record_request_log(path: str) -> bool:
    if path == "/healthz" or path.startswith("/admin/static/"):
        return False
    return path == "/" or path.startswith("/v1/") or path.startswith("/admin")


def request_log_source(path: str) -> str:
    if path.startswith("/v1/"):
        return "api"
    if path.startswith("/admin"):
        return "admin"
    return "root"


def insert_request_log(data: dict[str, Any]) -> None:
    with SessionLocal() as db:
        try:
            db.add(RequestLog(**data))
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            raise
