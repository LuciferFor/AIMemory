from celery import Celery

from aimemory.core.config import get_settings
from aimemory.core.logging import configure_logging

settings = get_settings()
configure_logging(settings)

celery_app = Celery(
    "aimemory",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["aimemory.worker.tasks"],
)
celery_app.conf.update(
    task_always_eager=settings.celery_task_always_eager,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_hijack_root_logger=False,
)
