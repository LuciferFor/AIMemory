import logging
import time
import uuid

from aimemory.db.session import SessionLocal
from aimemory.repositories.memories import (
    get_embedding_job,
    get_memory_for_embedding,
    mark_embedding_failed,
    mark_embedding_skipped,
    mark_embedding_started,
    mark_embedding_succeeded,
)
from aimemory.services.embedding import (
    EmbeddingProviderError,
    OpenAICompatibleEmbeddingClient,
    memory_embedding_input,
)
from aimemory.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="aimemory.generate_memory_embedding",
    max_retries=5,
    default_retry_delay=10,
)
def generate_memory_embedding(self, memory_id: str, job_id: str | None = None) -> dict[str, str]:
    start = time.perf_counter()
    db = SessionLocal()
    memory = None
    job = None
    try:
        parsed_memory_id = uuid.UUID(memory_id)
        parsed_job_id = uuid.UUID(job_id) if job_id else None

        memory = get_memory_for_embedding(db, parsed_memory_id)
        job = get_embedding_job(db, parsed_job_id)
        if memory is None:
            reason = "Memory was deleted or does not exist."
            mark_embedding_skipped(job, reason)
            db.commit()
            logger.info(
                "embedding.job_skipped",
                extra=_job_log_extra("embedding.job_skipped", memory_id, job_id, job, start, reason=reason),
            )
            return {"status": "skipped"}

        mark_embedding_started(job)
        db.commit()
        logger.info(
            "embedding.job_started",
            extra=_job_log_extra("embedding.job_started", memory_id, job_id, job, start),
        )

        source_text = memory_embedding_input(memory.title, memory.content)
        vector = OpenAICompatibleEmbeddingClient().embed(source_text)
        db.refresh(memory)
        if memory.deleted_at is not None:
            reason = "Memory was deleted while embedding was generated."
            mark_embedding_skipped(job, reason)
            db.commit()
            logger.info(
                "embedding.job_skipped",
                extra=_job_log_extra("embedding.job_skipped", memory_id, job_id, job, start, reason=reason),
            )
            return {"status": "skipped"}
        if memory_embedding_input(memory.title, memory.content) != source_text:
            reason = "Memory changed while embedding was generated."
            mark_embedding_skipped(job, reason)
            db.commit()
            logger.info(
                "embedding.job_skipped",
                extra=_job_log_extra("embedding.job_skipped", memory_id, job_id, job, start, reason=reason),
            )
            return {"status": "skipped"}

        mark_embedding_succeeded(memory, job, vector)
        db.commit()
        logger.info(
            "embedding.job_succeeded",
            extra=_job_log_extra("embedding.job_succeeded", memory_id, job_id, job, start, vector_dim=len(vector)),
        )
        return {"status": "succeeded"}
    except EmbeddingProviderError as exc:
        retrying = self.request.retries < self.max_retries
        mark_embedding_failed(memory, job, str(exc), retrying=retrying)
        db.commit()
        if retrying:
            countdown = min(300, 10 * (2 ** self.request.retries))
            logger.warning(
                "embedding.job_retrying",
                extra=_job_log_extra(
                    "embedding.job_retrying",
                    memory_id,
                    job_id,
                    job,
                    start,
                    retry_countdown_seconds=countdown,
                    error_type=exc.__class__.__name__,
                    error=str(exc),
                ),
            )
            raise self.retry(exc=exc, countdown=countdown)
        logger.exception(
            "embedding.job_failed",
            extra=_job_log_extra(
                "embedding.job_failed",
                memory_id,
                job_id,
                job,
                start,
                error_type=exc.__class__.__name__,
                error=str(exc),
            ),
        )
        return {"status": "failed"}
    except Exception as exc:
        db.rollback()
        logger.exception(
            "embedding.job_unexpected_failed",
            extra=_job_log_extra(
                "embedding.job_unexpected_failed",
                memory_id,
                job_id,
                job,
                start,
                error_type=exc.__class__.__name__,
                error=str(exc),
            ),
        )
        raise
    finally:
        db.close()


def _job_log_extra(event: str, memory_id: str, job_id: str | None, job, start: float, **extra):
    return {
        "event": event,
        "memory_id": memory_id,
        "embedding_job_id": job_id,
        "attempts": job.attempts if job is not None else None,
        "job_status": job.status if job is not None else None,
        "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        **extra,
    }
