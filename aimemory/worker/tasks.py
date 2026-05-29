import logging
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
    db = SessionLocal()
    memory = None
    job = None
    try:
        parsed_memory_id = uuid.UUID(memory_id)
        parsed_job_id = uuid.UUID(job_id) if job_id else None

        memory = get_memory_for_embedding(db, parsed_memory_id)
        job = get_embedding_job(db, parsed_job_id)
        if memory is None:
            mark_embedding_skipped(job, "Memory was deleted or does not exist.")
            db.commit()
            return {"status": "skipped"}

        mark_embedding_started(job)
        db.commit()

        source_text = memory_embedding_input(memory.title, memory.content)
        vector = OpenAICompatibleEmbeddingClient().embed(source_text)
        db.refresh(memory)
        if memory.deleted_at is not None:
            mark_embedding_skipped(job, "Memory was deleted while embedding was generated.")
            db.commit()
            return {"status": "skipped"}
        if memory_embedding_input(memory.title, memory.content) != source_text:
            mark_embedding_skipped(job, "Memory changed while embedding was generated.")
            db.commit()
            return {"status": "skipped"}

        mark_embedding_succeeded(memory, job, vector)
        db.commit()
        return {"status": "succeeded"}
    except EmbeddingProviderError as exc:
        retrying = self.request.retries < self.max_retries
        mark_embedding_failed(memory, job, str(exc), retrying=retrying)
        db.commit()
        if retrying:
            countdown = min(300, 10 * (2 ** self.request.retries))
            raise self.retry(exc=exc, countdown=countdown)
        logger.exception("Embedding job failed permanently for memory %s", memory_id)
        return {"status": "failed"}
    except Exception:
        db.rollback()
        logger.exception("Unexpected embedding job failure for memory %s", memory_id)
        raise
    finally:
        db.close()
