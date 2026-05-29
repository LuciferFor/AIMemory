import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse

from aimemory.api.deps import get_current_user
from aimemory.db.session import get_db
from aimemory.models.user import User
from aimemory.repositories.memories import (
    create_embedding_job,
    search_memories,
    soft_delete_memory,
    upsert_memory,
)
from aimemory.schemas.memory import (
    MemoryDeleteRequest,
    MemoryDeleteResponse,
    MemorySearchItem,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryUpsertRequest,
    MemoryUpsertResponse,
    ScoreParts,
)
from aimemory.services.embedding import EmbeddingProviderError, OpenAICompatibleEmbeddingClient
from aimemory.services.text import normalize_query
from aimemory.worker.tasks import generate_memory_embedding

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/memories", tags=["memories"])


@router.post("", response_model=MemoryUpsertResponse)
def write_memory(
    payload: MemoryUpsertRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemoryUpsertResponse:
    try:
        memory, action = upsert_memory(db, current_user.id, payload)
        job = create_embedding_job(db, memory.id)
        db.commit()
    except IntegrityError:
        db.rollback()
        memory, action = upsert_memory(db, current_user.id, payload)
        job = create_embedding_job(db, memory.id)
        db.commit()

    try:
        generate_memory_embedding.delay(str(memory.id), str(job.id))
    except Exception as exc:  # Celery broker failures should not lose the memory.
        logger.warning("Failed to enqueue embedding job for memory %s: %s", memory.id, exc)

    return MemoryUpsertResponse(
        memory_id=memory.id,
        external_id=memory.external_id,
        action=action,
        embedding_status=memory.embedding_status,
    )


@router.post("/search", response_model=MemorySearchResponse)
def search_memory(
    payload: MemorySearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemorySearchResponse:
    normalized_query = normalize_query(payload.query)
    query_vector = None

    try:
        query_vector = OpenAICompatibleEmbeddingClient().embed(normalized_query)
    except EmbeddingProviderError as exc:
        logger.info("Embedding search fallback to text-only: %s", exc)

    results = search_memories(db, current_user.id, payload, normalized_query, query_vector)
    return MemorySearchResponse(
        items=[
            MemorySearchItem(
                memory_id=result.memory_id,
                external_id=result.external_id,
                title=result.title,
                content=result.content,
                metadata=result.metadata,
                created_at=result.created_at,
                updated_at=result.updated_at,
                score=result.score,
                score_parts=ScoreParts(**result.score_parts),
                embedding_status=result.embedding_status,
            )
            for result in results
        ]
    )


@router.delete("", response_model=MemoryDeleteResponse)
def delete_memory(
    payload: MemoryDeleteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MemoryDeleteResponse:
    deleted = soft_delete_memory(db, current_user.id, payload.agent_id, payload.external_id)
    db.commit()
    return MemoryDeleteResponse(deleted=deleted)


health_router = APIRouter(tags=["health"])


@health_router.get("/", include_in_schema=False)
@health_router.head("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)


@health_router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
