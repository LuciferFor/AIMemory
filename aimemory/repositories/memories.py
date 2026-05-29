import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from aimemory.models.embedding_job import EmbeddingJob
from aimemory.models.memory import Memory
from aimemory.schemas.memory import MemorySearchRequest, MemoryUpsertRequest
from aimemory.services.text import build_search_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    memory_id: uuid.UUID
    external_id: str
    title: str
    content: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    score: float
    score_parts: dict[str, float]
    embedding_status: str


def utcnow() -> datetime:
    return datetime.now(UTC)


def upsert_memory(db: Session, user_id: uuid.UUID, payload: MemoryUpsertRequest) -> tuple[Memory, str]:
    existing = db.scalar(
        select(Memory).where(
            Memory.user_id == user_id,
            Memory.agent_id == payload.agent_id,
            Memory.external_id == payload.external_id,
            Memory.deleted_at.is_(None),
        )
    )

    if existing:
        existing.title = payload.title
        existing.content = payload.content
        existing.metadata_json = payload.metadata
        existing.occurred_at = payload.occurred_at
        existing.search_text = build_search_text(payload.title, payload.content)
        existing.embedding = None
        existing.embedding_status = "pending"
        existing.embedding_error = None
        existing.updated_at = utcnow()
        db.add(existing)
        db.flush()
        return existing, "updated"

    memory = Memory(
        user_id=user_id,
        agent_id=payload.agent_id,
        external_id=payload.external_id,
        title=payload.title,
        content=payload.content,
        metadata_json=payload.metadata,
        occurred_at=payload.occurred_at,
        search_text=build_search_text(payload.title, payload.content),
        embedding_status="pending",
    )
    db.add(memory)
    db.flush()
    return memory, "created"


def soft_delete_memory(db: Session, user_id: uuid.UUID, agent_id: str, external_id: str) -> bool:
    memory = db.scalar(
        select(Memory).where(
            Memory.user_id == user_id,
            Memory.agent_id == agent_id,
            Memory.external_id == external_id,
            Memory.deleted_at.is_(None),
        )
    )
    if memory is None:
        return False

    memory.deleted_at = utcnow()
    memory.updated_at = utcnow()
    db.add(memory)
    db.flush()
    return True


def create_embedding_job(db: Session, memory_id: uuid.UUID) -> EmbeddingJob:
    job = EmbeddingJob(memory_id=memory_id, status="pending", attempts=0)
    db.add(job)
    db.flush()
    return job


def get_memory_for_embedding(db: Session, memory_id: uuid.UUID) -> Memory | None:
    return db.scalar(
        select(Memory).where(
            Memory.id == memory_id,
            Memory.deleted_at.is_(None),
        )
    )


def get_embedding_job(db: Session, job_id: uuid.UUID | None) -> EmbeddingJob | None:
    if job_id is None:
        return None
    return db.get(EmbeddingJob, job_id)


def mark_embedding_started(job: EmbeddingJob | None) -> None:
    if job is not None:
        job.status = "running"
        job.attempts += 1
        job.updated_at = utcnow()


def mark_embedding_succeeded(memory: Memory, job: EmbeddingJob | None, vector: list[float]) -> None:
    memory.embedding = vector
    memory.embedding_status = "ready"
    memory.embedding_error = None
    memory.updated_at = utcnow()
    if job is not None:
        job.status = "succeeded"
        job.last_error = None
        job.updated_at = utcnow()


def mark_embedding_failed(memory: Memory | None, job: EmbeddingJob | None, error: str, retrying: bool) -> None:
    if memory is not None:
        memory.embedding_status = "pending" if retrying else "failed"
        memory.embedding_error = error[:2000]
        memory.updated_at = utcnow()
    if job is not None:
        job.status = "retrying" if retrying else "failed"
        job.last_error = error[:2000]
        job.updated_at = utcnow()


def mark_embedding_skipped(job: EmbeddingJob | None, reason: str) -> None:
    if job is not None:
        job.status = "skipped"
        job.last_error = reason[:2000]
        job.updated_at = utcnow()


def vector_to_sql(value: list[float]) -> str:
    return "[" + ",".join(f"{part:.10f}" for part in value) + "]"


def search_memories(
    db: Session,
    user_id: uuid.UUID,
    payload: MemorySearchRequest,
    normalized_query: str,
    query_vector: list[float] | None,
) -> list[SearchResult]:
    params: dict[str, Any] = {
        "user_id": user_id,
        "agent_id": payload.agent_id,
        "query": normalized_query,
        "like_query": f"%{normalized_query}%",
        "top_k": payload.top_k,
        "candidate_limit": max(payload.top_k * 8, 50),
    }
    where_sql = _build_common_where(payload, params)
    if query_vector:
        params["query_vector"] = vector_to_sql(query_vector)
        sql = _vector_search_sql(where_sql)
    else:
        sql = _text_search_sql(where_sql)

    rows = db.execute(text(sql), params).mappings().all()
    return [_row_to_result(row) for row in rows]


def _build_common_where(payload: MemorySearchRequest, params: dict[str, Any]) -> str:
    clauses = [
        "m.user_id = :user_id",
        "m.agent_id = :agent_id",
        "m.deleted_at IS NULL",
    ]
    if payload.metadata_filter:
        clauses.append('m."metadata" @> CAST(:metadata_filter AS jsonb)')
        params["metadata_filter"] = json.dumps(payload.metadata_filter)
    if payload.since:
        clauses.append("COALESCE(m.occurred_at, m.created_at) >= :since")
        params["since"] = payload.since
    if payload.until:
        clauses.append("COALESCE(m.occurred_at, m.created_at) <= :until")
        params["until"] = payload.until
    return " AND ".join(clauses)


def _vector_search_sql(where_sql: str) -> str:
    return f"""
WITH vector_candidates AS (
    SELECT
        m.id,
        GREATEST(0.0, 1.0 - (m.embedding <=> CAST(:query_vector AS vector))) AS semantic_score
    FROM memories m
    WHERE {where_sql}
      AND m.embedding IS NOT NULL
    ORDER BY m.embedding <=> CAST(:query_vector AS vector)
    LIMIT :candidate_limit
),
text_candidates AS (
    SELECT
        m.id,
        ts_rank_cd(to_tsvector('simple', m.search_text), plainto_tsquery('simple', :query)) AS keyword_score,
        similarity(m.search_text, :query) AS fuzzy_score
    FROM memories m
    WHERE {where_sql}
      AND (
        to_tsvector('simple', m.search_text) @@ plainto_tsquery('simple', :query)
        OR m.search_text % :query
        OR m.search_text ILIKE :like_query
      )
    ORDER BY keyword_score DESC, fuzzy_score DESC
    LIMIT :candidate_limit
),
candidate_ids AS (
    SELECT id FROM vector_candidates
    UNION
    SELECT id FROM text_candidates
),
scored AS (
    SELECT
        m.id AS memory_id,
        m.external_id,
        m.title,
        m.content,
        m."metadata" AS metadata,
        m.created_at,
        m.updated_at,
        m.embedding_status,
        COALESCE(vc.semantic_score, 0.0) AS semantic_score,
        COALESCE(ts_rank_cd(to_tsvector('simple', m.search_text), plainto_tsquery('simple', :query)), 0.0) AS keyword_score,
        COALESCE(similarity(m.search_text, :query), 0.0) AS fuzzy_score
    FROM memories m
    JOIN candidate_ids c ON c.id = m.id
    LEFT JOIN vector_candidates vc ON vc.id = m.id
)
SELECT
    *,
    (
        0.65 * LEAST(GREATEST(semantic_score, 0.0), 1.0)
        + 0.25 * LEAST(GREATEST(keyword_score, 0.0), 1.0)
        + 0.10 * LEAST(GREATEST(fuzzy_score, 0.0), 1.0)
    ) AS score
FROM scored
ORDER BY score DESC, updated_at DESC
LIMIT :top_k
"""


def _text_search_sql(where_sql: str) -> str:
    return f"""
WITH text_candidates AS (
    SELECT
        m.id,
        ts_rank_cd(to_tsvector('simple', m.search_text), plainto_tsquery('simple', :query)) AS keyword_score,
        similarity(m.search_text, :query) AS fuzzy_score
    FROM memories m
    WHERE {where_sql}
      AND (
        to_tsvector('simple', m.search_text) @@ plainto_tsquery('simple', :query)
        OR m.search_text % :query
        OR m.search_text ILIKE :like_query
      )
    ORDER BY keyword_score DESC, fuzzy_score DESC, m.updated_at DESC
    LIMIT :candidate_limit
),
scored AS (
    SELECT
        m.id AS memory_id,
        m.external_id,
        m.title,
        m.content,
        m."metadata" AS metadata,
        m.created_at,
        m.updated_at,
        m.embedding_status,
        0.0 AS semantic_score,
        COALESCE(tc.keyword_score, 0.0) AS keyword_score,
        COALESCE(tc.fuzzy_score, 0.0) AS fuzzy_score
    FROM memories m
    JOIN text_candidates tc ON tc.id = m.id
)
SELECT
    *,
    (
        0.25 * LEAST(GREATEST(keyword_score, 0.0), 1.0)
        + 0.10 * LEAST(GREATEST(fuzzy_score, 0.0), 1.0)
    ) AS score
FROM scored
ORDER BY score DESC, updated_at DESC
LIMIT :top_k
"""


def _row_to_result(row: Any) -> SearchResult:
    return SearchResult(
        memory_id=row["memory_id"],
        external_id=row["external_id"],
        title=row["title"],
        content=row["content"],
        metadata=row["metadata"] or {},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        score=float(row["score"] or 0.0),
        score_parts={
            "semantic": float(row["semantic_score"] or 0.0),
            "keyword": float(row["keyword_score"] or 0.0),
            "fuzzy": float(row["fuzzy_score"] or 0.0),
        },
        embedding_status=row["embedding_status"],
    )
