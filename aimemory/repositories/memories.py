import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from aimemory.models.memory import Memory
from aimemory.schemas.memory import MemorySearchRequest, MemoryUpsertRequest
from aimemory.services.text import build_search_text, split_query_terms

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
        existing.embedding_status = "disabled"
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
        embedding_status="disabled",
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


def search_memories(
    db: Session,
    user_id: uuid.UUID,
    payload: MemorySearchRequest,
    normalized_query: str,
) -> list[SearchResult]:
    query_terms = split_query_terms(normalized_query) or [normalized_query]
    params: dict[str, Any] = {
        "user_id": user_id,
        "agent_id": payload.agent_id,
        "query": normalized_query,
        "terms": query_terms,
        "like_query": f"%{normalized_query}%",
        "top_k": payload.top_k,
        "candidate_limit": max(payload.top_k * 8, 50),
    }
    where_sql = _build_common_where(payload, params)
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


def _text_search_sql(where_sql: str) -> str:
    return f"""
WITH query_terms AS (
    SELECT term
    FROM unnest(CAST(:terms AS text[])) AS query_term(term)
    WHERE length(term) > 0
),
term_stats AS (
    SELECT GREATEST(count(*), 1)::float AS term_count FROM query_terms
),
text_candidates AS (
    SELECT
        m.id,
        COALESCE(ts_rank_cd(to_tsvector('simple', m.search_text), plainto_tsquery('simple', :query)), 0.0) AS keyword_score,
        GREATEST(
            COALESCE(similarity(m.search_text, :query), 0.0),
            COALESCE(similarity(m.title, :query), 0.0)
        ) AS fuzzy_score,
        CASE
            WHEN m.search_text ILIKE :like_query OR m.title ILIKE :like_query THEN 1.0
            ELSE 0.0
        END AS exact_score,
        GREATEST(
            CASE WHEN m.title ILIKE :like_query THEN 1.0 ELSE 0.0 END,
            COALESCE(similarity(m.title, :query), 0.0)
        ) AS title_score,
        COALESCE(
            (
                SELECT count(*)::float
                FROM query_terms qt
                WHERE m.search_text ILIKE ('%' || qt.term || '%')
                   OR m."metadata"::text ILIKE ('%' || qt.term || '%')
            ) / NULLIF(ts.term_count, 0.0),
            0.0
        ) AS term_score,
        CASE WHEN m."metadata"::text ILIKE :like_query THEN 1.0 ELSE 0.0 END AS metadata_score,
        CASE
            WHEN COALESCE(m.occurred_at, m.updated_at, m.created_at) >= now() - interval '30 days' THEN 1.0
            WHEN COALESCE(m.occurred_at, m.updated_at, m.created_at) >= now() - interval '180 days' THEN 0.5
            WHEN COALESCE(m.occurred_at, m.updated_at, m.created_at) >= now() - interval '365 days' THEN 0.25
            ELSE 0.0
        END AS recency_score
    FROM memories m
    CROSS JOIN term_stats ts
    WHERE {where_sql}
      AND (
        to_tsvector('simple', m.search_text) @@ plainto_tsquery('simple', :query)
        OR m.search_text % :query
        OR m.title % :query
        OR m.search_text ILIKE :like_query
        OR m.title ILIKE :like_query
        OR m."metadata"::text ILIKE :like_query
        OR EXISTS (
            SELECT 1
            FROM query_terms qt
            WHERE m.search_text ILIKE ('%' || qt.term || '%')
               OR m."metadata"::text ILIKE ('%' || qt.term || '%')
        )
      )
    ORDER BY keyword_score DESC, title_score DESC, term_score DESC, fuzzy_score DESC, m.updated_at DESC
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
        'disabled' AS embedding_status,
        0.0 AS semantic_score,
        COALESCE(tc.keyword_score, 0.0) AS keyword_score,
        COALESCE(tc.fuzzy_score, 0.0) AS fuzzy_score,
        COALESCE(tc.exact_score, 0.0) AS exact_score,
        COALESCE(tc.title_score, 0.0) AS title_score,
        COALESCE(tc.term_score, 0.0) AS term_score,
        COALESCE(tc.metadata_score, 0.0) AS metadata_score,
        COALESCE(tc.recency_score, 0.0) AS recency_score
    FROM memories m
    JOIN text_candidates tc ON tc.id = m.id
)
SELECT
    *,
    LEAST(
        1.0,
        0.30 * LEAST(GREATEST(keyword_score, 0.0), 1.0)
        + 0.20 * LEAST(GREATEST(fuzzy_score, 0.0), 1.0)
        + 0.20 * LEAST(GREATEST(term_score, 0.0), 1.0)
        + 0.15 * LEAST(GREATEST(title_score, 0.0), 1.0)
        + 0.10 * LEAST(GREATEST(exact_score, 0.0), 1.0)
        + 0.03 * LEAST(GREATEST(metadata_score, 0.0), 1.0)
        + 0.02 * LEAST(GREATEST(recency_score, 0.0), 1.0)
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
            "semantic": 0.0,
            "keyword": float(row["keyword_score"] or 0.0),
            "fuzzy": float(row["fuzzy_score"] or 0.0),
        },
        embedding_status="disabled",
    )
