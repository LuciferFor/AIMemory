import json
import logging
import re
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session, defer

from aimemory.models.memory import Memory
from aimemory.models.memory_attachment import MemoryAttachment
from aimemory.schemas.memory import MemorySearchRequest, MemoryUpsertRequest
from aimemory.repositories.memory_categories import get_or_create_category
from aimemory.services.attachments import (
    DecodedAttachment,
    attachment_search_text,
    decode_attachment_inputs,
)
from aimemory.services.text import build_search_text, is_numeric_proper_noun_term, normalize_query, split_query_terms

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchAttachment:
    attachment_id: uuid.UUID
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    description: str | None
    ocr_text: str | None = None


@dataclass(frozen=True)
class SearchResult:
    memory_id: uuid.UUID
    external_id: str
    category: str
    title: str
    content: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    score: float
    score_parts: dict[str, float]
    embedding_status: str
    matched_terms: list[str] = field(default_factory=list)
    matched_fields: list[str] = field(default_factory=list)
    attachments: list[SearchAttachment] = field(default_factory=list)


MIN_SEARCH_SCORE = 0.12
HIGH_FREQUENCY_ABSOLUTE_MATCHES = 8
HIGH_FREQUENCY_RATIO = 0.30
HIGH_FREQUENCY_MIN_MATCHES = 3
SIMILAR_MEMORY_DEDUPE_LIMIT = 300
SIMILAR_MEMORY_MIN_SCORE = 0.88
_DEDUPE_TEXT_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)
_AUTO_MEMORY_SOURCES = {"conversation_compaction", "conversation_compression"}
_AUTO_MEMORY_SOURCE_METADATA = {"openclaw_aimemory_plugin"}


def utcnow() -> datetime:
    return datetime.now(UTC)


def upsert_memory(db: Session, user_id: uuid.UUID, payload: MemoryUpsertRequest) -> tuple[Memory, str]:
    decoded_attachments = decode_attachment_inputs(payload.attachments)
    category, _ = get_or_create_category(db, user_id, payload.category)
    existing = db.scalar(
        select(Memory).where(
            Memory.user_id == user_id,
            Memory.agent_id == payload.agent_id,
            Memory.external_id == payload.external_id,
            Memory.deleted_at.is_(None),
        )
    )

    if existing:
        _update_memory_from_payload(db, existing, user_id, category.id, payload, decoded_attachments)
        return existing, "updated"

    similar_existing = find_similar_auto_memory(db, user_id, category.id, payload)
    if similar_existing:
        _update_memory_from_payload(db, similar_existing, user_id, category.id, payload, decoded_attachments)
        logger.info(
            "memory.upsert.similar_dedupe",
            extra={
                "event": "memory.upsert.similar_dedupe",
                "user_id": user_id,
                "agent_id": payload.agent_id,
                "incoming_external_id": payload.external_id,
                "existing_external_id": similar_existing.external_id,
                "memory_id": similar_existing.id,
            },
        )
        return similar_existing, "updated"

    attachment_text = "\n".join(attachment.search_text for attachment in decoded_attachments)
    memory = Memory(
        user_id=user_id,
        category_id=category.id,
        agent_id=payload.agent_id,
        external_id=payload.external_id,
        title=payload.title,
        content=payload.content,
        metadata_json=payload.metadata,
        occurred_at=payload.occurred_at,
        search_text=build_search_text(payload.title, payload.content, attachment_text),
        embedding_status="disabled",
    )
    db.add(memory)
    db.flush()
    if decoded_attachments:
        _add_attachments(db, memory, user_id, decoded_attachments)
        db.flush()
    return memory, "created"


def _update_memory_from_payload(
    db: Session,
    memory: Memory,
    user_id: uuid.UUID,
    category_id: uuid.UUID,
    payload: MemoryUpsertRequest,
    decoded_attachments: list[DecodedAttachment],
) -> None:
    memory.category_id = category_id
    memory.title = payload.title
    memory.content = payload.content
    memory.metadata_json = payload.metadata
    memory.occurred_at = payload.occurred_at
    memory.embedding = None
    memory.embedding_status = "disabled"
    memory.embedding_error = None
    memory.updated_at = utcnow()
    db.add(memory)
    db.flush()
    if payload.attachments is not None:
        _replace_attachments(db, memory, user_id, decoded_attachments)
        attachment_text = "\n".join(attachment.search_text for attachment in decoded_attachments)
    else:
        attachment_text = attachment_search_text(_active_attachments(db, memory.id))
    memory.search_text = build_search_text(payload.title, payload.content, attachment_text)
    db.add(memory)
    db.flush()


def find_similar_auto_memory(
    db: Session,
    user_id: uuid.UUID,
    category_id: uuid.UUID,
    payload: MemoryUpsertRequest,
) -> Memory | None:
    if not should_attempt_auto_memory_dedupe(payload):
        return None
    if payload.attachments:
        return None

    candidates = db.scalars(
        select(Memory)
        .options(defer(Memory.embedding))
        .where(
            Memory.user_id == user_id,
            Memory.category_id == category_id,
            Memory.agent_id == payload.agent_id,
            Memory.deleted_at.is_(None),
        )
        .order_by(Memory.updated_at.desc())
        .limit(SIMILAR_MEMORY_DEDUPE_LIMIT)
    ).all()

    best_memory: Memory | None = None
    best_score = 0.0
    for candidate in candidates:
        score = duplicate_memory_score(payload.title, payload.content, candidate.title, candidate.content)
        if score > best_score:
            best_memory = candidate
            best_score = score
    return best_memory if best_memory is not None and best_score >= SIMILAR_MEMORY_MIN_SCORE else None


def should_attempt_auto_memory_dedupe(payload: MemoryUpsertRequest) -> bool:
    if str(payload.external_id or "").startswith("auto-"):
        return True
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    if metadata.get("extract_reason"):
        return True
    if metadata.get("source") in _AUTO_MEMORY_SOURCES:
        return True
    source_metadata = metadata.get("source_metadata")
    return isinstance(source_metadata, dict) and source_metadata.get("source") in _AUTO_MEMORY_SOURCE_METADATA


def is_probable_duplicate_memory(title: str, content: str, existing_title: str, existing_content: str) -> bool:
    return duplicate_memory_score(title, content, existing_title, existing_content) >= SIMILAR_MEMORY_MIN_SCORE


def duplicate_memory_score(title: str, content: str, existing_title: str, existing_content: str) -> float:
    title_score = _text_similarity(title, existing_title)
    content_score = _text_similarity(content, existing_content)
    combined_score = _text_similarity(f"{title}\n{content}", f"{existing_title}\n{existing_content}")
    token_score = _token_overlap_score(f"{title}\n{content}", f"{existing_title}\n{existing_content}")
    same_title = _dedupe_text(title) == _dedupe_text(existing_title)

    if same_title and content_score >= 0.58:
        return max(0.94, content_score, combined_score)
    if title_score >= 0.92 and content_score >= 0.70:
        return max(title_score, content_score, combined_score)
    if combined_score >= 0.92:
        return combined_score
    if title_score >= 0.84 and token_score >= 0.78:
        return max(token_score, combined_score)
    return max(0.0, min(title_score, content_score), combined_score * 0.85, token_score * 0.8)


def _text_similarity(left: str, right: str) -> float:
    normalized_left = _dedupe_text(left)
    normalized_right = _dedupe_text(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def _dedupe_text(value: str) -> str:
    normalized = normalize_query(value)
    return _DEDUPE_TEXT_RE.sub("", normalized)


def _token_overlap_score(left: str, right: str) -> float:
    left_terms = set(split_query_terms(left))
    right_terms = set(split_query_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(len(left_terms), len(right_terms))


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
    for attachment in _active_attachments(db, memory.id):
        attachment.deleted_at = utcnow()
        db.add(attachment)
    db.add(memory)
    db.flush()
    return True


def get_attachment_for_user(db: Session, user_id: uuid.UUID, attachment_id: uuid.UUID) -> MemoryAttachment | None:
    return db.scalar(
        select(MemoryAttachment)
        .join(Memory)
        .where(
            MemoryAttachment.id == attachment_id,
            MemoryAttachment.user_id == user_id,
            MemoryAttachment.deleted_at.is_(None),
            Memory.user_id == user_id,
            Memory.deleted_at.is_(None),
        )
    )


def get_attachment_for_admin(db: Session, attachment_id: uuid.UUID) -> MemoryAttachment | None:
    return db.scalar(
        select(MemoryAttachment)
        .join(Memory)
        .where(
            MemoryAttachment.id == attachment_id,
            MemoryAttachment.deleted_at.is_(None),
            Memory.deleted_at.is_(None),
        )
    )


def search_memories(
    db: Session,
    user_id: uuid.UUID,
    category_id: uuid.UUID,
    payload: MemorySearchRequest,
    normalized_query: str,
    query_terms: list[str] | None = None,
    min_matched_terms: int | None = None,
) -> list[SearchResult]:
    query_terms = query_terms if query_terms is not None else split_query_terms(normalized_query)
    if not query_terms:
        return []
    required_matched_terms = min_matched_terms if min_matched_terms is not None else (1 if len(query_terms) == 1 else 2)
    params: dict[str, Any] = {
        "user_id": user_id,
        "category_id": category_id,
        "agent_id": payload.agent_id,
        "query": normalized_query,
        "terms": query_terms,
        "like_query": f"%{normalized_query}%",
        "top_k": payload.top_k,
        "candidate_limit": max(payload.top_k * 8, 50),
        "min_matched_terms": required_matched_terms,
        "min_score": MIN_SEARCH_SCORE,
    }
    where_sql = _build_common_where(payload, params)
    sql = _text_search_sql(where_sql)
    rows = db.execute(text(sql), params).mappings().all()
    results = [_row_to_result(row) for row in rows]
    results = [result for result in results if result.matched_terms and result.score >= MIN_SEARCH_SCORE]
    attachments_by_memory = _attachments_for_memories(db, [result.memory_id for result in results])
    return [replace(result, attachments=attachments_by_memory.get(result.memory_id, [])) for result in results]


def filter_high_frequency_terms(
    db: Session,
    user_id: uuid.UUID,
    category_id: uuid.UUID,
    agent_id: str,
    query_terms: list[str],
) -> tuple[list[str], list[str]]:
    if not query_terms:
        return [], []

    rows = db.execute(
        text(
            """
WITH query_terms AS (
    SELECT term
    FROM unnest(CAST(:terms AS text[])) AS query_term(term)
    WHERE length(term) > 0
),
total AS (
    SELECT count(*)::float AS total_count
    FROM memories
    WHERE user_id = :user_id
      AND category_id = :category_id
      AND agent_id = :agent_id
      AND deleted_at IS NULL
),
term_counts AS (
    SELECT
        qt.term,
        count(m.id)::int AS match_count,
        total.total_count
    FROM query_terms qt
    CROSS JOIN total
    LEFT JOIN memories m
      ON m.user_id = :user_id
     AND m.category_id = :category_id
     AND m.agent_id = :agent_id
     AND m.deleted_at IS NULL
     AND (
        m.search_text ILIKE ('%' || qt.term || '%')
        OR m."metadata"::text ILIKE ('%' || qt.term || '%')
     )
    GROUP BY qt.term, total.total_count
)
SELECT term, match_count, total_count
FROM term_counts
"""
        ),
        {
            "user_id": user_id,
            "category_id": category_id,
            "agent_id": agent_id,
            "terms": query_terms,
        },
    ).mappings().all()

    high_frequency_terms: set[str] = set()
    for row in rows:
        term = str(row["term"] or "")
        if is_numeric_proper_noun_term(term):
            continue
        match_count = int(row["match_count"] or 0)
        total_count = float(row["total_count"] or 0.0)
        ratio = (match_count / total_count) if total_count else 0.0
        if match_count >= HIGH_FREQUENCY_ABSOLUTE_MATCHES or (
            match_count >= HIGH_FREQUENCY_MIN_MATCHES and ratio >= HIGH_FREQUENCY_RATIO
        ):
            high_frequency_terms.add(term)

    effective_terms = [term for term in query_terms if term not in high_frequency_terms]
    ignored_terms = [f"{term}:高频弱词" for term in query_terms if term in high_frequency_terms]
    return effective_terms, ignored_terms


def _build_common_where(payload: MemorySearchRequest, params: dict[str, Any]) -> str:
    clauses = [
        "m.user_id = :user_id",
        "m.category_id = :category_id",
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
term_matches AS (
    SELECT
        m.id,
        COALESCE(
            array_agg(DISTINCT qt.term ORDER BY qt.term) FILTER (
                WHERE m.search_text ILIKE ('%' || qt.term || '%')
                   OR m."metadata"::text ILIKE ('%' || qt.term || '%')
            ),
            ARRAY[]::text[]
        ) AS matched_terms,
        count(DISTINCT qt.term) FILTER (
            WHERE m.search_text ILIKE ('%' || qt.term || '%')
               OR m."metadata"::text ILIKE ('%' || qt.term || '%')
        )::float AS matched_term_count,
        count(DISTINCT qt.term) FILTER (
            WHERE m.title ILIKE ('%' || qt.term || '%')
        )::float AS title_term_matches,
        count(DISTINCT qt.term) FILTER (
            WHERE m.content ILIKE ('%' || qt.term || '%')
        )::float AS content_term_matches,
        count(DISTINCT qt.term) FILTER (
            WHERE m."metadata"::text ILIKE ('%' || qt.term || '%')
        )::float AS metadata_term_matches,
        count(DISTINCT qt.term) FILTER (
            WHERE m.search_text ILIKE ('%' || qt.term || '%')
              AND NOT (
                m.title ILIKE ('%' || qt.term || '%')
                OR m.content ILIKE ('%' || qt.term || '%')
              )
        )::float AS attachment_term_matches
    FROM memories m
    CROSS JOIN query_terms qt
    WHERE {where_sql}
    GROUP BY m.id
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
            WHEN m.search_text ILIKE :like_query OR m.title ILIKE :like_query OR m."metadata"::text ILIKE :like_query THEN 1.0
            ELSE 0.0
        END AS exact_score,
        GREATEST(
            tm.title_term_matches / NULLIF(ts.term_count, 0.0),
            CASE WHEN m.title ILIKE :like_query THEN 1.0 ELSE 0.0 END,
            COALESCE(similarity(m.title, :query), 0.0) * 0.5
        ) AS title_score,
        COALESCE(tm.matched_term_count / NULLIF(ts.term_count, 0.0), 0.0) AS term_score,
        COALESCE(tm.content_term_matches / NULLIF(ts.term_count, 0.0), 0.0) AS content_score,
        GREATEST(
            COALESCE(tm.metadata_term_matches / NULLIF(ts.term_count, 0.0), 0.0),
            CASE WHEN m."metadata"::text ILIKE :like_query THEN 1.0 ELSE 0.0 END
        ) AS metadata_score,
        CASE
            WHEN COALESCE(m.occurred_at, m.updated_at, m.created_at) >= now() - interval '30 days' THEN 1.0
            WHEN COALESCE(m.occurred_at, m.updated_at, m.created_at) >= now() - interval '180 days' THEN 0.5
            WHEN COALESCE(m.occurred_at, m.updated_at, m.created_at) >= now() - interval '365 days' THEN 0.25
            ELSE 0.0
        END AS recency_score,
        tm.matched_terms,
        ARRAY_REMOVE(ARRAY[
            CASE WHEN tm.title_term_matches > 0 THEN '标题'::text END,
            CASE WHEN tm.content_term_matches > 0 THEN '正文'::text END,
            CASE WHEN tm.metadata_term_matches > 0 THEN '元数据'::text END,
            CASE WHEN tm.attachment_term_matches > 0 THEN '附件'::text END
        ], NULL) AS matched_fields
    FROM memories m
    CROSS JOIN term_stats ts
    JOIN term_matches tm ON tm.id = m.id
    WHERE {where_sql}
      AND tm.matched_term_count >= :min_matched_terms
    ORDER BY keyword_score DESC, title_score DESC, term_score DESC, fuzzy_score DESC, m.updated_at DESC
    LIMIT :candidate_limit
),
scored AS (
    SELECT
        m.id AS memory_id,
        m.external_id,
        c.name AS category,
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
        COALESCE(tc.content_score, 0.0) AS content_score,
        COALESCE(tc.metadata_score, 0.0) AS metadata_score,
        COALESCE(tc.recency_score, 0.0) AS recency_score,
        tc.matched_terms,
        tc.matched_fields
    FROM memories m
    JOIN memory_categories c ON c.id = m.category_id
    JOIN text_candidates tc ON tc.id = m.id
),
weighted AS (
    SELECT
    *,
    LEAST(
        1.0,
        0.24 * LEAST(GREATEST(title_score, 0.0), 1.0)
        + 0.24 * LEAST(GREATEST(term_score, 0.0), 1.0)
        + 0.18 * LEAST(GREATEST(content_score, 0.0), 1.0)
        + 0.16 * LEAST(GREATEST(metadata_score, 0.0), 1.0)
        + 0.08 * LEAST(GREATEST(exact_score, 0.0), 1.0)
        + 0.05 * LEAST(GREATEST(keyword_score, 0.0), 1.0)
        + 0.03 * LEAST(GREATEST(fuzzy_score, 0.0), 1.0)
        + 0.02 * LEAST(GREATEST(recency_score, 0.0), 1.0)
    ) AS score
    FROM scored
)
SELECT *
FROM weighted
WHERE score >= :min_score
ORDER BY score DESC, updated_at DESC
LIMIT :top_k
"""


def _row_to_result(row: Any) -> SearchResult:
    return SearchResult(
        memory_id=row["memory_id"],
        external_id=row["external_id"],
        category=row["category"],
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
            "term": float(row["term_score"] or 0.0),
            "title": float(row["title_score"] or 0.0),
            "content": float(row["content_score"] or 0.0),
            "metadata": float(row["metadata_score"] or 0.0),
            "exact": float(row["exact_score"] or 0.0),
            "recency": float(row["recency_score"] or 0.0),
        },
        embedding_status="disabled",
        matched_terms=list(row["matched_terms"] or []),
        matched_fields=list(row["matched_fields"] or []),
        attachments=[],
    )


def _replace_attachments(
    db: Session,
    memory: Memory,
    user_id: uuid.UUID,
    decoded_attachments: list[DecodedAttachment],
) -> None:
    now = utcnow()
    for attachment in _active_attachments(db, memory.id):
        attachment.deleted_at = now
        db.add(attachment)
    _add_attachments(db, memory, user_id, decoded_attachments)


def _add_attachments(
    db: Session,
    memory: Memory,
    user_id: uuid.UUID,
    decoded_attachments: list[DecodedAttachment],
) -> None:
    for decoded in decoded_attachments:
        db.add(
            MemoryAttachment(
                memory_id=memory.id,
                user_id=user_id,
                filename=decoded.filename,
                mime_type=decoded.mime_type,
                size_bytes=decoded.size_bytes,
                sha256=decoded.sha256,
                image_bytes=decoded.image_bytes,
                description=decoded.description,
                ocr_text=decoded.ocr_text,
                metadata_json=decoded.metadata,
            )
        )


def _active_attachments(db: Session, memory_id: uuid.UUID) -> list[MemoryAttachment]:
    return list(
        db.scalars(
            select(MemoryAttachment)
            .options(defer(MemoryAttachment.image_bytes))
            .where(
                MemoryAttachment.memory_id == memory_id,
                MemoryAttachment.deleted_at.is_(None),
            )
            .order_by(MemoryAttachment.created_at)
        ).all()
    )


def _attachments_for_memories(
    db: Session,
    memory_ids: list[uuid.UUID],
) -> dict[uuid.UUID, list[SearchAttachment]]:
    if not memory_ids:
        return {}
    rows = db.execute(
        select(
            MemoryAttachment.memory_id,
            MemoryAttachment.id,
            MemoryAttachment.filename,
            MemoryAttachment.mime_type,
            MemoryAttachment.size_bytes,
            MemoryAttachment.sha256,
            MemoryAttachment.description,
            MemoryAttachment.ocr_text,
        )
        .where(
            MemoryAttachment.memory_id.in_(memory_ids),
            MemoryAttachment.deleted_at.is_(None),
        )
        .order_by(MemoryAttachment.created_at)
    ).mappings().all()
    grouped: dict[uuid.UUID, list[SearchAttachment]] = {}
    for attachment in rows:
        grouped.setdefault(attachment["memory_id"], []).append(
            SearchAttachment(
                attachment_id=attachment["id"],
                filename=attachment["filename"],
                mime_type=attachment["mime_type"],
                size_bytes=attachment["size_bytes"],
                sha256=attachment["sha256"],
                description=attachment["description"],
                ocr_text=attachment["ocr_text"],
            )
        )
    return grouped
