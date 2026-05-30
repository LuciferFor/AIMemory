import uuid

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aimemory.db.base import Base


class AiMemoryReviewRun(Base):
    __tablename__ = "ai_memory_review_runs"
    __table_args__ = (
        Index("ix_ai_memory_review_runs_created_at", "created_at"),
        Index("ix_ai_memory_review_runs_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider_config_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("llm_provider_configs.id"), nullable=True)
    admin_username: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", server_default="pending")
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual", server_default="manual")
    selection_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    request_summary: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    response_summary: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    prompt_preview: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)

    provider_config = relationship("LlmProviderConfig", back_populates="review_runs")
    suggestions = relationship("AiMemoryReviewSuggestion", back_populates="run")


class AiMemoryReviewSuggestion(Base):
    __tablename__ = "ai_memory_review_suggestions"
    __table_args__ = (
        Index("ix_ai_memory_review_suggestions_run_id", "run_id"),
        Index("ix_ai_memory_review_suggestions_status", "status"),
        Index("ix_ai_memory_review_suggestions_type", "suggestion_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ai_memory_review_runs.id"), nullable=False)
    suggestion_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", server_default="pending")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    memory_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))
    target_memory_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    proposed_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    original_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    applied_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ignored_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)

    run = relationship("AiMemoryReviewRun", back_populates="suggestions")
