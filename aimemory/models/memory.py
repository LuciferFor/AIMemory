import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aimemory.core.config import get_settings
from aimemory.db.base import Base

settings = get_settings()


class Memory(Base):
    __tablename__ = "memories"
    __table_args__ = (
        Index(
            "uq_memories_active_external_id",
            "user_id",
            "agent_id",
            "external_id",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
        Index("ix_memories_user_agent_deleted", "user_id", "agent_id", "deleted_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_id: Mapped[str] = mapped_column(String(256), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    search_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    embedding = mapped_column(Vector(settings.embedding_dim), nullable=True)
    embedding_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", server_default="pending")
    embedding_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurred_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    deleted_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="memories")
    embedding_jobs = relationship("EmbeddingJob", back_populates="memory")
