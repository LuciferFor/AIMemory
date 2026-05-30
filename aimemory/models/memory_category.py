import uuid

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aimemory.db.base import Base


class MemoryCategory(Base):
    __tablename__ = "memory_categories"
    __table_args__ = (
        Index("ix_memory_categories_user_id", "user_id"),
        Index("ix_memory_categories_normalized_name", "normalized_name"),
        Index(
            "uq_memory_categories_active_user_normalized_name",
            "user_id",
            "normalized_name",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    merged_into_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("memory_categories.id"), nullable=True)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    deleted_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="memory_categories")
    memories = relationship("Memory", back_populates="category")
    merged_into = relationship("MemoryCategory", remote_side=[id])
