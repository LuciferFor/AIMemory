import uuid

import sqlalchemy as sa
from sqlalchemy import DateTime, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aimemory.db.base import Base


class SearchStopword(Base):
    __tablename__ = "search_stopwords"
    __table_args__ = (
        Index("ix_search_stopwords_user_id", "user_id"),
        Index("ix_search_stopwords_term", "term"),
        Index(
            "uq_search_stopwords_active_user_term",
            "user_id",
            "term",
            unique=True,
            postgresql_where=sa.text("deleted_at IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    term: Mapped[str] = mapped_column(String(128), nullable=False)
    note: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="search_stopwords")
