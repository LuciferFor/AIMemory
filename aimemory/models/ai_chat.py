import uuid

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aimemory.db.base import Base


class AiChatThread(Base):
    __tablename__ = "ai_chat_threads"
    __table_args__ = (
        Index("ix_ai_chat_threads_admin_updated", "admin_username", "updated_at"),
        Index("ix_ai_chat_threads_deleted_at", "deleted_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    admin_username: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(128), nullable=False, default="新对话", server_default="新对话")
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    deleted_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)

    messages = relationship("AiChatMessage", back_populates="thread", order_by="AiChatMessage.created_at")


class AiChatMessage(Base):
    __tablename__ = "ai_chat_messages"
    __table_args__ = (
        Index("ix_ai_chat_messages_thread_created", "thread_id", "created_at"),
        Index("ix_ai_chat_messages_role", "role"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ai_chat_threads.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    thread = relationship("AiChatThread", back_populates="messages")
