import uuid

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aimemory.db.base import Base


class LlmProviderConfig(Base):
    __tablename__ = "llm_provider_configs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, default="default", server_default="default")
    base_url: Mapped[str] = mapped_column(String(512), nullable=False, default="https://api.deepseek.com")
    model: Mapped[str] = mapped_column(String(128), nullable=False, default="deepseek-v4-flash")
    encrypted_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_key_hint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    timeout_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=30000, server_default="30000")
    max_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=4096, server_default="4096")
    temperature: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0")
    extra_body_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    review_prompt_injection: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=text("true"))
    query_analysis_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=text("true"))
    query_analysis_max_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=256, server_default="256")
    query_analysis_timeout_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=3000, server_default="3000")
    ai_chat_allow_select: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=text("true"))
    ai_chat_allow_insert: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    ai_chat_allow_update: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    ai_chat_allow_delete: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    review_runs = relationship("AiMemoryReviewRun", back_populates="provider_config")
