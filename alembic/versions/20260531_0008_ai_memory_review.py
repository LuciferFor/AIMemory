"""add ai memory review

Revision ID: 20260531_0008
Revises: 20260530_0007
Create Date: 2026-05-31 00:20:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260531_0008"
down_revision = "20260530_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.create_table(
        "llm_provider_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("name", sa.String(length=64), server_default="default", nullable=False),
        sa.Column("base_url", sa.String(length=512), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("encrypted_api_key", sa.Text(), nullable=True),
        sa.Column("api_key_hint", sa.String(length=64), nullable=True),
        sa.Column("timeout_ms", sa.Integer(), server_default="30000", nullable=False),
        sa.Column("max_output_tokens", sa.Integer(), server_default="4096", nullable=False),
        sa.Column("temperature", sa.Float(), server_default="0", nullable=False),
        sa.Column("extra_body_json", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_llm_provider_configs_name"),
    )
    op.create_table(
        "ai_memory_review_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("provider_config_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("admin_username", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("source", sa.String(length=32), server_default="manual", nullable=False),
        sa.Column("selection_json", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("request_summary", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("response_summary", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("prompt_preview", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["provider_config_id"], ["llm_provider_configs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ai_memory_review_runs_created_at", "ai_memory_review_runs", ["created_at"], unique=False)
    op.create_index("ix_ai_memory_review_runs_status", "ai_memory_review_runs", ["status"], unique=False)
    op.create_table(
        "ai_memory_review_suggestions",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("suggestion_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("memory_ids", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("target_memory_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("proposed_json", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("original_json", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ignored_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["ai_memory_review_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ai_memory_review_suggestions_run_id", "ai_memory_review_suggestions", ["run_id"], unique=False)
    op.create_index("ix_ai_memory_review_suggestions_status", "ai_memory_review_suggestions", ["status"], unique=False)
    op.create_index("ix_ai_memory_review_suggestions_type", "ai_memory_review_suggestions", ["suggestion_type"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_ai_memory_review_suggestions_type", table_name="ai_memory_review_suggestions")
    op.drop_index("ix_ai_memory_review_suggestions_status", table_name="ai_memory_review_suggestions")
    op.drop_index("ix_ai_memory_review_suggestions_run_id", table_name="ai_memory_review_suggestions")
    op.drop_table("ai_memory_review_suggestions")
    op.drop_index("ix_ai_memory_review_runs_status", table_name="ai_memory_review_runs")
    op.drop_index("ix_ai_memory_review_runs_created_at", table_name="ai_memory_review_runs")
    op.drop_table("ai_memory_review_runs")
    op.drop_table("llm_provider_configs")
