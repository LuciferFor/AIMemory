"""add ai chat

Revision ID: 20260531_0009
Revises: 20260531_0008
Create Date: 2026-05-31 01:40:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260531_0009"
down_revision = "20260531_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ai_chat_threads",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("admin_username", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=128), server_default="新对话", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ai_chat_threads_admin_updated", "ai_chat_threads", ["admin_username", "updated_at"], unique=False)
    op.create_index("ix_ai_chat_threads_deleted_at", "ai_chat_threads", ["deleted_at"], unique=False)
    op.create_table(
        "ai_chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("thread_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["thread_id"], ["ai_chat_threads.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ai_chat_messages_role", "ai_chat_messages", ["role"], unique=False)
    op.create_index("ix_ai_chat_messages_thread_created", "ai_chat_messages", ["thread_id", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_ai_chat_messages_thread_created", table_name="ai_chat_messages")
    op.drop_index("ix_ai_chat_messages_role", table_name="ai_chat_messages")
    op.drop_table("ai_chat_messages")
    op.drop_index("ix_ai_chat_threads_deleted_at", table_name="ai_chat_threads")
    op.drop_index("ix_ai_chat_threads_admin_updated", table_name="ai_chat_threads")
    op.drop_table("ai_chat_threads")
