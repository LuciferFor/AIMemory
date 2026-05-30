"""initial schema

Revision ID: 20260529_0001
Revises:
Create Date: 2026-05-29 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "20260529_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_name", "users", ["name"], unique=True)

    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("key_prefix", sa.String(length=24), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"], unique=True)
    op.create_index("ix_api_keys_key_prefix", "api_keys", ["key_prefix"], unique=True)
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"], unique=False)

    op.create_table(
        "memories",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_id", sa.String(length=128), nullable=False),
        sa.Column("external_id", sa.String(length=256), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.Column("embedding_status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("embedding_error", sa.Text(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memories_user_id", "memories", ["user_id"], unique=False)
    op.create_index("ix_memories_user_agent_deleted", "memories", ["user_id", "agent_id", "deleted_at"], unique=False)
    op.execute(
        "CREATE UNIQUE INDEX uq_memories_active_external_id "
        "ON memories (user_id, agent_id, external_id) WHERE deleted_at IS NULL"
    )
    op.execute(
        "CREATE INDEX ix_memories_embedding_hnsw "
        "ON memories USING hnsw (embedding vector_cosine_ops) "
        "WHERE embedding IS NOT NULL AND deleted_at IS NULL"
    )
    op.execute(
        "CREATE INDEX ix_memories_search_tsv "
        "ON memories USING gin (to_tsvector('simple', search_text))"
    )
    op.execute(
        "CREATE INDEX ix_memories_search_trgm "
        "ON memories USING gin (search_text gin_trgm_ops)"
    )

    op.create_table(
        "embedding_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["memory_id"], ["memories.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_embedding_jobs_memory_id", "embedding_jobs", ["memory_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_embedding_jobs_memory_id", table_name="embedding_jobs")
    op.drop_table("embedding_jobs")
    op.execute("DROP INDEX IF EXISTS ix_memories_search_trgm")
    op.execute("DROP INDEX IF EXISTS ix_memories_search_tsv")
    op.execute("DROP INDEX IF EXISTS ix_memories_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS uq_memories_active_external_id")
    op.drop_index("ix_memories_user_agent_deleted", table_name="memories")
    op.drop_index("ix_memories_user_id", table_name="memories")
    op.drop_table("memories")
    op.drop_index("ix_api_keys_user_id", table_name="api_keys")
    op.drop_index("ix_api_keys_key_prefix", table_name="api_keys")
    op.drop_index("ix_api_keys_key_hash", table_name="api_keys")
    op.drop_table("api_keys")
    op.drop_index("ix_users_name", table_name="users")
    op.drop_table("users")
