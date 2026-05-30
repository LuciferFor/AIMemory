"""add memory categories

Revision ID: 20260530_0007
Revises: 20260530_0006
Create Date: 2026-05-30 20:20:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260530_0007"
down_revision = "20260530_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.create_table(
        "memory_categories",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("normalized_name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("merged_into_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["merged_into_id"], ["memory_categories.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_categories_user_id", "memory_categories", ["user_id"], unique=False)
    op.create_index("ix_memory_categories_normalized_name", "memory_categories", ["normalized_name"], unique=False)
    op.execute(
        "CREATE UNIQUE INDEX uq_memory_categories_active_user_normalized_name "
        "ON memory_categories (user_id, normalized_name) WHERE deleted_at IS NULL"
    )

    op.add_column("memories", sa.Column("category_id", postgresql.UUID(as_uuid=True), nullable=True))

    op.execute(
        """
WITH source AS (
    SELECT
        user_id,
        COALESCE(NULLIF(BTRIM("metadata"->>'category'), ''), '未分类') AS name
    FROM memories
),
normalized AS (
    SELECT
        user_id,
        MIN(name) AS name,
        LEFT(LOWER(BTRIM(name)), 128) AS normalized_name
    FROM source
    GROUP BY user_id, LEFT(LOWER(BTRIM(name)), 128)
)
INSERT INTO memory_categories (user_id, name, normalized_name)
SELECT user_id, LEFT(name, 128), normalized_name
FROM normalized
ON CONFLICT DO NOTHING
"""
    )
    op.execute(
        """
UPDATE memories AS m
SET category_id = c.id
FROM memory_categories AS c
WHERE c.user_id = m.user_id
  AND c.deleted_at IS NULL
  AND c.normalized_name = LEFT(
    LOWER(BTRIM(COALESCE(NULLIF(BTRIM(m."metadata"->>'category'), ''), '未分类'))),
    128
  )
"""
    )
    op.alter_column("memories", "category_id", nullable=False)
    op.create_index("ix_memories_category_id", "memories", ["category_id"], unique=False)
    op.create_foreign_key("fk_memories_category_id_memory_categories", "memories", "memory_categories", ["category_id"], ["id"])


def downgrade() -> None:
    op.drop_constraint("fk_memories_category_id_memory_categories", "memories", type_="foreignkey")
    op.drop_index("ix_memories_category_id", table_name="memories")
    op.drop_column("memories", "category_id")
    op.execute("DROP INDEX IF EXISTS uq_memory_categories_active_user_normalized_name")
    op.drop_index("ix_memory_categories_normalized_name", table_name="memory_categories")
    op.drop_index("ix_memory_categories_user_id", table_name="memory_categories")
    op.drop_table("memory_categories")
