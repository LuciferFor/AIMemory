"""add memory attachments

Revision ID: 20260530_0003
Revises: 20260530_0002
Create Date: 2026-05-30 14:45:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260530_0003"
down_revision = "20260530_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_attachments",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("mime_type", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("image_bytes", sa.LargeBinary(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("ocr_text", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["memory_id"], ["memories.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_attachments_memory_id", "memory_attachments", ["memory_id"], unique=False)
    op.create_index("ix_memory_attachments_sha256", "memory_attachments", ["sha256"], unique=False)
    op.create_index("ix_memory_attachments_user_id", "memory_attachments", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_memory_attachments_user_id", table_name="memory_attachments")
    op.drop_index("ix_memory_attachments_sha256", table_name="memory_attachments")
    op.drop_index("ix_memory_attachments_memory_id", table_name="memory_attachments")
    op.drop_table("memory_attachments")
