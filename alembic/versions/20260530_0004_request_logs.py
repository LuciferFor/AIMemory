"""add request logs

Revision ID: 20260530_0004
Revises: 20260530_0003
Create Date: 2026-05-30 18:10:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260530_0004"
down_revision = "20260530_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "request_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("method", sa.String(length=16), nullable=False),
        sa.Column("path", sa.String(length=512), nullable=False),
        sa.Column("route_path", sa.String(length=512), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("client_ip", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=512), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("api_key_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("api_key_prefix", sa.String(length=24), nullable=True),
        sa.Column("admin_username", sa.String(length=128), nullable=True),
        sa.Column("error_type", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_request_logs_created_at", "request_logs", ["created_at"], unique=False)
    op.create_index("ix_request_logs_status_code", "request_logs", ["status_code"], unique=False)
    op.create_index("ix_request_logs_source", "request_logs", ["source"], unique=False)
    op.create_index("ix_request_logs_path", "request_logs", ["path"], unique=False)
    op.create_index("ix_request_logs_user_id", "request_logs", ["user_id"], unique=False)
    op.create_index("ix_request_logs_request_id", "request_logs", ["request_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_request_logs_request_id", table_name="request_logs")
    op.drop_index("ix_request_logs_user_id", table_name="request_logs")
    op.drop_index("ix_request_logs_path", table_name="request_logs")
    op.drop_index("ix_request_logs_source", table_name="request_logs")
    op.drop_index("ix_request_logs_status_code", table_name="request_logs")
    op.drop_index("ix_request_logs_created_at", table_name="request_logs")
    op.drop_table("request_logs")
