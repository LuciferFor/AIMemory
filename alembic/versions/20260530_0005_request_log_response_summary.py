"""add request log response summary

Revision ID: 20260530_0005
Revises: 20260530_0004
Create Date: 2026-05-30 18:40:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260530_0005"
down_revision = "20260530_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("request_logs", sa.Column("response_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("request_logs", "response_summary")
