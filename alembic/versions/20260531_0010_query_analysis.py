"""add query analysis config

Revision ID: 20260531_0010
Revises: 20260531_0009
Create Date: 2026-05-31 02:25:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "20260531_0010"
down_revision = "20260531_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "llm_provider_configs",
        sa.Column("query_analysis_enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
    )
    op.add_column(
        "llm_provider_configs",
        sa.Column("query_analysis_max_output_tokens", sa.Integer(), server_default="256", nullable=False),
    )
    op.add_column(
        "llm_provider_configs",
        sa.Column("query_analysis_timeout_ms", sa.Integer(), server_default="3000", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("llm_provider_configs", "query_analysis_timeout_ms")
    op.drop_column("llm_provider_configs", "query_analysis_max_output_tokens")
    op.drop_column("llm_provider_configs", "query_analysis_enabled")
