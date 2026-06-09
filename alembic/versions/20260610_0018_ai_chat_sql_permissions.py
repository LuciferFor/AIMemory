"""add ai chat sql permissions

Revision ID: 20260610_0018
Revises: 20260603_0017
Create Date: 2026-06-10 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "20260610_0018"
down_revision = "20260603_0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "llm_provider_configs",
        sa.Column("ai_chat_allow_select", sa.Boolean(), server_default=sa.text("true"), nullable=False),
    )
    op.add_column(
        "llm_provider_configs",
        sa.Column("ai_chat_allow_insert", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.add_column(
        "llm_provider_configs",
        sa.Column("ai_chat_allow_update", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.add_column(
        "llm_provider_configs",
        sa.Column("ai_chat_allow_delete", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("llm_provider_configs", "ai_chat_allow_delete")
    op.drop_column("llm_provider_configs", "ai_chat_allow_update")
    op.drop_column("llm_provider_configs", "ai_chat_allow_insert")
    op.drop_column("llm_provider_configs", "ai_chat_allow_select")
