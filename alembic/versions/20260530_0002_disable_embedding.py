"""disable embedding workflow

Revision ID: 20260530_0002
Revises: 20260529_0001
Create Date: 2026-05-30 00:00:00.000000
"""

from alembic import op

revision = "20260530_0002"
down_revision = "20260529_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE memories ALTER COLUMN embedding_status SET DEFAULT 'disabled'")
    op.execute("UPDATE memories SET embedding_status = 'disabled', embedding_error = NULL WHERE deleted_at IS NULL")
    op.execute(
        "UPDATE embedding_jobs SET status = 'skipped', last_error = 'Embedding workflow disabled.' "
        "WHERE status IN ('pending', 'running', 'retrying')"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE memories ALTER COLUMN embedding_status SET DEFAULT 'pending'")
