"""add per-user search stopwords

Revision ID: 20260530_0006
Revises: 20260530_0005
Create Date: 2026-05-30 19:40:00.000000
"""

import uuid
import unicodedata

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260530_0006"
down_revision = "20260530_0005"
branch_labels = None
depends_on = None

DEFAULT_STOPWORDS = ("skill", "assistant", "user", "xxx", "aimemory", "openclaw")


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).lower().strip().split())


def upgrade() -> None:
    op.create_table(
        "search_stopwords",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("term", sa.String(length=128), nullable=False),
        sa.Column("note", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_search_stopwords_user_id", "search_stopwords", ["user_id"], unique=False)
    op.create_index("ix_search_stopwords_term", "search_stopwords", ["term"], unique=False)
    op.execute(
        "CREATE UNIQUE INDEX uq_search_stopwords_active_user_term "
        "ON search_stopwords (user_id, term) WHERE deleted_at IS NULL"
    )

    stopword_table = sa.table(
        "search_stopwords",
        sa.column("id", postgresql.UUID(as_uuid=True)),
        sa.column("user_id", postgresql.UUID(as_uuid=True)),
        sa.column("term", sa.String(length=128)),
        sa.column("note", sa.String(length=256)),
    )
    users = op.get_bind().execute(sa.text("SELECT id, name FROM users")).mappings().all()
    rows = []
    for user in users:
        terms = {_normalize(term) for term in DEFAULT_STOPWORDS}
        terms.add(_normalize(user["name"]))
        for term in sorted(value for value in terms if value and not value.isdigit()):
            rows.append(
                {
                    "id": uuid.uuid4(),
                    "user_id": user["id"],
                    "term": term,
                    "note": "默认停用词",
                }
            )
    if rows:
        op.bulk_insert(stopword_table, rows)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_search_stopwords_active_user_term")
    op.drop_index("ix_search_stopwords_term", table_name="search_stopwords")
    op.drop_index("ix_search_stopwords_user_id", table_name="search_stopwords")
    op.drop_table("search_stopwords")
