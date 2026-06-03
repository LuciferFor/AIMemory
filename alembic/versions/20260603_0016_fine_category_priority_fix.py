"""fix fine category priority

Revision ID: 20260603_0016
Revises: 20260603_0015
Create Date: 2026-06-03 16:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260603_0016"
down_revision: str | None = "20260603_0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _move_by_patterns(category_name: str, patterns: Sequence[str]) -> None:
    clauses = []
    params: dict[str, str] = {"category_name": category_name}
    for index, pattern in enumerate(patterns):
        key = f"p{index}"
        params[key] = f"%{pattern}%"
        clauses.append(f"(m.title ILIKE :{key} OR m.content ILIKE :{key})")
    op.execute(
        sa.text(
            f"""
WITH target AS (
    SELECT user_id, id AS category_id
    FROM memory_categories
    WHERE normalized_name = :category_name
      AND deleted_at IS NULL
)
UPDATE memories AS m
SET category_id = target.category_id,
    updated_at = now()
FROM target
WHERE m.user_id = target.user_id
  AND m.deleted_at IS NULL
  AND ({' OR '.join(clauses)})
  AND m.category_id <> target.category_id
"""
        ).bindparams(**params)
    )


def upgrade() -> None:
    _move_by_patterns("角色关系", ["用户关系", "角色关系", "关系定义", "关系边界", "主仆关系", "身份与关系", "相处原则", "不抢名分"])
    _move_by_patterns("自动化任务", ["OpenClaw 启动时自动拉起", "自动拉起 OneBot", "supervisor", "自动恢复"])


def downgrade() -> None:
    pass
