"""add character category hints

Revision ID: 20260603_0017
Revises: 20260603_0016
Create Date: 2026-06-03 16:20:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260603_0017"
down_revision: str | None = "20260603_0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _update_description(name: str, description: str) -> None:
    op.execute(
        sa.text(
            """
UPDATE memory_categories
SET description = :description,
    updated_at = now()
WHERE normalized_name = :name
  AND deleted_at IS NULL
"""
        ).bindparams(name=name, description=description)
    )


def upgrade() -> None:
    _update_description(
        "角色关系",
        "用户与 AI、AI 与 AI 之间的关系、边界、称呼归属、相处原则；绯夜、鸦羽绯夜、月见绫音、绫音、鸦羽之间的关系问题优先归入这里。",
    )
    _update_description(
        "角色设定",
        "AI 角色身份、性格、视觉核心设定、固定人设和角色特征；绯夜、鸦羽绯夜、月见绫音、绫音的人设资料归入这里。",
    )
    _update_description(
        "游戏资料",
        "游戏资料与偏好；命运2/Destiny 2、通行证、装备、仓库容量、版本系统、任务、赛季系统、角色配装等。绯夜、月见绫音、鸦羽不是游戏名，不要归入这里。",
    )


def downgrade() -> None:
    pass
