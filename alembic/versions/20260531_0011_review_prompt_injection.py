"""add review prompt injection

Revision ID: 20260531_0011
Revises: 20260531_0010
Create Date: 2026-05-31 04:40:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "20260531_0011"
down_revision = "20260531_0010"
branch_labels = None
depends_on = None

DEFAULT_REVIEW_PROMPT_INJECTION = (
    "整理长期记忆时请遵循：\n"
    "1. 先判断每条记忆的长期价值，只提出必要、低风险的整理建议。\n"
    "2. 保留具体事实、偏好、约束、关系、项目背景和可执行指令；删除口水话、重复表达、临时状态和过期上下文。\n"
    "3. rewrite 要让标题简短明确，正文自然、可检索，并保持第三方视角。\n"
    "4. merge 只用于事实高度重复或互补的记忆，目标记忆选择信息最完整的一条。\n"
    "5. move_category 优先使用已有分类；只有明显不合适时才提出新分类。\n"
    "6. soft_delete 仅用于明显无价值、错误、过期或已被其他记忆完整覆盖的内容。"
)


def upgrade() -> None:
    op.add_column(
        "llm_provider_configs",
        sa.Column("review_prompt_injection", sa.Text(), server_default="", nullable=False),
    )
    op.get_bind().execute(
        sa.text(
            "UPDATE llm_provider_configs "
            "SET review_prompt_injection = :prompt "
            "WHERE review_prompt_injection = ''"
        ),
        {"prompt": DEFAULT_REVIEW_PROMPT_INJECTION},
    )


def downgrade() -> None:
    op.drop_column("llm_provider_configs", "review_prompt_injection")
