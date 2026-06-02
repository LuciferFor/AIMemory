"""add entertainment category hints

Revision ID: 20260602_0012
Revises: 20260531_0011
Create Date: 2026-06-02 19:40:00.000000
"""

from alembic import op

revision = "20260602_0012"
down_revision = "20260531_0011"
branch_labels = None
depends_on = None


ENTERTAINMENT_DESCRIPTION = (
    "游戏、动漫、影视、角色/图片娱乐偏好；命运2/Destiny 2、通行证、装备、仓库容量、"
    "版本系统、游戏任务等都属于这里。"
)


def upgrade() -> None:
    op.execute(
        f"""
INSERT INTO memory_categories (user_id, name, normalized_name, description)
SELECT u.id, '娱乐偏好', '娱乐偏好', '{ENTERTAINMENT_DESCRIPTION}'
FROM users AS u
WHERE NOT EXISTS (
    SELECT 1
    FROM memory_categories AS c
    WHERE c.user_id = u.id
      AND c.normalized_name = '娱乐偏好'
      AND c.deleted_at IS NULL
)
"""
    )
    op.execute(
        f"""
UPDATE memory_categories
SET description = CASE
    WHEN description IS NULL OR BTRIM(description) = '' THEN '{ENTERTAINMENT_DESCRIPTION}'
    WHEN description NOT ILIKE '%命运2%' AND description NOT ILIKE '%Destiny 2%'
        THEN description || E'\\n' || '{ENTERTAINMENT_DESCRIPTION}'
    ELSE description
END,
updated_at = now()
WHERE normalized_name = '娱乐偏好'
  AND deleted_at IS NULL
"""
    )
    op.execute(
        """
WITH target AS (
    SELECT user_id, id AS category_id
    FROM memory_categories
    WHERE normalized_name = '娱乐偏好'
      AND deleted_at IS NULL
),
matched AS (
    SELECT m.id, target.category_id
    FROM memories AS m
    JOIN target ON target.user_id = m.user_id
    WHERE (
        m.title ILIKE '%命运2%'
        OR m.content ILIKE '%命运2%'
        OR m.title ILIKE '%Destiny 2%'
        OR m.content ILIKE '%Destiny 2%'
    )
)
UPDATE memories AS m
SET category_id = matched.category_id,
    updated_at = now()
FROM matched
WHERE m.id = matched.id
  AND m.category_id <> matched.category_id
"""
    )


def downgrade() -> None:
    pass
