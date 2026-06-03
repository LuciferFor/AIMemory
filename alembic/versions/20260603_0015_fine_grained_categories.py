"""fine grained memory categories

Revision ID: 20260603_0015
Revises: 20260603_0014
Create Date: 2026-06-03 15:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260603_0015"
down_revision: str | None = "20260603_0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


CATEGORIES: tuple[tuple[str, str], ...] = (
    ("回答风格", "中文优先、称呼、语气、自然收尾、回复方式、抱怨或亲近程度等长期回答偏好。"),
    ("角色设定", "AI 角色身份、性格、视觉核心设定、固定人设和角色特征。"),
    ("角色关系", "用户与 AI、AI 与 AI 之间的关系、边界、称呼归属、相处原则。"),
    ("群聊规则", "QQ群、例外群、@触发、非主人参与者、群消息过滤和群内回应规则。"),
    ("游戏资料", "游戏资料与偏好；命运2/Destiny 2、通行证、装备、仓库容量、版本系统、任务、赛季系统、角色配装等。"),
    ("出图偏好", "长期画风、角色外观、服装、黑丝、二次元风格、视觉身份等稳定出图偏好；不保存一次性出图要求。"),
    ("出图流程", "Grok/Codex 双路出图、图片发送、生成脚本、拒绝后的安全降级、出图相关执行流程。"),
    ("技术资料", "账号、接口地址、脚本路径、文件位置、状态验证、固定配置、依赖服务等可复用技术资料。"),
    ("故障排查", "登录态失效、连接失败、进程崩溃、报错、故障恢复和排查方法。"),
    ("自动化任务", "heartbeat、定时任务、自动拉起、supervisor、守护进程和后台自动恢复。"),
    ("工作流程", "部署诊断、操作步骤、项目处理流程、任务执行约定和长期工作习惯。"),
    ("生活偏好", "饮食、作息、健康提醒和日常生活偏好。"),
    ("其它", "无法归入已有细分类的兜底分类；不要优先使用。"),
)


def _ensure_category(name: str, description: str) -> None:
    op.execute(
        sa.text(
            """
INSERT INTO memory_categories (user_id, name, normalized_name, description)
SELECT u.id, :name, :name, :description
FROM users AS u
WHERE NOT EXISTS (
    SELECT 1
    FROM memory_categories AS c
    WHERE c.user_id = u.id
      AND c.normalized_name = :name
      AND c.deleted_at IS NULL
)
"""
        ).bindparams(name=name, description=description)
    )
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


def _move_by_patterns(category_name: str, patterns: Sequence[str]) -> None:
    clauses = []
    params: dict[str, str] = {"category_name": category_name}
    for index, pattern in enumerate(patterns):
        key = f"p{index}"
        params[key] = f"%{pattern}%"
        clauses.append(f"(m.title ILIKE :{key} OR m.content ILIKE :{key})")
    if not clauses:
        return
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
    for name, description in CATEGORIES:
        _ensure_category(name, description)

    # Broad rules first; specific drawers below can override them.
    _move_by_patterns("工作流程", ["部署诊断", "处理流程", "操作步骤", "流程", "约定"])
    _move_by_patterns("自动化任务", ["Heartbeat", "HEARTBEAT", "心跳", "自动拉起", "supervisor", "自动恢复", "唤醒"])
    _move_by_patterns("技术资料", ["文件位置", "脚本路径", "账号与连接信息", "状态验证", "本地 TTS", "API 地址", "配置"])
    _move_by_patterns("故障排查", ["登录态失效", "连接失败", "进程崩溃", "报错", "故障", "失效"])
    _move_by_patterns("群聊规则", ["QQ群", "例外群", "群聊", "群消息", "非主人参与者", "纯@", "被@"])
    _move_by_patterns("游戏资料", ["命运2", "Destiny 2", "猎杀通行证", "仓库扩容", "同调系统", "挽歌", "混乱无序", "噬菌体"])
    _move_by_patterns("出图流程", ["默认双路出图", "Grok 后台", "原图发送", "出图安全", "fallback", "硬编码无关主题", "生成脚本"])
    _move_by_patterns("出图偏好", ["出图风格", "画风转换", "黑丝", "二次元黑系", "视觉身份"])
    _move_by_patterns("角色设定", ["性格偏好", "角色核心设定", "固定人设", "纯血吸血鬼", "月见绫音喜欢安静"])
    _move_by_patterns("角色关系", ["用户关系", "角色关系", "关系定义", "关系边界", "主仆关系", "身份与关系", "相处原则", "不抢名分"])
    _move_by_patterns("回答风格", ["说话风格", "回答风格", "回应风格", "中文优先", "自然收尾", "称呼与语气", "抱怨语气", "不硬回"])
    _move_by_patterns("生活偏好", ["饮食", "热食", "粥", "馄饨", "牛肉面", "作息"])

    op.execute(
        """
UPDATE memory_categories AS c
SET deleted_at = now(),
    updated_at = now()
WHERE c.normalized_name IN ('技术记忆', '自动化', '自检测试', '娱乐偏好')
  AND c.deleted_at IS NULL
  AND NOT EXISTS (
      SELECT 1
      FROM memories AS m
      WHERE m.category_id = c.id
        AND m.deleted_at IS NULL
  )
"""
    )


def downgrade() -> None:
    pass
