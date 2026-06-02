"""add memory agents and devices

Revision ID: 20260602_0013
Revises: 20260602_0012
Create Date: 2026-06-02 23:50:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260602_0013"
down_revision = "20260602_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "memory_agents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_id", sa.String(length=128), nullable=False),
        sa.Column("display_name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_agents_user_id", "memory_agents", ["user_id"], unique=False)
    op.create_index(
        "uq_memory_agents_active_user_agent_id",
        "memory_agents",
        ["user_id", "agent_id"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    op.create_table(
        "memory_devices",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("device_id", sa.String(length=128), nullable=False),
        sa.Column("display_name", sa.String(length=128), nullable=False),
        sa.Column("agent_id", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_devices_user_id", "memory_devices", ["user_id"], unique=False)
    op.create_index("ix_memory_devices_user_agent_id", "memory_devices", ["user_id", "agent_id"], unique=False)
    op.create_index(
        "uq_memory_devices_active_user_device_id",
        "memory_devices",
        ["user_id", "device_id"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    op.add_column("request_logs", sa.Column("agent_id", sa.String(length=128), nullable=True))
    op.add_column("request_logs", sa.Column("agent_name", sa.String(length=128), nullable=True))
    op.add_column("request_logs", sa.Column("device_id", sa.String(length=128), nullable=True))
    op.add_column("request_logs", sa.Column("device_name", sa.String(length=128), nullable=True))
    op.create_index("ix_request_logs_agent_id", "request_logs", ["agent_id"], unique=False)
    op.create_index("ix_request_logs_device_id", "request_logs", ["device_id"], unique=False)

    op.execute(
        """
INSERT INTO memory_agents (id, user_id, agent_id, display_name, description)
SELECT gen_random_uuid(), source.user_id, source.agent_id,
       CASE
           WHEN source.agent_id = 'feiye-31-11' THEN '绯夜'
           WHEN source.agent_id = '5df9cbfb-d31b-46dd-972b-05d466d2257c' THEN '31.9 OpenClaw'
           ELSE source.agent_id
       END AS display_name,
       '由历史记忆自动导入，可在后台修改。'
FROM (
    SELECT DISTINCT user_id, agent_id
    FROM memories
    WHERE agent_id IS NOT NULL AND BTRIM(agent_id) <> ''
) AS source
WHERE NOT EXISTS (
    SELECT 1
    FROM memory_agents AS existing
    WHERE existing.user_id = source.user_id
      AND existing.agent_id = source.agent_id
      AND existing.deleted_at IS NULL
)
"""
    )
    op.execute(
        """
INSERT INTO memory_devices (id, user_id, device_id, display_name, agent_id, description)
SELECT gen_random_uuid(), agent.user_id, 'openclaw-31-11', '31.11 OpenClaw', agent.agent_id,
       '31.11 OpenClaw 设备，由迁移自动创建。'
FROM memory_agents AS agent
WHERE agent.agent_id = 'feiye-31-11'
  AND agent.deleted_at IS NULL
  AND NOT EXISTS (
      SELECT 1 FROM memory_devices AS device
      WHERE device.user_id = agent.user_id
        AND device.device_id = 'openclaw-31-11'
        AND device.deleted_at IS NULL
  )
"""
    )
    op.execute(
        """
INSERT INTO memory_devices (id, user_id, device_id, display_name, agent_id, description)
SELECT gen_random_uuid(), agent.user_id, 'openclaw-31-9', '31.9 OpenClaw', agent.agent_id,
       '31.9 OpenClaw 设备，由迁移自动创建。'
FROM memory_agents AS agent
WHERE agent.agent_id = '5df9cbfb-d31b-46dd-972b-05d466d2257c'
  AND agent.deleted_at IS NULL
  AND NOT EXISTS (
      SELECT 1 FROM memory_devices AS device
      WHERE device.user_id = agent.user_id
        AND device.device_id = 'openclaw-31-9'
        AND device.deleted_at IS NULL
  )
"""
    )


def downgrade() -> None:
    op.drop_index("ix_request_logs_device_id", table_name="request_logs")
    op.drop_index("ix_request_logs_agent_id", table_name="request_logs")
    op.drop_column("request_logs", "device_name")
    op.drop_column("request_logs", "device_id")
    op.drop_column("request_logs", "agent_name")
    op.drop_column("request_logs", "agent_id")
    op.drop_index("uq_memory_devices_active_user_device_id", table_name="memory_devices")
    op.drop_index("ix_memory_devices_user_agent_id", table_name="memory_devices")
    op.drop_index("ix_memory_devices_user_id", table_name="memory_devices")
    op.drop_table("memory_devices")
    op.drop_index("uq_memory_agents_active_user_agent_id", table_name="memory_agents")
    op.drop_index("ix_memory_agents_user_id", table_name="memory_agents")
    op.drop_table("memory_agents")
