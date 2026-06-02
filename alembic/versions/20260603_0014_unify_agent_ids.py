"""unify agent ids

Revision ID: 20260603_0014
Revises: 20260602_0013
Create Date: 2026-06-03 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "20260603_0014"
down_revision: str | None = "20260602_0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


OLD_AGENT_31_9 = "5df9cbfb-d31b-46dd-972b-05d466d2257c"
NEW_AGENT_31_9 = "am_my7vyfwqm53vf7jdwlwzxfmw"
OLD_AGENT_FEIYE = "feiye-31-11"
NEW_AGENT_FEIYE = "am_bgwjsmilavsj9sv3pinfaxjk"


def _rename_agent(old: str, new: str) -> None:
    op.execute(
        sa.text(
            """
UPDATE memory_agents
SET legacy_agent_id = COALESCE(legacy_agent_id, agent_id),
    agent_id = :new,
    updated_at = NOW()
WHERE agent_id = :old
  AND deleted_at IS NULL
"""
        ).bindparams(old=old, new=new)
    )
    op.execute(sa.text("UPDATE memory_devices SET agent_id = :new, updated_at = NOW() WHERE agent_id = :old").bindparams(old=old, new=new))
    op.execute(sa.text("UPDATE memories SET agent_id = :new, updated_at = NOW() WHERE agent_id = :old").bindparams(old=old, new=new))
    op.execute(sa.text("UPDATE request_logs SET agent_id = :new WHERE agent_id = :old").bindparams(old=old, new=new))


def upgrade() -> None:
    op.add_column("memory_agents", sa.Column("legacy_agent_id", sa.String(length=128), nullable=True))
    _rename_agent(OLD_AGENT_31_9, NEW_AGENT_31_9)
    _rename_agent(OLD_AGENT_FEIYE, NEW_AGENT_FEIYE)


def downgrade() -> None:
    op.execute(
        sa.text(
            """
UPDATE memory_agents
SET agent_id = :old,
    updated_at = NOW()
WHERE agent_id = :new
  AND legacy_agent_id = :old
"""
        ).bindparams(old=OLD_AGENT_31_9, new=NEW_AGENT_31_9)
    )
    op.execute(
        sa.text(
            """
UPDATE memory_agents
SET agent_id = :old,
    updated_at = NOW()
WHERE agent_id = :new
  AND legacy_agent_id = :old
"""
        ).bindparams(old=OLD_AGENT_FEIYE, new=NEW_AGENT_FEIYE)
    )
    op.execute(sa.text("UPDATE memory_devices SET agent_id = :old WHERE agent_id = :new").bindparams(old=OLD_AGENT_31_9, new=NEW_AGENT_31_9))
    op.execute(sa.text("UPDATE memory_devices SET agent_id = :old WHERE agent_id = :new").bindparams(old=OLD_AGENT_FEIYE, new=NEW_AGENT_FEIYE))
    op.execute(sa.text("UPDATE memories SET agent_id = :old WHERE agent_id = :new").bindparams(old=OLD_AGENT_31_9, new=NEW_AGENT_31_9))
    op.execute(sa.text("UPDATE memories SET agent_id = :old WHERE agent_id = :new").bindparams(old=OLD_AGENT_FEIYE, new=NEW_AGENT_FEIYE))
    op.execute(sa.text("UPDATE request_logs SET agent_id = :old WHERE agent_id = :new").bindparams(old=OLD_AGENT_31_9, new=NEW_AGENT_31_9))
    op.execute(sa.text("UPDATE request_logs SET agent_id = :old WHERE agent_id = :new").bindparams(old=OLD_AGENT_FEIYE, new=NEW_AGENT_FEIYE))
    op.drop_column("memory_agents", "legacy_agent_id")
