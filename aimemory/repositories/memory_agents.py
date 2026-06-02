import unicodedata
import uuid
from dataclasses import dataclass
import re
import secrets
import string

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aimemory.models.memory import Memory
from aimemory.models.memory_agent import MemoryAgent
from aimemory.models.memory_device import MemoryDevice


AGENT_ID_PREFIX = "am_"
AGENT_ID_LENGTH = 27
_AGENT_ID_RE = re.compile(r"^am_[a-z0-9]{24}$")
_AGENT_ID_ALPHABET = string.ascii_lowercase + string.digits
AGENT_ID_31_9 = "am_my7vyfwqm53vf7jdwlwzxfmw"
AGENT_ID_FEIYE_31_11 = "am_bgwjsmilavsj9sv3pinfaxjk"


@dataclass(frozen=True)
class ResolvedMemoryIdentity:
    agent_id: str
    agent_name: str | None = None
    device_id: str | None = None
    device_name: str | None = None
    source: str = "agent"


def normalize_identifier(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).strip().split())[:128]


def generate_agent_id() -> str:
    return AGENT_ID_PREFIX + "".join(secrets.choice(_AGENT_ID_ALPHABET) for _ in range(24))


def is_unified_agent_id(value: object) -> bool:
    return bool(_AGENT_ID_RE.fullmatch(normalize_identifier(value)))


def default_agent_display_name(agent_id: str) -> str:
    if agent_id in {"feiye-31-11", AGENT_ID_FEIYE_31_11}:
        return "绯夜"
    if agent_id in {"5df9cbfb-d31b-46dd-972b-05d466d2257c", AGENT_ID_31_9}:
        return "31.9 OpenClaw"
    return agent_id


def get_active_agent(db: Session, user_id: uuid.UUID, agent_id: str) -> MemoryAgent | None:
    cleaned = normalize_identifier(agent_id)
    if not cleaned:
        return None
    return db.scalar(
        select(MemoryAgent).where(
            MemoryAgent.user_id == user_id,
            MemoryAgent.agent_id == cleaned,
            MemoryAgent.deleted_at.is_(None),
        )
    )


def get_or_create_agent(
    db: Session,
    user_id: uuid.UUID,
    agent_id: str,
    display_name: str | None = None,
    description: str | None = None,
) -> tuple[MemoryAgent, bool]:
    cleaned = normalize_identifier(agent_id)
    if not cleaned:
        raise ValueError("智能体 ID 不能为空。")
    if not is_unified_agent_id(cleaned):
        raise ValueError("智能体 ID 必须使用 am_ 开头的统一格式。")
    existing = get_active_agent(db, user_id, cleaned)
    if existing is not None and hasattr(existing, "agent_id"):
        return existing, False

    if not hasattr(db, "add") or not hasattr(db, "flush"):
        agent = MemoryAgent(
            user_id=user_id,
            agent_id=cleaned,
            display_name=(normalize_identifier(display_name) or default_agent_display_name(cleaned))[:128],
            description=description.strip() if description else None,
            is_active=True,
        )
        return agent, True

    agent = MemoryAgent(
        user_id=user_id,
        agent_id=cleaned,
        display_name=(normalize_identifier(display_name) or default_agent_display_name(cleaned))[:128],
        description=description.strip() if description else None,
        is_active=True,
    )
    db.add(agent)
    db.flush()
    return agent, True


def create_agent_with_generated_id(
    db: Session,
    user_id: uuid.UUID,
    *,
    display_name: str,
    description: str | None = None,
) -> MemoryAgent:
    for _ in range(20):
        agent_id = generate_agent_id()
        if get_active_agent(db, user_id, agent_id) is None:
            agent, _created = get_or_create_agent(
                db,
                user_id,
                agent_id,
                display_name=display_name,
                description=description,
            )
            return agent
    raise RuntimeError("生成智能体 ID 失败，请重试。")


def get_active_device(db: Session, user_id: uuid.UUID, device_id: str) -> MemoryDevice | None:
    cleaned = normalize_identifier(device_id)
    if not cleaned:
        return None
    return db.scalar(
        select(MemoryDevice).where(
            MemoryDevice.user_id == user_id,
            MemoryDevice.device_id == cleaned,
            MemoryDevice.deleted_at.is_(None),
        )
    )


def resolve_memory_identity(
    db: Session,
    user_id: uuid.UUID,
    *,
    agent_id: str | None,
    device_id: str | None = None,
    create_missing_agent: bool = True,
) -> ResolvedMemoryIdentity:
    cleaned_device_id = normalize_identifier(device_id)
    cleaned_agent_id = normalize_identifier(agent_id)
    if cleaned_device_id:
        device = get_active_device(db, user_id, cleaned_device_id)
        if device is None:
            raise ValueError("设备不存在。")
        if not getattr(device, "is_active", True):
            raise PermissionError("设备已停用。")
        agent = get_active_agent(db, user_id, device.agent_id)
        if agent is None or not hasattr(agent, "agent_id"):
            if not create_missing_agent:
                raise ValueError("设备绑定的智能体不存在。")
            agent, _ = get_or_create_agent(db, user_id, device.agent_id)
        if not getattr(agent, "is_active", True):
            raise PermissionError("设备绑定的智能体已停用。")
        return ResolvedMemoryIdentity(
            agent_id=agent.agent_id,
            agent_name=getattr(agent, "display_name", agent.agent_id),
            device_id=device.device_id,
            device_name=getattr(device, "display_name", device.device_id),
            source="device",
        )

    if not cleaned_agent_id:
        raise ValueError("必须提供 agent_id 或 device_id。")
    agent = get_active_agent(db, user_id, cleaned_agent_id)
    if agent is not None and not hasattr(agent, "agent_id"):
        return ResolvedMemoryIdentity(
            agent_id=cleaned_agent_id,
            agent_name=cleaned_agent_id,
            source="agent",
        )
    if agent is None:
        if not create_missing_agent:
            raise ValueError("智能体不存在。")
        agent, _ = get_or_create_agent(db, user_id, cleaned_agent_id)
    if not getattr(agent, "is_active", True):
        raise PermissionError("智能体已停用。")
    return ResolvedMemoryIdentity(
        agent_id=getattr(agent, "agent_id", cleaned_agent_id),
        agent_name=getattr(agent, "display_name", cleaned_agent_id),
        source="agent",
    )


def agent_labels_for_user(db: Session, user_id: uuid.UUID) -> dict[str, str]:
    agents = db.scalars(
        select(MemoryAgent).where(
            MemoryAgent.user_id == user_id,
            MemoryAgent.deleted_at.is_(None),
        )
    ).all()
    return {agent.agent_id: agent.display_name for agent in agents}


def agent_memory_counts(db: Session) -> dict[tuple[uuid.UUID, str], int]:
    rows = db.execute(
        select(Memory.user_id, Memory.agent_id, func.count(Memory.id))
        .where(Memory.deleted_at.is_(None))
        .group_by(Memory.user_id, Memory.agent_id)
    ).all()
    return {(user_id, agent_id): int(count or 0) for user_id, agent_id, count in rows}
