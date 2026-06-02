import unicodedata
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aimemory.models.memory import Memory
from aimemory.models.memory_category import MemoryCategory

UNCATEGORIZED_CATEGORY = "其它"


@dataclass(frozen=True)
class CategorySummary:
    id: uuid.UUID
    name: str
    description: str | None
    memory_count: int


def normalize_category_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).strip().lower().split())


def display_category_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).strip().split())


def get_active_category(db: Session, user_id: uuid.UUID, category: str) -> MemoryCategory | None:
    normalized = normalize_category_name(category)
    if not normalized:
        return None
    return db.scalar(
        select(MemoryCategory).where(
            MemoryCategory.user_id == user_id,
            MemoryCategory.normalized_name == normalized,
            MemoryCategory.deleted_at.is_(None),
        )
    )


def get_or_create_category(
    db: Session,
    user_id: uuid.UUID,
    category: str,
    description: str | None = None,
) -> tuple[MemoryCategory, bool]:
    normalized = normalize_category_name(category)
    name = display_category_name(category)
    if not normalized or not name:
        raise ValueError("分类不能为空。")

    existing = get_active_category(db, user_id, normalized)
    if existing is not None:
        return existing, False

    created = MemoryCategory(
        user_id=user_id,
        name=name[:128],
        normalized_name=normalized[:128],
        description=description.strip() if description else None,
    )
    db.add(created)
    db.flush()
    return created, True


def list_category_summaries(db: Session, user_id: uuid.UUID) -> list[CategorySummary]:
    rows = db.execute(
        select(
            MemoryCategory.id,
            MemoryCategory.name,
            MemoryCategory.description,
            func.count(Memory.id).label("memory_count"),
        )
        .outerjoin(Memory, (Memory.category_id == MemoryCategory.id) & (Memory.deleted_at.is_(None)))
        .where(MemoryCategory.user_id == user_id, MemoryCategory.deleted_at.is_(None))
        .group_by(MemoryCategory.id)
        .order_by(MemoryCategory.name)
    ).mappings().all()
    return [
        CategorySummary(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            memory_count=int(row["memory_count"] or 0),
        )
        for row in rows
    ]
