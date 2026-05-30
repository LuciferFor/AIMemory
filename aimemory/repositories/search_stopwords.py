import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from aimemory.models.search_stopword import SearchStopword
from aimemory.models.user import User
from aimemory.services.text import is_numeric_term, normalize_query

DEFAULT_SEARCH_STOPWORDS = ("skill", "assistant", "user", "xxx", "aimemory", "openclaw")


def normalize_stopword(term: str) -> str:
    return normalize_query(term)


def default_stopword_terms_for_user(user_name: str) -> list[str]:
    terms = {normalize_stopword(term) for term in DEFAULT_SEARCH_STOPWORDS}
    if user_name:
        terms.add(normalize_stopword(user_name))
    return sorted(term for term in terms if term and not is_numeric_term(term))


def active_search_stopword_terms(db: Session, user_id: uuid.UUID) -> set[str]:
    return set(
        db.scalars(
            select(SearchStopword.term).where(
                SearchStopword.user_id == user_id,
                SearchStopword.deleted_at.is_(None),
            )
        ).all()
    )


def add_default_search_stopwords(db: Session, user: User) -> None:
    existing_terms = active_search_stopword_terms(db, user.id)
    for term in default_stopword_terms_for_user(user.name):
        if term not in existing_terms:
            db.add(SearchStopword(user_id=user.id, term=term, note="默认停用词"))


def add_search_stopword(db: Session, user_id: uuid.UUID, term: str, note: str | None = None) -> SearchStopword:
    normalized = normalize_stopword(term)
    stopword = SearchStopword(user_id=user_id, term=normalized, note=note.strip() if note else None)
    db.add(stopword)
    return stopword
