from datetime import UTC, datetime

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from aimemory.core.security import hash_api_key
from aimemory.db.session import get_db
from aimemory.models.api_key import ApiKey
from aimemory.models.user import User

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )

    key_hash = hash_api_key(credentials.credentials)
    api_key = db.scalar(select(ApiKey).where(ApiKey.key_hash == key_hash))
    if api_key is None or api_key.revoked_at is not None or not api_key.user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    request.state.request_log_user_id = api_key.user_id
    request.state.request_log_api_key_id = api_key.id
    request.state.request_log_api_key_prefix = api_key.key_prefix
    api_key.last_used_at = datetime.now(UTC)
    db.add(api_key)
    db.commit()
    return api_key.user
