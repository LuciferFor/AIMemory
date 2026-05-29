import hmac
import secrets
from typing import Any
from urllib.parse import parse_qs, urlencode

from fastapi import HTTPException, Request, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from starlette.responses import RedirectResponse, Response

from aimemory.core.config import Settings, get_settings

COOKIE_NAME = "aimemory_admin"
COOKIE_SALT = "aimemory-admin-session"


def get_serializer(settings: Settings | None = None) -> URLSafeTimedSerializer:
    settings = settings or get_settings()
    return URLSafeTimedSerializer(settings.admin_session_secret, salt=COOKIE_SALT)


async def read_urlencoded_form(request: Request) -> dict[str, str]:
    body = (await request.body()).decode("utf-8")
    parsed = parse_qs(body, keep_blank_values=True)
    return {key: values[0] if values else "" for key, values in parsed.items()}


def verify_admin_credentials(username: str, password: str, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    return hmac.compare_digest(username, settings.admin_username) and hmac.compare_digest(
        password,
        settings.admin_password,
    )


def create_session_payload(username: str) -> dict[str, str]:
    return {
        "sub": username,
        "csrf": secrets.token_urlsafe(24),
    }


def get_admin_session(request: Request, settings: Settings | None = None) -> dict[str, Any] | None:
    settings = settings or get_settings()
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None

    try:
        payload = get_serializer(settings).loads(token, max_age=settings.admin_session_max_age_seconds)
    except (BadSignature, SignatureExpired):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("sub") != settings.admin_username:
        return None
    if not payload.get("csrf"):
        return None
    return payload


def set_admin_session(response: Response, payload: dict[str, str], settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    token = get_serializer(settings).dumps(payload)
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=settings.admin_session_max_age_seconds,
        httponly=True,
        secure=settings.admin_cookie_secure,
        samesite="lax",
        path="/admin",
    )


def clear_admin_session(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/admin")


def login_redirect(request: Request, error: str | None = None) -> RedirectResponse:
    query = {"next": str(request.url.path)}
    if request.url.query:
        query["next"] = f"{request.url.path}?{request.url.query}"
    if error:
        query["error"] = error
    return RedirectResponse(f"/admin/login?{urlencode(query)}", status_code=status.HTTP_303_SEE_OTHER)


def verify_csrf(session: dict[str, Any], form: dict[str, str]) -> None:
    expected = str(session.get("csrf", ""))
    actual = form.get("csrf_token", "")
    if not expected or not hmac.compare_digest(expected, actual):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF 校验失败。")
