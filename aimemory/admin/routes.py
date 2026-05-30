import uuid
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.templating import Jinja2Templates
from redis import Redis
from sqlalchemy import func, or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload
from starlette.responses import RedirectResponse, Response

from aimemory.admin.auth import (
    clear_admin_session,
    create_session_payload,
    get_admin_session,
    login_redirect,
    read_urlencoded_form,
    set_admin_session,
    verify_admin_credentials,
    verify_csrf,
)
from aimemory.core.config import get_settings
from aimemory.core.security import api_key_prefix, generate_api_key, hash_api_key
from aimemory.db.session import get_db
from aimemory.models.api_key import ApiKey
from aimemory.models.memory import Memory
from aimemory.models.memory_attachment import MemoryAttachment
from aimemory.models.memory_category import MemoryCategory
from aimemory.models.request_log import RequestLog
from aimemory.models.search_stopword import SearchStopword
from aimemory.models.user import User
from aimemory.repositories.memories import get_attachment_for_admin, utcnow
from aimemory.repositories.memory_categories import (
    display_category_name,
    get_or_create_category,
    normalize_category_name,
)
from aimemory.repositories.search_stopwords import (
    add_default_search_stopwords,
    add_search_stopword,
    normalize_stopword,
)
from aimemory.services.attachments import attachment_search_text
from aimemory.services.text import build_search_text, is_numeric_term

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
ADMIN_TIMEZONE = ZoneInfo("Asia/Shanghai")
ADMIN_ASSET_VERSION = "20260530-2148"

STATUS_LABELS = {
    "active": "启用",
    "disabled": "禁用",
    "revoked": "已吊销",
    "ok": "正常",
    "warning": "警告",
    "error": "异常",
}
DELETED_MODE_LABELS = {
    "active": "仅未删除",
    "deleted": "仅已删除",
    "all": "全部",
}
templates.env.globals["status_label"] = lambda value: STATUS_LABELS.get(value, value)
templates.env.globals["deleted_mode_label"] = lambda value: DELETED_MODE_LABELS.get(value, value)


def request_log_business(log: RequestLog) -> dict[str, str]:
    path = str(getattr(log, "path", "") or "")
    method = str(getattr(log, "method", "") or "").upper()
    source = str(getattr(log, "source", "") or "")
    status_code = int(getattr(log, "status_code", 0) or 0)

    if path == "/v1/memories/context" and method == "POST":
        label = "请求记忆"
    elif path == "/v1/memories/search" and method == "POST":
        label = "搜索记忆"
    elif path == "/v1/memories" and method == "POST":
        label = "保存记忆"
    elif path == "/v1/memories" and method == "DELETE":
        label = "删除记忆"
    elif path == "/v1/memories/categories" and method == "GET":
        label = "获取分类"
    elif path == "/v1/memories/write-policy" and method == "GET":
        label = "保存规则"
    elif path.startswith("/v1/memories/attachments/") and method == "GET":
        label = "下载附件"
    elif source == "admin":
        label = "管理后台"
    elif source == "root":
        label = "入口跳转"
    elif source == "api":
        label = "业务接口"
    else:
        label = "普通请求"

    if status_code >= 500:
        return {"label": f"{label}报错", "class": "failed"}
    if status_code >= 400:
        return {"label": f"{label}失败", "class": "warning"}
    if status_code >= 300:
        return {"label": label, "class": "pending"}
    return {"label": label, "class": "ready"}


def short_middle(value: object, head: int = 10, tail: int = 6) -> str:
    text_value = str(value or "")
    if len(text_value) <= head + tail + 3:
        return text_value
    return f"{text_value[:head]}...{text_value[-tail:]}"


def parse_datetime_value(value: object | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(ADMIN_TIMEZONE)


def compact_datetime(value: object | None) -> str:
    parsed = parse_datetime_value(value)
    if parsed is None:
        return str(value or "")
    return parsed.strftime("%m-%d %H:%M")


def full_datetime(value: object | None) -> str:
    parsed = parse_datetime_value(value)
    if parsed is None:
        return str(value or "")
    return f"{parsed.strftime('%Y-%m-%d %H:%M:%S')} 北京时间"


templates.env.globals["short_middle"] = short_middle
templates.env.globals["compact_datetime"] = compact_datetime
templates.env.globals["full_datetime"] = full_datetime
templates.env.globals["admin_asset_version"] = ADMIN_ASSET_VERSION


def parse_optional_uuid(value: str | uuid.UUID | None) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    return uuid.UUID(cleaned)


def redirect_to(path: str, **params: str) -> RedirectResponse:
    query = urlencode({key: value for key, value in params.items() if value})
    url = f"{path}?{query}" if query else path
    return RedirectResponse(url, status_code=status.HTTP_303_SEE_OTHER)


def require_admin_context(request: Request) -> dict[str, Any] | RedirectResponse:
    session = get_admin_session(request)
    if session is None:
        return login_redirect(request)
    mark_admin_request(request, session)
    return session


def mark_admin_request(request: Request, session: dict[str, Any] | None) -> None:
    if session and session.get("sub"):
        request.state.request_log_admin_username = session["sub"]


def base_context(request: Request, session: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "request": request,
        "admin_username": session["sub"],
        "csrf_token": session["csrf"],
        "notice": request.query_params.get("notice"),
        "error": request.query_params.get("error"),
        **extra,
    }


def parse_date(value: str | None, end_of_day: bool = False) -> datetime | None:
    if not value:
        return None
    parsed_date = datetime.fromisoformat(value)
    if parsed_date.tzinfo is None:
        parsed_date = parsed_date.replace(tzinfo=UTC)
    if len(value) == 10:
        parsed_time = time.max if end_of_day else time.min
        parsed_date = datetime.combine(parsed_date.date(), parsed_time, tzinfo=UTC)
    return parsed_date


def active_attachments(memory: Memory) -> list[MemoryAttachment]:
    return [attachment for attachment in memory.attachments if attachment.deleted_at is None]


@router.get("/login")
def login_page(request: Request) -> Response:
    session = get_admin_session(request)
    mark_admin_request(request, session)
    if session:
        return redirect_to("/admin")
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "request": request,
            "error": request.query_params.get("error"),
            "next_url": request.query_params.get("next", "/admin"),
        },
    )


@router.post("/login")
async def login_submit(request: Request) -> Response:
    form = await read_urlencoded_form(request)
    username = form.get("username", "")
    password = form.get("password", "")
    next_url = form.get("next", "/admin")
    if not next_url.startswith("/admin"):
        next_url = "/admin"

    if not verify_admin_credentials(username, password):
        return redirect_to("/admin/login", error="用户名或密码错误。", next=next_url)

    request.state.request_log_admin_username = username
    response = redirect_to(next_url)
    set_admin_session(response, create_session_payload(username))
    return response


@router.post("/logout")
async def logout_submit(request: Request) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    response = redirect_to("/admin/login", notice="已退出登录。")
    clear_admin_session(response)
    return response


@router.get("")
def dashboard(request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session

    stats = {
        "users": db.scalar(select(func.count(User.id))) or 0,
        "api_keys": db.scalar(select(func.count(ApiKey.id))) or 0,
        "memories": db.scalar(select(func.count(Memory.id)).where(Memory.deleted_at.is_(None))) or 0,
        "deleted_memories": db.scalar(select(func.count(Memory.id)).where(Memory.deleted_at.is_not(None))) or 0,
    }
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        base_context(request, session, stats=stats),
    )


@router.get("/users")
def users_page(request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session

    users = db.scalars(select(User).order_by(User.name)).all()
    return templates.TemplateResponse(request, "users.html", base_context(request, session, users=users))


@router.post("/users")
async def create_user(request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    name = form.get("name", "").strip()
    if not name:
        return redirect_to("/admin/users", error="请输入用户名。")

    user = User(name=name)
    db.add(user)
    try:
        db.flush()
        add_default_search_stopwords(db, user)
        db.commit()
    except IntegrityError:
        db.rollback()
        return redirect_to("/admin/users", error="用户已存在。")
    return redirect_to("/admin/users", notice="用户已创建。")


@router.post("/users/{user_id}/toggle")
async def toggle_user(user_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在。")
    user.is_active = not user.is_active
    user.updated_at = utcnow()
    db.add(user)
    db.commit()
    return redirect_to("/admin/users", notice="用户状态已更新。")


@router.get("/search-stopwords")
def search_stopwords_page(
    request: Request,
    user_id: str = "",
    db: Session = Depends(get_db),
) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    try:
        selected_user_id = parse_optional_uuid(user_id)
    except ValueError:
        return redirect_to("/admin/search-stopwords", error="用户筛选参数无效。")

    users = db.scalars(select(User).order_by(User.name)).all()
    query = (
        select(SearchStopword, User.name.label("user_name"))
        .join(User)
        .where(SearchStopword.deleted_at.is_(None))
        .order_by(User.name, SearchStopword.term)
    )
    if selected_user_id:
        query = query.where(SearchStopword.user_id == selected_user_id)
    rows = [
        {"stopword": stopword, "user_name": user_name}
        for stopword, user_name in db.execute(query).all()
    ]
    return templates.TemplateResponse(
        request,
        "search_stopwords.html",
        base_context(
            request,
            session,
            users=users,
            rows=rows,
            selected_user_id=str(selected_user_id) if selected_user_id else "",
        ),
    )


@router.post("/search-stopwords")
async def create_search_stopword(request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    selected_user_id = form.get("selected_user_id", "")
    try:
        user_id = uuid.UUID(form.get("user_id", ""))
    except ValueError:
        return redirect_to("/admin/search-stopwords", user_id=selected_user_id, error="请选择有效用户。")

    user = db.get(User, user_id)
    if user is None:
        return redirect_to("/admin/search-stopwords", user_id=selected_user_id, error="用户不存在。")

    term = normalize_stopword(form.get("term", ""))
    if not term:
        return redirect_to("/admin/search-stopwords", user_id=selected_user_id, error="请输入停用词。")
    if len(term) > 128:
        return redirect_to("/admin/search-stopwords", user_id=selected_user_id, error="停用词不能超过 128 个字符。")
    if is_numeric_term(term):
        return redirect_to("/admin/search-stopwords", user_id=selected_user_id, error="纯数字会被系统自动忽略，无需添加。")

    note = form.get("note", "").strip()
    add_search_stopword(db, user.id, term, note)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return redirect_to("/admin/search-stopwords", user_id=selected_user_id, error="这个停用词已经存在。")
    return redirect_to("/admin/search-stopwords", user_id=str(user.id), notice="停用词已添加。")


@router.post("/search-stopwords/{stopword_id}/delete")
async def delete_search_stopword(stopword_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    selected_user_id = form.get("selected_user_id", "")
    stopword = db.get(SearchStopword, stopword_id)
    if stopword is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="停用词不存在。")
    if stopword.deleted_at is None:
        stopword.deleted_at = utcnow()
        db.add(stopword)
        db.commit()
    return redirect_to("/admin/search-stopwords", user_id=selected_user_id, notice="停用词已删除。")


@router.get("/api-keys")
def api_keys_page(
    request: Request,
    user_id: str = "",
    db: Session = Depends(get_db),
) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    try:
        selected_user_id = parse_optional_uuid(user_id)
    except ValueError:
        return redirect_to("/admin/api-keys", error="用户筛选参数无效。")
    return render_api_keys(request, session, db, user_id=selected_user_id)


def render_api_keys(
    request: Request,
    session: dict[str, Any],
    db: Session,
    user_id: uuid.UUID | None = None,
    created_key: str | None = None,
) -> Response:
    users = db.scalars(select(User).order_by(User.name)).all()
    query = select(ApiKey, User.name.label("user_name")).join(User).order_by(ApiKey.created_at.desc())
    if user_id:
        query = query.where(ApiKey.user_id == user_id)
    rows = [
        {"api_key": api_key, "user_name": user_name}
        for api_key, user_name in db.execute(query).all()
    ]
    return templates.TemplateResponse(
        request,
        "api_keys.html",
        base_context(
            request,
            session,
            users=users,
            rows=rows,
            selected_user_id=str(user_id) if user_id else "",
            created_key=created_key,
        ),
    )


@router.post("/api-keys")
async def create_api_key(request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    try:
        user_id = uuid.UUID(form.get("user_id", ""))
    except ValueError:
        return redirect_to("/admin/api-keys", error="请选择有效用户。")

    user = db.get(User, user_id)
    if user is None:
        return redirect_to("/admin/api-keys", error="用户不存在。")

    settings = get_settings()
    raw_key = generate_api_key(settings.api_key_prefix)
    api_key = ApiKey(
        user_id=user.id,
        key_hash=hash_api_key(raw_key),
        key_prefix=api_key_prefix(raw_key),
        label=form.get("label", "").strip() or None,
    )
    db.add(api_key)
    db.commit()
    return render_api_keys(request, session, db, user_id=user.id, created_key=raw_key)


@router.post("/api-keys/{api_key_id}/label")
async def update_api_key_label(api_key_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    label = form.get("label", "").strip()
    selected_user_id = form.get("selected_user_id", "")
    if len(label) > 128:
        return redirect_to("/admin/api-keys", user_id=selected_user_id, error="标签不能超过 128 个字符。")

    api_key = db.get(ApiKey, api_key_id)
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="接口密钥不存在。")
    api_key.label = label or None
    db.add(api_key)
    db.commit()
    return redirect_to("/admin/api-keys", user_id=selected_user_id, notice="接口密钥标签已更新。")


@router.post("/api-keys/{api_key_id}/revoke")
async def revoke_api_key(api_key_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    api_key = db.get(ApiKey, api_key_id)
    if api_key is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="接口密钥不存在。")
    api_key.revoked_at = utcnow()
    db.add(api_key)
    db.commit()
    return redirect_to("/admin/api-keys", notice="接口密钥已吊销。")


@router.get("/categories")
def categories_page(
    request: Request,
    user_id: str = "",
    db: Session = Depends(get_db),
) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    try:
        selected_user_id = parse_optional_uuid(user_id)
    except ValueError:
        return redirect_to("/admin/categories", error="用户筛选参数无效。")

    query = (
        select(MemoryCategory, User.name.label("user_name"), func.count(Memory.id).label("memory_count"))
        .join(User, User.id == MemoryCategory.user_id)
        .outerjoin(Memory, (Memory.category_id == MemoryCategory.id) & (Memory.deleted_at.is_(None)))
        .where(MemoryCategory.deleted_at.is_(None))
        .group_by(MemoryCategory.id, User.name)
        .order_by(User.name, MemoryCategory.name)
    )
    if selected_user_id:
        query = query.where(MemoryCategory.user_id == selected_user_id)
    rows = [
        {"category": category, "user_name": user_name, "memory_count": int(memory_count or 0)}
        for category, user_name, memory_count in db.execute(query).all()
    ]
    users = db.scalars(select(User).order_by(User.name)).all()
    return templates.TemplateResponse(
        request,
        "categories.html",
        base_context(
            request,
            session,
            users=users,
            rows=rows,
            filters={"user_id": str(selected_user_id) if selected_user_id else ""},
        ),
    )


@router.post("/categories")
async def create_category(request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    try:
        user_id = uuid.UUID(form.get("user_id", ""))
    except ValueError:
        return redirect_to("/admin/categories", error="请选择用户。")
    name = display_category_name(form.get("name", ""))
    description = form.get("description", "").strip() or None
    if not name:
        return redirect_to("/admin/categories", user_id=str(user_id), error="分类名称不能为空。")
    if len(name) > 128:
        return redirect_to("/admin/categories", user_id=str(user_id), error="分类名称不能超过 128 个字符。")
    user = db.get(User, user_id)
    if user is None:
        return redirect_to("/admin/categories", error="用户不存在。")
    try:
        get_or_create_category(db, user_id, name, description)
        db.commit()
    except IntegrityError:
        db.rollback()
        return redirect_to("/admin/categories", user_id=str(user_id), error="分类已存在。")
    return redirect_to("/admin/categories", user_id=str(user_id), notice="分类已创建。")


@router.post("/categories/{category_id}/update")
async def update_category(category_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    category = db.get(MemoryCategory, category_id)
    if category is None or category.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="分类不存在。")
    name = display_category_name(form.get("name", ""))
    description = form.get("description", "").strip() or None
    if not name:
        return redirect_to("/admin/categories", user_id=str(category.user_id), error="分类名称不能为空。")
    if len(name) > 128:
        return redirect_to("/admin/categories", user_id=str(category.user_id), error="分类名称不能超过 128 个字符。")
    category.name = name
    category.normalized_name = normalize_category_name(name)
    category.description = description
    category.updated_at = utcnow()
    db.add(category)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return redirect_to("/admin/categories", user_id=str(category.user_id), error="分类已存在。")
    return redirect_to("/admin/categories", user_id=str(category.user_id), notice="分类已更新。")


@router.post("/categories/{category_id}/delete")
async def delete_category(category_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    category = db.get(MemoryCategory, category_id)
    if category is None or category.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="分类不存在。")
    active_count = db.scalar(select(func.count(Memory.id)).where(Memory.category_id == category.id, Memory.deleted_at.is_(None))) or 0
    if active_count:
        return redirect_to("/admin/categories", user_id=str(category.user_id), error="分类下还有记忆，请先合并到其他分类。")
    category.deleted_at = utcnow()
    category.updated_at = utcnow()
    db.add(category)
    db.commit()
    return redirect_to("/admin/categories", user_id=str(category.user_id), notice="分类已删除。")


@router.post("/categories/{category_id}/merge")
async def merge_category(category_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    source = db.get(MemoryCategory, category_id)
    if source is None or source.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="分类不存在。")
    try:
        target_id = uuid.UUID(form.get("target_category_id", ""))
    except ValueError:
        return redirect_to("/admin/categories", user_id=str(source.user_id), error="请选择目标分类。")
    target = db.get(MemoryCategory, target_id)
    if target is None or target.deleted_at is not None or target.user_id != source.user_id:
        return redirect_to("/admin/categories", user_id=str(source.user_id), error="目标分类无效。")
    if target.id == source.id:
        return redirect_to("/admin/categories", user_id=str(source.user_id), error="不能合并到自身。")

    now = utcnow()
    for memory in db.scalars(select(Memory).where(Memory.category_id == source.id)).all():
        memory.category_id = target.id
        memory.updated_at = now
        db.add(memory)
    source.merged_into_id = target.id
    source.deleted_at = now
    source.updated_at = now
    db.add(source)
    db.commit()
    return redirect_to("/admin/categories", user_id=str(source.user_id), notice="分类已合并。")


@router.get("/memories")
def memories_page(
    request: Request,
    user_id: str = "",
    category_id: str = "",
    agent_id: str = "",
    q: str = "",
    deleted: str = Query(default="active", pattern="^(active|deleted|all)$"),
    since: str = "",
    until: str = "",
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session

    try:
        selected_user_id = parse_optional_uuid(user_id)
    except ValueError:
        return redirect_to("/admin/memories", error="用户筛选参数无效。")
    try:
        selected_category_id = parse_optional_uuid(category_id)
    except ValueError:
        return redirect_to("/admin/memories", error="分类筛选参数无效。")

    query = (
        select(Memory, User.name.label("user_name"), MemoryCategory.name.label("category_name"))
        .select_from(Memory)
        .join(User, User.id == Memory.user_id)
        .join(MemoryCategory, MemoryCategory.id == Memory.category_id)
        .options(selectinload(Memory.attachments).defer(MemoryAttachment.image_bytes))
        .order_by(Memory.updated_at.desc())
    )
    if selected_user_id:
        query = query.where(Memory.user_id == selected_user_id)
    if selected_category_id:
        query = query.where(Memory.category_id == selected_category_id)
    if agent_id.strip():
        query = query.where(Memory.agent_id == agent_id.strip())
    if deleted == "active":
        query = query.where(Memory.deleted_at.is_(None))
    elif deleted == "deleted":
        query = query.where(Memory.deleted_at.is_not(None))
    if q.strip():
        like = f"%{q.strip()}%"
        query = query.where(
            or_(
                Memory.external_id.ilike(like),
                Memory.title.ilike(like),
                Memory.content.ilike(like),
                Memory.search_text.ilike(like),
            )
        )
    since_dt = parse_date(since)
    until_dt = parse_date(until, end_of_day=True)
    if since_dt:
        query = query.where(Memory.created_at >= since_dt)
    if until_dt:
        query = query.where(Memory.created_at <= until_dt)

    rows = [
        {
            "memory": memory,
            "user_name": user_name,
            "category_name": category_name,
            "attachments": [attachment for attachment in memory.attachments if attachment.deleted_at is None],
        }
        for memory, user_name, category_name in db.execute(query.limit(limit)).all()
    ]
    users = db.scalars(select(User).order_by(User.name)).all()
    category_query = select(MemoryCategory).where(MemoryCategory.deleted_at.is_(None)).order_by(MemoryCategory.name)
    if selected_user_id:
        category_query = category_query.where(MemoryCategory.user_id == selected_user_id)
    categories = db.scalars(category_query).all()
    agents = db.scalars(select(Memory.agent_id).distinct().order_by(Memory.agent_id)).all()
    return templates.TemplateResponse(
        request,
        "memories.html",
        base_context(
            request,
            session,
            users=users,
            categories=categories,
            agents=agents,
            rows=rows,
            filters={
                "user_id": str(selected_user_id) if selected_user_id else "",
                "category_id": str(selected_category_id) if selected_category_id else "",
                "agent_id": agent_id,
                "q": q,
                "deleted": deleted,
                "since": since,
                "until": until,
                "limit": limit,
            },
        ),
    )


@router.get("/request-logs")
def request_logs_page(
    request: Request,
    source: str = Query(default="", pattern="^(|api|admin|root)$"),
    method: str = "",
    status_code: str = "",
    q: str = "",
    user_id: str = "",
    since: str = "",
    until: str = "",
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session

    try:
        selected_user_id = parse_optional_uuid(user_id)
    except ValueError:
        return redirect_to("/admin/request-logs", error="用户筛选参数无效。")

    parsed_status_code: int | None = None
    if status_code.strip():
        try:
            parsed_status_code = int(status_code)
        except ValueError:
            return redirect_to("/admin/request-logs", error="状态码筛选参数无效。")

    normalized_method = method.strip().upper()
    query = select(RequestLog, User.name.label("user_name")).outerjoin(User, User.id == RequestLog.user_id).order_by(RequestLog.created_at.desc())
    if source:
        query = query.where(RequestLog.source == source)
    if normalized_method:
        query = query.where(RequestLog.method == normalized_method)
    if parsed_status_code is not None:
        query = query.where(RequestLog.status_code == parsed_status_code)
    if selected_user_id:
        query = query.where(RequestLog.user_id == selected_user_id)
    if q.strip():
        like = f"%{q.strip()}%"
        query = query.where(or_(RequestLog.path.ilike(like), RequestLog.route_path.ilike(like), RequestLog.request_id.ilike(like)))
    since_dt = parse_date(since)
    until_dt = parse_date(until, end_of_day=True)
    if since_dt:
        query = query.where(RequestLog.created_at >= since_dt)
    if until_dt:
        query = query.where(RequestLog.created_at <= until_dt)

    rows = [
        {"log": log, "user_name": user_name, "business": request_log_business(log)}
        for log, user_name in db.execute(query.limit(limit)).all()
    ]
    users = db.scalars(select(User).order_by(User.name)).all()
    return templates.TemplateResponse(
        request,
        "request_logs.html",
        base_context(
            request,
            session,
            rows=rows,
            users=users,
            filters={
                "source": source,
                "method": normalized_method,
                "status_code": status_code,
                "q": q,
                "user_id": str(selected_user_id) if selected_user_id else "",
                "since": since,
                "until": until,
                "limit": limit,
            },
        ),
    )


@router.get("/memories/{memory_id}")
def memory_detail_page(memory_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session

    row = db.execute(
        select(Memory, User.name.label("user_name"), MemoryCategory.name.label("category_name"))
        .select_from(Memory)
        .join(User, User.id == Memory.user_id)
        .join(MemoryCategory, MemoryCategory.id == Memory.category_id)
        .options(selectinload(Memory.attachments).defer(MemoryAttachment.image_bytes))
        .where(Memory.id == memory_id)
    ).one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记忆不存在。")
    memory, user_name, category_name = row
    return templates.TemplateResponse(
        request,
        "memory_detail.html",
        base_context(
            request,
            session,
            memory=memory,
            user_name=user_name,
            category_name=category_name,
            attachments=active_attachments(memory),
        ),
    )


@router.post("/memories/{memory_id}/update")
async def update_memory(memory_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    title = form.get("title", "").strip()
    content = form.get("content", "").strip()
    detail_path = f"/admin/memories/{memory_id}"
    if not title:
        return redirect_to(detail_path, error="标题不能为空。")
    if len(title) > 512:
        return redirect_to(detail_path, error="标题不能超过 512 个字符。")
    if not content:
        return redirect_to(detail_path, error="正文不能为空。")

    memory = db.scalar(
        select(Memory)
        .options(selectinload(Memory.attachments).defer(MemoryAttachment.image_bytes))
        .where(Memory.id == memory_id)
    )
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记忆不存在。")

    memory.title = title
    memory.content = content
    memory.search_text = build_search_text(title, content, attachment_search_text(active_attachments(memory)))
    memory.updated_at = utcnow()
    db.add(memory)
    db.commit()
    return redirect_to(detail_path, notice="记忆已更新。")


@router.post("/memories/{memory_id}/delete")
async def delete_memory(memory_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    memory = db.get(Memory, memory_id)
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="记忆不存在。")
    if memory.deleted_at is None:
        memory.deleted_at = utcnow()
        memory.updated_at = utcnow()
        db.add(memory)
        db.commit()
    return redirect_to("/admin/memories", notice="记忆已删除。")


@router.get("/attachments/{attachment_id}")
def attachment_preview(attachment_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    attachment = get_attachment_for_admin(db, attachment_id)
    if attachment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="附件不存在。")
    return Response(content=attachment.image_bytes, media_type=attachment.mime_type)


@router.get("/jobs")
def jobs_page(
    request: Request,
    db: Session = Depends(get_db),
) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    return templates.TemplateResponse(
        request,
        "jobs.html",
        base_context(request, session),
    )


@router.get("/health")
def health_page(request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session

    settings = get_settings()
    checks: list[dict[str, str]] = []
    try:
        db.execute(text("select 1")).scalar_one()
        checks.append({"name": "PostgreSQL", "status": "ok", "detail": "已连接"})
    except Exception as exc:
        checks.append({"name": "PostgreSQL", "status": "error", "detail": str(exc)})

    try:
        Redis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=2).ping()
        checks.append({"name": "Redis", "status": "ok", "detail": "已连接"})
    except Exception as exc:
        checks.append({"name": "Redis", "status": "error", "detail": str(exc)})

    checks.append(
        {
            "name": "后台会话密钥",
            "status": "ok" if not settings.admin_session_secret.startswith("change-me") else "warning",
            "detail": "已配置" if not settings.admin_session_secret.startswith("change-me") else "正在使用默认值",
        }
    )
    return templates.TemplateResponse(request, "health.html", base_context(request, session, checks=checks))
