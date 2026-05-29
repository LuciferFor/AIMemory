import uuid
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.templating import Jinja2Templates
from redis import Redis
from sqlalchemy import func, or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
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
from aimemory.models.embedding_job import EmbeddingJob
from aimemory.models.memory import Memory
from aimemory.models.user import User
from aimemory.repositories.memories import create_embedding_job, utcnow
from aimemory.worker.tasks import generate_memory_embedding

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

STATUS_LABELS = {
    "active": "启用",
    "disabled": "禁用",
    "revoked": "已吊销",
    "ready": "已就绪",
    "pending": "待处理",
    "running": "运行中",
    "retrying": "重试中",
    "succeeded": "成功",
    "failed": "失败",
    "skipped": "已跳过",
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


def redirect_to(path: str, **params: str) -> RedirectResponse:
    query = urlencode({key: value for key, value in params.items() if value})
    url = f"{path}?{query}" if query else path
    return RedirectResponse(url, status_code=status.HTTP_303_SEE_OTHER)


def require_admin_context(request: Request) -> dict[str, Any] | RedirectResponse:
    session = get_admin_session(request)
    if session is None:
        return login_redirect(request)
    return session


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


@router.get("/login")
def login_page(request: Request) -> Response:
    if get_admin_session(request):
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

    status_rows = db.execute(
        select(Memory.embedding_status, func.count(Memory.id))
        .where(Memory.deleted_at.is_(None))
        .group_by(Memory.embedding_status)
    ).all()
    embedding_counts = {status_name or "unknown": count for status_name, count in status_rows}
    stats = {
        "users": db.scalar(select(func.count(User.id))) or 0,
        "api_keys": db.scalar(select(func.count(ApiKey.id))) or 0,
        "memories": db.scalar(select(func.count(Memory.id)).where(Memory.deleted_at.is_(None))) or 0,
        "deleted_memories": db.scalar(select(func.count(Memory.id)).where(Memory.deleted_at.is_not(None))) or 0,
        "jobs": db.scalar(select(func.count(EmbeddingJob.id))) or 0,
        "pending": embedding_counts.get("pending", 0),
        "failed": embedding_counts.get("failed", 0),
        "ready": embedding_counts.get("ready", 0),
    }
    recent_jobs = db.scalars(select(EmbeddingJob).order_by(EmbeddingJob.created_at.desc()).limit(8)).all()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        base_context(request, session, stats=stats, recent_jobs=recent_jobs),
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

    db.add(User(name=name))
    try:
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


@router.get("/api-keys")
def api_keys_page(
    request: Request,
    user_id: uuid.UUID | None = None,
    db: Session = Depends(get_db),
) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    return render_api_keys(request, session, db, user_id=user_id)


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


@router.get("/memories")
def memories_page(
    request: Request,
    user_id: uuid.UUID | None = None,
    agent_id: str = "",
    q: str = "",
    embedding_status: str = "",
    deleted: str = Query(default="active", pattern="^(active|deleted|all)$"),
    since: str = "",
    until: str = "",
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session

    query = select(Memory, User.name.label("user_name")).join(User).order_by(Memory.updated_at.desc())
    if user_id:
        query = query.where(Memory.user_id == user_id)
    if agent_id.strip():
        query = query.where(Memory.agent_id == agent_id.strip())
    if embedding_status.strip():
        query = query.where(Memory.embedding_status == embedding_status.strip())
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
        {"memory": memory, "user_name": user_name}
        for memory, user_name in db.execute(query.limit(limit)).all()
    ]
    users = db.scalars(select(User).order_by(User.name)).all()
    agents = db.scalars(select(Memory.agent_id).distinct().order_by(Memory.agent_id)).all()
    return templates.TemplateResponse(
        request,
        "memories.html",
        base_context(
            request,
            session,
            users=users,
            agents=agents,
            rows=rows,
            filters={
                "user_id": str(user_id) if user_id else "",
                "agent_id": agent_id,
                "q": q,
                "embedding_status": embedding_status,
                "deleted": deleted,
                "since": since,
                "until": until,
                "limit": limit,
            },
        ),
    )


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


@router.post("/memories/{memory_id}/requeue")
async def requeue_memory(memory_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    memory = db.get(Memory, memory_id)
    if memory is None or memory.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="未找到可用记忆。")
    memory.embedding_status = "pending"
    memory.embedding_error = None
    job = create_embedding_job(db, memory.id)
    db.commit()
    generate_memory_embedding.delay(str(memory.id), str(job.id))
    return redirect_to("/admin/memories", notice="向量任务已加入队列。")


@router.get("/jobs")
def jobs_page(
    request: Request,
    job_status: str = "",
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    query = (
        select(EmbeddingJob, Memory.external_id, Memory.agent_id, User.name.label("user_name"))
        .join(Memory, EmbeddingJob.memory_id == Memory.id)
        .join(User, Memory.user_id == User.id)
        .order_by(EmbeddingJob.updated_at.desc())
        .limit(limit)
    )
    if job_status.strip():
        query = query.where(EmbeddingJob.status == job_status.strip())
    rows = [
        {
            "job": job,
            "external_id": external_id,
            "agent_id": agent_id,
            "user_name": user_name,
        }
        for job, external_id, agent_id, user_name in db.execute(query).all()
    ]
    return templates.TemplateResponse(
        request,
        "jobs.html",
        base_context(request, session, rows=rows, filters={"job_status": job_status, "limit": limit}),
    )


@router.post("/jobs/{job_id}/requeue")
async def requeue_job(job_id: uuid.UUID, request: Request, db: Session = Depends(get_db)) -> Response:
    session = require_admin_context(request)
    if isinstance(session, RedirectResponse):
        return session
    form = await read_urlencoded_form(request)
    verify_csrf(session, form)
    job = db.get(EmbeddingJob, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在。")
    memory = db.get(Memory, job.memory_id)
    if memory is None or memory.deleted_at is not None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="未找到可用记忆。")
    job.status = "pending"
    job.last_error = None
    memory.embedding_status = "pending"
    memory.embedding_error = None
    db.add_all([job, memory])
    db.commit()
    generate_memory_embedding.delay(str(memory.id), str(job.id))
    return redirect_to("/admin/jobs", notice="任务已重新入队。")


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

    embedding_ready = bool(
        settings.embedding_base_url
        and settings.embedding_api_key
        and settings.embedding_api_key != "replace-me"
    )
    checks.append(
        {
            "name": "向量服务",
            "status": "ok" if embedding_ready else "warning",
            "detail": settings.embedding_model if embedding_ready else "接口密钥未配置",
        }
    )
    checks.append(
        {
            "name": "后台会话密钥",
            "status": "ok" if not settings.admin_session_secret.startswith("change-me") else "warning",
            "detail": "已配置" if not settings.admin_session_secret.startswith("change-me") else "正在使用默认值",
        }
    )
    return templates.TemplateResponse(request, "health.html", base_context(request, session, checks=checks))
