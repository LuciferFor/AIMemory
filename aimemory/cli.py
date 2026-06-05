from datetime import UTC, datetime

import typer
from sqlalchemy import select

from aimemory.core.config import get_settings
from aimemory.core.security import api_key_prefix, generate_api_key, hash_api_key
from aimemory.db.session import SessionLocal
from aimemory.models.api_key import ApiKey
from aimemory.models.memory import Memory
from aimemory.models.memory_attachment import MemoryAttachment
from aimemory.models.memory_category import MemoryCategory
from aimemory.models.user import User
from aimemory.repositories.search_stopwords import add_default_search_stopwords
from aimemory.services.memory_policy import technical_or_operational_memory_skip_reason

app = typer.Typer(help="AIMemory administration commands.")


@app.command("create-user")
def create_user(name: str) -> None:
    with SessionLocal() as db:
        existing = db.scalar(select(User).where(User.name == name))
        if existing:
            typer.echo(f"User already exists: {existing.id}")
            return

        user = User(name=name)
        db.add(user)
        db.flush()
        add_default_search_stopwords(db, user)
        db.commit()
        typer.echo(f"Created user {name}: {user.id}")


@app.command("create-api-key")
def create_api_key(name: str, label: str | None = typer.Option(None, "--label", "-l")) -> None:
    settings = get_settings()
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.name == name))
        if user is None:
            raise typer.BadParameter(f"User does not exist: {name}")

        raw_key = generate_api_key(settings.api_key_prefix)
        api_key = ApiKey(
            user_id=user.id,
            key_hash=hash_api_key(raw_key),
            key_prefix=api_key_prefix(raw_key),
            label=label,
        )
        db.add(api_key)
        db.commit()

        typer.echo("API key created. Store it now; it will not be shown again.")
        typer.echo(raw_key)


@app.command("revoke-api-key")
def revoke_api_key(prefix: str) -> None:
    with SessionLocal() as db:
        api_key = db.scalar(select(ApiKey).where(ApiKey.key_prefix == prefix))
        if api_key is None:
            raise typer.BadParameter(f"API key prefix not found: {prefix}")
        api_key.revoked_at = datetime.now(UTC)
        db.add(api_key)
        db.commit()
        typer.echo(f"Revoked API key prefix: {prefix}")


@app.command("requeue-pending")
def requeue_pending(limit: int = typer.Option(100, "--limit", min=1, max=1000)) -> None:
    _ = limit
    typer.echo("Embedding workflow is disabled. Memories are searchable immediately with text indexes.")


@app.command("cleanup-technical-memories")
def cleanup_technical_memories(
    user: str | None = typer.Option(None, "--user", "-u", help="Only scan one username."),
    agent_id: str | None = typer.Option(None, "--agent-id", help="Only scan one agent_id."),
    apply_changes: bool = typer.Option(False, "--apply", help="Soft-delete matched memories."),
    sample_limit: int = typer.Option(20, "--sample-limit", min=1, max=100),
) -> None:
    """Soft-delete existing config/fix/troubleshooting memories. Dry-run by default."""

    with SessionLocal() as db:
        stmt = (
            select(Memory, MemoryCategory.name, User.name)
            .join(User, Memory.user_id == User.id)
            .join(MemoryCategory, Memory.category_id == MemoryCategory.id)
            .where(Memory.deleted_at.is_(None))
            .order_by(Memory.updated_at.desc())
        )
        if user:
            stmt = stmt.where(User.name == user)
        if agent_id:
            stmt = stmt.where(Memory.agent_id == agent_id)

        matches: list[tuple[Memory, str, str, str]] = []
        for memory, category_name, user_name in db.execute(stmt).all():
            reason = technical_or_operational_memory_skip_reason(
                title=memory.title,
                content=memory.content,
                category=category_name,
                metadata=memory.metadata_json,
            )
            if reason:
                matches.append((memory, category_name, user_name, reason))

        typer.echo(f"Matched technical/config/fix memories: {len(matches)}")
        if not apply_changes:
            typer.echo("Dry-run only. Re-run with --apply to soft-delete matched memories.")

        for memory, category_name, user_name, reason in matches[:sample_limit]:
            typer.echo(f"- {user_name} / {memory.agent_id} / {category_name} / {memory.title} / {reason}")

        if not apply_changes:
            return

        now = datetime.now(UTC)
        for memory, _category_name, _user_name, _reason in matches:
            memory.deleted_at = now
            memory.updated_at = now
            db.add(memory)
            attachments = db.scalars(
                select(MemoryAttachment).where(
                    MemoryAttachment.memory_id == memory.id,
                    MemoryAttachment.deleted_at.is_(None),
                )
            ).all()
            for attachment in attachments:
                attachment.deleted_at = now
                db.add(attachment)
        db.commit()
        typer.echo(f"Soft-deleted memories: {len(matches)}")


if __name__ == "__main__":
    app()
