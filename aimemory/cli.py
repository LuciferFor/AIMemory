from datetime import UTC, datetime

import typer
from sqlalchemy import select

from aimemory.core.config import get_settings
from aimemory.core.security import api_key_prefix, generate_api_key, hash_api_key
from aimemory.db.session import SessionLocal
from aimemory.models.api_key import ApiKey
from aimemory.models.embedding_job import EmbeddingJob
from aimemory.models.user import User
from aimemory.worker.tasks import generate_memory_embedding

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
    with SessionLocal() as db:
        jobs = (
            db.scalars(
                select(EmbeddingJob)
                .where(EmbeddingJob.status.in_(["pending", "retrying"]))
                .order_by(EmbeddingJob.created_at)
                .limit(limit)
            )
            .all()
        )
        for job in jobs:
            generate_memory_embedding.delay(str(job.memory_id), str(job.id))
        typer.echo(f"Requeued {len(jobs)} embedding jobs.")


if __name__ == "__main__":
    app()
