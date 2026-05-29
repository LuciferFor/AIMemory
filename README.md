# AIMemory

AIMemory is a FastAPI service for long-term AI memory. It stores memories by
`user_id + agent_id`, supports idempotent writes through `external_id`, creates
embeddings asynchronously with a Celery worker, and searches with hybrid vector,
full-text, and fuzzy matching.

## Stack

- Python 3.12+ with FastAPI
- PostgreSQL 16 with pgvector and pg_trgm
- Redis and Celery for embedding jobs
- External OpenAI-compatible embeddings API
- Alembic migrations

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

Create a user and API key:

```bash
docker compose exec api aimemory create-user default
docker compose exec api aimemory create-api-key default --label local
```

Insert memory:

```bash
curl -X POST http://localhost:10011/v1/memories \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "assistant",
    "external_id": "mem-001",
    "title": "User likes concise answers",
    "content": "The user prefers direct answers with short implementation notes."
  }'
```

Search memory:

```bash
curl -X POST http://localhost:10011/v1/memories/search \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"assistant","query":"short replies preference","top_k":5}'
```

Delete memory:

```bash
curl -X DELETE http://localhost:10011/v1/memories \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"assistant","external_id":"mem-001"}'
```

## Admin UI

The admin web UI runs on the same service port:

```text
http://localhost:10011/admin/login
```

Set these values before exposing it:

```bash
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<strong-password>
ADMIN_SESSION_SECRET=<random-secret>
ADMIN_COOKIE_SECURE=false
```

Use `ADMIN_COOKIE_SECURE=true` after putting the service behind HTTPS.

## Configuration

All files are UTF-8. Database initialization in Docker Compose uses UTF8.

Important environment variables:

- `DATABASE_URL`: SQLAlchemy PostgreSQL URL.
- `REDIS_URL`: Celery broker/result backend URL.
- `EMBEDDING_BASE_URL`: OpenAI-compatible API base, for example `https://api.openai.com/v1`.
- `EMBEDDING_API_KEY`: embedding provider API key.
- `EMBEDDING_MODEL`: embedding model name.
- `EMBEDDING_DIM`: one fixed vector dimension for this deployment.
- `EMBEDDING_INCLUDE_DIMENSIONS`: send `dimensions` in embedding requests when supported.
- `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `ADMIN_SESSION_SECRET`: admin web login.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
pytest
```

For local PostgreSQL:

```bash
alembic upgrade head
uvicorn aimemory.main:create_app --factory --reload --port 10011
celery -A aimemory.worker.celery_app:celery_app worker --loglevel=INFO
```

Useful admin commands:

```bash
aimemory create-user <name>
aimemory create-api-key <name> --label <label>
aimemory revoke-api-key <key-prefix>
aimemory requeue-pending --limit 100
```
