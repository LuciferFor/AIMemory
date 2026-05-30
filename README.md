# AIMemory

AIMemory is a FastAPI service for long-term AI memory. It stores memories by
`user_id + agent_id`, supports idempotent writes through `external_id`, and
searches with PostgreSQL full-text, trigram fuzzy matching, and local query
tokenization. It does not require an external AI or embedding provider.

## Stack

- Python 3.12+ with FastAPI
- PostgreSQL 16 with pg_trgm text/fuzzy indexes
- Redis for runtime health checks and future async work
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

Build prompt-ready memory context:

```bash
curl -X POST http://localhost:10011/v1/memories/context \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"assistant","query":"short replies preference","top_k":8,"max_chars":3000}'
```

The response includes `context_text`, which can be inserted into the model's
system/developer context before the current user message. AIMemory does not call
the main language model; clients still control the model request.

Get the standard write policy for extracting memories before context compression:

```bash
curl -X GET http://localhost:10011/v1/memories/write-policy \
  -H "Authorization: Bearer <api_key>"
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
- `REDIS_URL`: Redis URL for health checks and future async work.
- `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `ADMIN_SESSION_SECRET`: admin web login.
- `LOG_LEVEL`: logging level, default `INFO`.
- `LOG_FORMAT`: `json` for Docker/production logs, or `text` for local debugging.
- `SLOW_REQUEST_MS`: request duration threshold logged as warning.

## Logging

API logs are written to stdout. In Docker, use:

```bash
docker compose logs -f api
```

Each API response includes `X-Request-ID`; pass the same header from a client to
trace a request through logs. Normal `/healthz` and `/admin/static/*` requests are
not logged at INFO level.

Logs intentionally do not include API keys, admin passwords, full memory content,
or metadata bodies. Business logs include IDs, status, counts, and timings.

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
```

Useful admin commands:

```bash
aimemory create-user <name>
aimemory create-api-key <name> --label <label>
aimemory revoke-api-key <key-prefix>
```
