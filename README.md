# AIMemory

AIMemory is a FastAPI service for long-term AI memory. It stores memories by
`user_id + agent_id + category`, supports idempotent writes through
`external_id`, and searches inside one memory category with PostgreSQL
full-text, trigram fuzzy matching, and local query tokenization. It does not
require an external AI or embedding provider.

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
docker compose exec api aimemory create-user lucifer
docker compose exec api aimemory create-api-key lucifer --label local
```

Insert memory:

```bash
curl -X POST http://localhost:10011/v1/memories \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "assistant",
    "external_id": "mem-001",
    "category": "回答偏好",
    "title": "User likes concise answers",
    "content": "The user prefers direct answers with short implementation notes."
  }'
```

Search memory:

```bash
curl -X POST http://localhost:10011/v1/memories/search \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"assistant","category":"回答偏好","query":"short replies preference","top_k":5}'
```

List existing categories for the API user:

```bash
curl -X GET http://localhost:10011/v1/memories/categories \
  -H "Authorization: Bearer <api_key>"
```

Insert memory with an image attachment. AIMemory accepts base64 in the request,
decodes it, and stores binary image bytes in PostgreSQL; image retrieval is based
on the text description, OCR text, and tags you submit:

```bash
curl -X POST http://localhost:10011/v1/memories \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "assistant",
    "external_id": "image-memory-001",
    "category": "视觉参考",
    "title": "Reference image",
    "content": "A reusable visual reference.",
    "attachments": [{
      "filename": "reference.png",
      "mime_type": "image/png",
      "data_base64": "<base64>",
      "description": "A dark interface screenshot with an error message.",
      "ocr_text": "Connection failed",
      "metadata": {"tags": ["screenshot", "error"]}
    }]
  }'
```

Search responses include attachment metadata and a download URL, not base64.

Build prompt-ready memory context:

```bash
curl -X POST http://localhost:10011/v1/memories/context \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"assistant","category":"回答偏好","query":"short replies preference","top_k":8,"max_chars":3000}'
```

The response includes `context_text`, which can be inserted into the model's
system/developer context before the current user message. Clients should choose
the category before calling this endpoint; AIMemory only searches within that
category. AIMemory does not call the main language model; clients still control
the model request.

Get the standard write policy for extracting memories before context compression:

```bash
curl -X GET http://localhost:10011/v1/memories/write-policy \
  -H "Authorization: Bearer <api_key>"
```

The write-policy response includes the current category list. Memory extraction
clients should choose an existing category first, and create a short new category
only when none fits.

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
AI_CONFIG_ENCRYPTION_SECRET=<random-secret-for-ai-config>
ADMIN_COOKIE_SECURE=false
```

Use `ADMIN_COOKIE_SECURE=true` after putting the service behind HTTPS.

The admin UI also includes optional AI memory review. Configure it from
`/admin/ai-settings` with any OpenAI-compatible `/chat/completions` provider.
The default values target DeepSeek: `https://api.deepseek.com` and
`deepseek-v4-flash`. Review actions send the selected full memory content to
that provider and only write changes after an admin applies each suggestion.

## Configuration

All files are UTF-8. Database initialization in Docker Compose uses UTF8.

Important environment variables:

- `DATABASE_URL`: SQLAlchemy PostgreSQL URL.
- `REDIS_URL`: Redis URL for health checks and future async work.
- `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `ADMIN_SESSION_SECRET`: admin web login.
- `AI_CONFIG_ENCRYPTION_SECRET`: encrypts the AI provider API key stored from the admin UI.
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

## OpenClaw Plugin

The OpenClaw automatic memory plugin lives in:

```text
clients/openclaw-aimemory
```

It is an OpenClaw native plugin that fetches the category list, asks the current
OpenClaw model to choose one category, calls `/v1/memories/context`, and injects
the returned memory context. By default it runs for private/direct/DM sessions
across all channels, while group/channel memory injection stays disabled for
privacy.

Install it on an OpenClaw machine with:

```bash
cd clients/openclaw-aimemory
python3 install.py
```

Keep the AIMemory API key in `~/.openclaw/aimemory.env`; the installer does not
write secrets into OpenClaw config.
