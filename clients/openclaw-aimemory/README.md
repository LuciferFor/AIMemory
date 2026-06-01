# AIMemory OpenClaw Plugin

This is the OpenClaw client plugin for AIMemory. It injects long-term memory
context before model calls and writes durable memories on explicit remember
requests or before context compaction.

## Behavior

- Runs on all OpenClaw channels, but only for private/direct/DM sessions by
  default.
- Calls `GET /v1/memories/categories`, asks the current OpenClaw model to pick
  one existing category, then calls `POST /v1/memories/context` in
  `before_prompt_build`.
- Builds the AIMemory query only from the current user input. Model replies,
  previous messages, static prompt files such as `IDENTITY.md`, `SOUL.md`, and
  `MEMORY.md` are not sent to AIMemory for retrieval.
- Injects returned `context_text` into the current model turn.
- Calls `GET /v1/memories/write-policy` and `POST /v1/memories` when saving;
  extracted memories must include `category`. Before compaction, only
  structured user/assistant messages are extracted by default, and the
  extraction prompt asks the model to save memories in a third-person style.
- Does not print API keys or full memory content in logs.
- Keeps group/channel memory injection disabled unless explicitly configured.

## Install

Make sure `~/.openclaw/aimemory.env` exists:

```bash
AIMEMORY_BASE_URL=http://192.168.31.11:10011
AIMEMORY_API_KEY=<api-key>
AIMEMORY_AGENT_ID=5df9cbfb-d31b-46dd-972b-05d466d2257c
```

Install into OpenClaw:

```bash
cd /path/to/AIMemory/clients/openclaw-aimemory
python3 install.py
```

The installer:

- copies this plugin to `~/.openclaw/plugins/aimemory`
- updates `~/.openclaw/openclaw.json`
- links the plugin with `openclaw plugins install --force --link`
- keeps the API key only in `~/.openclaw/aimemory.env`

Restart OpenClaw gateway after installation.

For OpenClaw builds where manual Codex app-server compaction does not fire
plugin compaction hooks, patch the bundled Codex compaction bridge inside the
gateway container:

```bash
python3 /home/node/.openclaw/plugins/aimemory/patch_codex_appserver_compaction.py --app-root /app
```

Then restart the gateway container.

## Configuration

The installer writes:

```json
{
  "plugins": {
    "entries": {
      "aimemory": {
        "enabled": true,
        "hooks": {
          "allowConversationAccess": true
        },
        "config": {
          "enabled": true,
          "baseUrl": "http://192.168.31.11:10011",
          "agentId": "5df9cbfb-d31b-46dd-972b-05d466d2257c",
          "envFile": "~/.openclaw/aimemory.env",
          "allowedChatTypes": ["direct", "private", "dm"],
          "topK": 8,
          "maxChars": 3000,
          "timeoutMs": 3000,
          "saveOnExplicitRemember": true,
          "saveBeforeCompaction": true,
          "useBackendExtraction": true,
          "watchCodexCompaction": true,
          "compactionWatcherIntervalMs": 5000,
          "includePromptInMemoryQuery": false,
          "includeUnstructuredTranscriptForCompaction": false,
          "logging": true
        }
      }
    }
  }
}
```

`includePromptInMemoryQuery` is kept for old configs but retrieval no longer
uses prompt/history text; AIMemory queries are based on the current user input
only.

## Verify

```bash
openclaw plugins inspect aimemory --runtime --json
openclaw plugins doctor
```

Then send a private message to OpenClaw. AIMemory API logs should show:

```text
GET /v1/memories/categories
POST /v1/memories/context
```

If the model cannot choose a clear category, the plugin skips memory context for
that turn instead of doing a cross-category search.

Check the AIMemory request log `query_preview` after a private message. It
should show the current user request and recent ordinary dialogue only, not
static identity/soul/memory prompt text.

## Test

```bash
npm test
```
