# AIMemory OpenClaw Plugin

This is the OpenClaw client plugin for AIMemory. It injects long-term memory
context before model calls and writes durable memories on explicit remember
requests or before context compaction.

## Behavior

- Runs on all OpenClaw channels, but only for private/direct/DM sessions by
  default.
- Calls `POST /v1/memories/context` with the current user input and lets the
  AIMemory server choose the category. The plugin preloads context in
  `message_received` and reuses it in `before_prompt_build`, so AIMemory request
  logs still show a `/context` request even when no memory is returned.
- Builds the AIMemory query only from the current user input. Model replies,
  previous messages, static prompt files such as `IDENTITY.md`, `SOUL.md`, and
  `MEMORY.md` are not sent to AIMemory for retrieval.
- Injects returned `context_text` into the current model turn.
- Calls `GET /v1/memories/write-policy` and `POST /v1/memories` when saving;
  extracted memories may include `category`, but AIMemory can classify them on
  the server when admin AI is configured. Before compaction, only structured
  user/assistant messages are extracted by default, and the extraction prompt
  asks the model to save memories in a third-person style.
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
          "allowConversationAccess": true,
          "allowPromptInjection": true
        },
        "config": {
          "enabled": true,
          "baseUrl": "http://192.168.31.11:10011",
          "agentId": "5df9cbfb-d31b-46dd-972b-05d466d2257c",
          "envFile": "~/.openclaw/aimemory.env",
          "allowedChatTypes": ["direct", "private", "dm", "webchat", "dashboard", "local", "embedded"],
          "topK": 8,
          "maxChars": 3000,
          "timeoutMs": 3000,
          "fallbackCategory": "其它",
          "preloadContextOnMessageReceived": true,
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

`fallbackCategory` is kept for old configs but no longer drives retrieval;
AIMemory server-side AI now selects the category.

## Verify

```bash
openclaw plugins inspect aimemory --runtime --json
openclaw plugins doctor
```

Then send a private message to OpenClaw. AIMemory API logs should show:

```text
POST /v1/memories/context
```

The request log should show the server-selected category source, selected
category, keyword analysis, and whether memory context was empty or injected.

Check the AIMemory request log `query_preview` after a private message. It
should show the current user request and recent ordinary dialogue only, not
static identity/soul/memory prompt text.

## Test

```bash
npm test
```
