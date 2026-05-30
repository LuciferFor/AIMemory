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
- Injects returned `context_text` into the current model turn.
- Calls `GET /v1/memories/write-policy` and `POST /v1/memories` when saving;
  extracted memories must include `category`.
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

## Configuration

The installer writes:

```json
{
  "plugins": {
    "entries": {
      "aimemory": {
        "enabled": true,
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
          "logging": true
        }
      }
    }
  }
}
```

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

## Test

```bash
npm test
```
