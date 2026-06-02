#!/usr/bin/env python3
"""Patch OpenClaw's bundled Codex app-server turn path to request AIMemory.

OpenClaw's native Codex conversation binding sends user turns directly to the
Codex app-server and does not pass through the normal prompt hooks. This bridge
adds a small pre-turn call that asks AIMemory for context using only the current
user prompt, then prepends the returned context to the Codex turn input.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil


PATCH_MARKER = "aimemory-codex-appserver-context-bridge"


def find_conversation_bundle(app_root: Path) -> Path:
    dist_dir = app_root / "dist"
    candidates = sorted(dist_dir.glob("conversation-binding-*.js"))
    for candidate in candidates:
        text = candidate.read_text(encoding="utf-8")
        if "function runBoundTurn(params)" in text and '"turn/start"' in text:
            return candidate
    raise FileNotFoundError(f"OpenClaw Codex conversation bundle not found under {dist_dir}")


def patch_bundle(path: Path) -> tuple[bool, Path | None]:
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return False, None

    helper_anchor = "async function runBoundTurn(params) {"
    helper = f"""const AIMEMORY_CODEX_CONTEXT_BRIDGE = "{PATCH_MARKER}";
async function buildAIMemoryCodexContextPrompt(params) {{
\ttry {{
\t\tconst bridge = await import("file:///home/node/.openclaw/plugins/aimemory/lib/codex_appserver_context_bridge.mjs");
\t\tif (typeof bridge.buildCodexAppServerPromptWithMemory === "function") {{
\t\t\treturn await bridge.buildCodexAppServerPromptWithMemory(params);
\t\t}}
\t}} catch (error) {{
\t\tconsole.warn("[aimemory] codex app-server context bridge failed", {{
\t\t\tsessionFile: params?.sessionFile,
\t\t\terror: error instanceof Error ? error.message : String(error)
\t\t}});
\t}}
\treturn String(params?.prompt || "");
}}
{helper_anchor}"""
    if helper_anchor not in text:
        raise RuntimeError("runBoundTurn anchor not found")
    text = text.replace(helper_anchor, helper, 1)

    input_anchor = """input: buildCodexConversationTurnInput({
\t\t\t\tprompt: params.prompt,
\t\t\t\tevent: params.event
\t\t\t}),"""
    input_replacement = """input: buildCodexConversationTurnInput({
\t\t\t\tprompt: await buildAIMemoryCodexContextPrompt({
\t\t\t\t\tprompt: params.prompt,
\t\t\t\t\tevent: params.event,
\t\t\t\t\tsessionFile: params.data.sessionFile,
\t\t\t\t\tworkspaceDir: params.data.workspaceDir,
\t\t\t\t\tagentDir: params.data.agentDir,
\t\t\t\t\tthreadId
\t\t\t\t}),
\t\t\t\tevent: params.event
\t\t\t}),"""
    if input_anchor not in text:
        raise RuntimeError("turn input anchor not found")
    text = text.replace(input_anchor, input_replacement, 1)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{PATCH_MARKER}-{stamp}")
    shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")
    return True, backup


def patch_source(path: Path) -> tuple[bool, Path | None]:
    if not path.exists():
        return False, None
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return False, None

    helper_anchor = "async function runBoundTurn(params: {"
    helper = f"""const AIMEMORY_CODEX_CONTEXT_BRIDGE = "{PATCH_MARKER}";

async function buildAIMemoryCodexContextPrompt(params: {{
  prompt: string;
  event?: unknown;
  sessionFile?: string;
  workspaceDir?: string;
  agentDir?: string;
  threadId?: string;
}}): Promise<string> {{
  try {{
    const bridge = await import("file:///home/node/.openclaw/plugins/aimemory/lib/codex_appserver_context_bridge.mjs");
    const runner = (bridge as {{
      buildCodexAppServerPromptWithMemory?: (params: unknown) => Promise<string>;
    }}).buildCodexAppServerPromptWithMemory;
    if (typeof runner === "function") {{
      return await runner(params);
    }}
  }} catch (error) {{
    console.warn("[aimemory] codex app-server context bridge failed", {{
      sessionFile: params.sessionFile,
      error: error instanceof Error ? error.message : String(error),
    }});
  }}
  return params.prompt;
}}

{helper_anchor}"""
    if helper_anchor not in text:
        raise RuntimeError("source runBoundTurn anchor not found")
    text = text.replace(helper_anchor, helper, 1)

    input_anchor = """input: buildCodexConversationTurnInput({
          prompt: params.prompt,
          event: params.event,
        }),"""
    input_replacement = """input: buildCodexConversationTurnInput({
          prompt: await buildAIMemoryCodexContextPrompt({
            prompt: params.prompt,
            event: params.event,
            sessionFile: params.data.sessionFile,
            workspaceDir: params.data.workspaceDir,
            agentDir: params.data.agentDir,
            threadId,
          }),
          event: params.event,
        }),"""
    if input_anchor not in text:
        raise RuntimeError("source turn input anchor not found")
    text = text.replace(input_anchor, input_replacement, 1)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{PATCH_MARKER}-{stamp}")
    shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")
    return True, backup


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch OpenClaw Codex app-server turns to request AIMemory context.")
    parser.add_argument("--app-root", type=Path, default=Path("/app"))
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    bundle = args.bundle or find_conversation_bundle(args.app_root)
    bundle_changed, bundle_backup = patch_bundle(bundle)
    source = args.app_root / "extensions" / "codex" / "src" / "conversation-binding.ts"
    source_changed, source_backup = patch_source(source)
    print(f"bundle={bundle}")
    print(f"bundle_changed={str(bundle_changed).lower()}")
    if bundle_backup:
        print(f"bundle_backup={bundle_backup}")
    if source.exists():
        print(f"source={source}")
        print(f"source_changed={str(source_changed).lower()}")
        if source_backup:
            print(f"source_backup={source_backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
