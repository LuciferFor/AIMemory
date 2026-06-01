#!/usr/bin/env python3
"""Patch OpenClaw's bundled Codex app-server compaction bridge.

Some OpenClaw builds run manual Codex app-server compaction through the native
`thread/compact/start` path without projecting the compaction item through the
agent event projector. In that path the normal plugin `before_compaction` /
`after_compaction` hooks are not called, so AIMemory never sees the transcript.

This patch wires the bundled Codex native compaction function directly into the
AIMemory extraction bridge. It is intentionally small and idempotent so it can
be re-applied after an OpenClaw container/image update.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil


PATCH_MARKER = "aimemory-codex-appserver-direct-extract-bridge"
LEGACY_PATCH_MARKER = "aimemory-codex-appserver-compaction-bridge"


def find_compact_bundle(app_root: Path) -> Path:
    dist_dir = app_root / "dist"
    candidates = sorted(dist_dir.glob("compact-*.js"))
    for candidate in candidates:
        text = candidate.read_text(encoding="utf-8")
        if "thread/compact/start" in text and "started codex app-server compaction" in text:
            return candidate
    raise FileNotFoundError(f"OpenClaw Codex compact bundle not found under {dist_dir}")


def patch_bundle(path: Path) -> tuple[bool, Path | None]:
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return False, None
    if LEGACY_PATCH_MARKER in text:
        backups = sorted(path.parent.glob(f"{path.name}.bak-{LEGACY_PATCH_MARKER}-*"))
        if not backups:
            raise RuntimeError("legacy AIMemory bridge patch found but no backup bundle is available")
        shutil.copy2(backups[-1], path)
        text = path.read_text(encoding="utf-8")

    helper_anchor = "const warnedIgnoredCompactionOverrides = /* @__PURE__ */ new Set();"
    helper = f"""{helper_anchor}
const AIMEMORY_CODEX_COMPACTION_BRIDGE = "{PATCH_MARKER}";
async function runAIMemoryCodexCompactionBridge(params) {{
\ttry {{
\t\tconst bridge = await import("file:///home/node/.openclaw/plugins/aimemory/lib/codex_appserver_bridge.mjs");
\t\tif (typeof bridge.runCodexAppServerCompactionBridge === "function") {{
\t\t\tawait bridge.runCodexAppServerCompactionBridge(params);
\t\t}}
\t}} catch (error) {{
\t\tlog.warn("aimemory codex app-server direct extract bridge failed", {{
\t\t\tsessionId: params.sessionId,
\t\t\tsessionKey: params.sessionKey,
\t\t\treason: formatCompactionError(error)
\t\t}});
\t}}
}}"""
    if helper_anchor not in text:
        raise RuntimeError("compaction helper anchor not found")
    text = text.replace(helper_anchor, helper, 1)

    before_anchor = '\ttry {\n\t\tawait client.request("thread/compact/start", { threadId: binding.threadId });'
    before_replacement = "\tawait runAIMemoryCodexCompactionBridge(params);\n" + before_anchor
    if before_anchor not in text:
        raise RuntimeError("compaction start anchor not found")
    text = text.replace(before_anchor, before_replacement, 1)

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

    helper_anchor = "const warnedIgnoredCompactionOverrides = new Set<string>();"
    helper = f"""{helper_anchor}
const AIMEMORY_CODEX_COMPACTION_BRIDGE = "{PATCH_MARKER}";

async function runAIMemoryCodexCompactionBridge(
  params: CompactEmbeddedPiSessionParams,
): Promise<void> {{
  try {{
    const bridge = await import("file:///home/node/.openclaw/plugins/aimemory/lib/codex_appserver_bridge.mjs");
    const runner = (bridge as {{
      runCodexAppServerCompactionBridge?: (params: CompactEmbeddedPiSessionParams) => Promise<unknown>;
    }}).runCodexAppServerCompactionBridge;
    if (typeof runner === "function") {{
      await runner(params);
    }}
  }} catch (error) {{
    embeddedAgentLog.warn("aimemory codex app-server direct extract bridge failed", {{
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      reason: formatCompactionError(error),
    }});
  }}
}}"""
    if helper_anchor not in text:
        raise RuntimeError("source compaction helper anchor not found")
    text = text.replace(helper_anchor, helper, 1)

    before_anchor = """  try {
    await client.request("thread/compact/start", {
      threadId: binding.threadId,
    });"""
    before_replacement = "  await runAIMemoryCodexCompactionBridge(params);\n" + before_anchor
    if before_anchor not in text:
        raise RuntimeError("source compaction start anchor not found")
    text = text.replace(before_anchor, before_replacement, 1)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{PATCH_MARKER}-{stamp}")
    shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")
    return True, backup


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch OpenClaw Codex app-server compaction to fire AIMemory hooks.")
    parser.add_argument("--app-root", type=Path, default=Path("/app"))
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    bundle = args.bundle or find_compact_bundle(args.app_root)
    bundle_changed, bundle_backup = patch_bundle(bundle)
    source = args.app_root / "extensions" / "codex" / "src" / "app-server" / "compact.ts"
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
