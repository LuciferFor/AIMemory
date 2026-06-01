#!/usr/bin/env python3
"""Install the AIMemory OpenClaw plugin into ~/.openclaw.

All file IO is UTF-8. The API key stays in ~/.openclaw/aimemory.env; this
script only writes non-secret plugin config.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys


DEFAULT_AGENT_ID = "5df9cbfb-d31b-46dd-972b-05d466d2257c"
DEFAULT_BASE_URL = "http://192.168.31.11:10011"


def copy_plugin(source_dir: Path, install_dir: Path) -> None:
    if install_dir.exists():
        shutil.rmtree(install_dir)
    ignore = shutil.ignore_patterns("node_modules", ".git", "__pycache__", "*.pyc")
    shutil.copytree(source_dir, install_dir, ignore=ignore)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.bak-aimemory-{stamp}")
    shutil.copy2(path, backup)
    return backup


def update_openclaw_config(config_path: Path, args: argparse.Namespace) -> Path | None:
    config = load_json(config_path)
    backup = backup_file(config_path)
    plugins = config.setdefault("plugins", {})
    entries = plugins.setdefault("entries", {})
    entry = entries.setdefault("aimemory", {})
    entry["enabled"] = True
    hooks = entry.setdefault("hooks", {})
    hooks["allowConversationAccess"] = True
    hooks["allowPromptInjection"] = True
    plugin_config = entry.setdefault("config", {})
    plugin_config.update(
        {
            "enabled": True,
            "baseUrl": args.base_url,
            "agentId": args.agent_id,
            "envFile": str(args.env_file),
            "allowedChatTypes": ["direct", "private", "dm", "webchat", "dashboard", "local", "embedded"],
            "topK": args.top_k,
            "maxChars": args.max_chars,
            "timeoutMs": args.timeout_ms,
            "fallbackCategory": "未分类",
            "preloadContextOnMessageReceived": True,
            "saveOnExplicitRemember": True,
            "saveBeforeCompaction": True,
            "useBackendExtraction": True,
            "watchCodexCompaction": True,
            "compactionWatcherIntervalMs": 5000,
            "includePromptInMemoryQuery": False,
            "includeUnstructuredTranscriptForCompaction": False,
            "logging": True,
        }
    )

    allow = plugins.get("allow")
    if isinstance(allow, list) and "aimemory" not in allow:
        allow.append("aimemory")

    write_json(config_path, config)
    return backup


def run_openclaw_install(openclaw_bin: str, install_dir: Path) -> None:
    subprocess.run(
        [openclaw_bin, "plugins", "install", "--link", str(install_dir)],
        check=True,
    )
    subprocess.run([openclaw_bin, "plugins", "enable", "aimemory"], check=False)
    subprocess.run([openclaw_bin, "plugins", "registry", "--refresh"], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install AIMemory OpenClaw plugin.")
    parser.add_argument("--openclaw-home", type=Path, default=Path.home() / ".openclaw")
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--install-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--agent-id", default=DEFAULT_AGENT_ID)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--openclaw-bin", default="openclaw")
    parser.add_argument("--skip-openclaw-install", action="store_true")
    args = parser.parse_args()

    openclaw_home = args.openclaw_home.expanduser().resolve()
    args.install_dir = (args.install_dir or openclaw_home / "plugins" / "aimemory").expanduser().resolve()
    args.config = (args.config or openclaw_home / "openclaw.json").expanduser().resolve()
    args.env_file = (args.env_file or openclaw_home / "aimemory.env").expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()

    if not (source_dir / "openclaw.plugin.json").exists():
        print(f"source-dir is not an AIMemory OpenClaw plugin: {source_dir}", file=sys.stderr)
        return 2

    copy_plugin(source_dir, args.install_dir)
    backup = update_openclaw_config(args.config, args)
    if not args.env_file.exists():
        print(f"warning: env file not found: {args.env_file}", file=sys.stderr)
        print("create it with AIMEMORY_BASE_URL, AIMEMORY_API_KEY, AIMEMORY_AGENT_ID", file=sys.stderr)

    if not args.skip_openclaw_install:
        run_openclaw_install(args.openclaw_bin, args.install_dir)

    print(f"installed: {args.install_dir}")
    print(f"config: {args.config}")
    if backup:
        print(f"backup: {backup}")
    print("restart OpenClaw gateway for the plugin to take effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
