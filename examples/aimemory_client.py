"""
AIMemory client for model-side memory access.

Usage:
    Set AIMEMORY_API_KEY in the environment, or replace DEFAULT_API_KEY below.
    Import this file from your AI runtime and call search_memories/write_memory.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("AIMEMORY_BASE_URL", "http://192.168.31.11:10011")
DEFAULT_API_KEY = "aim_jmiJcOXdxPC7i-_JtDRp4YysqlZRz8r08rrz83jkQ5M"
API_KEY = os.getenv("AIMEMORY_API_KEY", DEFAULT_API_KEY)

# This identifies this AI/agent's isolated memory space.
AGENT_ID = os.getenv("AIMEMORY_AGENT_ID", "5df9cbfb-d31b-46dd-972b-05d466d2257c")


class AIMemoryError(RuntimeError):
    pass


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        f"{BASE_URL}{path}",
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise AIMemoryError(f"AIMemory HTTP {exc.code}: {detail}") from exc


def search_memories(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search relevant memories before answering the user."""
    response = _request(
        "POST",
        "/v1/memories/search",
        {
            "agent_id": AGENT_ID,
            "query": query,
            "top_k": top_k,
        },
    )
    return response.get("items", [])


def build_memory_context(query: str, top_k: int = 5) -> str:
    """Return a compact context block that can be inserted into a model prompt."""
    memories = search_memories(query, top_k=top_k)
    if not memories:
        return ""

    lines = ["可参考的长期记忆："]
    for index, memory in enumerate(memories, start=1):
        title = memory.get("title", "")
        content = memory.get("content", "")
        score = memory.get("score", 0)
        lines.append(f"{index}. {title}（相关度 {score:.3f}）\n{content}")
    return "\n\n".join(lines)


def write_memory(
    title: str,
    content: str,
    external_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or update one memory after the model decides it is worth saving."""
    if external_id is None:
        external_id = f"mem-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    return _request(
        "POST",
        "/v1/memories",
        {
            "agent_id": AGENT_ID,
            "external_id": external_id,
            "title": title,
            "content": content,
            "metadata": metadata or {},
        },
    )


def delete_memory(external_id: str) -> bool:
    """Soft-delete one memory by external_id."""
    response = _request(
        "DELETE",
        "/v1/memories",
        {
            "agent_id": AGENT_ID,
            "external_id": external_id,
        },
    )
    return bool(response.get("deleted"))


if __name__ == "__main__":
    # Tiny smoke example. Remove this block if your AI runtime imports the module.
    print(build_memory_context("用户有什么偏好", top_k=3))
