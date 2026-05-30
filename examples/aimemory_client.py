"""
AIMemory client for model-side memory access.

Usage:
    Set AIMEMORY_API_KEY in the environment before use.
    Import this file from your AI runtime and call search_memories/write_memory.
"""

from __future__ import annotations

import json
import base64
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("AIMEMORY_BASE_URL", "http://192.168.31.11:10011")
DEFAULT_API_KEY = ""
API_KEY = os.getenv("AIMEMORY_API_KEY", DEFAULT_API_KEY)

# This identifies this AI/agent's isolated memory space.
AGENT_ID = os.getenv("AIMEMORY_AGENT_ID", "5df9cbfb-d31b-46dd-972b-05d466d2257c")


class AIMemoryError(RuntimeError):
    pass


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if not API_KEY:
        raise AIMemoryError("AIMEMORY_API_KEY is required")

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


def search_memories(query: str, category: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search relevant memories in one category before answering the user."""
    response = _request(
        "POST",
        "/v1/memories/search",
        {
            "agent_id": AGENT_ID,
            "category": category,
            "query": query,
            "top_k": top_k,
        },
    )
    return response.get("items", [])


def build_server_memory_context(
    query: str,
    category: str,
    top_k: int = 8,
    max_chars: int = 3000,
) -> str:
    """Ask AIMemory to return the standard prompt-ready memory context."""
    response = _request(
        "POST",
        "/v1/memories/context",
        {
            "agent_id": AGENT_ID,
            "category": category,
            "query": query,
            "top_k": top_k,
            "max_chars": max_chars,
        },
    )
    return response.get("context_text", "")


def build_memory_context(query: str, category: str, top_k: int = 5) -> str:
    """Return a compact context block that can be inserted into a model prompt."""
    memories = search_memories(query, category=category, top_k=top_k)
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
    category: str,
    external_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    attachments: list[dict[str, Any]] | None = None,
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
            "category": category,
            "title": title,
            "content": content,
            "metadata": metadata or {},
            **({"attachments": attachments} if attachments is not None else {}),
        },
    )


def image_attachment_from_file(
    path: str | Path,
    description: str | None = None,
    ocr_text: str | None = None,
    metadata: dict[str, Any] | None = None,
    mime_type: str | None = None,
) -> dict[str, Any]:
    """Build one AIMemory image attachment from a local image file."""
    image_path = Path(path)
    detected_mime = mime_type or mimetypes.guess_type(image_path.name)[0]
    if detected_mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise AIMemoryError(f"Unsupported image MIME type: {detected_mime}")
    return {
        "filename": image_path.name,
        "mime_type": detected_mime,
        "data_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        "description": description,
        "ocr_text": ocr_text,
        "metadata": metadata or {},
    }


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


def get_write_policy() -> dict[str, Any]:
    """Return the standard prompt for extracting memories before context compression."""
    return _request("GET", "/v1/memories/write-policy")


def list_categories() -> list[dict[str, Any]]:
    """Return existing memory categories for the current API user."""
    response = _request("GET", "/v1/memories/categories")
    return response.get("items", [])


if __name__ == "__main__":
    # Tiny smoke example. Remove this block if your AI runtime imports the module.
    print(build_server_memory_context("用户有什么偏好", category="偏好", top_k=3))
