import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from aimemory.models.llm_provider_config import LlmProviderConfig


class OpenAICompatibleError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    usage: dict[str, Any]


def normalize_base_url(base_url: str) -> str:
    return str(base_url or "").strip().rstrip("/")


def chat_completion(
    config: LlmProviderConfig,
    api_key: str,
    messages: list[dict[str, str]],
) -> ChatCompletionResult:
    key = str(api_key or "").strip()
    if not key:
        raise OpenAICompatibleError("AI API Key 未配置。")

    base_url = normalize_base_url(config.base_url)
    if not base_url:
        raise OpenAICompatibleError("AI Base URL 未配置。")

    payload: dict[str, Any] = {}
    if isinstance(config.extra_body_json, dict):
        payload.update(config.extra_body_json)
    payload.update(
        {
            "model": config.model,
            "messages": messages,
            "temperature": float(config.temperature or 0.0),
            "max_tokens": int(config.max_output_tokens or 4096),
            "stream": False,
            "response_format": {"type": "json_object"},
        }
    )
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=max(int(config.timeout_ms or 30000) / 1000, 0.5)) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise OpenAICompatibleError(f"AI 请求失败：HTTP {exc.code} {safe_error_body(body)}") from exc
    except urllib.error.URLError as exc:
        raise OpenAICompatibleError(f"AI 请求失败：{exc.reason}") from exc
    except TimeoutError as exc:
        raise OpenAICompatibleError("AI 请求超时。") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OpenAICompatibleError("AI 返回不是合法 JSON。") from exc

    choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message") if isinstance(first_choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    if not str(content or "").strip():
        raise OpenAICompatibleError("AI 返回内容为空。")
    usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
    return ChatCompletionResult(content=str(content), usage=usage)


def safe_error_body(value: str) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(r"(?i)(authorization|api[_-]?key|token|password)(\s*[=:]\s*)\S+", r"\1\2***", text)
    text = re.sub(r"(?i)bearer\s+[a-z0-9._-]+", "Bearer ***", text)
    text = re.sub(r"(?i)sk-[a-z0-9._-]+", "sk-***", text)
    return text[:500]
