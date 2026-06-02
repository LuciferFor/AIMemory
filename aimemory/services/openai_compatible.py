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


def token_usage_summary(usage: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}

    prompt_tokens = usage_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion_tokens = usage_int(usage.get("completion_tokens") or usage.get("output_tokens"))
    total_tokens = usage_int(usage.get("total_tokens"))
    if not total_tokens and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens

    details = usage.get("prompt_tokens_details")
    cached_tokens = usage_int(details.get("cached_tokens")) if isinstance(details, dict) else 0

    summary: dict[str, int] = {}
    if prompt_tokens:
        summary["prompt_tokens"] = prompt_tokens
    if completion_tokens:
        summary["completion_tokens"] = completion_tokens
    if total_tokens:
        summary["total_tokens"] = total_tokens
    if cached_tokens:
        summary["cached_tokens"] = cached_tokens
    return summary


def usage_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def chat_completion(
    config: LlmProviderConfig,
    api_key: str,
    messages: list[dict[str, str]],
    *,
    response_format: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout_ms: int | None = None,
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
    selected_temperature = config.temperature if temperature is None else temperature
    selected_max_tokens = config.max_output_tokens if max_tokens is None else max_tokens
    payload.update(
        {
            "model": config.model,
            "messages": messages,
            "temperature": float(selected_temperature or 0.0),
            "max_tokens": int(selected_max_tokens or 4096),
            "stream": False,
        }
    )
    if response_format is not None:
        payload["response_format"] = response_format
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
        selected_timeout_ms = config.timeout_ms if timeout_ms is None else timeout_ms
        request_timeout_ms = int(selected_timeout_ms or 30000)
        with urllib.request.urlopen(request, timeout=max(request_timeout_ms / 1000, 0.5)) as response:
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
