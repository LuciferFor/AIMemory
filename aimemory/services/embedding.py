import logging
import time
from typing import Any

import httpx

from aimemory.core.config import Settings, get_settings
from aimemory.core.logging import sanitize_for_log, url_host

logger = logging.getLogger(__name__)


class EmbeddingProviderError(RuntimeError):
    pass


class EmbeddingProviderNotConfigured(EmbeddingProviderError):
    pass


class OpenAICompatibleEmbeddingClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def embed(self, text: str) -> list[float]:
        if not self.settings.embedding_base_url or not self.settings.embedding_api_key:
            raise EmbeddingProviderNotConfigured("Embedding provider is not configured.")

        start = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.settings.embedding_model,
            "input": text,
        }
        if self.settings.embedding_include_dimensions:
            payload["dimensions"] = self.settings.embedding_dim

        headers = {
            "Authorization": f"Bearer {self.settings.embedding_api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.settings.embedding_base_url.rstrip('/')}/embeddings"

        try:
            with httpx.Client(timeout=self.settings.embedding_timeout_seconds) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            status_code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            _log_embedding_event(
                "embedding.request_failed",
                self.settings,
                start,
                logging.WARNING,
                error_type=exc.__class__.__name__,
                status_code=status_code,
                error=sanitize_for_log(str(exc)),
            )
            raise EmbeddingProviderError(f"Embedding request failed: {sanitize_for_log(str(exc))}") from exc
        except ValueError as exc:
            _log_embedding_event(
                "embedding.invalid_json",
                self.settings,
                start,
                logging.WARNING,
                error_type=exc.__class__.__name__,
            )
            raise EmbeddingProviderError("Embedding provider returned invalid JSON.") from exc

        try:
            embedding = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            _log_embedding_event(
                "embedding.invalid_response",
                self.settings,
                start,
                logging.WARNING,
                error_type=exc.__class__.__name__,
            )
            raise EmbeddingProviderError("Embedding provider response did not include data[0].embedding.") from exc

        if not isinstance(embedding, list) or not embedding:
            _log_embedding_event("embedding.empty_vector", self.settings, start, logging.WARNING)
            raise EmbeddingProviderError("Embedding provider returned an empty embedding.")

        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            _log_embedding_event(
                "embedding.invalid_vector",
                self.settings,
                start,
                logging.WARNING,
                error_type=exc.__class__.__name__,
            )
            raise EmbeddingProviderError("Embedding provider returned a non-numeric embedding.") from exc
        if len(vector) != self.settings.embedding_dim:
            _log_embedding_event(
                "embedding.dimension_mismatch",
                self.settings,
                start,
                logging.WARNING,
                vector_dim=len(vector),
            )
            raise EmbeddingProviderError(
                f"Embedding dimension mismatch: expected {self.settings.embedding_dim}, got {len(vector)}."
            )
        level = logging.WARNING if _elapsed_ms(start) >= self.settings.slow_embedding_ms else logging.INFO
        _log_embedding_event(
            "embedding.request_succeeded",
            self.settings,
            start,
            level,
            vector_dim=len(vector),
        )
        return vector


def memory_embedding_input(title: str, content: str) -> str:
    return f"title: {title}\ncontent: {content}"


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _log_embedding_event(
    event: str,
    settings: Settings,
    start: float,
    level: int,
    **extra: Any,
) -> None:
    logger.log(
        level,
        event,
        extra={
            "event": event,
            "provider_host": url_host(settings.embedding_base_url),
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "duration_ms": _elapsed_ms(start),
            **extra,
        },
    )
