from typing import Any

import httpx

from aimemory.core.config import Settings, get_settings


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
            raise EmbeddingProviderError(f"Embedding request failed: {exc}") from exc
        except ValueError as exc:
            raise EmbeddingProviderError("Embedding provider returned invalid JSON.") from exc

        try:
            embedding = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbeddingProviderError("Embedding provider response did not include data[0].embedding.") from exc

        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingProviderError("Embedding provider returned an empty embedding.")

        vector = [float(value) for value in embedding]
        if len(vector) != self.settings.embedding_dim:
            raise EmbeddingProviderError(
                f"Embedding dimension mismatch: expected {self.settings.embedding_dim}, got {len(vector)}."
            )
        return vector


def memory_embedding_input(title: str, content: str) -> str:
    return f"title: {title}\ncontent: {content}"
