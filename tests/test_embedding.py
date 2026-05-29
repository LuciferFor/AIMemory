import pytest
import httpx

from aimemory.core.config import Settings
from aimemory.services.embedding import (
    EmbeddingProviderNotConfigured,
    EmbeddingProviderError,
    OpenAICompatibleEmbeddingClient,
    memory_embedding_input,
)


def test_embedding_requires_provider_configuration() -> None:
    settings = Settings(embedding_base_url="", embedding_api_key="", embedding_dim=2)
    client = OpenAICompatibleEmbeddingClient(settings)

    with pytest.raises(EmbeddingProviderNotConfigured):
        client.embed("hello")


def test_memory_embedding_input_is_stable() -> None:
    assert memory_embedding_input("Title", "Body") == "title: Title\ncontent: Body"


def test_embedding_error_message_is_redacted(monkeypatch) -> None:
    class _FailingClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, *args, **kwargs):
            request = httpx.Request("POST", "https://example.test/embeddings?api_key=supersecret")
            raise httpx.ConnectError("token=tokensecret api_key=supersecret", request=request)

    monkeypatch.setattr("aimemory.services.embedding.httpx.Client", _FailingClient)
    settings = Settings(
        embedding_base_url="https://example.test/v1?api_key=supersecret",
        embedding_api_key="real-secret",
        embedding_dim=2,
    )
    client = OpenAICompatibleEmbeddingClient(settings)

    with pytest.raises(EmbeddingProviderError) as exc:
        client.embed("private input text")

    message = str(exc.value)
    assert "supersecret" not in message
    assert "tokensecret" not in message
    assert "private input text" not in message
