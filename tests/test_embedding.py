import pytest

from aimemory.core.config import Settings
from aimemory.services.embedding import (
    EmbeddingProviderNotConfigured,
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
