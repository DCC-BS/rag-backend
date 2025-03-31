"""Integration tests for the BentoEmbeddings class."""

import os

import pytest
import requests

from src.rag.core.bento_embeddings import BentoEmbeddings


def is_bento_api_available(url: str) -> bool:
    """Check if the BentoML API is available."""
    try:
        response = requests.get(f"{url}/readyz", timeout=2)
    except (requests.RequestException, ConnectionError):
        return False
    else:
        return response.status_code == 200


@pytest.mark.skipif(
    not is_bento_api_available(os.environ.get("BENTO_API_URL", "http://localhost:50001")),
    reason="BentoML API is not available",
)
class TestBentoEmbeddingsIntegration:
    """Integration tests for BentoEmbeddings."""

    @pytest.fixture
    def api_url(self) -> str:
        """Get the API URL from environment or use a default."""
        return os.environ.get("BENTO_API_URL", "http://localhost:50001")

    @pytest.fixture
    def bento_embeddings(self, api_url: str) -> BentoEmbeddings:
        """Create a BentoEmbeddings instance."""
        return BentoEmbeddings(api_url=api_url)

    def test_embed_documents(self, bento_embeddings: BentoEmbeddings) -> None:
        """Test embedding documents with actual API."""
        texts = ["This is a test document.", "Another test document."]
        embeddings = bento_embeddings.embed_documents(texts)

        # Check that we got the right number of embeddings
        assert len(embeddings) == len(texts)

        # Check that embeddings are lists of floats
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(isinstance(val, float) for emb in embeddings for val in emb)

        # Check that all embeddings have the same dimension
        embedding_dim = len(embeddings[0])
        assert all(len(emb) == embedding_dim for emb in embeddings)

    def test_embed_query(self, bento_embeddings: BentoEmbeddings) -> None:
        """Test embedding a query with actual API."""
        text = "This is a test query."
        embedding = bento_embeddings.embed_query(text)

        # Check that embedding is a list of floats
        assert isinstance(embedding, list)
        assert all(isinstance(val, float) for val in embedding)

    @pytest.mark.asyncio
    async def test_aembed_documents(self, bento_embeddings: BentoEmbeddings) -> None:
        """Test async embedding documents with actual API."""
        texts = ["This is a test document.", "Another test document."]
        embeddings = await bento_embeddings.aembed_documents(texts)

        # Check that we got the right number of embeddings
        assert len(embeddings) == len(texts)

        # Check that embeddings are lists of floats
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(isinstance(val, float) for emb in embeddings for val in emb)

        # Check that all embeddings have the same dimension
        embedding_dim = len(embeddings[0])
        assert all(len(emb) == embedding_dim for emb in embeddings)

    @pytest.mark.asyncio
    async def test_aembed_query(self, bento_embeddings: BentoEmbeddings) -> None:
        """Test async embedding a query with actual API."""
        text = "This is a test query."
        embedding = await bento_embeddings.aembed_query(text)

        # Check that embedding is a list of floats
        assert isinstance(embedding, list)
        assert all(isinstance(val, float) for val in embedding)
