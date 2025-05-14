from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.rag.core.bento_embeddings import (
    AsyncBentoClientProtocol,
    BentoClientFactory,
    BentoClientProtocol,
    BentoEmbeddings,
)


class MockSyncClient:
    """Mock implementation of a sync BentoML client for testing."""

    def __init__(self, is_client_ready: bool = True, embeddings: dict[str, np.ndarray[Any, Any]] | None = None):
        self._is_ready = is_client_ready
        self.embeddings = embeddings or {}

        # Default test embeddings if none provided
        if not self.embeddings:
            self.embeddings = {
                "documents": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
                "queries": np.array([[0.7, 0.8, 0.9]], dtype=np.float32),
            }

    def is_ready(self) -> bool:
        return self._is_ready

    def encode_documents(self, documents: list[str]) -> np.ndarray[Any, Any]:
        """Return mock document embeddings."""
        # Return the corresponding number of embeddings from our mock data
        num_docs = len(documents)
        return cast(np.ndarray[Any, Any], self.embeddings["documents"][:num_docs])

    def encode_queries(self, queries: list[str]) -> np.ndarray[Any, Any]:
        """Return mock query embeddings."""
        return cast(np.ndarray[Any, Any], self.embeddings["queries"])


class MockAsyncClient:
    """Mock implementation of an async BentoML client for testing."""

    def __init__(self, is_client_ready: bool = True, embeddings: dict[str, np.ndarray[Any, Any]] | None = None):
        self._is_ready = is_client_ready
        self.embeddings = embeddings or {}

        # Default test embeddings if none provided
        if not self.embeddings:
            self.embeddings = {
                "documents": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
                "queries": np.array([[0.7, 0.8, 0.9]], dtype=np.float32),
            }

    async def is_ready(self) -> bool:
        return self._is_ready

    async def encode_documents(self, documents: list[str]) -> np.ndarray[Any, Any]:
        """Return mock document embeddings."""
        # Return the corresponding number of embeddings from our mock data
        num_docs = len(documents)
        return cast(np.ndarray[Any, Any], self.embeddings["documents"][:num_docs])

    async def encode_queries(self, queries: list[str]) -> np.ndarray[Any, Any]:
        """Return mock query embeddings."""
        return cast(np.ndarray[Any, Any], self.embeddings["queries"])


class MockBentoClientFactory:
    """Mock factory that creates mock clients for testing."""

    def __init__(self, sync_client=None, async_client=None):
        self.sync_client = sync_client or MockSyncClient()
        self.async_client = async_client or MockAsyncClient()

    def create_sync_client(self, url: str) -> BentoClientProtocol:
        return self.sync_client

    def create_async_client(self, url: str) -> AsyncBentoClientProtocol:
        return self.async_client


class TestBentoEmbeddings:
    """Test suite for BentoEmbeddings class."""

    @pytest.fixture
    def mock_client_factory(self):
        """Create a mock client factory for testing."""
        return MockBentoClientFactory()

    @pytest.fixture
    def bento_embeddings(self, mock_client_factory):
        """Create a BentoEmbeddings instance for testing."""
        return BentoEmbeddings(api_url="http://fake-api.example.com", batch_size=2, client_factory=mock_client_factory)

    def test_init(self):
        """Test initialization of BentoEmbeddings."""
        embeddings = BentoEmbeddings(api_url="http://test-api.com", batch_size=10)
        assert embeddings.api_url == "http://test-api.com"
        assert embeddings.batch_size == 10
        assert isinstance(embeddings.client_factory, BentoClientFactory)

    def test_embed_documents(self, bento_embeddings):
        """Test the embed_documents method."""
        texts = ["This is a test document", "This is another test document"]
        embeddings = bento_embeddings.embed_documents(texts)

        # Validate the output
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(isinstance(val, float) for emb in embeddings for val in emb)

    def test_embed_query(self, bento_embeddings):
        """Test the embed_query method."""
        text = "This is a test query"
        embedding = bento_embeddings.embed_query(text)

        # Validate the output
        assert isinstance(embedding, list)
        assert all(isinstance(val, float) for val in embedding)
        assert len(embedding) == 3  # Based on our mock data

    @pytest.mark.asyncio
    async def test_aembed_documents(self, bento_embeddings):
        """Test the async embed_documents method."""
        texts = ["This is a test document", "This is another test document"]
        embeddings = await bento_embeddings.aembed_documents(texts)

        # Validate the output
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(isinstance(val, float) for emb in embeddings for val in emb)

    @pytest.mark.asyncio
    async def test_aembed_query(self, bento_embeddings):
        """Test the async embed_query method."""
        text = "This is a test query"
        embedding = await bento_embeddings.aembed_query(text)

        # Validate the output
        assert isinstance(embedding, list)
        assert all(isinstance(val, float) for val in embedding)
        assert len(embedding) == 3  # Based on our mock data

    def test_service_not_ready(self, mock_client_factory):
        """Test behavior when BentoML service is not ready."""
        # Create a mock with not-ready client
        mock_client_factory.sync_client = MockSyncClient(is_client_ready=False)
        embeddings = BentoEmbeddings(api_url="http://fake-api.example.com", client_factory=mock_client_factory)

        # Check that RuntimeError is raised when service is not ready
        with pytest.raises(RuntimeError) as exc_info:
            embeddings.embed_documents(["test"])

        assert "not ready" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_async_service_not_ready(self, mock_client_factory):
        """Test behavior when async BentoML service is not ready."""
        # Create a mock with not-ready client
        mock_client_factory.async_client = MockAsyncClient(is_client_ready=False)
        embeddings = BentoEmbeddings(api_url="http://fake-api.example.com", client_factory=mock_client_factory)

        # Check that RuntimeError is raised when service is not ready
        with pytest.raises(RuntimeError) as exc_info:
            await embeddings.aembed_documents(["test"])

        assert "not ready" in str(exc_info.value)

    def test_batching(self, mock_client_factory):
        """Test that documents are correctly batched during embedding."""
        # Create a client that can track calls
        mock_sync_client = MockSyncClient()
        mock_response = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mock_sync_client.encode_documents = MagicMock(return_value=mock_response)
        mock_client_factory.sync_client = mock_sync_client

        embeddings = BentoEmbeddings(
            api_url="http://fake-api.example.com", batch_size=2, client_factory=mock_client_factory
        )

        # Embed 5 documents with batch_size=2
        docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        embeddings.embed_documents(docs)

        # Should be called 3 times (ceiling of 5/2)
        assert mock_sync_client.encode_documents.call_count == 3

        # Check the batches
        calls = mock_sync_client.encode_documents.call_args_list
        assert calls[0][1]["documents"] == ["doc1", "doc2"]
        assert calls[1][1]["documents"] == ["doc3", "doc4"]
        assert calls[2][1]["documents"] == ["doc5"]
