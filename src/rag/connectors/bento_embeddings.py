import asyncio
from typing import (
    Any,
    Protocol,
    override,
)

import bentoml
import numpy as np
import pyarrow
from lancedb.rerankers import Reranker
from langchain_core.embeddings import Embeddings


class BentoClientProtocol(Protocol):
    def is_ready(self) -> bool:
        """Check if the client is ready."""
        ...

    def encode_documents(self, documents: list[str]) -> np.ndarray[Any, Any]:
        """Encode documents."""
        ...

    def encode_queries(self, queries: list[str]) -> np.ndarray[Any, Any]:
        """Encode queries."""
        ...

    def rerank(self, documents: list[str], query: str) -> dict[str, tuple[str, float, str]]:
        """Rerank documents."""
        ...


class AsyncBentoClientProtocol(Protocol):
    async def is_ready(self) -> bool:
        """Check if the client is ready."""
        ...

    async def encode_documents(self, documents: list[str]) -> np.ndarray[Any, Any]:
        """Encode documents."""
        ...

    async def encode_queries(self, queries: list[str]) -> np.ndarray[Any, Any]:
        """Encode queries."""
        ...


# We need to use a class factory approach since we don't know the actual methods available
# on the BentoML clients at type checking time
def sync_client_adapter_factory(client: bentoml.SyncHTTPClient) -> BentoClientProtocol:
    """Create an adapter that wraps a sync client."""

    class SyncClientAdapter:
        def is_ready(self) -> bool:
            return client.is_ready()

        def encode_documents(self, documents: list[str]) -> np.ndarray[Any, Any]:
            return client.encode_documents(documents=documents)

        def encode_queries(self, queries: list[str]) -> np.ndarray[Any, Any]:
            return client.encode_queries(queries=queries)

        def rerank(self, documents: list[str], query: str) -> dict[str, tuple[str, float, str]]:
            return client.rerank(documents=documents, query=query)

    return SyncClientAdapter()


def async_client_adapter_factory(client: bentoml.AsyncHTTPClient) -> AsyncBentoClientProtocol:
    """Create an adapter that wraps an async client."""

    class AsyncClientAdapter:
        async def is_ready(self) -> bool:
            return await client.is_ready()

        async def encode_documents(self, documents: list[str]) -> np.ndarray[Any, Any]:
            return await client.encode_documents(documents=documents)

        async def encode_queries(self, queries: list[str]) -> np.ndarray[Any, Any]:
            return await client.encode_queries(queries=queries)

        async def rerank(self, documents: list[str], query: str) -> dict[str, tuple[str, float, str]]:
            return await client.rerank(documents=documents, query=query)

    return AsyncClientAdapter()


class BentoClientFactory:
    """Factory for creating BentoML clients."""

    @staticmethod
    def create_sync_client(url: str) -> BentoClientProtocol:
        """Create a synchronous BentoML client."""
        client = bentoml.SyncHTTPClient(url=url)
        return sync_client_adapter_factory(client)

    @staticmethod
    def create_async_client(url: str) -> AsyncBentoClientProtocol:
        """Create an asynchronous BentoML client."""
        client = bentoml.AsyncHTTPClient(url=url)
        return async_client_adapter_factory(client)


class BentoEmbeddings(Embeddings):
    def __init__(self, api_url: str, batch_size: int = 32, client_factory: BentoClientFactory | None = None) -> None:
        """
        Initializes the BentoEmbeddings.

        Args:
            api_url: The URL of the BentoML API.
            batch_size: The batch size for processing documents.
            client_factory: Factory for creating BentoML clients. Defaults to BentoClientFactory.
        """
        self.api_url: str = api_url
        self.batch_size: int = batch_size
        self.client_factory: BentoClientFactory = client_factory or BentoClientFactory()

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of documents.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embeddings, one for each document.
        """
        embeddings: list[list[float]] = []
        client = self.client_factory.create_sync_client(self.api_url)

        if not client.is_ready():
            raise RuntimeError(f"BentoML service at {self.api_url} is not ready.")

        for i in range(0, len(texts), self.batch_size):
            batch: list[str] = texts[i : i + self.batch_size]
            batch_embeddings: np.ndarray[Any, Any] = client.encode_documents(documents=batch)
            embeddings.extend(batch_embeddings.tolist())

        return embeddings

    @override
    def embed_query(self, text: str) -> list[float]:
        """
        Embeds a single query.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding.
        """
        client = self.client_factory.create_sync_client(self.api_url)

        if not client.is_ready():
            raise RuntimeError(f"BentoML service at {self.api_url} is not ready.")

        embeddings: np.ndarray[Any, Any] = client.encode_queries(queries=[text])
        return embeddings[0].tolist()

    @override
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Asynchronously embeds a list of documents.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embeddings, one for each document.
        """
        embeddings: list[list[float]] = []
        client = self.client_factory.create_async_client(self.api_url)

        if not await client.is_ready():
            raise RuntimeError(f"BentoML service at {self.api_url} is not ready.")

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            batch_embeddings: np.ndarray[Any, Any] = await client.encode_documents(documents=batch)
            return batch_embeddings.tolist()

        tasks = []
        for i in range(0, len(texts), self.batch_size):
            batch: list[str] = texts[i : i + self.batch_size]
            tasks.append(embed_batch(batch))

        results = await asyncio.gather(*tasks)
        for result in results:
            if isinstance(result, list):
                if result and isinstance(result[0], list):
                    embeddings.extend(result)
                elif result:
                    embeddings.append(result)

        return embeddings

    @override
    async def aembed_query(self, text: str) -> list[float]:
        """
        Asynchronously embeds a single query.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding.
        """
        client = self.client_factory.create_async_client(self.api_url)

        if not await client.is_ready():
            raise RuntimeError(f"BentoML service at {self.api_url} is not ready.")

        embeddings: np.ndarray[Any, Any] = await client.encode_queries(queries=[text])
        return embeddings[0].tolist()


class BentoMLReranker(Reranker):
    def __init__(self, api_url: str, column: str, return_score: str = "relevance") -> None:
        super().__init__(return_score)
        self.api_url = api_url
        self.column: str = column
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = bentoml.SyncHTTPClient(url=self.api_url)
        return self._client

    def _rerank(self, query: str, result_set: pyarrow.Table) -> pyarrow.Table:
        if self.client.is_ready():
            documents = result_set[self.column].to_pylist()
            if len(documents) > 0:
                ranks: dict[str, tuple[str, float, str]] = self.client.rerank(documents=documents, query=query)
                scores: list[float | Any] = [score for _, score, _ in ranks.values()]
                result_set = result_set.append_column("_relevance_score", pyarrow.array(scores, type=pyarrow.float32()))
            else:
                result_set = result_set.append_column("_relevance_score", pyarrow.array([], type=pyarrow.float32()))
        return result_set

    def rerank_hybrid(self, query: str, vector_results: pyarrow.Table, fts_results: pyarrow.Table) -> pyarrow.Table:
        combined_results = self.merge_results(vector_results, fts_results)
        combined_results = self._rerank(result_set=combined_results, query=query)

        if self.score == "relevance":
            combined_results = self._keep_relevance_score(combined_results)
        elif self.score == "all":
            raise NotImplementedError("return_score='all' not implemented for BentoML Reranker")
        combined_results = combined_results.sort_by([("_relevance_score", "descending")])
        return combined_results

    def rerank_vector(self, query: str, vector_results: pyarrow.Table) -> pyarrow.Table:
        vector_results = self._rerank(result_set=vector_results, query=query)
        if self.score == "relevance":
            vector_results = vector_results.drop_columns(["_distance"])

        vector_results = vector_results.sort_by([("_relevance_score", "descending")])
        return vector_results

    def rerank_fts(self, query: str, fts_results: pyarrow.Table) -> pyarrow.Table:
        fts_results = self._rerank(result_set=fts_results, query=query)
        if self.score == "relevance":
            fts_results = fts_results.drop_columns(["_score"])

        fts_results = fts_results.sort_by([("_relevance_score", "descending")])
        return fts_results
