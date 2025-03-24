import asyncio
from typing import Any, override

import bentoml
import numpy as np
import pyarrow
from langchain_core.embeddings import Embeddings

from lancedb.rerankers import Reranker


class BentoEmbeddings(Embeddings):
    def __init__(self, api_url: str) -> None:
        """
        Initializes the BentoEmbeddings.

        Args:
            api_url: The URL of the BentoML API.
        """
        self.api_url: str = api_url
        self.batch_size: int = 32

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
        with bentoml.SyncHTTPClient(url=self.api_url) as client:
            if not client.is_ready():
                raise RuntimeError(f"BentoML service at {self.api_url} is not ready.")

            for i in range(0, len(texts), self.batch_size):
                batch: list[str] = texts[i : i + self.batch_size]
                batch_embeddings: np.ndarray[Any, Any] = client.encode_documents(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
                    documents=batch,
                )
                embeddings.extend(batch_embeddings.tolist())  # pyright: ignore[reportArgumentType, reportUnknownMemberType]

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
        with bentoml.SyncHTTPClient(url=self.api_url) as client:
            if client.is_ready():
                embeddings: np.ndarray[Any, Any] = client.encode_queries(queries=[text])  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
                return embeddings[0].tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            else:
                raise RuntimeError(f"BentoML service at {self.api_url} is not ready.")

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
        async with bentoml.AsyncHTTPClient(self.api_url) as client:
            if not await client.is_ready():
                raise RuntimeError(f"BentoML service at {self.api_url} is not ready.")

            async def embed_batch(batch: list[str]) -> list[list[float]]:
                batch_embeddings: np.ndarray[Any, Any] = await client.encode_documents(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
                    documents=batch,
                )
                return batch_embeddings.tolist()  # pyright: ignore[reportReturnType, reportUnknownMemberType, reportUnknownVariableType]

            tasks: list[Any] = []
            for i in range(0, len(texts), self.batch_size):
                batch: list[str] = texts[i : i + self.batch_size]
                tasks.append(embed_batch(batch))

            results: list[list[float]] = await asyncio.gather(*tasks)  # pyright: ignore[reportAny]
            for result in results:
                embeddings.extend([result])

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
        async with bentoml.AsyncHTTPClient(url=self.api_url) as client:
            if await client.is_ready():
                embeddings: np.ndarray[Any, Any] = await client.encode_queries(queries=[text])  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
                return embeddings[0].tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            else:
                raise RuntimeError(f"BentoML service at {self.api_url} is not ready.")


class BentoMLReranker(Reranker):
    def __init__(self, api_url: str, column: str, return_score: str = "relevance") -> None:
        super().__init__(return_score)
        self.client: bentoml.SyncHTTPClient = bentoml.SyncHTTPClient(url=api_url)
        self.column: str = column

    def _rerank(self, query: str, result_set: pyarrow.Table) -> pyarrow.Table:
        if self.client.is_ready():
            documents = result_set[self.column].to_pylist()
            ranks: dict[str, tuple[str, float, str]] = self.client.rerank(documents=documents, query=query)
            scores: list[float | Any] = [score for _, score, _ in ranks.values()]
            result_set = result_set.append_column("_relevance_score", pyarrow.array(scores, type=pyarrow.float32()))
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
