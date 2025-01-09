from typing import List

import bentoml
import asyncio
import numpy as np
from langchain_core.embeddings import Embeddings


class BentoEmbeddings(Embeddings):
    def __init__(self, api_url: str):
        """
        Initializes the BentoEmbeddings.

        Args:
            api_url: The URL of the BentoML API.
        """
        self.api_url = api_url
        self.batch_size: int = 32

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embeddings, one for each document.
        """
        embeddings: List[List[float]] = []
        with bentoml.SyncHTTPClient(self.api_url) as client:
            if not client.is_ready():
                raise RuntimeError(
                    f"BentoML service at {self.api_url} is not ready."
                )

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_embeddings: np.ndarray = client.encode_documents(
                    documents=batch,
                )
                embeddings.extend(batch_embeddings.tolist())

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding.
        """
        with bentoml.SyncHTTPClient(self.api_url) as client:
            if client.is_ready():
                embeddings: np.ndarray = client.encode_queries(queries=[text])
                return embeddings[0].tolist()
            else:
                raise RuntimeError(
                    f"BentoML service at {self.api_url} is not ready."
                )

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously embeds a list of documents.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embeddings, one for each document.
        """
        embeddings: List[List[float]] = []
        async with bentoml.AsyncHTTPClient(self.api_url) as client:
            if not await client.is_ready():
                raise RuntimeError(
                    f"BentoML service at {self.api_url} is not ready."
                )

            async def embed_batch(batch: List[str]) -> List[List[float]]:
                batch_embeddings: np.ndarray = await client.encode_documents(
                    documents=batch,
                )
                return batch_embeddings.tolist()

            tasks = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                tasks.append(embed_batch(batch))

            results = await asyncio.gather(*tasks)
            for result in results:
                embeddings.extend(result)

        return embeddings
    

    async def aembed_query(self, text: str) -> List[float]:
        """
        Asynchronously embeds a single query.

        Args:
            text: The query string to embed.

        Returns:
            A list of floats representing the embedding.
        """
        async with bentoml.AsyncHTTPClient(self.api_url) as client:
            if await client.is_ready():
                embeddings: np.ndarray = await client.encode_queries(queries=[text])
                return embeddings[0].tolist()
            else:
                raise RuntimeError(
                    f"BentoML service at {self.api_url} is not ready."
                )