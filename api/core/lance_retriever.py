from dataclasses import dataclass
from typing import Any

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from typing_extensions import override

from lancedb.db import Table
from lancedb.rerankers import Reranker

# pyright: basic


@dataclass
class LanceDBRetrieverConfig:
    fts_col: str = "text"
    vector_col: str = "vector"
    k: int = 5
    docs_before_rerank: int = 20
    filter: str = ""


class LanceDBRetriever(BaseRetriever):
    table: Table = Field(description="LanceDB table to search")
    reranker: Reranker = Field(description="Reranker to use for search results")
    embeddings: Embeddings = Field(description="Embeddings model to use")
    config: LanceDBRetrieverConfig = Field(
        default_factory=LanceDBRetrieverConfig, description="Retriever configuration"
    )

    @override
    def _get_relevant_documents(
        self, query: str, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        vector: list[float] = self.embeddings.embed_query(text=query)
        results: list[dict[Any, Any]] = (
            self.table.search(query_type="hybrid")
            .vector(vector)
            .text(query)
            .where(self.config.filter, prefilter=True)
            .limit(self.config.docs_before_rerank)
            .rerank(self.reranker)
            .limit(self.config.k)
            .to_list()
        )
        return [
            Document(page_content=result["text"], metadata=result["metadata"]) for result in results
        ]
