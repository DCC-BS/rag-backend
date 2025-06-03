from dataclasses import dataclass
from typing import Any, override

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from lancedb.rerankers import Reranker
from lancedb.table import Table


@dataclass
class LanceDBRetrieverConfig:
    fts_col: str = "text"
    vector_col: str = "vector"
    k: int = 5
    docs_before_rerank: int = 20


class LanceDBRetriever(BaseRetriever):
    table: Table = Field(description="LanceDB table to search")
    reranker: Reranker = Field(description="Reranker to use for search results")
    embeddings: Embeddings = Field(description="Embeddings model to use")
    config: LanceDBRetrieverConfig = Field(
        default_factory=LanceDBRetrieverConfig, description="Retriever configuration"
    )

    @override
    def _get_relevant_documents(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        user_organizations: list[str],
    ) -> list[Document]:
        filter_query: str = f"metadata.organization IN ('{",".join(user_organizations)}')"
        vector: list[float] = self.embeddings.embed_query(text=query)

        results: list[dict[Any, Any]] = (
            self.table.search(query_type="hybrid")
            .vector(vector)
            .text(text=query)
            .where(where=filter_query, prefilter=True)
            .limit(limit=self.config.docs_before_rerank)
            .rerank(reranker=self.reranker)
            .limit(limit=self.config.k)
            .to_list()
        )
        return [Document(page_content=result["text"], metadata=result["metadata"]) for result in results]
