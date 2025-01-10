from typing import List

from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from lancedb.db import Table
from lancedb.rerankers import Reranker


class LanceDBRetriever(BaseRetriever):
    table: Table
    reranker: Reranker
    embeddings: Embeddings
    fts_col: str = "text"
    vector_col: str = "vector"
    k: int = 5
    docs_before_rerank: int = 20
    filter: str = ""

    def _get_relevant_documents(self, query: str) -> List[Document]:
        vector = self.embeddings.embed_query(query)
        results = (
            self.table.search(
                query_type="hybrid",
            )
            .vector(vector)
            .text(query)
            .where(self.filter, prefilter=True)
            .limit(self.docs_before_rerank)
            .rerank(self.reranker)
            .limit(self.k)
            .to_list()
        )
        return [
            Document(page_content=result["text"], metadata=result["metadata"])
            for result in results
        ]
