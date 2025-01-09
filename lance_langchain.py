from typing import List, Dict, Tuple
from langchain.schema import Document
from langchain.retrievers import BaseRetriever
from lancedb.db import Table
from lancedb.rerankers import Reranker
import pyarrow as pa
import bentoml

class BentoMLReranker(Reranker):
    def __init__(self, api_url, column, return_score="relevance"):
        super().__init__(return_score)
        self.client = bentoml.SyncHTTPClient(api_url)
        self.column = column
    
    def _rerank(self, query: str, result_set:  pa.Table) -> pa.Table:
        if self.client.is_ready():
            documents = result_set[self.column].to_pylist()
            ranks: Dict = self.client.rerank(documents=documents, query=query)
            scores = [score for _, score, _ in ranks.values()]
            result_set = result_set.append_column(
                "_relevance_score", pa.array(scores, type=pa.float32())
            )
        return result_set

    def rerank_hybrid(self, query: str, vector_results: pa.Table, fts_results: pa.Table) -> pa.Table:
        combined_results = self.merge_results(vector_results, fts_results)
        combined_results = self._rerank(combined_results, query)

        if self.score == "relevance":
            combined_results = self._keep_relevance_score(combined_results)
        elif self.score == "all":
            raise NotImplementedError(
                "return_score='all' not implemented for BentoML Reranker"
            )
        combined_results = combined_results.sort_by(
            [("_relevance_score", "descending")]
        )
        return combined_results
    
    def rerank_vector(self, query: str, vector_results: pa.Table) -> pa.Table:
        vector_results = self._rerank(vector_results, query)
        if self.score == "relevance":
            vector_results = vector_results.drop_columns(["_distance"])

        vector_results = vector_results.sort_by([("_relevance_score", "descending")])
        return vector_results

    def rerank_fts(self, query: str, fts_results: pa.Table) -> pa.Table:
        fts_results = self._rerank(fts_results, query)
        if self.score == "relevance":
            fts_results = fts_results.drop_columns(["_score"])

        fts_results = fts_results.sort_by([("_relevance_score", "descending")])
        return fts_results

class LanceDBRetriever(BaseRetriever):
    def __init__(self, table: Table, reranker: Reranker, vector_col: str, fts_col: str, k: int = 10, filter: str = ""):
        super().__init__()
        self.table = table
        self.reranker = reranker
        self.k = k
        self.vector_col = vector_col
        self.fts_col = fts_col
        self.filter = filter

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = (
            self.table
            .search(
                query,
                query_type="hybrid",
                vector_column_name=self.vector_col,
                fts_columns=self.fts_col,
            )
            .where(self.filter, prefilter=True)
            .rerank(self.reranker)
            .limit(self.k)
        )
        return [Document(page_content=result[self.fts_col], metadata=result.metadata) for result in results]
