from collections.abc import Sequence
from typing import Any, override

import cohere
from cohere.v2.types.v2rerank_response import V2RerankResponse
from langchain.schema import (
    BaseRetriever,
    Document as LangChainDocument,
)
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from openai import Client
from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Engine, Integer, String, Text, bindparam, create_engine, select, text
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import TextClause

from rag.models.db_document import DocumentChunk
from rag.utils.db import get_db_url
from rag.utils.logger import get_logger
from rag.utils.model_clients import get_embedding_client, get_reranker_client


class PGRoleRetriever(BaseRetriever):
    """
    A LangChain retriever that performs role-based hybrid search
    on a PostgreSQL database with pgvector and pg_search.
    """

    # This SQL query uses CTEs to perform BM25 and vector searches,
    # then combines them using a FULL OUTER JOIN and scores them
    # with Reciprocal Rank Fusion (RRF).
    # We only return the id of the chunk, not the full document.
    # This allows us to use sqlalchemy with the ORM to get the documents as objects.
    # Using autocommit=False prevents SQLAlchemy from interpreting colons in user input
    _id_query: TextClause = text("""
            WITH bm25_ranked AS (
                SELECT id, RANK() OVER (ORDER BY score DESC) as rank
                FROM (
                    SELECT id, paradedb.score(id) AS score
                    FROM document_chunks dc
                    WHERE chunk_text @@@ :query_text
                    AND (:document_ids IS NULL OR dc.document_id = ANY(:document_ids))
                    ORDER BY score DESC
                    LIMIT :bm25_limit
                ) AS bm25_scores
            ),
            semantic_ranked AS (
                SELECT id, RANK() OVER (ORDER BY embedding <=> :embedding) AS rank
                FROM document_chunks dc
                WHERE (:document_ids IS NULL OR dc.document_id = ANY(:document_ids))
                ORDER BY embedding <=> :embedding
                LIMIT :vector_limit
            )
            SELECT
                COALESCE(sr.id, br.id) AS chunk_id,
                COALESCE(1.0 / (60 + sr.rank), 0.0) + COALESCE(1.0 / (60 + br.rank), 0.0) AS score
            FROM semantic_ranked sr
            FULL OUTER JOIN bm25_ranked br ON sr.id = br.id
            JOIN document_chunks dc ON dc.id = COALESCE(sr.id, br.id)
            JOIN documents d ON d.id = dc.document_id
            WHERE d.access_roles && :access_roles
            AND (:document_ids IS NULL OR d.id = ANY(:document_ids))
            ORDER BY score DESC
            LIMIT :final_limit;
        """)

    id_query: TextClause = _id_query.bindparams(
        bindparam("query_text", type_=String),
        bindparam("bm25_limit", type_=Integer),
        bindparam("embedding", type_=Vector(dim=1024)),
        bindparam("vector_limit", type_=Integer),
        bindparam("access_roles", type_=ARRAY(Text)),
        bindparam("document_ids", type_=ARRAY(Integer)),
        bindparam("final_limit", type_=Integer),
    )

    def __init__(
        self,
        reranker_api: str,
        embedding_api: str,
        embedding_instructions: str,
        bm25_limit: int = 20,
        vector_limit: int = 20,
        top_k: int = 5,
        use_reranker: bool = True,
    ):
        super().__init__()

        # Setup embedding client and model
        embedding_client_info = get_embedding_client()
        self._embedding_client: Client = embedding_client_info.client
        self._embedding_model: str = embedding_client_info.model

        # Setup reranker client and model
        reranker_client_info = get_reranker_client()
        self._reranker_client: cohere.ClientV2 = reranker_client_info.client
        self._reranker_model: str = reranker_client_info.model

        db_url: str = get_db_url()
        self._engine: Engine = create_engine(url=db_url)
        self._logger = get_logger()
        self._bm25_limit: int = bm25_limit
        self._vector_limit: int = vector_limit
        self._top_k: int = top_k
        self._embedding_instructions: str = embedding_instructions
        self._use_reranker: bool = use_reranker

    def _create_query_embedding(self, query: str) -> list[float]:
        """Create an embedding for the given query."""
        try:
            response = self._embedding_client.embeddings.create(
                input=self._embedding_instructions + query, model=self._embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            self._logger.exception("Failed to create query embedding", error=e)
            raise

    def _execute_hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        user_roles: list[str],
        document_ids: list[int] | None = None,
    ) -> list[int]:
        """Execute the hybrid search query and return ordered chunk IDs."""
        params = {
            "query_text": '"' + query + '"',
            "bm25_limit": self._bm25_limit,
            "embedding": query_embedding,
            "vector_limit": self._vector_limit,
            "access_roles": user_roles,
            "document_ids": document_ids,
            "final_limit": max(self._bm25_limit, self._vector_limit),
        }

        try:
            with Session(self._engine) as session:
                id_results = session.execute(self.id_query, params).all()
                return [res.chunk_id for res in id_results]
        except Exception as e:
            self._logger.exception("Failed to execute hybrid search", error=e)
            raise

    def _fetch_document_chunks(self, chunk_ids: list[int]) -> list[DocumentChunk]:
        """Fetch document chunks by their IDs in the specified order."""
        if not chunk_ids:
            return []

        try:
            with Session(self._engine) as session:
                statement: Select[Any] = (
                    select(DocumentChunk)
                    .options(selectinload(DocumentChunk.document))
                    .where(DocumentChunk.id.in_(chunk_ids))
                )

                fetched_chunks: Sequence[DocumentChunk] = session.execute(statement).scalars().all()
                chunks_by_id: dict[int, DocumentChunk] = {chunk.id: chunk for chunk in fetched_chunks}

                # Maintain the original order from the search results
                return [chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id]
        except Exception as e:
            self._logger.exception("Failed to fetch document chunks", chunk_ids=chunk_ids, error=e)
            raise

    def _rerank_documents(self, query: str, chunks: list[DocumentChunk], top_k: int) -> list[DocumentChunk]:
        """Rerank documents using the reranker model."""
        if not chunks:
            return []

        try:
            docs_to_rerank: list[str] = [chunk.chunk_text for chunk in chunks]

            reranked_result: V2RerankResponse = self._reranker_client.rerank(
                model=self._reranker_model, query=query, documents=docs_to_rerank, top_n=top_k
            )

            return [chunks[doc.index] for doc in reranked_result.results]
        except Exception as e:
            self._logger.exception("Failed to rerank documents", error=e)
            raise

    def _create_langchain_document(self, chunk: DocumentChunk) -> LangChainDocument:
        """Create a LangChain document from a DocumentChunk."""
        return LangChainDocument(
            page_content=chunk.chunk_text,
            metadata={
                "file_name": chunk.document.file_name,
                "document_path": chunk.document.document_path,
                "mime_type": chunk.document.mime_type,
                "page": chunk.page_number,
                "num_pages": chunk.document.num_pages,
                "access_roles": chunk.document.access_roles,
                "created_at": chunk.document.created_at.isoformat(),
                "id": chunk.document.id,
                "chunk_id": chunk.id,
            },
        )

    def _convert_chunks_to_documents(self, chunks: list[DocumentChunk]) -> list[LangChainDocument]:
        """Convert a list of DocumentChunks to LangChain documents."""
        return [self._create_langchain_document(chunk) for chunk in chunks]

    @override
    def _get_relevant_documents(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        user_roles: list[str],
        document_ids: list[int] | None = None,
        top_k: int | None = None,
    ) -> list[LangChainDocument]:
        """
        Retrieves and reranks documents using a single hybrid search SQL query.
        """
        top_k = top_k or self._top_k

        query_embedding = self._create_query_embedding(query)

        chunk_ids = self._execute_hybrid_search(query, query_embedding, user_roles, document_ids)

        if not chunk_ids:
            self._logger.warning("No results found for query", query=query)
            return []

        ordered_chunks = self._fetch_document_chunks(chunk_ids)

        if self._use_reranker:
            final_chunks = self._rerank_documents(query, ordered_chunks, top_k)
        else:
            final_chunks = ordered_chunks[:top_k]

        return self._convert_chunks_to_documents(final_chunks)
