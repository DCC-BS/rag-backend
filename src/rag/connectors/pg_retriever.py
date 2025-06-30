from collections.abc import Sequence
from typing import Any, override

import cohere
import structlog
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

from rag.models.document import DocumentChunk
from rag.utils.db import get_db_url
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
                    FROM document_chunks
                    WHERE chunk_text @@@ :query_text
                    ORDER BY score DESC
                    LIMIT :bm25_limit
                ) AS bm25_scores
            ),
            semantic_ranked AS (
                SELECT id, RANK() OVER (ORDER BY embedding <=> :embedding) AS rank
                FROM document_chunks
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
            ORDER BY score DESC
            LIMIT :final_limit;
        """)

    id_query: TextClause = _id_query.bindparams(
        bindparam("query_text", type_=String),
        bindparam("bm25_limit", type_=Integer),
        bindparam("embedding", type_=Vector(dim=1024)),
        bindparam("vector_limit", type_=Integer),
        bindparam("access_roles", type_=ARRAY(Text)),
        bindparam("final_limit", type_=Integer),
    )

    def __init__(
        self,
        reranker_api: str,
        embedding_api: str,
        embedding_instructions: str,
        bm25_limit: int = 20,
        vector_limit: int = 20,
        rerank_top_k: int = 5,
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
        self._logger: structlog.stdlib.BoundLogger = structlog.get_logger()
        self._bm25_limit: int = bm25_limit
        self._vector_limit: int = vector_limit
        self._rerank_top_k: int = rerank_top_k
        self._embedding_instructions: str = embedding_instructions

    @override
    def _get_relevant_documents(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, user_roles: list[str]
    ) -> list[LangChainDocument]:
        """
        Retrieves and reranks documents using a single hybrid search SQL query.
        """

        query_embedding: list[float] = (
            self._embedding_client.embeddings.create(
                input=self._embedding_instructions + query, model=self._embedding_model
            )
            .data[0]
            .embedding
        )
        params = {
            "query_text": '"' + query + '"',
            "bm25_limit": self._bm25_limit,
            "embedding": query_embedding,  # pgvector expects a string
            "vector_limit": self._vector_limit,
            "access_roles": user_roles,
            "final_limit": max(self._bm25_limit, self._vector_limit),
        }

        with Session(self._engine) as session:
            id_results = session.execute(self.id_query, params).all()
            if not id_results:
                self._logger.warning(f"No results found for query: {query}")
                return []
            ordered_chunk_ids: list[int] = [res.chunk_id for res in id_results]
            statement: Select[Any] = (
                select(DocumentChunk)
                .options(selectinload(DocumentChunk.document))
                .where(DocumentChunk.id.in_(ordered_chunk_ids))
            )

            fetched_chunks: Sequence[DocumentChunk] = session.execute(statement).scalars().all()
            chunks_by_id: dict[int, DocumentChunk] = {chunk.id: chunk for chunk in fetched_chunks}
            ordered_chunks: list[DocumentChunk] = [
                chunks_by_id[cid] for cid in ordered_chunk_ids if cid in chunks_by_id
            ]

        # TODO: Right now vllm reranker is not working, so we just return the ordered chunks
        reranked_docs: list[LangChainDocument] = [
            LangChainDocument(
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
            for chunk in ordered_chunks
        ]
        return reranked_docs[: self._rerank_top_k]

        docs_to_rerank: list[str] = [chunk.chunk_text for chunk in ordered_chunks]

        reranked_docs_result: V2RerankResponse = self._reranker_client.rerank(
            model=self._reranker_model, query=query, documents=docs_to_rerank, top_n=self._rerank_top_k
        )

        reranked_docs: list[LangChainDocument] = [
            LangChainDocument(
                page_content=ordered_chunks[doc.index].chunk_text,
                metadata={
                    "file_name": ordered_chunks[doc.index].document.file_name,
                    "document_path": ordered_chunks[doc.index].document.document_path,
                    "mime_type": ordered_chunks[doc.index].document.mime_type,
                    "page": ordered_chunks[doc.index].page_number,
                    "num_pages": ordered_chunks[doc.index].document.num_pages,
                    "access_roles": ordered_chunks[doc.index].document.access_roles,
                    "created_at": ordered_chunks[doc.index].document.created_at.isoformat(),
                    "id": ordered_chunks[doc.index].document.id,
                    "chunk_id": ordered_chunks[doc.index].id,
                },
            )
            for doc in reranked_docs_result.results
        ]

        return reranked_docs
