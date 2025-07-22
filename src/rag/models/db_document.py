import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, TIMESTAMP, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__: str = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_name: Mapped[str] = mapped_column(String(255))
    document_path: Mapped[str] = mapped_column(String(1024), unique=True)
    mime_type: Mapped[str] = mapped_column(String(255))
    num_pages: Mapped[int] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.datetime.now(datetime.UTC)
    )
    access_roles: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)

    # Establish the one-to-many relationship
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__: str = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(dim=1024))
    page_number: Mapped[int] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.datetime.now(datetime.UTC)
    )

    # Establish the many-to-one relationship
    document: Mapped["Document"] = relationship(back_populates="chunks")


Index("idx_documents_access_roles", Document.access_roles, postgresql_using="gin")
Index(
    "idx_document_chunks_embedding",
    DocumentChunk.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)

Index(
    "search_idx",
    DocumentChunk.chunk_text,
    postgresql_using="bm25",
    postgresql_with={"key_field": "id"},
    postgresql_ops=["id", "chunk_text", "page_number", "created_at"],
)
