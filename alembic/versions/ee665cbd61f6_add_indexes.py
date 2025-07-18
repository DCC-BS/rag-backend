"""Add indexes

Revision ID: ee665cbd61f6
Revises: 7643df89db15
Create Date: 2025-06-23 13:55:24.275932

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ee665cbd61f6"
down_revision: str | Sequence[str] | None = "7643df89db15"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_index(
        "idx_document_chunks_embedding",
        "document_chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "search_idx",
        "document_chunks",
        ["id", "chunk_text", "page_number", "created_at"],
        unique=False,
        postgresql_using="bm25",
        postgresql_with={"key_field": "id"},
    )
    op.create_index("idx_documents_access_roles", "documents", ["access_roles"], unique=False, postgresql_using="gin")
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index("idx_documents_access_roles", table_name="documents", postgresql_using="gin")
    op.drop_index(
        "search_idx", table_name="document_chunks", postgresql_using="bm25", postgresql_with={"key_field": "id"}
    )
    op.drop_index(
        "idx_document_chunks_embedding",
        table_name="document_chunks",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    # ### end Alembic commands ###
