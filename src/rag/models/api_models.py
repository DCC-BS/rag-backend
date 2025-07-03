from typing import Any

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Document metadata response model."""

    id: int
    file_name: str
    document_path: str
    mime_type: str
    num_pages: int | None
    page: int | None
    created_at: str
    access_roles: list[str]


class DocumentListResponse(BaseModel):
    """Response model for document list endpoint."""

    documents: list[DocumentMetadata]
    total_count: int


class UploadDocumentRequest(BaseModel):
    """Request model for document upload."""

    access_role: str


class UpdateDocumentRequest(BaseModel):
    """Request model for document update."""

    access_role: str


class DocumentOperationResponse(BaseModel):
    """Generic response model for document operations."""

    message: str
    document_id: str | None = None
    file_name: str | None = None
    additional_info: dict[str, Any] | None = None
