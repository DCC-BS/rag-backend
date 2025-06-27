import tempfile
from pathlib import Path
from typing import Any, NoReturn

import structlog
from fastapi import HTTPException, UploadFile, status
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from rag.models.document import Document
from rag.utils.config import AppConfig, ConfigurationManager
from rag.utils.db import get_db_url
from rag.utils.s3 import S3Utils


class DocumentManagementService:
    """Service for managing documents in S3 storage and database."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialize the document management service.

        Args:
            config: Application configuration object
        """
        self.config: AppConfig = config or ConfigurationManager.get_config()
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()

        db_url: str = get_db_url()
        self.engine: Engine = create_engine(url=db_url)

        # S3 utilities setup
        self.s3_utils = S3Utils(config)

    @staticmethod
    def _raise_document_not_found() -> NoReturn:
        """Raise HTTPException for document not found."""
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    @staticmethod
    def _raise_access_denied(detail: str = "Access denied to this document") -> NoReturn:
        """Raise HTTPException for access denied."""
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)

    @staticmethod
    def _raise_invalid_document_path() -> NoReturn:
        """Raise HTTPException for invalid document path format."""
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid document path format")

    def get_user_documents(self, access_roles: list[str]) -> list[dict[str, Any]]:
        """Get documents metadata for user's access roles.

        Args:
            access_roles: List of access roles for the user

        Returns:
            List of document metadata dictionaries
        """
        try:
            with Session(self.engine) as session:
                # Query documents that have at least one matching access role
                stmt = select(Document).where(Document.access_roles.op("&&")(access_roles))
                documents = session.execute(stmt).scalars().all()

                return [
                    {
                        "id": doc.id,
                        "file_name": doc.file_name,
                        "document_path": doc.document_path,
                        "mime_type": doc.mime_type,
                        "num_pages": doc.num_pages,
                        "created_at": doc.created_at.isoformat(),
                        "access_roles": doc.access_roles,
                    }
                    for doc in documents
                ]
        except Exception as e:
            self.logger.exception("Failed to get user documents", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve documents"
            ) from e

    def get_document_by_id(self, document_id: int, access_roles: list[str]) -> bytes:
        """Get document content from S3 by document ID.

        Args:
            document_id: ID of the document
            access_roles: List of access roles for the user

        Returns:
            Document content as bytes

        Raises:
            HTTPException: If document not found or access denied
        """
        try:
            with Session(self.engine) as session:
                # Find document and verify access
                stmt = select(Document).where(Document.id == document_id)
                document = session.execute(stmt).scalar_one_or_none()

                if not document:
                    self._raise_document_not_found()

                # Check if user has access to this document
                if not any(role in document.access_roles for role in access_roles):
                    self._raise_access_denied()

                # Extract bucket and object key from document_path
                # document_path format: s3://bucket-name/object-key
                if not document.document_path.startswith("s3://"):
                    self._raise_invalid_document_path()

                path_parts = document.document_path[5:].split("/", 1)  # Remove s3:// prefix
                if len(path_parts) != 2:
                    self._raise_invalid_document_path()

                bucket_name, object_key = path_parts

                # Download from S3
                return self.s3_utils.download_s3_object(bucket_name, object_key)

        except HTTPException:
            raise
        except Exception as e:
            self.logger.exception("Failed to get document by ID", document_id=document_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve document"
            ) from e

    def upload_document(self, file: UploadFile, access_role: str, user_access_roles: list[str]) -> dict[str, Any]:
        """Upload a new document to S3.

        Args:
            file: File to upload
            access_role: Target access role for the document
            user_access_roles: User's access roles for validation

        Returns:
            Dictionary with upload result information

        Raises:
            HTTPException: If access denied or upload fails
        """
        # Validate user has access to the specified role
        if access_role not in user_access_roles:
            self._raise_access_denied(f"Access denied to role: {access_role}")

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File name is required")

        try:
            # Get bucket name for the role
            bucket_name = self.s3_utils.get_bucket_name(access_role)

            # Ensure bucket exists
            self.s3_utils.ensure_bucket_exists(bucket_name)

            # Normalize the file name for S3
            normalized_filename = S3Utils.normalize_path(file.filename)
            object_key = normalized_filename

            # Create temporary file to upload
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                # Read file content
                content = file.file.read()
                tmp_file.write(content)
                tmp_file_path = Path(tmp_file.name)

            try:
                # Upload to S3
                success = self.s3_utils.upload_file(tmp_file_path, bucket_name, object_key)

                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload file to S3"
                    )

                return {
                    "message": "File uploaded successfully",
                    "bucket_name": bucket_name,
                    "object_key": object_key,
                    "original_filename": file.filename,
                    "normalized_filename": normalized_filename,
                    "size": len(content),
                }

            finally:
                # Clean up temporary file
                tmp_file_path.unlink(missing_ok=True)

        except HTTPException:
            raise
        except Exception as e:
            self.logger.exception("Failed to upload document", filename=file.filename, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload document"
            ) from e

    def delete_document(self, document_id: int, access_role: str, user_access_roles: list[str]) -> dict[str, str]:
        """Delete a document from both S3 and database.

        Args:
            document_id: ID of the document to delete
            access_role: Access role context for the operation
            user_access_roles: User's access roles for validation

        Returns:
            Dictionary with deletion result

        Raises:
            HTTPException: If document not found, access denied, or deletion fails
        """
        # Validate user has access to the specified role
        if access_role not in user_access_roles:
            self._raise_access_denied(f"Access denied to role: {access_role}")

        try:
            with Session(self.engine) as session:
                # Find document and verify access
                stmt = select(Document).where(Document.id == document_id)
                document = session.execute(stmt).scalar_one_or_none()

                if not document:
                    self._raise_document_not_found()

                # Check if user has access to this document through the specified role
                if access_role not in document.access_roles:
                    self._raise_access_denied("Access denied to this document in the specified role")

                # Extract bucket and object key from document_path
                if not document.document_path.startswith("s3://"):
                    self._raise_invalid_document_path()

                path_parts = document.document_path[5:].split("/", 1)  # Remove s3:// prefix
                if len(path_parts) != 2:
                    self._raise_invalid_document_path()

                bucket_name, object_key = path_parts

                # Delete from S3
                self.s3_utils.delete_s3_object(bucket_name, object_key)

                # Delete from database (this will cascade to document chunks)
                session.delete(document)
                session.commit()

                return {
                    "message": "Document deleted successfully",
                    "document_id": str(document_id),
                    "file_name": document.file_name,
                }

        except HTTPException:
            raise
        except Exception as e:
            self.logger.exception("Failed to delete document", document_id=document_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete document"
            ) from e

    def update_document(
        self, document_id: int, file: UploadFile, access_role: str, user_access_roles: list[str]
    ) -> dict[str, Any]:
        """Update an existing document in S3.

        Args:
            document_id: ID of the document to update
            file: New file content
            access_role: Access role context for the operation
            user_access_roles: User's access roles for validation

        Returns:
            Dictionary with update result information

        Raises:
            HTTPException: If document not found, access denied, or update fails
        """
        # Validate user has access to the specified role
        if access_role not in user_access_roles:
            self._raise_access_denied(f"Access denied to role: {access_role}")

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File name is required")

        try:
            with Session(self.engine) as session:
                # Find document and verify access
                stmt = select(Document).where(Document.id == document_id)
                document = session.execute(stmt).scalar_one_or_none()

                if not document:
                    self._raise_document_not_found()

                # Check if user has access to this document through the specified role
                if access_role not in document.access_roles:
                    self._raise_access_denied("Access denied to this document in the specified role")

                # Extract bucket and object key from document_path
                if not document.document_path.startswith("s3://"):
                    self._raise_invalid_document_path()

                path_parts = document.document_path[5:].split("/", 1)  # Remove s3:// prefix
                if len(path_parts) != 2:
                    self._raise_invalid_document_path()

                bucket_name, object_key = path_parts

                # Create temporary file for upload
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    # Read file content
                    content = file.file.read()
                    tmp_file.write(content)
                    tmp_file_path = Path(tmp_file.name)

                try:
                    # Upload new version to S3 (overwrite existing)
                    success = self.s3_utils.upload_file(tmp_file_path, bucket_name, object_key)

                    if not success:
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update file in S3"
                        )

                    # The document will be reprocessed automatically by the ingestion service
                    # when it detects the updated file

                    return {
                        "message": "Document updated successfully",
                        "document_id": document_id,
                        "file_name": document.file_name,
                        "original_filename": file.filename,
                        "size": len(content),
                    }

                finally:
                    # Clean up temporary file
                    tmp_file_path.unlink(missing_ok=True)

        except HTTPException:
            raise
        except Exception as e:
            self.logger.exception("Failed to update document", document_id=document_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update document"
            ) from e
