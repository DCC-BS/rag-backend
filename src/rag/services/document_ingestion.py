import asyncio
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError
from openai import Client
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, selectinload

from rag.connectors.docling_loader import DoclingAPILoader
from rag.models.db_document import Document, DocumentChunk
from rag.utils.config import AppConfig, ConfigurationManager
from rag.utils.db import get_db_url
from rag.utils.logger import get_logger
from rag.utils.model_clients import get_embedding_client
from rag.utils.s3 import S3DocumentTagger, S3FileClassifier, S3Utils


class S3DocumentIngestionService:
    """Service for ingesting documents from S3/MinIO storage into the database with embeddings."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialize the S3 document ingestion service.

        Args:
            config: Application configuration object
        """
        self.config: AppConfig = config or ConfigurationManager.get_config()
        self.logger = get_logger()

        db_url: str = get_db_url()
        self.engine: Engine = create_engine(url=db_url)

        # Setup embedding client and model
        embedding_client_info = get_embedding_client(self.config)
        self.embedding_client: Client = embedding_client_info.client
        self.embedding_model: str = embedding_client_info.model

        # S3 utilities setup
        self.s3_utils = S3Utils(config)
        self.s3_tagger = S3DocumentTagger(self.s3_utils, config)
        self.file_classifier = S3FileClassifier()

        # Statistics tracking
        self.stats: dict[str, Any] = {
            "new": 0,
            "updated": 0,
            "deleted": 0,
            "orphaned_cleaned": 0,
            "symlinks_removed": 0,
            "temp_files_removed": 0,
            "not_supported": 0,
            "no_chunks": 0,
            "unprocessable": 0,
            "by_role": defaultdict(int),
        }

    def _update_stats(self, stat_key: str, access_role: str | None = None) -> None:
        """Update statistics counters."""
        self.stats[stat_key] += 1
        if access_role:
            self.stats["by_role"][access_role] += 1

    def _reset_stats(self) -> None:
        """Reset all statistics counters."""
        self.stats["new"] = 0
        self.stats["updated"] = 0
        self.stats["deleted"] = 0
        self.stats["orphaned_cleaned"] = 0
        self.stats["symlinks_removed"] = 0
        self.stats["temp_files_removed"] = 0
        self.stats["not_supported"] = 0
        self.stats["no_chunks"] = 0
        self.stats["unprocessable"] = 0
        self.stats["by_role"].clear()

    def check_document_exists(self, document_path: str) -> Document | None:
        """Check if document exists in database."""
        with Session(self.engine) as session:
            statement = select(Document).where(Document.document_path == document_path)
            return session.execute(statement).scalar_one_or_none()

    def need_document_update(self, document: Document, last_modified: datetime) -> bool:
        """Check if document needs to be updated based on S3 last modified time."""
        doc_created_time = document.created_at.replace(tzinfo=UTC)
        return last_modified > doc_created_time

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for a list of texts."""
        try:
            response = self.embedding_client.embeddings.create(input=texts, model=self.embedding_model)
            return [data.embedding for data in response.data]
        except Exception as e:
            self.logger.exception("Failed to create embeddings", error=str(e))
            raise

    def _store_document_and_chunks(
        self,
        session: Session,
        bucket_name: str,
        object_key: str,
        access_role: str,
        chunks: list[Any],
        embeddings: list[list[float]],
    ) -> None:
        """Store a document and its chunks in the database."""
        document_path = self.s3_utils.format_s3_path(bucket_name, object_key)

        # Create document record
        document = Document(
            file_name=Path(object_key).name,
            document_path=document_path,
            mime_type=chunks[0].metadata.get("mimetype", "unknown"),
            num_pages=chunks[0].metadata.get("num_pages"),
            access_roles=[access_role],
        )
        session.add(document)
        session.flush()  # Get the document ID

        # Create document chunks
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            doc_chunk = DocumentChunk(
                document_id=document.id,
                chunk_text=chunk.page_content,
                embedding=embedding,
                page_number=chunk.metadata.get("page_number"),
            )
            session.add(doc_chunk)

        session.commit()
        self.logger.info(f"Successfully stored document {object_key} with {len(chunks)} chunks")

    def _should_process_file(self, bucket_name: str, object_key: str) -> bool:
        """Check if a file should be processed based on type and filters."""
        s3_path = self.s3_utils.format_s3_path(bucket_name, object_key)

        # Check for symlinks and remove them
        if self.file_classifier.is_symlink_file(object_key):
            self.logger.info(f"Removing symlink file: {s3_path}")
            self.s3_utils.delete_object(bucket_name, object_key)
            self._update_stats("symlinks_removed")
            return False

        # Check for temporary MS Office files and remove them
        if self.file_classifier.is_temp_office_file(object_key):
            self.logger.info(f"Removing temporary MS Office file: {s3_path}")
            self.s3_utils.delete_object(bucket_name, object_key)
            self._update_stats("temp_files_removed")
            return False

        # Check if file type is supported
        if not self.file_classifier.is_supported_file_type(object_key, DoclingAPILoader.SUPPORTED_FORMATS):
            self.logger.debug(f"Unsupported file type: {s3_path}")
            # Only tag if not already tagged with an error
            if not self.s3_tagger.has_error_tag(bucket_name, object_key):
                self.s3_tagger.add_error_tag(
                    bucket_name,
                    object_key,
                    self.config.INGESTION.NOT_SUPPORTED_FILE_TAG,
                    "File type not supported by processing pipeline",
                )
                self._update_stats("not_supported")
            return False

        return True

    def process_s3_object(self, bucket_name: str, object_key: str, last_modified: datetime) -> None:
        """Process a single S3 object: check if new/updated and process accordingly."""
        # Check if file should be processed (handles filtering and cleanup)
        if not self._should_process_file(bucket_name, object_key):
            return

        s3_path = self.s3_utils.format_s3_path(bucket_name, object_key)

        # Extract access role from bucket name
        access_role = self.s3_utils.extract_access_role_from_bucket(bucket_name)
        if not access_role:
            return

        document_path = self.s3_utils.format_s3_path(bucket_name, object_key)
        existing_document = self.check_document_exists(document_path)

        # Check if already processed or has error tags (unless we have a database record)
        if not existing_document:
            if self.s3_tagger.is_document_processed(bucket_name, object_key):
                self.logger.debug(f"Document {s3_path} already processed")
                return
            if self.s3_tagger.has_error_tag(bucket_name, object_key):
                self.logger.debug(f"Document {s3_path} already has error tag")
                return

        if existing_document:
            if self.need_document_update(existing_document, last_modified):
                self._handle_document_update(bucket_name, object_key, access_role)
            else:
                self.logger.debug(f"Document {s3_path} is up to date")
        else:
            self._handle_new_document(bucket_name, object_key, access_role)

    def scan_bucket(self, bucket_name: str) -> None:
        """Scan a bucket for documents and process them."""
        self.logger.info(f"Scanning S3 bucket: {bucket_name}")

        try:
            paginator = self.s3_utils.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=bucket_name)

            objects_to_process = []
            for page in page_iterator:
                for obj in page.get("Contents", []):
                    objects_to_process.append({"key": obj["Key"], "last_modified": obj["LastModified"]})

            total_objects = len(objects_to_process)
            self.logger.info(f"Found {total_objects} objects in bucket {bucket_name}")

            for i, obj_info in enumerate(objects_to_process):
                self.logger.info(f"Processing object {i + 1}/{total_objects}: {obj_info["key"]}")
                self.process_s3_object(bucket_name, obj_info["key"], obj_info["last_modified"])

        except ClientError as e:
            self.logger.exception(f"Failed to scan bucket {bucket_name}", error=str(e))
            raise

    def log_statistics(self) -> None:
        """Log the collected statistics."""
        # Convert defaultdict to dict for cleaner logging
        stats_to_log = self.stats.copy()
        stats_to_log["by_role"] = dict(self.stats["by_role"])
        self.logger.info(
            "Ingestion statistics",
            new_documents=stats_to_log["new"],
            updated_documents=stats_to_log["updated"],
            deleted_documents=stats_to_log["deleted"],
            orphaned_cleaned=stats_to_log["orphaned_cleaned"],
            symlinks_removed=stats_to_log["symlinks_removed"],
            temp_files_removed=stats_to_log["temp_files_removed"],
            not_supported=stats_to_log["not_supported"],
            no_chunks=stats_to_log["no_chunks"],
            unprocessable=stats_to_log["unprocessable"],
            roles=stats_to_log["by_role"],
        )

    def initial_scan(self) -> None:
        """Perform initial scan of all S3 buckets for configured access roles.

        Args:
            cleanup_orphaned: Whether to also clean up orphaned documents during the scan
        """
        self.logger.info("Starting initial S3 document scan")

        # Reset stats for the scan
        self._reset_stats()

        self.logger.info("Running orphaned document cleanup before scanning")
        cleanup_stats = self.cleanup_orphaned_documents()
        self.stats["orphaned_cleaned"] = cleanup_stats["deleted_documents"]

        # Ensure buckets exist
        self.s3_utils.ensure_buckets_exist_for_roles()

        # Scan each bucket
        for role in self.config.ROLES:
            bucket_name = self.s3_utils.get_bucket_name(role)
            self.scan_bucket(bucket_name)

        self.log_statistics()
        self.logger.info("Initial S3 document scan completed")

    async def watch_s3_changes(self) -> None:
        """Watch for S3 changes by periodically scanning buckets."""
        self.logger.info("Starting S3 change monitoring")

        while True:
            try:
                for role in self.config.ROLES:
                    bucket_name = self.s3_utils.get_bucket_name(role)

                    # Simple approach: scan for unprocessed objects
                    paginator = self.s3_utils.s3_client.get_paginator("list_objects_v2")
                    page_iterator = paginator.paginate(Bucket=bucket_name)

                    for page in page_iterator:
                        for obj in page.get("Contents", []):
                            # Check if object is unprocessed and doesn't have error tags
                            if not self.s3_tagger.is_document_processed(
                                bucket_name, obj["Key"]
                            ) and not self.s3_tagger.has_error_tag(bucket_name, obj["Key"]):
                                s3_path = self.s3_utils.format_s3_path(bucket_name, obj["Key"])
                                self.logger.info(f"Found unprocessed object: {s3_path}")
                                self.process_s3_object(bucket_name, obj["Key"], obj["LastModified"])

                # Wait before next scan
                await asyncio.sleep(self.config.INGESTION.SCAN_INTERVAL)

            except Exception as e:
                self.logger.exception("Error during S3 change monitoring", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying

    async def run(self) -> None:
        """Run the S3 ingestion service."""
        self.logger.info("Starting S3 document ingestion service")

        try:
            # Perform initial scan
            self.initial_scan()

            # Start change monitoring if enabled
            if self.config.INGESTION.WATCH_ENABLED:
                self.logger.info("S3 document ingestion service is running. Press Ctrl+C to stop.")
                try:
                    await self.watch_s3_changes()
                except KeyboardInterrupt:
                    self.logger.info("Received shutdown signal")
            else:
                self.logger.info("Watch mode disabled, exiting after initial scan")

        except Exception as e:
            self.logger.exception("S3 document ingestion service failed", error=str(e))
            raise

    def process_s3_document(self, bucket_name: str, object_key: str, access_role: str) -> None:
        """Process a single S3 document: download, extract text, create chunks, embeddings, and store in DB."""
        s3_path = self.s3_utils.format_s3_path(bucket_name, object_key)
        try:
            self.logger.debug(f"Starting processing for S3 document: {s3_path}")

            # Download the file content
            file_content = self.s3_utils.download_object(bucket_name, object_key)

            # Create a temporary file for docling processing
            with tempfile.NamedTemporaryFile(suffix=Path(object_key).suffix, delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name

            try:
                # Process with docling
                loader = DoclingAPILoader(file_path=tmp_file_path, organization=access_role)
                chunks = list(loader.lazy_load())

                if not chunks:
                    self.logger.warning(f"No chunks extracted from {s3_path}")
                    self.s3_tagger.add_error_tag(
                        bucket_name,
                        object_key,
                        self.config.INGESTION.NO_CHUNKS_TAG,
                        "No chunks could be extracted from document",
                    )
                    self._update_stats("no_chunks")
                    return

                self.logger.info(f"Extracted {len(chunks)} chunks from {s3_path}")

                # Create embeddings
                chunk_texts = [chunk.page_content for chunk in chunks]
                embeddings = self.create_embeddings(chunk_texts)

                # Store in database
                with Session(self.engine) as session:
                    self._store_document_and_chunks(session, bucket_name, object_key, access_role, chunks, embeddings)

                # Mark as processed
                self.s3_tagger.mark_document_processed(bucket_name, object_key)

            finally:
                # Clean up temporary file
                Path(tmp_file_path).unlink(missing_ok=True)

        except Exception as e:
            self.logger.exception(f"Failed to process S3 document {s3_path}", error=str(e))
            # Tag as unprocessable for any other errors
            self.s3_tagger.add_error_tag(bucket_name, object_key, self.config.INGESTION.UNPROCESSABLE_TAG, str(e))
            self._update_stats("unprocessable")
            raise

    def delete_document(self, document_path: str) -> None:
        """Delete a document and all its chunks from the database."""
        try:
            with Session(self.engine) as session:
                # Find the document
                statement = (
                    select(Document)
                    .options(selectinload(Document.chunks))
                    .where(Document.document_path == document_path)
                )
                document = session.execute(statement).scalar_one_or_none()

                if document:
                    # Delete document (chunks will be deleted automatically due to cascade)
                    session.delete(document)
                    session.commit()
                    self.logger.info(f"Deleted document {document_path} from database")
                else:
                    self.logger.warning(f"Document {document_path} not found in database")

        except Exception as e:
            self.logger.exception(f"Failed to delete document {document_path}", error=str(e))
            raise

    def _handle_document_update(self, bucket_name: str, object_key: str, access_role: str) -> None:
        """Handle updating an existing document."""
        document_path = self.s3_utils.format_s3_path(bucket_name, object_key)
        self.logger.info(f"Document {document_path} needs update")

        # Remove processed tag and delete from database
        self.s3_tagger.remove_processed_tag(bucket_name, object_key)
        self.delete_document(document_path)

        # Reprocess the document
        self.process_s3_document(bucket_name, object_key, access_role)
        self._update_stats("updated", access_role)

    def _handle_new_document(self, bucket_name: str, object_key: str, access_role: str) -> None:
        """Handle processing a new document."""
        s3_path = self.s3_utils.format_s3_path(bucket_name, object_key)
        self.logger.info(f"New document found: {s3_path}")
        self.process_s3_document(bucket_name, object_key, access_role)
        self._update_stats("new", access_role)

    def _is_document_orphaned(self, document: Document, cleanup_stats: dict[str, int]) -> bool:
        """Check if a single document is orphaned (S3 object no longer exists).

        Args:
            document: Document to check
            cleanup_stats: Statistics dictionary to update

        Returns:
            True if document is orphaned, False otherwise
        """
        # Parse document path
        parsed_path = self.s3_utils.parse_document_path(document.document_path)
        if not parsed_path:
            cleanup_stats["invalid_paths"] += 1
            return True

        bucket_name, object_key = parsed_path

        try:
            # Check if S3 object exists
            if not self.s3_utils.object_exists(bucket_name, object_key):
                self.logger.info(
                    f"Document orphaned - S3 object not found: {document.document_path}",
                    document_id=document.id,
                    file_name=document.file_name,
                )
                return True
            else:
                self.logger.debug(f"Document valid - S3 object exists: {document.document_path}")
                return False

        except Exception as e:
            self.logger.warning(
                f"Error checking S3 object existence for {document.document_path}",
                error=str(e),
                document_id=document.id,
            )
            cleanup_stats["s3_check_errors"] += 1
            return False

    def _delete_orphaned_documents(
        self, session: Session, orphaned_documents: list[Document], cleanup_stats: dict[str, int]
    ) -> None:
        """Delete a list of orphaned documents from the database.

        Args:
            session: Database session
            orphaned_documents: List of orphaned documents to delete
            cleanup_stats: Statistics dictionary to update
        """
        self.logger.info(f"Deleting {len(orphaned_documents)} orphaned documents")

        for document in orphaned_documents:
            try:
                # Delete document (chunks will be deleted automatically due to cascade)
                session.delete(document)
                cleanup_stats["deleted_documents"] += 1
                self.logger.info(
                    f"Deleted orphaned document: {document.document_path}",
                    document_id=document.id,
                    file_name=document.file_name,
                )
            except Exception as e:
                self.logger.exception(
                    f"Failed to delete orphaned document {document.id}",
                    document_path=document.document_path,
                    error=str(e),
                )

        # Commit all deletions
        session.commit()
        self.logger.info(f"Successfully deleted {cleanup_stats["deleted_documents"]} orphaned documents")

    def cleanup_orphaned_documents(self) -> dict[str, int]:
        """Check all documents in database and remove those that no longer exist in S3.

        Returns:
            Dictionary with cleanup statistics
        """
        self.logger.info("Starting cleanup of orphaned documents")

        cleanup_stats = {
            "total_documents": 0,
            "orphaned_documents": 0,
            "invalid_paths": 0,
            "s3_check_errors": 0,
            "deleted_documents": 0,
        }

        try:
            with Session(self.engine) as session:
                # Get all documents from database
                statement = select(Document)
                documents = session.execute(statement).scalars().all()

                cleanup_stats["total_documents"] = len(documents)
                self.logger.info(f"Found {len(documents)} documents in database to validate")

                # Find orphaned documents
                orphaned_documents = []
                for document in documents:
                    if self._is_document_orphaned(document, cleanup_stats):
                        orphaned_documents.append(document)
                        cleanup_stats["orphaned_documents"] += 1

                # Delete orphaned documents if any found
                if orphaned_documents:
                    self._delete_orphaned_documents(session, orphaned_documents, cleanup_stats)
                else:
                    self.logger.info("No orphaned documents found")

        except Exception as e:
            self.logger.exception("Failed to cleanup orphaned documents", error=str(e))
            raise

        # Log final statistics
        self.logger.info(
            "Orphaned document cleanup completed",
            total_documents=cleanup_stats["total_documents"],
            orphaned_documents=cleanup_stats["orphaned_documents"],
            invalid_paths=cleanup_stats["invalid_paths"],
            s3_check_errors=cleanup_stats["s3_check_errors"],
            deleted_documents=cleanup_stats["deleted_documents"],
        )

        return cleanup_stats
