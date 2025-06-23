import asyncio
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, override

import structlog
from openai import Client
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, selectinload
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rag.connectors.docling_loader import DoclingLoader
from rag.models.document import Document, DocumentChunk
from rag.utils.config import AppConfig, ConfigurationManager
from rag.utils.db import get_db_url


class DocumentIngestionService:
    """Service for ingesting documents into the database with embeddings."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialize the document ingestion service.

        Args:
            config: Application configuration object
        """
        self.config: AppConfig = config or ConfigurationManager.get_config()
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()

        db_url: str = get_db_url()
        self.engine: Engine = create_engine(url=db_url)

        self.embedding_client: Client = Client(
            base_url=self.config.EMBEDDINGS.API_URL, api_key=os.getenv("OPENAI_API_KEY", "none")
        )
        model_data = self.embedding_client.models.list().data
        self.embedding_model: str = (model_data[0].id if model_data else None) or "Qwen/Qwen3-Embedding-0.6B"

        self.observer = Observer()
        self.event_handler = None  # Will be initialized later
        self.stats: dict[str, Any] = {
            "new": 0,
            "updated": 0,
            "deleted": 0,
            "by_role": defaultdict(int),
        }

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA256 hash string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def get_access_role_from_path(self, file_path: Path, data_dir: Path) -> str:
        """Extract access role from file path based on directory structure.

        Args:
            file_path: Path to the file
            data_dir: Base data directory path

        Returns:
            Access role string
        """
        try:
            relative_path = file_path.relative_to(data_dir)
            # First directory level determines the access role
            return relative_path.parts[0]
        except ValueError:
            self.logger.warning(f"File {file_path} is not under data directory {data_dir}")
            return "unknown"

    def check_document_exists(self, document_path: str) -> Document | None:
        """Check if document exists in database.

        Args:
            document_path: Path to the document

        Returns:
            Document object if exists, None otherwise
        """
        with Session(self.engine) as session:
            statement = select(Document).where(Document.document_path == document_path)
            return session.execute(statement).scalar_one_or_none()

    def need_document_update(self, document: Document, file_path: Path) -> bool:
        """Check if document needs to be updated based on file modification.

        Args:
            document: Existing document object
            file_path: Path to the file

        Returns:
            True if document needs update, False otherwise
        """
        file_modified_time = file_path.stat().st_mtime
        doc_created_time = document.created_at.timestamp()
        return file_modified_time > doc_created_time

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        try:
            response = self.embedding_client.embeddings.create(input=texts, model=self.embedding_model)
            return [data.embedding for data in response.data]
        except Exception as e:
            self.logger.exception("Failed to create embeddings", error=str(e))
            raise

    def process_document(self, file_path: Path, access_role: str) -> None:
        """Process a single document: extract text, create chunks, embeddings, and store in DB.

        Args:
            file_path: Path to the document file
            access_role: Access role for the document
        """
        try:
            self.logger.info(f"Processing document: {file_path}")

            # Load and process document
            loader = DoclingLoader(file_path=str(file_path), organization=access_role)

            # Get all chunks from the document
            chunks = list(loader.lazy_load())
            if not chunks:
                self.logger.warning(f"No chunks extracted from {file_path}")
                return

            self.logger.info(f"Extracted {len(chunks)} chunks from {file_path}")

            # Create embeddings for all chunks
            chunk_texts = [chunk.page_content for chunk in chunks]
            embeddings = self.create_embeddings(chunk_texts)

            # Store in database
            with Session(self.engine) as session:
                # Create document record
                document = Document(
                    file_name=file_path.name,
                    document_path=str(file_path),
                    mime_type=chunks[0].metadata.get("mimetype", "unknown"),
                    num_pages=len({
                        chunk.metadata.get("page_number") for chunk in chunks if chunk.metadata.get("page_number")
                    }),
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
                self.logger.info(f"Successfully stored document {file_path} with {len(chunks)} chunks")

        except Exception as e:
            self.logger.exception(f"Failed to process document {file_path}", error=str(e))
            raise

    def delete_document(self, document_path: str) -> None:
        """Delete a document and all its chunks from the database.

        Args:
            document_path: Path to the document
        """
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

    def process_file(self, file_path: Path) -> None:
        """Process a single file: check if new/updated and process accordingly.

        Args:
            file_path: Path to the file to process
        """
        if not file_path.exists() or not file_path.is_file():
            return

        # Skip unsupported file types
        if file_path.suffix.lower() not in DoclingLoader.SUPPORTED_FORMATS:
            return

        data_dir = Path(self.config.INGESTION.DATA_DIR)
        access_role = self.get_access_role_from_path(file_path, data_dir)
        document_path = str(file_path)

        # Check if document exists
        existing_document = self.check_document_exists(document_path)

        if existing_document:
            # Check if document needs update
            if self.need_document_update(existing_document, file_path):
                self.logger.info(f"Document {file_path} needs update")
                self.delete_document(document_path)
                self.process_document(file_path, access_role)
                self.stats["updated"] += 1
                self.stats["by_role"][access_role] += 1
            else:
                self.logger.debug(f"Document {file_path} is up to date")
        else:
            # New document
            self.logger.info(f"New document found: {file_path}")
            self.process_document(file_path, access_role)
            self.stats["new"] += 1
            self.stats["by_role"][access_role] += 1

    def scan_directory(self, directory: Path) -> None:
        """Scan a directory for documents and process them.

        Args:
            directory: Directory to scan
        """
        self.logger.info(f"Scanning directory: {directory}")

        if not directory.exists():
            self.logger.warning(f"Directory {directory} does not exist")
            return

        # Find all supported files
        files_to_process = [
            file_path
            for file_path in directory.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in DoclingLoader.SUPPORTED_FORMATS
        ]
        total_files = len(files_to_process)
        self.logger.info(f"Found {total_files} documents to process.")

        for i, file_path in enumerate(files_to_process):
            self.logger.info(f"Processing document {i + 1}/{total_files}: {file_path}")
            self.process_file(file_path)

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
            roles=stats_to_log["by_role"],
        )

    def initial_scan(self) -> None:
        """Perform initial scan of all configured data directories."""
        self.logger.info("Starting initial document scan")
        # Reset stats for the scan
        self.stats["new"] = 0
        self.stats["updated"] = 0
        self.stats["by_role"].clear()

        data_dir = Path(self.config.INGESTION.DATA_DIR)
        if not data_dir.exists():
            self.logger.error(f"Data directory {data_dir} does not exist")
            return

        self.scan_directory(data_dir)
        self.log_statistics()
        self.logger.info("Initial document scan completed")

    def start_file_watcher(self) -> None:
        """Start the file system watcher."""
        data_dir = Path(self.config.INGESTION.DATA_DIR)
        if not data_dir.exists():
            self.logger.error(f"Data directory {data_dir} does not exist")
            return

        # Initialize event handler if not already done
        if self.event_handler is None:
            self.event_handler = DocumentFileHandler(self)

        self.observer.schedule(self.event_handler, str(data_dir), recursive=True)
        self.observer.start()
        self.logger.info(f"Started file watcher for {data_dir}")

    def stop_file_watcher(self) -> None:
        """Stop the file system watcher."""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            self.logger.info("Stopped file watcher")

    async def run(self) -> None:
        """Run the ingestion service."""
        self.logger.info("Starting document ingestion service")

        try:
            # Perform initial scan
            self.initial_scan()

            # Start file watcher
            self.start_file_watcher()

            # Keep the service running
            self.logger.info("Document ingestion service is running. Press Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Received shutdown signal")
            finally:
                self.stop_file_watcher()

        except Exception as e:
            self.logger.exception("Document ingestion service failed", error=str(e))
            raise


class DocumentFileHandler(FileSystemEventHandler):
    """File system event handler for document changes."""

    def __init__(self, ingestion_service: DocumentIngestionService) -> None:
        """Initialize the file handler.

        Args:
            ingestion_service: Reference to the ingestion service
        """
        self.ingestion_service = ingestion_service
        self.logger = structlog.get_logger()
        super().__init__()

    @override
    def on_created(self, event: Any) -> None:
        """Handle file creation events."""
        if not event.is_directory:
            file_path = Path(event.src_path)
            self.logger.info(f"File created: {file_path}")
            self.ingestion_service.process_file(file_path)

    @override
    def on_modified(self, event: Any) -> None:
        """Handle file modification events."""
        if not event.is_directory:
            file_path = Path(event.src_path)
            self.logger.info(f"File modified: {file_path}")
            self.ingestion_service.process_file(file_path)

    @override
    def on_deleted(self, event: Any) -> None:
        """Handle file deletion events."""
        if not event.is_directory:
            file_path = event.src_path
            self.logger.info(f"File deleted: {file_path}")
            self.ingestion_service.delete_document(file_path)
            self.ingestion_service.stats["deleted"] += 1
            self.ingestion_service.log_statistics()
