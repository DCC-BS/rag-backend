import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, override

import httpx
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from structlog.stdlib import BoundLogger

from rag.utils.config import AppConfig, ConfigurationManager
from rag.utils.logger import get_logger
from rag.utils.model_clients import EmbeddingClient, get_embedding_client


@dataclass
class PageContent:
    """Represents a page with its number and content."""

    page_number: int
    content: str


@dataclass
class PageChunk:
    """Represents a chunk of content with its associated page number."""

    page_number: int
    content: str


@dataclass
class FinalChunk:
    """Represents a final chunk with page range information and content."""

    page_info: int
    content: str


class DoclingAPILoader(BaseLoader):
    """Loader for documents using Docling Serve API."""

    SUPPORTED_FORMATS: ClassVar[set[str]] = {".pdf", ".docx", ".pptx", ".html", ".adoc", ".md", ".xlsx", ".csv"}

    def __init__(
        self,
        file_path: str | list[str],
        organization: str,
        logger: BoundLogger | None = None,
    ) -> None:
        """Initialize the DoclingAPILoader.

        Args:
            file_path: Path to the file or list of file paths
            organization: Organization name
            logger: Logger instance
        """
        self.logger = logger or get_logger()
        self._file_paths: list[str] = file_path if isinstance(file_path, list) else [file_path]
        self._organization: str = organization
        self._config: AppConfig = ConfigurationManager.get_config()
        self._embedding_client: EmbeddingClient = get_embedding_client(self._config)

    @override
    def lazy_load(self) -> Iterator[LCDocument]:
        """Lazy load documents from files."""
        for source in self._file_paths:
            yield from self._process_file(source)

    def _process_file(self, source: str) -> Iterator[LCDocument]:
        """Process a single file and yield documents."""
        path_source = Path(source)

        if not self._is_valid_file(path_source, source):
            return

        # Convert document using API
        markdown_content = self._convert_document_via_api(source)
        if not markdown_content:
            return

        # Split by page breaks and process
        yield from self._process_markdown_content(markdown_content, path_source, source)

    def _is_valid_file(self, path_source: Path, source: str) -> bool:
        """Check if file is valid for processing."""
        if not path_source.exists():
            self.logger.warning(f"File {source} does not exist")
            return False

        if not path_source.is_file():
            self.logger.warning(f"Path {source} is not a file")
            return False

        if path_source.suffix.lower() not in self.SUPPORTED_FORMATS:
            self.logger.warning(f"File {source} has an unsupported format")
            return False

        if path_source.is_symlink() or path_source.name.startswith("~"):
            self.logger.warning(f"Skipping symlink {source}")
            return False

        return True

    def _convert_document_via_api(self, source: str) -> str | None:
        """Convert document to markdown using Docling Serve API."""
        page_break_placeholder = self._config.DOCLING.PAGE_BREAK_PLACEHOLDER
        try:
            with open(source, "rb") as file:
                file_mime_type = self._get_mimetype(Path(source))
                files = {"files": (Path(source).name, file, file_mime_type)}

                parameters = {
                    "to_formats": ["md"],
                    "image_export_mode": "placeholder",
                    "do_ocr": True,
                    "force_ocr": False,
                    "table_mode": "accurate",
                    "abort_on_error": False,
                    "md_page_break_placeholder": page_break_placeholder,
                    "ocr_lang": ["de", "en", "fr"],
                    "pdf_backend": "pypdfium2",
                }

                with httpx.Client(timeout=300.0) as client:
                    response = client.post(
                        f"{self._config.DOCLING.API_URL}/v1/convert/file",
                        files=files,
                        data=parameters,
                    )

                if response.status_code != 200:
                    self.logger.error(f"Failed to convert document {source}: {response.status_code} - {response.text}")
                    return None

                result: dict[str, str | dict[str, str]] = response.json()
                if result.get("status") == "success" and "document" in result:
                    document: str | dict[str, str] = result["document"]
                    if isinstance(document, dict):
                        return document.get("md_content", "")
                    else:
                        self.logger.error(f"Failed to convert document {source}: {document}")
                        return None
                else:
                    self.logger.error(
                        f"Failed to convert document {source}: {result.get("status")}. \n Errors: {result.get("errors")}"
                    )
                    return None

        except Exception:
            self.logger.exception(f"Error converting document {source} via API.")
            return None

    def _get_token_count(self, text: str) -> int:
        """Get token count for text using LLM API."""
        try:
            embedding_base_url: str = self._config.EMBEDDINGS.API_URL.rstrip("/v1")
            model: str = self._embedding_client.model
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{embedding_base_url}/tokenize",
                    json={"prompt": text, "model": model},
                    headers={"Authorization": f"Bearer {os.getenv("OPENAI_API_KEY", "none")}"},
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("count", 0)
                else:
                    self.logger.warning(f"Tokenization failed: {response.status_code}")
                    # Fallback: rough estimation (1 token ≈ 4 characters)
                    return len(text) // 4

        except Exception as e:
            self.logger.warning(f"Error tokenizing text: {e}")
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4

    def _process_markdown_content(self, markdown_content: str, path_source: Path, source: str) -> Iterator[LCDocument]:
        """Process markdown content by splitting into pages and chunking."""
        page_break_placeholder = self._config.DOCLING.PAGE_BREAK_PLACEHOLDER

        # Split by page breaks and track page numbers
        pages = markdown_content.split(page_break_placeholder)

        # Process PPTX files page by page
        if path_source.suffix.lower() == ".pptx":
            yield from self._process_pptx_pages(pages, path_source, source)
        else:
            # For other formats, chunk hierarchically across pages
            yield from self._process_document_pages(pages, path_source, source)

    def _process_pptx_pages(self, pages: list[str], path_source: Path, source: str) -> Iterator[LCDocument]:
        """Process PPTX files page by page."""
        for page_no, page_content in enumerate(pages, 1):
            content = page_content.strip()
            if not content:
                continue

            meta: dict[str, str | int] = {
                "organization": self._organization,
                "filename": path_source.name,
                "source": source,
                "mimetype": "pptx",
                "page_number": page_no,
                "num_pages": len(pages),
            }
            yield LCDocument(page_content=content, metadata=meta)

    def _process_document_pages(self, pages: list[str], path_source: Path, source: str) -> Iterator[LCDocument]:
        """Process document pages with hierarchical header-based chunking."""
        # Filter out empty pages and track their numbers
        non_empty_pages: list[PageContent] = []
        for page_no, page_content in enumerate(pages, 1):
            if page_content.strip():
                non_empty_pages.append(PageContent(page_number=page_no, content=page_content))

        if not non_empty_pages:
            return

        # Process pages individually or combine adjacent pages if needed
        yield from self._chunk_pages_with_tracking(non_empty_pages, path_source, source, len(pages))

    def _chunk_pages_with_tracking(
        self, pages: list[PageContent], path_source: Path, source: str, total_pages: int
    ) -> Iterator[LCDocument]:
        """Chunk pages while tracking page numbers directly."""
        # First pass: chunk each page individually and collect with page numbers
        page_chunks: list[PageChunk] = []

        for page in pages:
            chunks = self._chunk_content_hierarchically(page.content.strip())
            for chunk in chunks:
                if chunk.strip():
                    page_chunks.append(PageChunk(page_number=page.page_number, content=chunk.strip()))

        # Second pass: combine small adjacent chunks if possible (within 2-page limit)
        final_chunks: list[FinalChunk] = []

        i = 0
        while i < len(page_chunks):
            current_chunk = page_chunks[i]
            current_tokens = self._get_token_count(current_chunk.content)

            has_next_chunk = i + 1 < len(page_chunks)

            # If chunk is too small, try to combine with next chunk
            if current_tokens < self._config.DOCLING.MIN_TOKENS and has_next_chunk:
                next_chunk = page_chunks[i + 1]
                combined_content = current_chunk.content + "\n\n" + next_chunk.content
                combined_tokens = self._get_token_count(combined_content)

                # Check if we can combine (within token limit and max 2 pages)
                page_difference = abs(next_chunk.page_number - current_chunk.page_number)
                if combined_tokens <= self._config.DOCLING.MAX_TOKENS and page_difference <= 1:
                    # Determine page info
                    page_info = current_chunk.page_number

                    final_chunks.append(FinalChunk(page_info=page_info, content=combined_content))
                    i += 2  # Skip next chunk since we combined it
                else:
                    final_chunks.append(FinalChunk(page_info=current_chunk.page_number, content=current_chunk.content))
                    i += 1
            else:
                final_chunks.append(FinalChunk(page_info=current_chunk.page_number, content=current_chunk.content))
                i += 1

        # Yield final chunks as documents
        for final_chunk in final_chunks:
            chunk_tokens = self._get_token_count(final_chunk.content)

            # Validate chunk size
            if chunk_tokens > self._config.DOCLING.MAX_TOKENS:
                raise ValueError(
                    f"Chunk has {chunk_tokens} tokens, which is greater than max_tokens {self._config.DOCLING.MAX_TOKENS}"
                )

            meta: dict[str, Any] = {
                "organization": self._organization,
                "filename": path_source.name,
                "source": source,
                "mimetype": self._get_mimetype(path_source),
                "page_number": final_chunk.page_info,
                "num_pages": total_pages,
            }

            yield LCDocument(page_content=final_chunk.content, metadata=meta)

    def _chunk_content_hierarchically(self, content: str) -> list[str]:
        """Chunk content hierarchically by markdown headers using pattern matching."""
        max_tokens = self._config.DOCLING.MAX_TOKENS
        return self._split_content_by_headers(content, max_tokens, header_level=1)

    def _split_content_by_headers(self, content: str, max_tokens: int, header_level: int) -> list[str]:
        """Split content by headers of specified level, recursively handling oversized chunks."""
        # Split by current header level
        chunks: list[str] = self._split_by_header_pattern(content, header_level)

        # If no split occurred and content is too large, try next level or fallback
        if len(chunks) == 1 and self._get_token_count(content) > max_tokens:
            if header_level < 3:
                return self._split_content_by_headers(content, max_tokens, header_level=header_level + 1)
            else:
                return self._split_by_lines(content, max_tokens)

        # Process each chunk recursively if needed
        result_chunks: list[str] = []
        for chunk in chunks:
            token_count = self._get_token_count(text=chunk)
            if token_count <= max_tokens:
                result_chunks.append(chunk)
            else:
                # Recursively split oversized chunk
                result_chunks.extend(self._split_content_by_headers(chunk, max_tokens, header_level + 1))

        # Combine consecutive small chunks
        min_tokens = self._config.DOCLING.MIN_TOKENS
        return self._combine_small_chunks(result_chunks, max_tokens, min_tokens)

    def _split_by_header_pattern(self, content: str, level: int) -> list[str]:
        """Split content by markdown headers of specified level."""
        header_pattern = rf"^{"#" * level}\s+(.+)$"
        lines: list[str] = content.split("\n")
        chunks: list[str] = []
        current_chunk: list[str] = []

        for line in lines:
            if re.match(header_pattern, line, re.MULTILINE) and current_chunk:
                # Start new chunk
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        # If no headers found, return original content
        return chunks if len(chunks) > 1 else [content]

    def _split_by_lines(self, content: str, max_tokens: int) -> list[str]:
        """Split content by lines when headers don't work."""
        lines: list[str] = content.split("\n")
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens: int = 0

        for line in lines:
            line_tokens: int = self._get_token_count(line)
            if line_tokens > max_tokens:
                # Single line is too long, do sliding window on the line
                if len(current_chunk) > 0:
                    chunks.append("\n".join(current_chunk))
                    current_chunk: list[str] = []
                    current_tokens = 0

                for start_pos in range(0, len(line), max_tokens):
                    end_pos: int = start_pos + max_tokens if start_pos + max_tokens < len(line) else len(line)
                    chunk: str = line[start_pos:end_pos]
                    chunks.append(chunk)
            elif current_tokens + line_tokens > max_tokens and current_chunk:
                # Add current chunk to final chunks and reset current chunk and tokens
                chunks.append("\n".join(current_chunk))
                # Reset current chunk and tokens
                current_chunk: list[str] = [line]
                current_tokens = line_tokens
            else:
                # Add line to current chunk
                current_chunk.append(line)
                current_tokens += line_tokens

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _combine_small_chunks(self, chunks: list[str], max_tokens: int, min_tokens: int) -> list[str]:
        """Combine consecutive small chunks that fit within max_tokens."""
        if not chunks:
            return chunks

        combined_chunks: list[str] = []
        current_combined: str = chunks[0]
        current_tokens: int = self._get_token_count(current_combined)

        for i in range(1, len(chunks)):
            chunk = chunks[i]
            chunk_tokens: int = self._get_token_count(chunk)
            if chunk_tokens > max_tokens:
                raise ValueError(f"Chunk has {chunk_tokens} tokens, which is greater than max_tokens {max_tokens}")

            if current_tokens < min_tokens and current_tokens + chunk_tokens < max_tokens:
                current_combined += "\n\n" + chunk
                current_tokens += chunk_tokens
            else:
                combined_chunks.append(current_combined)
                current_combined = chunk
                current_tokens = chunk_tokens

        combined_chunks.append(current_combined)
        return combined_chunks

    def _get_mimetype(self, path_source: Path) -> str:
        """Get MIME type based on file extension."""
        extension = path_source.suffix.lower()
        mimetypes = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".html": "text/html",
            ".adoc": "text/asciidoc",
            ".md": "text/markdown",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".csv": "text/csv",
        }
        return mimetypes.get(extension, "application/octet-stream")
