import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, override

import structlog
from docling.chunking import HybridChunker  # pyright: ignore[reportPrivateImportUsage]
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from transformers.models.auto.tokenization_auto import AutoTokenizer

from rag.utils.config import AppConfig, ConfigurationManager
from rag.utils.model_clients import get_embedding_client

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_converter(config: AppConfig) -> DocumentConverter:
    """Create a document converter with default settings."""
    accelerator_device: AcceleratorDevice = AcceleratorDevice.CUDA if config.DOCLING.USE_GPU else AcceleratorDevice.CPU
    accelerator_options: AcceleratorOptions = AcceleratorOptions(
        num_threads=int(config.DOCLING.NUM_THREADS), device=accelerator_device
    )
    return DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.HTML,
            # InputFormat.XLSX,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    do_table_structure=True,
                    table_structure_options=TableStructureOptions(mode=TableFormerMode.ACCURATE),
                    accelerator_options=accelerator_options,
                )
            )
        },
    )


class DoclingLoader(BaseLoader):
    """Loader for documents using Docling."""

    SUPPORTED_FORMATS: ClassVar[set[str]] = {".pdf", ".docx", ".pptx", ".html", ".xlsx"}

    def __init__(
        self,
        file_path: str | list[str],
        organization: str,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize the DoclingLoader.

        Args:
            file_path: Path to the file or list of file paths
            organization: Organization name
            logger: Logger instance
        """
        self.logger: structlog.stdlib.BoundLogger = logger or structlog.get_logger()
        self._file_paths: list[str] = file_path if isinstance(file_path, list) else [file_path]
        self._organization: str = organization
        self._config: AppConfig = ConfigurationManager.get_config()

        self._converter: DocumentConverter = create_converter(self._config)
        embedding_model_id: str = get_embedding_client(self._config).model
        max_tokens: int = self._config.DOCLING.MAX_TOKENS

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=embedding_model_id),
            max_tokens=max_tokens,
        )
        self.chunker: HybridChunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,
        )

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

        result: ConversionResult = self._converter.convert(source, raises_on_error=False)
        if result.status != ConversionStatus.SUCCESS:
            self.logger.error(event=f"Failed to convert document {source}. Document skipped.")
            return

        dl_doc: DoclingDocument = result.document

        if path_source.suffix.lower() == ".pptx":
            yield from self._process_pptx(dl_doc, path_source, source)
        else:
            yield from self._process_document(dl_doc, source)

    def _is_valid_file(self, path_source: Path, source: str) -> bool:
        """Check if file is valid for processing.

        Args:
            path_source: Path object of the source file
            source: Source file path as string

        Returns:
            True if the file is valid, False otherwise
        """
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

    def _process_pptx(self, dl_doc: DoclingDocument, path_source: Path, source: str) -> Iterator[LCDocument]:
        """Process PPTX files.

        Args:
            dl_doc: Docling document
            path_source: Path object of the source file
            source: Source file path as string

        Yields:
            LangChain documents
        """
        for page_no in range(1, dl_doc.num_pages() + 1):
            content: str = dl_doc.export_to_markdown(page_no=page_no)
            if len(content.strip()) == 0:
                continue

            meta: dict[str, str | int] = {
                "organization": self._organization,
                "filename": path_source.name,
                "source": source,
                "mimetype": "pptx",
                "page_number": page_no,
            }
            yield LCDocument(page_content=content, metadata=meta)

    def _process_document(self, dl_doc: DoclingDocument, source: str) -> Iterator[LCDocument]:
        """Process documents using chunker.

        Args:
            dl_doc: Docling document
            source: Source file path as string

        Yields:
            LangChain documents
        """
        chunks: Iterator[BaseChunk] = self.chunker.chunk(dl_doc)

        for chunk in chunks:
            if len(chunk.text.strip()) == 0:
                continue

            meta: dict[str, Any] = chunk.meta.model_dump()
            prov: list[dict[str, Any]] = meta["doc_items"][0]["prov"]
            meta_data: dict[str, Any] = {
                "organization": self._organization,
                "filename": meta["origin"]["filename"],
                "source": source,
                "mimetype": meta["origin"]["mimetype"],
                "page_number": prov[0]["page_no"] if len(prov) > 0 else None,
            }
            text = self.chunker.contextualize(chunk=chunk)
            yield LCDocument(page_content=text, metadata=meta_data)
