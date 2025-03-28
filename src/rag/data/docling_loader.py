from collections.abc import Iterator
from pathlib import Path
from typing import Any, override

import structlog
from docling.chunking import HybridChunker  # pyright: ignore[reportPrivateImportUsage]
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.base import BaseChunk
from docling_core.types.doc.document import DoclingDocument
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from transformers.models.auto.tokenization_auto import AutoTokenizer

from rag.utils.config import AppConfig, ConfigurationManager


class DoclingLoader(BaseLoader):
    def __init__(self, file_path: str | list[str], organization: str) -> None:
        config: AppConfig = ConfigurationManager.get_config()
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()

        self._file_paths: list[str] = file_path if isinstance(file_path, list) else [file_path]
        self._organization: str = organization
        accelerator_device: AcceleratorDevice = (
            AcceleratorDevice.CUDA if config.DOCLING.USE_GPU else AcceleratorDevice.CPU
        )
        accelerator_options: AcceleratorOptions = AcceleratorOptions(
            num_threads=int(config.DOCLING.NUM_THREADS), device=accelerator_device
        )
        self._converter: DocumentConverter = DocumentConverter(
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

        EMBED_MODEL_ID = "jinaai/jina-embeddings-v3"
        MAX_TOKENS = 8192
        tokenizer: Any = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=EMBED_MODEL_ID)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        self.chunker: HybridChunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=MAX_TOKENS,
            merge_peers=True,
        )

    @override
    def lazy_load(self) -> Iterator[LCDocument]:
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
        """Check if file is valid for processing."""
        if not path_source.exists():
            self.logger.warning(f"File {source} does not exist")
            return False

        if not path_source.is_file():
            self.logger.warning(f"Path {source} is not a file")
            return False

        if path_source.suffix.lower() not in [
            ".pdf",
            ".docx",
            ".pptx",
            ".html",
            ".xlsx",
        ]:
            self.logger.warning(f"File {source} has an unsupported format")
            return False

        if path_source.is_symlink() or path_source.name.startswith("~"):
            self.logger.warning(f"Skipping symlink {source}")
            return False

        return True

    def _process_pptx(self, dl_doc: DoclingDocument, path_source: Path, source: str) -> Iterator[LCDocument]:
        """Process PPTX files."""
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
        """Process documents using chunker."""
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
            yield LCDocument(page_content=chunk.text, metadata=meta_data)
