from typing import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker import HierarchicalChunker
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
)
import logging
from config import get_config


class DoclingLoader(BaseLoader):
    def __init__(self, file_path: str | list[str], organization: str) -> None:
        config = get_config()
        self.logger = logging.getLogger(config.APP_NAME)

        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._organization = organization
        self._converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.PPTX,
                InputFormat.HTML,
                InputFormat.XLSX,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=PdfPipelineOptions(
                        do_table_structure=True,
                        table_structure_options=TableStructureOptions(
                            mode=TableFormerMode.ACCURATE
                        ),
                    )
                )
            },
        )

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            source = Path(source)
            if not source.exists():
                self.logger.warning(f"File {source} does not exist")
                continue
            if not source.is_file():
                self.logger.warning(f"Path {source} is not a file")
                continue
            if source.suffix.lower() not in [".pdf", ".docx", ".pptx", ".html", ".xlsx"]:
                self.logger.warning(f"File {source} has an unsupported format")
                continue
            if source.is_symlink() or source.name.startswith("~"):
                self.logger.warning(f"Skipping symlink {source}")

            dl_doc = self._converter.convert(source).document
            chunks = HierarchicalChunker().chunk(dl_doc)
            for chunk in chunks:
                meta = chunk.meta.model_dump()
                meta["organization"] = self._organization
                yield LCDocument(page_content=chunk.text, metadata=meta)
