from pathlib import Path
from typing import Iterator

from docling.chunking import HybridChunker
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from transformers import AutoTokenizer

from utils.config import get_config
from utils.logging import setup_logger


class DoclingLoader(BaseLoader):
    def __init__(self, file_path: str | list[str], organization: str) -> None:
        config = get_config()
        self.logger = setup_logger()

        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._organization = organization
        accelerator_device = (
            AcceleratorDevice.CUDA if config.DOCLING.USE_GPU else AcceleratorDevice.CPU
        )
        accelerator_options = AcceleratorOptions(
            num_threads=int(config.DOCLING.NUM_THREADS), device=accelerator_device
        )
        self._converter = DocumentConverter(
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
                        table_structure_options=TableStructureOptions(
                            mode=TableFormerMode.ACCURATE
                        ),
                        accelerator_options=accelerator_options,
                    )
                )
            },
        )

        EMBED_MODEL_ID = "jinaai/jina-embeddings-v3"
        MAX_TOKENS = 8192
        tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=MAX_TOKENS,
            merge_peers=True,
        )

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            path_source = Path(source)
            if not path_source.exists():
                self.logger.warning(f"File {source} does not exist")
                continue
            if not path_source.is_file():
                self.logger.warning(f"Path {source} is not a file")
                continue
            if path_source.suffix.lower() not in [
                ".pdf",
                ".docx",
                ".pptx",
                ".html",
                ".xlsx",
            ]:
                self.logger.warning(f"File {source} has an unsupported format")
                continue
            if path_source.is_symlink() or path_source.name.startswith("~"):
                self.logger.warning(f"Skipping symlink {source}")
                continue
            result = self._converter.convert(source, raises_on_error=True)
            if not result.status == ConversionStatus.SUCCESS:
                self.logger.error(
                    f"Failed to convert document {source}. Document skipped."
                )
                continue

            dl_doc = result.document

            if path_source.suffix.lower() == ".pptx":
                for page_no in range(1, dl_doc.num_pages() + 1):
                    content = dl_doc.export_to_markdown(page_no=page_no)
                    if len(content.strip()) == 0:
                        continue
                    meta = dict(
                        organization=self._organization,
                        filename=path_source.name,
                        source=source,
                        mimetype="pptx",
                        page_number=page_no,
                    )
                    yield LCDocument(page_content=content, metadata=meta)

            chunks = self.chunker.chunk(dl_doc)

            for chunk in chunks:
                if len(chunk.text.strip()) == 0:
                    continue
                meta = chunk.meta.model_dump()
                prov = meta["doc_items"][0]["prov"]
                meta = dict(
                    organization=self._organization,
                    filename=meta["origin"]["filename"],
                    source=source,
                    mimetype=meta["origin"]["mimetype"],
                    page_number=prov[0]["page_no"] if len(prov) > 0 else None,
                )
                meta["organization"] = self._organization
                yield LCDocument(page_content=chunk.text, metadata=meta)
