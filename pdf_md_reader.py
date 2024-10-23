import os
import re
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Union

import pymupdf
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from pymupdf import Document as FitzDocument
from pymupdf4llm import IdentifyHeaders, to_markdown
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean
from unstructured.partition.md import partition_md

from utils import config_loader


class PDFMarkdownReader(BaseLoader, ABC):
    """Read PDF files using PyMuPDF library."""

    def __init__(self, file_path: Union[str, Path]):
        """Initialize with file path."""
        self.file_path = str(file_path)
        if "~" in self.file_path:
            self.file_path = os.path.expanduser(self.file_path)

        self.config = config_loader("conf/conf.yaml")["DOC_STORE"]
        self.file_path = Path(file_path)
        if not isinstance(self.file_path, str) and not isinstance(self.file_path, Path):
            raise TypeError("file_path must be a string or Path.")

    def load(self, extra_info: Optional[Dict] = None) -> List[Document]:
        """Loads list of documents from PDF file and also accepts extra information in dict format.

        Args:
            extra_info (Optional[Dict], optional): A dictionary containing extra information. Defaults to None.

        Returns:
            List[Document]: A list of Document objects.
        """
        if not extra_info:
            extra_info = {}

        if extra_info and not isinstance(extra_info, dict):
            raise TypeError("extra_info must be a dictionary.")

        hdr_info = IdentifyHeaders(self.file_path)

        doc: FitzDocument = pymupdf.open(self.file_path)
        documents = []

        # Process each page separately
        for page_num in range(len(doc)):
            # Convert single page to markdown
            page_text = to_markdown(
                doc,
                hdr_info=hdr_info,
                write_images=False,
                show_progress=False,
                table_strategy="lines",
                pages=[page_num],
            )

            elements = partition_md(text=page_text)
            chunked_elements = chunk_by_title(
                elements,
                max_characters=self.config["MAX_CHUNK_SIZE"],
                multipage_sections=True,
                overlap=self.config["SPLIT_OVERLAP"],
                combine_text_under_n_chars=self.config["MIN_CHUNK_SIZE"],
            )

            for element in chunked_elements:
                metadata = element.metadata.to_dict()
                extra_info = self._process_doc_meta(doc, extra_info)
                metadata.update(extra_info)
                metadata["page_number"] = page_num + 1
                metadata_to_drop = [
                    "file_directory",
                    "filename",
                    "keywords",
                    "link_urls",
                    "link_texts",
                    "emphasized_text_tags",
                    "creator",
                    "emphasized_text_contents",
                    "encryption",
                    "format",
                    "orig_elements",
                    "producer",
                    "subject",
                    "text_as_html",
                    "trapped",
                ]
                for key in metadata_to_drop:
                    if key in metadata:
                        del metadata[key]

                clean_text = clean(element.text, extra_whitespace=True, dashes=True)
                documents.append(Document(page_content=clean_text, metadata=metadata))
        return documents

    def _remove_empty_lines(self, text: str):
        """Cleans newlines from text."""
        return re.sub(r"[\s]*\n[\s]*(\n[\s]*)+", "\n", text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalizes whitespace in text."""
        return " ".join(text.split())

    def _process_doc_meta(
        self,
        doc: FitzDocument,
        extra_info: Optional[Dict] = None,
    ):
        """Processes metas of a PDF document."""
        extra_info.update(doc.metadata)
        extra_info["total_pages"] = len(doc)
        extra_info["source"] = str(self.file_path)

        return extra_info
