import os
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pymupdf
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from pymupdf import Document as FitzDocument
from pymupdf4llm import IdentifyHeaders, to_markdown


class PDFMarkdownReader(BaseLoader, ABC):
    """Read PDF files using PyMuPDF library."""

    def __init__(self, file_path: Union[str, Path]):
        """Initialize with file path."""
        self.file_path = str(file_path)
        if "~" in self.file_path:
            self.file_path = os.path.expanduser(self.file_path)

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

        extra_info = self._process_doc_meta(doc, self.file_path, extra_info)

        text = to_markdown(
            doc,
            hdr_info=hdr_info,
            write_images=False,
            show_progress=False,
            table_strategy="lines",
        )
        return [Document(page_content=text, metadata=extra_info)]

    def _process_doc_meta(
        self,
        doc: FitzDocument,
        file_path: Union[Path, str],
        extra_info: Optional[Dict] = None,
    ):
        """Processes metas of a PDF document."""
        extra_info.update(doc.metadata)
        extra_info["total_pages"] = len(doc)
        extra_info["source"] = str(file_path)

        return extra_info
