import os
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean
from unstructured.partition.docx import partition_docx

from utils import config_loader


class DocxUnstructuredReader(BaseLoader, ABC):
    """Read Docx files using unstructured library."""

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
        """Loads list of documents from Docx file and also accepts extra information in dict format.

        Args:
            extra_info (Optional[Dict], optional): A dictionary containing extra information. Defaults to None.

        Returns:
            List[Document]: A list of Document objects.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File {self.file_path} does not exist.")

        elements = partition_docx(self.file_path.as_posix())
        chunked_elements = chunk_by_title(
            elements,
            max_characters=self.config["MAX_CHUNK_SIZE"],
            multipage_sections=True,
            overlap=self.config["SPLIT_OVERLAP"],
            combine_text_under_n_chars=self.config["MIN_CHUNK_SIZE"],
        )
        documents = []
        for element in chunked_elements:
            metadata = element.metadata.to_dict()
            metadata["source"] = self.file_path.as_posix()
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
