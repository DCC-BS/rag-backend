"""Unit tests for the DoclingLoader class."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConversionResult
from langchain_core.documents import Document as LCDocument

from rag.documents.docling_loader import (
    DefaultDocumentConverterFactory,
    DefaultTokenizerFactory,
    DoclingLoader,
    DocumentConverterFactory,
    TokenizerFactory,
)


class TestDefaultDocumentConverterFactory:
    """Tests for the DefaultDocumentConverterFactory class."""

    @patch("rag.documents.docling_loader.ConfigurationManager")
    def test_create_converter_with_default_config(self, mock_config_manager):
        """Test creating a converter with the default configuration."""
        mock_config = MagicMock()
        mock_config.DOCLING.USE_GPU = False
        mock_config.DOCLING.NUM_THREADS = "4"
        mock_config_manager.get_config.return_value = mock_config

        factory = DefaultDocumentConverterFactory()
        converter = factory.create_converter()

        assert converter is not None
        mock_config_manager.get_config.assert_called_once()

    def test_create_converter_with_custom_config(self):
        """Test creating a converter with a custom configuration."""
        mock_config = MagicMock()
        mock_config.DOCLING.USE_GPU = True
        mock_config.DOCLING.NUM_THREADS = "8"

        factory = DefaultDocumentConverterFactory(config=mock_config)
        converter = factory.create_converter()

        assert converter is not None


class TestDefaultTokenizerFactory:
    """Tests for the DefaultTokenizerFactory class."""

    @patch("rag.documents.docling_loader.AutoTokenizer")
    def test_create_tokenizer_with_default_model(self, mock_auto_tokenizer):
        """Test creating a tokenizer with the default model."""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        factory = DefaultTokenizerFactory()
        tokenizer = factory.create_tokenizer()

        assert tokenizer is mock_tokenizer
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            pretrained_model_name_or_path="jinaai/jina-embeddings-v3"
        )

    @patch("rag.documents.docling_loader.AutoTokenizer")
    def test_create_tokenizer_with_custom_model(self, mock_auto_tokenizer):
        """Test creating a tokenizer with a custom model."""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        custom_model = "custom/model"

        factory = DefaultTokenizerFactory(model_id=custom_model)
        tokenizer = factory.create_tokenizer()

        assert tokenizer is mock_tokenizer
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(pretrained_model_name_or_path=custom_model)


class TestDoclingLoader:
    """Tests for the DoclingLoader class."""

    # Define instance variables at class level to avoid linter errors
    mock_converter_factory = None
    mock_tokenizer_factory = None
    mock_converter = None
    mock_tokenizer = None
    mock_chunker = None
    mock_logger = None
    loader = None

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.mock_converter_factory = MagicMock(spec=DocumentConverterFactory)
        self.mock_tokenizer_factory = MagicMock(spec=TokenizerFactory)
        self.mock_converter = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_chunker = MagicMock()
        self.mock_logger = MagicMock()

        self.mock_converter_factory.create_converter.return_value = self.mock_converter
        self.mock_tokenizer_factory.create_tokenizer.return_value = self.mock_tokenizer

        with patch("rag.documents.docling_loader.HybridChunker", return_value=self.mock_chunker):
            self.loader = DoclingLoader(
                file_path="test.pdf",
                organization="test-org",
                converter_factory=self.mock_converter_factory,
                tokenizer_factory=self.mock_tokenizer_factory,
                logger=self.mock_logger,
            )

        yield
        # Reset all mocks after each test if needed
        self.mock_converter_factory.reset_mock()
        self.mock_tokenizer_factory.reset_mock()
        self.mock_converter.reset_mock()
        self.mock_tokenizer.reset_mock()
        self.mock_chunker.reset_mock()
        self.mock_logger.reset_mock()

    def test_init_with_single_file_path(self):
        """Test initialization with a single file path."""
        assert self.loader._file_paths == ["test.pdf"]
        assert self.loader._organization == "test-org"
        assert self.loader.logger is self.mock_logger
        assert self.loader._converter is self.mock_converter
        assert self.loader.chunker is self.mock_chunker

        self.mock_converter_factory.create_converter.assert_called_once()
        self.mock_tokenizer_factory.create_tokenizer.assert_called_once()

    def test_init_with_multiple_file_paths(self):
        """Test initialization with multiple file paths."""
        file_paths = ["test1.pdf", "test2.docx"]

        with patch("rag.documents.docling_loader.HybridChunker", return_value=self.mock_chunker):
            loader = DoclingLoader(
                file_path=file_paths,
                organization="test-org",
                converter_factory=self.mock_converter_factory,
                tokenizer_factory=self.mock_tokenizer_factory,
                logger=self.mock_logger,
            )

        assert loader._file_paths == file_paths

    @patch("rag.documents.docling_loader.structlog")
    def test_init_with_default_logger(self, mock_structlog):
        """Test initialization with the default logger."""
        mock_logger = MagicMock()
        mock_structlog.get_logger.return_value = mock_logger

        with patch("rag.documents.docling_loader.HybridChunker", return_value=self.mock_chunker):
            loader = DoclingLoader(
                file_path="test.pdf",
                organization="test-org",
                converter_factory=self.mock_converter_factory,
                tokenizer_factory=self.mock_tokenizer_factory,
            )

        assert loader.logger is mock_logger
        mock_structlog.get_logger.assert_called_once()

    def test_is_valid_file_with_non_existent_file(self):
        """Test checking if a non-existent file is valid."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        source = "nonexistent.pdf"

        result = self.loader._is_valid_file(mock_path, source)

        assert result is False
        self.mock_logger.warning.assert_called_once_with(f"File {source} does not exist")

    def test_is_valid_file_with_non_file(self):
        """Test checking if a non-file path is valid."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = False
        source = "directory/"

        result = self.loader._is_valid_file(mock_path, source)

        assert result is False
        self.mock_logger.warning.assert_called_once_with(f"Path {source} is not a file")

    def test_is_valid_file_with_unsupported_format(self):
        """Test checking if a file with an unsupported format is valid."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        mock_path.suffix = ".txt"
        source = "file.txt"

        result = self.loader._is_valid_file(mock_path, source)

        assert result is False
        self.mock_logger.warning.assert_called_once_with(f"File {source} has an unsupported format")

    def test_is_valid_file_with_symlink(self):
        """Test checking if a symlink file is valid."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        mock_path.suffix = ".pdf"
        mock_path.is_symlink.return_value = True
        mock_path.name = "file.pdf"
        source = "file.pdf"

        result = self.loader._is_valid_file(mock_path, source)

        assert result is False
        self.mock_logger.warning.assert_called_once_with(f"Skipping symlink {source}")

    def test_is_valid_file_with_temp_file(self):
        """Test checking if a temporary file is valid."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        mock_path.suffix = ".pdf"
        mock_path.is_symlink.return_value = False
        mock_path.name = "~file.pdf"
        source = "~file.pdf"

        result = self.loader._is_valid_file(mock_path, source)

        assert result is False
        self.mock_logger.warning.assert_called_once_with(f"Skipping symlink {source}")

    def test_is_valid_file_with_valid_file(self):
        """Test checking if a valid file is valid."""
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        mock_path.suffix = ".pdf"
        mock_path.is_symlink.return_value = False
        mock_path.name = "file.pdf"
        source = "file.pdf"

        result = self.loader._is_valid_file(mock_path, source)

        assert result is True

    @patch("rag.documents.docling_loader.Path")
    def test_process_file_with_invalid_file(self, mock_path_class):
        """Test processing an invalid file."""
        mock_path = MagicMock(spec=Path)
        mock_path_class.return_value = mock_path
        source = "invalid.pdf"

        with patch.object(self.loader, "_is_valid_file", return_value=False) as mock_is_valid_file:
            result = list(self.loader._process_file(source))

            assert result == []
            mock_is_valid_file.assert_called_once_with(mock_path, source)
            self.mock_converter.convert.assert_not_called()

    @patch("rag.documents.docling_loader.Path")
    def test_process_file_with_conversion_failure(self, mock_path_class):
        """Test processing a file with conversion failure."""
        mock_path = MagicMock(spec=Path)
        mock_path_class.return_value = mock_path
        source = "valid.pdf"

        mock_result = MagicMock(spec=ConversionResult)
        mock_result.status = ConversionStatus.FAILURE
        self.mock_converter.convert.return_value = mock_result

        with patch.object(self.loader, "_is_valid_file", return_value=True) as mock_is_valid_file:
            result = list(self.loader._process_file(source))

            assert result == []
            mock_is_valid_file.assert_called_once_with(mock_path, source)
            self.mock_converter.convert.assert_called_once_with(source, raises_on_error=False)
            self.mock_logger.error.assert_called_once()

    @patch("rag.documents.docling_loader.Path")
    def test_process_file_with_pptx(self, mock_path_class):
        """Test processing a PPTX file."""
        mock_path = MagicMock(spec=Path)
        mock_path.suffix = ".pptx"
        mock_path_class.return_value = mock_path
        source = "presentation.pptx"

        mock_doc = MagicMock()
        mock_result = MagicMock(spec=ConversionResult)
        mock_result.status = ConversionStatus.SUCCESS
        mock_result.document = mock_doc
        self.mock_converter.convert.return_value = mock_result

        mock_content = "Slide content"
        mock_doc.num_pages.return_value = 2
        mock_doc.export_to_markdown.side_effect = [mock_content, mock_content]

        expected_docs = [
            LCDocument(
                page_content=mock_content,
                metadata={
                    "organization": "test-org",
                    "filename": mock_path.name,
                    "source": source,
                    "mimetype": "pptx",
                    "page_number": 1,
                },
            ),
            LCDocument(
                page_content=mock_content,
                metadata={
                    "organization": "test-org",
                    "filename": mock_path.name,
                    "source": source,
                    "mimetype": "pptx",
                    "page_number": 2,
                },
            ),
        ]

        with (
            patch.object(self.loader, "_is_valid_file", return_value=True) as mock_is_valid_file,
            patch.object(self.loader, "_process_pptx", return_value=iter(expected_docs)) as mock_process_pptx,
        ):
            result = list(self.loader._process_file(source))

            assert result == expected_docs
            mock_is_valid_file.assert_called_once_with(mock_path, source)
            self.mock_converter.convert.assert_called_once_with(source, raises_on_error=False)
            mock_process_pptx.assert_called_once_with(mock_doc, mock_path, source)

    @patch("rag.documents.docling_loader.Path")
    def test_process_file_with_pdf(self, mock_path_class):
        """Test processing a PDF file."""
        mock_path = MagicMock(spec=Path)
        mock_path.suffix = ".pdf"
        mock_path_class.return_value = mock_path
        source = "document.pdf"

        mock_doc = MagicMock()
        mock_result = MagicMock(spec=ConversionResult)
        mock_result.status = ConversionStatus.SUCCESS
        mock_result.document = mock_doc
        self.mock_converter.convert.return_value = mock_result

        expected_docs = [
            LCDocument(
                page_content="Document content",
                metadata={
                    "organization": "test-org",
                    "filename": "document.pdf",
                    "source": source,
                    "mimetype": "application/pdf",
                    "page_number": 1,
                },
            )
        ]

        with (
            patch.object(self.loader, "_is_valid_file", return_value=True) as mock_is_valid_file,
            patch.object(self.loader, "_process_document", return_value=iter(expected_docs)) as mock_process_document,
        ):
            result = list(self.loader._process_file(source))

            assert result == expected_docs
            mock_is_valid_file.assert_called_once_with(mock_path, source)
            self.mock_converter.convert.assert_called_once_with(source, raises_on_error=False)
            mock_process_document.assert_called_once_with(mock_doc, source)

    def test_process_pptx(self):
        """Test processing a PPTX file."""
        mock_doc = MagicMock()
        mock_doc.num_pages.return_value = 2
        mock_doc.export_to_markdown.side_effect = ["Slide 1 content", "Slide 2 content"]

        mock_path = MagicMock(spec=Path)
        mock_path.name = "presentation.pptx"
        source = "presentation.pptx"

        result = list(self.loader._process_pptx(mock_doc, mock_path, source))

        assert len(result) == 2
        assert result[0].page_content == "Slide 1 content"
        assert result[0].metadata == {
            "organization": "test-org",
            "filename": "presentation.pptx",
            "source": source,
            "mimetype": "pptx",
            "page_number": 1,
        }
        assert result[1].page_content == "Slide 2 content"
        assert result[1].metadata == {
            "organization": "test-org",
            "filename": "presentation.pptx",
            "source": source,
            "mimetype": "pptx",
            "page_number": 2,
        }

    def test_process_pptx_skips_empty_content(self):
        """Test processing a PPTX file with empty content."""
        mock_doc = MagicMock()
        mock_doc.num_pages.return_value = 2
        mock_doc.export_to_markdown.side_effect = ["", "Slide 2 content"]

        mock_path = MagicMock(spec=Path)
        mock_path.name = "presentation.pptx"
        source = "presentation.pptx"

        result = list(self.loader._process_pptx(mock_doc, mock_path, source))

        assert len(result) == 1
        assert result[0].page_content == "Slide 2 content"
        assert result[0].metadata == {
            "organization": "test-org",
            "filename": "presentation.pptx",
            "source": source,
            "mimetype": "pptx",
            "page_number": 2,
        }

    def test_process_document(self):
        """Test processing a document using the chunker."""
        mock_doc = MagicMock()
        source = "document.pdf"

        mock_chunk1 = MagicMock()
        mock_chunk1.text = "Chunk 1 content"
        mock_meta1 = MagicMock()
        mock_meta1.model_dump.return_value = {
            "doc_items": [{"prov": [{"page_no": 1}]}],
            "origin": {"filename": "document.pdf", "mimetype": "application/pdf"},
        }
        mock_chunk1.meta = mock_meta1

        mock_chunk2 = MagicMock()
        mock_chunk2.text = "Chunk 2 content"
        mock_meta2 = MagicMock()
        mock_meta2.model_dump.return_value = {
            "doc_items": [{"prov": [{"page_no": 2}]}],
            "origin": {"filename": "document.pdf", "mimetype": "application/pdf"},
        }
        mock_chunk2.meta = mock_meta2

        self.mock_chunker.chunk.return_value = iter([mock_chunk1, mock_chunk2])

        result = list(self.loader._process_document(mock_doc, source))

        assert len(result) == 2
        assert result[0].page_content == "Chunk 1 content"
        assert result[0].metadata == {
            "organization": "test-org",
            "filename": "document.pdf",
            "source": source,
            "mimetype": "application/pdf",
            "page_number": 1,
        }
        assert result[1].page_content == "Chunk 2 content"
        assert result[1].metadata == {
            "organization": "test-org",
            "filename": "document.pdf",
            "source": source,
            "mimetype": "application/pdf",
            "page_number": 2,
        }

        self.mock_chunker.chunk.assert_called_once_with(mock_doc)

    def test_process_document_skips_empty_content(self):
        """Test processing a document with empty content."""
        mock_doc = MagicMock()
        source = "document.pdf"

        mock_chunk1 = MagicMock()
        mock_chunk1.text = ""
        mock_meta1 = MagicMock()
        mock_meta1.model_dump.return_value = {
            "doc_items": [{"prov": [{"page_no": 1}]}],
            "origin": {"filename": "document.pdf", "mimetype": "application/pdf"},
        }
        mock_chunk1.meta = mock_meta1

        mock_chunk2 = MagicMock()
        mock_chunk2.text = "Chunk 2 content"
        mock_meta2 = MagicMock()
        mock_meta2.model_dump.return_value = {
            "doc_items": [{"prov": [{"page_no": 2}]}],
            "origin": {"filename": "document.pdf", "mimetype": "application/pdf"},
        }
        mock_chunk2.meta = mock_meta2

        self.mock_chunker.chunk.return_value = iter([mock_chunk1, mock_chunk2])

        result = list(self.loader._process_document(mock_doc, source))

        assert len(result) == 1
        assert result[0].page_content == "Chunk 2 content"
        assert result[0].metadata == {
            "organization": "test-org",
            "filename": "document.pdf",
            "source": source,
            "mimetype": "application/pdf",
            "page_number": 2,
        }

    def test_process_document_with_empty_provenance(self):
        """Test processing a document with empty provenance information."""
        mock_doc = MagicMock()
        source = "document.pdf"

        mock_chunk = MagicMock()
        mock_chunk.text = "Chunk content"
        mock_meta = MagicMock()
        mock_meta.model_dump.return_value = {
            "doc_items": [{"prov": []}],
            "origin": {"filename": "document.pdf", "mimetype": "application/pdf"},
        }
        mock_chunk.meta = mock_meta

        self.mock_chunker.chunk.return_value = iter([mock_chunk])

        result = list(self.loader._process_document(mock_doc, source))

        assert len(result) == 1
        assert result[0].page_content == "Chunk content"
        assert result[0].metadata == {
            "organization": "test-org",
            "filename": "document.pdf",
            "source": source,
            "mimetype": "application/pdf",
            "page_number": None,
        }

    def test_lazy_load(self):
        """Test the lazy_load method."""
        self.loader._file_paths = ["file1.pdf", "file2.docx"]
        mock_docs1 = [LCDocument(page_content="File 1 content", metadata={"source": "file1.pdf"})]
        mock_docs2 = [LCDocument(page_content="File 2 content", metadata={"source": "file2.docx"})]

        with patch.object(
            self.loader, "_process_file", side_effect=[iter(mock_docs1), iter(mock_docs2)]
        ) as mock_process_file:
            result = list(self.loader.lazy_load())

            assert result == mock_docs1 + mock_docs2
            assert mock_process_file.call_count == 2
            mock_process_file.assert_any_call("file1.pdf")
            mock_process_file.assert_any_call("file2.docx")
