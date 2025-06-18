"""Test mocks for the SHRAGPipeline class and its dependencies."""

from collections.abc import Sequence
from typing import Any, Literal
from unittest.mock import MagicMock

from langchain.schema import Document
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from rag.core.rag_states import GradeAnswer, GradeDocuments, GradeHallucination, RAGState, RouteQuery


class MockLogger:
    """Mock implementation of the logger for testing."""

    def __init__(self):
        self.logs = []

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        self.logs.append({"level": "info", "message": message, "kwargs": kwargs})

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self.logs.append({"level": "warning", "message": message, "kwargs": kwargs})

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self.logs.append({"level": "error", "message": message, "kwargs": kwargs})

    def get_logs(self):
        """Get all logged messages."""
        return self.logs

    def clear(self):
        """Clear all logs."""
        self.logs = []


class MockRetriever:
    """Mock implementation of the retriever for testing."""

    def __init__(self, documents_to_return=None):
        self.documents_to_return = documents_to_return or []
        self.invoke_calls = []

    def invoke(
        self,
        input: str,
        user_organization: str | None = None,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Mock the invoke method of the retriever."""
        self.invoke_calls.append({
            "input": input,
            "user_organization": user_organization,
            "config": config,
            "kwargs": kwargs,
        })
        return self.documents_to_return

    def get_calls(self):
        """Get all calls to invoke."""
        return self.invoke_calls


class MockLLM:
    """Mock implementation of the LLM for testing."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.invoke_calls = []
        self.structured_output_calls = []
        self.mock_structured_output = MagicMock()

    def invoke(self, messages, config=None, **kwargs):
        """Mock the invoke method of the LLM."""
        self.invoke_calls.append({"messages": messages, "config": config, "kwargs": kwargs})
        mock_response = MagicMock()
        mock_response.content = self.responses.get("content", "Mock LLM response")
        return mock_response

    def with_structured_output(self, schema, **kwargs):
        """Mock the with_structured_output method."""
        self.structured_output_calls.append({"schema": schema, "kwargs": kwargs})

        # Configure the mock structured output behavior based on schema type
        if schema == RouteQuery:
            self.mock_structured_output.invoke.return_value = RouteQuery(next_step="retrieval")
        elif schema == GradeDocuments:
            self.mock_structured_output.invoke.return_value = GradeDocuments(binary_score="yes")
        elif schema == GradeHallucination:
            self.mock_structured_output.invoke.return_value = GradeHallucination(binary_score="no")
        elif schema == GradeAnswer:
            self.mock_structured_output.invoke.return_value = GradeAnswer(binary_score="yes")

        return self.mock_structured_output

    def get_invoke_calls(self):
        """Get all calls to invoke."""
        return self.invoke_calls

    def get_structured_output_calls(self):
        """Get all calls to with_structured_output."""
        return self.structured_output_calls

    def set_response(self, response_type, value):
        """Set a response for a specific type."""
        self.responses[response_type] = value


class MockConfig:
    """Mock implementation of the application config for testing."""

    def __init__(self):
        self.DOC_STORE = MagicMock()
        self.DOC_STORE.TABLE_NAME = "test_table"
        self.DOC_STORE.DOCUMENT_DESCRIPTION = "Test documents"

        self.LLM = MagicMock()
        self.LLM.MODEL = "test-model"
        self.LLM.TEMPERATURE = 0.0
        self.LLM.API_URL = "http://test-api-url"

        self.RETRIEVER = MagicMock()
        self.RETRIEVER.TOP_K = 3
        self.RETRIEVER.FETCH_FOR_RERANKING = 10
        self.RETRIEVER.MAX_RECURSION = 3

        self.EMBEDDINGS = MagicMock()
        self.EMBEDDINGS.API_URL = "http://test-embeddings-url"


def create_test_document(content="Test content", filename="test.txt", metadata=None):
    """Create a test document with content and metadata."""
    doc_metadata = {"filename": filename}
    if metadata:
        doc_metadata.update(metadata)
    return Document(page_content=content, metadata=doc_metadata)


class MockRAGState(RAGState):
    """Mock RAG state for testing."""

    input: str
    user_organization: str
    messages: Sequence[BaseMessage]
    context: list[Document] | None
    answer: str | None
    hallucination_score: Literal["yes", "no"] | None
    answer_score: Literal["yes", "no"] | None
    needs_rephrase: bool | None
    route_query: RouteQuery | None
