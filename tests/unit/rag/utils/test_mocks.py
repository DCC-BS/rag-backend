"""Mocks for testing the RAG pipeline."""

from typing import Any, Literal
from unittest.mock import Mock

from langchain.schema import Document

from rag.models.rag_states import (
    GradeAnswer,
    GradeHallucination,
    RouteQuery,
)


class MockLogger:
    """Mock logger for testing."""

    def __init__(self) -> None:
        self.info = Mock()
        self.warning = Mock()
        self.error = Mock()
        self.debug = Mock()


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self) -> None:
        self.invoke = Mock()

    def invoke(
        self,
        input: str,
        user_organization: str,
        config: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Mock implementation of the invoke method."""
        return [
            Document(page_content="Test content 1", metadata={"filename": "doc1.txt"}),
            Document(page_content="Test content 2", metadata={"filename": "doc2.txt"}),
        ]


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self) -> None:
        self.invoke = Mock()
        self.__or__ = Mock(return_value=self)
        self.with_structured_output = Mock(return_value=self)


def create_test_document(content: str = "Test content", metadata: dict[str, Any] | None = None) -> Document:
    """Create a test document."""
    if metadata is None:
        metadata = {"filename": "test.txt"}
    return Document(page_content=content, metadata=metadata)


def create_test_state() -> dict[str, Any]:
    """Create a test state dictionary."""
    return {
        "input": "What is the meaning of life?",
        "user_organization": "test_org",
        "messages": [],
        "context": [
            create_test_document("Life is about exploring and learning."),
            create_test_document("The meaning of life is 42."),
        ],
        "answer": "The meaning of life is 42, according to sources.",
    }


def create_mock_state_as_dict() -> dict[str, Any]:
    """Create a mock state as a dictionary."""
    return create_test_state()


def create_structured_output_llm(output_class: Any, return_values: list[Any]) -> MockLLM:
    """Create a mock LLM that returns structured output."""
    mock_llm = MockLLM()
    mock_llm.invoke.side_effect = return_values

    mock_with_structured = MockLLM()
    mock_with_structured.invoke.side_effect = return_values
    mock_llm.with_structured_output.return_value = mock_with_structured

    return mock_llm


def create_router_llm(next_steps: list[Literal["retrieval", "answer"]]) -> MockLLM:
    """Create a mock router LLM that returns the specified next steps."""
    return create_structured_output_llm(RouteQuery, [RouteQuery(next_step=step) for step in next_steps])


def create_hallucination_grader_llm(scores: list[Literal["yes", "no"]]) -> MockLLM:
    """Create a mock hallucination grader LLM that returns the specified scores."""
    return create_structured_output_llm(
        GradeHallucination, [GradeHallucination(binary_score=score) for score in scores]
    )


def create_answer_grader_llm(scores: list[Literal["yes", "no"]]) -> MockLLM:
    """Create a mock answer grader LLM that returns the specified scores."""
    return create_structured_output_llm(GradeAnswer, [GradeAnswer(binary_score=score) for score in scores])


def get_sample_yaml_config() -> str:
    """Get a sample YAML configuration for testing."""
    return """
APP_NAME: "Test App"
VERSION: "1.0.0"
DESCRIPTION: "Test Description"
DOC_STORE:
  TYPE: "lance"
  PATH: "/path/to/store"
  TABLE_NAME: "documents"
  MAX_CHUNK_SIZE: 1000
  MIN_CHUNK_SIZE: 100
  SPLIT_OVERLAP: 50
  DOCUMENT_DESCRIPTION: "Test docs"
RETRIEVER:
  TYPE: "semantic"
  FETCH_FOR_RERANKING: 10
  TOP_K: 5
  MAX_RECURSION: 3
EMBEDDINGS:
  API_URL: "http://localhost:8000"
LLM:
  MODEL: "test-model"
  API_URL: "http://localhost:8001"
  TEMPERATURE: 0.7
DOCLING:
  NUM_THREADS: 4
  USE_GPU: false
CHAT:
  EXAMPLE_QUERIES: ["How are you?", "What is this?"]
  DEFAULT_PROMPT: "Hello!"
ROLES: ["admin", "user"]
DATA_DIR: "/data"
DOC_SOURCES:
  Sozialhilfe: "${DATA_DIR}/SH"
  Ergänzungsleistungen: "${DATA_DIR}/EL"
  Ergänzungsleistungen2: "${DATA_DIR}/EL2"
LOGIN_CONFIG:
  credentials:
    usernames:
      admin:
        email: "admin@test.com"
        failed_login_attempts: 0
        name: "Admin"
        password: "password123"
        logged_in: false
        roles: ["admin"]
  cookie:
    expiry_days: 30
    key: "some_key"
    name: "session"
"""


def get_invalid_yaml_config() -> str:
    """Get an invalid YAML configuration for testing type validation."""
    return """
APP_NAME: "Test App"
VERSION: 1.0  # Should be string
DESCRIPTION: "Test Description"
DOC_STORE:
    TYPE: ["invalid"]  # Should be string
    PATH: 123  # Should be string
    TABLE_NAME: true  # Should be string
    MAX_CHUNK_SIZE: "1000"  # Should be integer
    MIN_CHUNK_SIZE: false  # Should be integer
    SPLIT_OVERLAP: 50.5  # Should be integer
    DOCUMENT_DESCRIPTION: null  # Should be string
RETRIEVER:
    TYPE: "semantic"
    FETCH_FOR_RERANKING: "10"  # Should be integer
    TOP_K: 5.5  # Should be integer
    MAX_RECURSION: true  # Should be integer
EMBEDDINGS:
    API_URL: ["invalid"]  # Should be string
LLM:
    MODEL: 123  # Should be string
    API_URL: null  # Should be string
    TEMPERATURE: "0.7"  # Should be float
DOCLING:
    NUM_THREADS: "4"  # Should be integer
    USE_GPU: "false"  # Should be boolean
CHAT:
    EXAMPLE_QUERIES: "not a list"  # Should be list
    DEFAULT_PROMPT: ["not", "a", "string"]  # Should be string
ROLES: "not a list"  # Should be list
DATA_DIR: 123  # Should be string
DOC_SOURCES:
    test1: 123  # Should be string
    test2: true  # Should be string
LOGIN_CONFIG:
    credentials:
        usernames:
            admin:
                email: 123  # Should be string
                failed_login_attempts: "0"  # Should be integer
                name: ["invalid"]  # Should be string
                password: null  # Should be string
                logged_in: "true"  # Should be boolean
                roles: "admin"  # Should be list
    cookie:
        expiry_days: "30"  # Should be integer
        key: true  # Should be string
        name: 123  # Should be string
"""


def get_incomplete_yaml_config() -> str:
    """Get an incomplete YAML configuration for testing missing fields."""
    return """
APP_NAME: "Test App"
VERSION: "1.0.0"
# Missing DESCRIPTION and other required fields
DOC_STORE:
    TYPE: "lance"
    # Missing required PATH field
    TABLE_NAME: "documents"
    MAX_CHUNK_SIZE: 1000
"""


def get_base_config_for_duplicates() -> str:
    """Get a base YAML configuration for testing duplicate entries."""
    return """
APP_NAME: "Base App"
VERSION: "1.0.0"
DESCRIPTION: "Base Description"
DOC_STORE:
  TYPE: "lance"
  PATH: "/path/to/store"
  TABLE_NAME: "documents"
  MAX_CHUNK_SIZE: 1000
  MIN_CHUNK_SIZE: 100
  SPLIT_OVERLAP: 50
  DOCUMENT_DESCRIPTION: "Test docs"
RETRIEVER:
  TYPE: "semantic"
  FETCH_FOR_RERANKING: 10
  TOP_K: 5
  MAX_RECURSION: 3
EMBEDDINGS:
  API_URL: "http://localhost:8000"
LLM:
  MODEL: "test-model"
  API_URL: "http://localhost:8001"
  TEMPERATURE: 0.7
DOCLING:
  NUM_THREADS: 4
  USE_GPU: false
CHAT:
  EXAMPLE_QUERIES: ["How are you?", "What is this?"]
  DEFAULT_PROMPT: "Hello!"
ROLES: ["admin", "user"]
DATA_DIR: "/data"
DOC_SOURCES:
  source1: "/path1"
  source2: "/path2"
LOGIN_CONFIG:
  credentials:
    usernames:
      admin:
        email: "admin@test.com"
        failed_login_attempts: 0
        name: "Admin"
        password: "password123"
        logged_in: false
        roles: ["admin"]
  cookie:
    expiry_days: 30
    key: "some_key"
    name: "session"
"""
