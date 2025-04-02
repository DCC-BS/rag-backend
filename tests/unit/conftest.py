"""Common pytest fixtures for unit tests."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from rag.core.rag_pipeline import SHRAGPipeline
from rag.core.rag_states import GradeAnswer, GradeDocuments, GradeHallucination, RouteQuery
from tests.unit.rag.core.test_mocks import MockConfig, MockLLM, MockLogger, MockRetriever, create_test_document


@pytest.fixture
def mock_retriever() -> MockRetriever:
    """Create a mock retriever using the custom MockRetriever class."""
    # Documents can be customized per test if needed by accessing the fixture instance
    return MockRetriever(
        documents_to_return=[
            create_test_document("Test content 1", "test1.txt"),
            create_test_document("Test content 2", "test2.txt"),
        ]
    )


@pytest.fixture
def mock_llm() -> MockLLM:
    """Create a mock LLM using the custom MockLLM class."""
    llm = MockLLM(responses={"content": "This is a test response"})

    # Default structured output behaviors are set in MockLLM,
    # but can be overridden here or in tests if needed.
    # Example override:
    # llm.set_structured_output_response(RouteQuery, RouteQuery(next_step="answer"))

    return llm


@pytest.fixture
def mock_query_rewriter_llm() -> MockLLM:
    """Create a separate mock LLM instance for query rewriting."""
    # Initialize with potentially different default behavior if needed
    llm = MockLLM(responses={"content": "<query>Rewritten Query</query>"})
    # Example: Configure specific behavior for query rewriting structured output if used
    # llm.set_structured_output_response(...)
    return llm


@pytest.fixture
def mock_config() -> MockConfig:
    """Create a mock config using the custom MockConfig class."""
    return MockConfig()


@pytest.fixture
def mock_memory() -> Mock:
    """Create a mock memory saver (using standard Mock as no custom class needed)."""
    return Mock(spec=MemorySaver)


@pytest.fixture
def mock_logger() -> MockLogger:
    """Create a mock logger using the custom MockLogger class."""
    return MockLogger()


@pytest.fixture
def test_state() -> dict[str, Any]:
    """Create a test state for the RAG pipeline."""
    return {
        "input": "Test query",
        "user_organization": "test_org",
        "messages": [SystemMessage(content="Test system prompt")],
        "context": [create_test_document("Test document content", "test.txt")],
        "answer": "Test answer",
        "route_query": None,
        "hallucination_score": None,
        "answer_score": None,
        "needs_rephrase": None,
        "requires_more_information": None,
        "num_steps": 0,
    }


@pytest.fixture
def runnable_config() -> RunnableConfig:
    """Create a runnable config for testing."""
    return RunnableConfig(configurable={"thread_id": "test-thread"})


@pytest.fixture
def rag_pipeline(
    mock_llm: MockLLM,
    mock_query_rewriter_llm: MockLLM,
    mock_retriever: MockRetriever,
    mock_memory: Mock,
    mock_config: MockConfig,
    mock_logger: MockLogger,
) -> SHRAGPipeline:
    """Create a RAG pipeline instance with mocked dependencies for testing."""
    # Mock the graph build process during instantiation
    with patch.object(SHRAGPipeline, "_build_graph") as mock_build_graph:
        pipeline = SHRAGPipeline(
            llm=mock_llm,
            query_rewriter_llm=mock_query_rewriter_llm,
            retriever=mock_retriever,
            memory=mock_memory,
            system_prompt="Test system prompt",
        )

        pipeline.logger = mock_logger
        pipeline.config = mock_config

        # Assign the structured LLM mocks (derived from the main mock_llm fixture)
        # The MockLLM's with_structured_output method returns a pre-configured MagicMock
        pipeline.structured_llm_router = pipeline.llm.with_structured_output(RouteQuery)
        pipeline.structured_llm_grade_documents = pipeline.llm.with_structured_output(GradeDocuments)
        pipeline.structured_llm_grade_hallucination = pipeline.llm.with_structured_output(GradeHallucination)
        pipeline.structured_llm_grade_answer = pipeline.llm.with_structured_output(GradeAnswer)

        pipeline.graph = MagicMock()
        mock_build_graph.return_value = pipeline.graph  # Ensure _build_graph returns the mock graph

        return pipeline


# Configuration test fixtures
@pytest.fixture
def sample_yaml_config() -> str:
    """Get a sample YAML configuration for testing."""
    from tests.unit.rag.utils.test_mocks import get_sample_yaml_config

    return get_sample_yaml_config()


@pytest.fixture
def invalid_yaml_config() -> str:
    """Get an invalid YAML configuration for testing type validation."""
    from tests.unit.rag.utils.test_mocks import get_invalid_yaml_config

    return get_invalid_yaml_config()


@pytest.fixture
def incomplete_yaml_config() -> str:
    """Get an incomplete YAML configuration for testing missing fields."""
    from tests.unit.rag.utils.test_mocks import get_incomplete_yaml_config

    return get_incomplete_yaml_config()


@pytest.fixture
def mock_conf_dir(tmp_path) -> Path:
    """Create a temporary conf directory with test config files."""
    conf_dir = tmp_path / "src" / "rag" / "conf"
    conf_dir.mkdir(parents=True)
    return conf_dir


@pytest.fixture
def base_config_for_duplicates() -> str:
    """Get a base YAML configuration for testing duplicate entries."""
    from tests.unit.rag.utils.test_mocks import get_base_config_for_duplicates

    return get_base_config_for_duplicates()
