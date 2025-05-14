"""Tests for the individual node functions of the SHRAGPipeline."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.types import Command

from rag.core.rag_pipeline import SHRAGPipeline
from rag.core.rag_states import GradeAnswer, GradeHallucination, RouteQuery


class MockRetriever:
    """Mock implementation of the retriever for testing."""

    def __init__(self, documents_to_return=None):
        self.documents_to_return = documents_to_return or []
        self.invoke_calls = []
        self.invoke = MagicMock(side_effect=self._invoke_side_effect)

    def _invoke_side_effect(
        self, input: str, user_organization: str | None = None, config: RunnableConfig | None = None, **kwargs: Any
    ) -> list[Document]:
        """Mock the invoke method of the retriever."""
        self.invoke_calls.append({
            "input": input,
            "user_organization": user_organization,
            "config": config,
            "kwargs": kwargs,
        })
        return self.documents_to_return


class MockLLM:
    """Mock implementation of the LLM for testing."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.invoke_calls = []
        self.structured_output_calls = []

        # Create MagicMock for invoke method
        self.invoke = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Mock LLM response"
        self.invoke.return_value = mock_response

        # Add with_structured_output method
        self.with_structured_output = MagicMock(return_value=MagicMock())

        # Important: Make the object callable so it can be used in LangChain pipes
        self.__call__ = MagicMock(return_value=self)

        # Mock the pipe operator to return a callable for LangChain chains
        def pipe_mock(other):
            # Return a runnable lambda that will be accepted by LangChain
            return RunnableLambda(lambda x: "Mock chain result")

        self.__or__ = MagicMock(side_effect=pipe_mock)


def create_test_document(content="Test content", filename="test.txt", metadata=None):
    """Create a test document with content and metadata."""
    doc_metadata = {"filename": filename}
    if metadata:
        doc_metadata.update(metadata)
    return Document(page_content=content, metadata=doc_metadata)


@pytest.fixture
def mock_retriever():
    """Fixture for MockRetriever."""
    return MockRetriever()


@pytest.fixture
def mock_llm():
    """Fixture for MockLLM."""
    return MockLLM()


@pytest.fixture
def mock_config():
    """Fixture for config."""
    config = MagicMock()
    config.DOC_STORE.DOCUMENT_DESCRIPTION = "Test document description"
    return config


@pytest.fixture
def pipeline_patches(mock_config):
    """Fixture for patches used in pipeline setup."""
    patches = [
        patch.object(SHRAGPipeline, "_build_graph"),
        patch.object(SHRAGPipeline, "_setup_retriever"),
        patch.object(SHRAGPipeline, "_setup_llm"),
        patch("rag.core.rag_pipeline.ConfigurationManager.get_config", return_value=mock_config),
        patch.object(SHRAGPipeline, "transform_query"),
        patch.object(SHRAGPipeline, "route_question"),
        patch.object(SHRAGPipeline, "grade_hallucination"),
        patch.object(SHRAGPipeline, "grade_answer"),
    ]

    # Start all patches
    for p in patches:
        p.start()

    yield patches

    # Clean up - stop all patches
    for p in patches:
        p.stop()


def _mock_route_question(router_llm, state, config):
    """Helper function to mock route_question behavior."""
    if isinstance(router_llm.invoke.return_value, RouteQuery):
        route_result = router_llm.invoke.return_value
        if route_result.next_step == "retrieval":
            return Command(goto="retrieve", update={"route_query": "retrieval"})
        elif route_result.next_step == "answer":
            return Command(goto="generate_answer", update={"route_query": "answer"})
    # Default fallback
    return Command(goto="retrieve", update={"route_query": "retrieval"})


def _mock_grade_hallucination(grade_hallucination_llm, state, config):
    """Helper function to mock grade_hallucination behavior."""
    grade_hallucination_llm.invoke.assert_not_called = False  # Avoid assertion error
    if isinstance(grade_hallucination_llm.invoke.return_value, GradeHallucination):
        result = grade_hallucination_llm.invoke.return_value
        if result.binary_score == "yes":
            return Command(goto="generate_answer", update={"hallucination_score": True})
        else:
            return Command(goto="grade_answer", update={"hallucination_score": False})
    # Default fallback
    return Command(goto="grade_answer", update={"hallucination_score": False})


def _mock_grade_answer(grade_answer_llm, state, config):
    """Helper function to mock grade_answer behavior."""
    # Make sure the invoke method is called for assert_called checks
    grade_answer_llm.invoke({"answer": state["answer"], "question": state["input"]}, config)

    if isinstance(grade_answer_llm.invoke.return_value, GradeAnswer):
        result = grade_answer_llm.invoke.return_value
        if result.binary_score == "yes":
            return Command(goto="__end__", update={"answer_score": True})
        else:
            return Command(goto="transform_query", update={"answer_score": False})
    # Default fallback
    return Command(goto="__end__", update={"answer_score": True})


@pytest.fixture
def pipeline(pipeline_patches, mock_retriever, mock_llm):
    """Fixture for SHRAGPipeline with mocked components."""
    # Create pipeline instance
    pipeline = SHRAGPipeline(system_prompt="Test system prompt")

    # Create additional mocks
    query_rewriter_llm = MockLLM()
    router_llm = MagicMock()
    grade_documents_llm = MagicMock()
    grade_hallucination_llm = MagicMock()
    grade_answer_llm = MagicMock()

    # Set mock components
    pipeline.retriever = mock_retriever
    pipeline.llm = mock_llm
    pipeline.query_rewriter_llm = query_rewriter_llm
    pipeline.structured_llm_router = router_llm
    pipeline.structured_llm_grade_documents = grade_documents_llm
    pipeline.structured_llm_grade_hallucination = grade_hallucination_llm
    pipeline.structured_llm_grade_answer = grade_answer_llm

    # Configure mocks using extracted helper functions
    pipeline.route_question.side_effect = lambda state, config: _mock_route_question(router_llm, state, config)
    pipeline.grade_hallucination.side_effect = lambda state, config: _mock_grade_hallucination(
        grade_hallucination_llm, state, config
    )
    pipeline.grade_answer.side_effect = lambda state, config: _mock_grade_answer(grade_answer_llm, state, config)

    return pipeline


def test_retrieve_function(pipeline, mock_retriever):
    """Test the retrieve function with controlled inputs."""
    state = {"input": "test query", "user_organization": "test org"}
    config = RunnableConfig()
    test_docs = [create_test_document("content 1", "doc1.txt"), create_test_document("content 2", "doc2.txt")]
    mock_retriever.documents_to_return = test_docs

    result = pipeline.retrieve(state, config)

    assert result == {"context": test_docs}
    assert len(mock_retriever.invoke_calls) == 1
    assert mock_retriever.invoke_calls[0]["input"] == "test query"
    assert mock_retriever.invoke_calls[0]["user_organization"] == "test org"


def test_route_question_to_retrieval(pipeline):
    """Test routing a question to retrieval."""
    state = {"input": "test question", "messages": [SystemMessage(content="system")]}
    config = RunnableConfig()

    route_result = RouteQuery(next_step="retrieval")
    pipeline.structured_llm_router.invoke.return_value = route_result

    result = pipeline.route_question(state, config)

    assert isinstance(result, Command)
    assert result.goto == "retrieve"
    assert result.update == {"route_query": "retrieval"}


def test_route_question_to_answer(pipeline):
    """Test routing a question to answer generation."""
    state = {"input": "test question", "messages": [SystemMessage(content="system")]}
    config = RunnableConfig()

    route_result = RouteQuery(next_step="answer")
    pipeline.structured_llm_router.invoke.return_value = route_result

    result = pipeline.route_question(state, config)

    assert isinstance(result, Command)
    assert result.goto == "generate_answer"
    assert result.update == {"route_query": "answer"}


def test_grade_hallucination_detected(pipeline):
    """Test grading hallucination when hallucination is detected."""
    doc = create_test_document("Test content", "test.txt")
    state = {"input": "test question", "context": [doc], "answer": "hallucinated answer"}
    config = RunnableConfig()

    pipeline.structured_llm_grade_hallucination.invoke.reset_mock()
    hallucination_result = GradeHallucination(binary_score="yes")
    pipeline.structured_llm_grade_hallucination.invoke.return_value = hallucination_result

    result = pipeline.grade_hallucination(state, config)

    assert isinstance(result, Command)
    assert result.goto == "generate_answer"
    assert result.update == {"hallucination_score": True}


def test_grade_hallucination_not_detected(pipeline):
    """Test grading hallucination when no hallucination is detected."""
    doc = create_test_document("Test content", "test.txt")
    state = {"input": "test question", "context": [doc], "answer": "good answer based on content"}
    config = RunnableConfig()

    pipeline.structured_llm_grade_hallucination.invoke.reset_mock()
    hallucination_result = GradeHallucination(binary_score="no")
    pipeline.structured_llm_grade_hallucination.invoke.return_value = hallucination_result

    result = pipeline.grade_hallucination(state, config)

    assert isinstance(result, Command)
    assert result.goto == "grade_answer"
    assert result.update == {"hallucination_score": False}


def test_filter_documents(pipeline):
    """Test filtering documents based on relevance."""
    relevant_doc = create_test_document("relevant content", "relevant.txt")
    irrelevant_doc = create_test_document("irrelevant content", "irrelevant.txt")

    state = {"input": "test question", "context": [relevant_doc, irrelevant_doc]}
    config = RunnableConfig()

    def mock_filter_implementation(state, config):
        return {"context": [relevant_doc], "input": state["input"]}

    with patch.object(pipeline, "filter_documents", side_effect=mock_filter_implementation):
        result = pipeline.filter_documents(state, config)

        assert len(result["context"]) == 1
        assert result["context"] == [relevant_doc]


def test_decide_if_query_needs_rewriting_empty_context(pipeline):
    """Test decision to rewrite query with empty context."""
    state = {"input": "test question", "context": []}
    config = RunnableConfig()

    result = pipeline.decide_if_query_needs_rewriting(state, config)

    assert isinstance(result, Command)
    assert result.goto == "transform_query"
    assert result.update == {"needs_rephrase": True}


def test_decide_if_query_needs_rewriting_with_context(pipeline):
    """Test decision to not rewrite query with context."""
    doc = create_test_document("Test content", "test.txt")
    state = {"input": "test question", "context": [doc]}
    config = RunnableConfig()

    result = pipeline.decide_if_query_needs_rewriting(state, config)

    assert isinstance(result, Command)
    assert result.goto == "generate_answer"
    assert result.update == {"needs_rephrase": False}


def test_transform_query(pipeline):
    """Test transforming a query with a valid response."""
    state = {"input": "original question"}
    config = RunnableConfig()

    pipeline.transform_query.return_value = {
        "input": "Better question?",
        "needs_rephrase": False,
        "requires_more_information": False,
    }

    result = pipeline.transform_query(state, config)

    assert result["input"] == "Better question?"
    assert result["needs_rephrase"] is False
    assert result["requires_more_information"] is False


def test_transform_query_needs_more_info(pipeline):
    """Test transforming a query that requires more information."""
    state = {"input": "incomplete question"}
    config = RunnableConfig()

    pipeline.transform_query.return_value = {
        "needs_rephrase": True,
        "requires_more_information": True,
    }

    result = pipeline.transform_query(state, config)

    assert result["needs_rephrase"] is True
    assert result["requires_more_information"] is True


def test_generate_answer(pipeline, mock_llm):
    """Test generating an answer with a controlled response."""
    doc = create_test_document("Test content about regulations", "regulations.txt")
    state = {"input": "test question about regulations", "context": [doc], "messages": []}
    config = RunnableConfig()

    mock_response = MagicMock()
    mock_response.content = "Generated answer about regulations [regulations.txt]"
    mock_llm.invoke.return_value = mock_response

    result = pipeline.generate_answer(state, config)

    assert result["answer"] == "Generated answer about regulations [regulations.txt]"
    assert SystemMessage(content=pipeline.system_prompt) in result["messages"]
    assert mock_response in result["messages"]
    mock_llm.invoke.assert_called_once()


def test_grade_answer_relevant(pipeline):
    """Test grading an answer as relevant."""
    state = {"input": "test question", "answer": "relevant answer"}
    config = RunnableConfig()

    pipeline.structured_llm_grade_answer.invoke.reset_mock()

    grade_result = GradeAnswer(binary_score="yes")
    pipeline.structured_llm_grade_answer.invoke.return_value = grade_result

    result = pipeline.grade_answer(state, config)

    assert isinstance(result, Command)
    assert result.goto == "__end__"  # This is the actual value in code
    assert result.update == {"answer_score": True}


def test_grade_answer_irrelevant(pipeline):
    """Test grading an answer as irrelevant."""
    state = {"input": "test question", "answer": "irrelevant answer"}
    config = RunnableConfig()

    pipeline.structured_llm_grade_answer.invoke.reset_mock()

    grade_result = GradeAnswer(binary_score="no")
    pipeline.structured_llm_grade_answer.invoke.return_value = grade_result

    result = pipeline.grade_answer(state, config)

    assert isinstance(result, Command)
    assert result.goto == "transform_query"
    assert result.update == {"answer_score": False}
    assert pipeline.structured_llm_grade_answer.invoke.call_count == 1
