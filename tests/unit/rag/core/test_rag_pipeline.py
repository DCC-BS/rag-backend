"""Tests for the Retrieval-Augmented Generation pipeline using pytest."""

from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain.schema import Document
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from rag.core.rag_pipeline import RetrieverProtocol, SHRAGPipeline
from tests.unit.rag.core.test_mocks import MockRAGState


@pytest.fixture
def pipeline():
    """Create a pipeline instance with mocked dependencies."""
    mock_logger = Mock()
    mock_retriever = Mock(spec=RetrieverProtocol)
    mock_llm = Mock()
    mock_query_rewriter_llm = Mock()
    mock_memory = Mock(spec=MemorySaver)
    mock_config = Mock()

    pipeline = SHRAGPipeline(
        llm=mock_llm,
        query_rewriter_llm=mock_query_rewriter_llm,
        retriever=mock_retriever,
        memory=mock_memory,
        system_prompt="Test system prompt",
    )

    pipeline.structured_llm_router = Mock()
    pipeline.structured_llm_grade_documents = Mock()
    pipeline.structured_llm_grade_hallucination = Mock()
    pipeline.structured_llm_grade_answer = Mock()

    pipeline._build_graph = MagicMock()
    pipeline.graph = Mock()

    pipeline.logger = mock_logger
    pipeline.config = mock_config

    return pipeline


def test_initialize_with_defaults():
    """Test that the pipeline can be initialized with default values."""
    with (
        patch("rag.core.rag_pipeline.ConfigurationManager.get_config"),
        patch("rag.core.rag_pipeline.structlog.get_logger"),
        patch.object(SHRAGPipeline, "_setup_retriever"),
        patch.object(SHRAGPipeline, "_setup_llm"),
        patch.object(SHRAGPipeline, "_build_graph"),
    ):
        pipeline = SHRAGPipeline()
        assert pipeline is not None


def test_retrieve(pipeline):
    """Test the retrieve method."""
    mock_state = cast(MockRAGState, cast(object, {"input": "test query", "user_organization": "test org"}))
    mock_config = RunnableConfig()
    mock_docs = [Document(page_content="test content", metadata={"filename": "test.txt"})]
    pipeline.retriever.invoke.return_value = mock_docs

    result = pipeline.retrieve(mock_state, mock_config)

    pipeline.retriever.invoke.assert_called_once_with(
        input="test query", user_organization="test org", config=mock_config
    )
    assert result == {"context": mock_docs}


def test_route_question_to_answer(pipeline):
    """Test routing a question to answer generation."""
    mock_state = cast(
        MockRAGState, cast(object, {"input": "test question", "messages": [SystemMessage(content="system prompt")]})
    )
    mock_config = RunnableConfig()

    mock_result = MagicMock()
    mock_result.next_step = "answer"

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_result

    with patch("rag.core.rag_pipeline.ChatPromptTemplate") as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)

        pipeline.structured_llm_router = mock_chain

        result = pipeline.route_question(mock_state, mock_config)

        assert isinstance(result, Command)
        assert result.goto == "generate_answer"
        assert result.update == {"route_query": "answer"}


def test_route_question_to_retrieval(pipeline):
    """Test routing a question to retrieval."""
    mock_state = cast(
        MockRAGState, cast(object, {"input": "test question", "messages": [SystemMessage(content="system prompt")]})
    )
    mock_config = RunnableConfig()

    mock_result = MagicMock()
    mock_result.next_step = "retrieval"

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_result

    with patch("rag.core.rag_pipeline.ChatPromptTemplate") as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)

        pipeline.structured_llm_router = mock_chain

        result = pipeline.route_question(mock_state, mock_config)

        assert isinstance(result, Command)
        assert result.goto == "retrieve"
        assert result.update == {"route_query": "retrieval"}


def test_filter_documents(pipeline):
    """Test filtering documents based on relevance."""
    doc1 = Document(page_content="relevant content", metadata={"filename": "doc1.txt"})
    doc2 = Document(page_content="irrelevant content", metadata={"filename": "doc2.txt"})

    mock_state = cast(MockRAGState, cast(object, {"input": "test question", "context": [doc1, doc2]}))
    mock_config = RunnableConfig()

    mock_result1 = MagicMock()
    mock_result1.binary_score = "yes"
    mock_result2 = MagicMock()
    mock_result2.binary_score = "no"

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [mock_result1, mock_result2]

    with patch("rag.core.rag_pipeline.ChatPromptTemplate") as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)

        pipeline.structured_llm_grade_documents = mock_chain

        result = pipeline.filter_documents(mock_state, mock_config)

        assert mock_chain.invoke.call_count == 2
        assert len(result["context"]) == 1
        assert result["context"][0] == doc1


def test_decide_if_query_needs_rewriting_with_empty_context(pipeline):
    """Test deciding if query needs rewriting when context is empty."""
    mock_state = cast(MockRAGState, cast(object, {"input": "test question", "context": []}))
    mock_config = RunnableConfig()

    result = pipeline.decide_if_query_needs_rewriting(mock_state, mock_config)

    assert isinstance(result, Command)
    assert result.goto == "transform_query"
    assert result.update == {"needs_rephrase": True}


def test_decide_if_query_needs_rewriting_with_context(pipeline):
    """Test deciding if query needs rewriting when context has documents."""
    doc = Document(page_content="test content", metadata={"filename": "test.txt"})
    mock_state = cast(MockRAGState, cast(object, {"input": "test question", "context": [doc]}))
    mock_config = RunnableConfig()

    result = pipeline.decide_if_query_needs_rewriting(mock_state, mock_config)

    assert isinstance(result, Command)
    assert result.goto == "generate_answer"
    assert result.update == {"needs_rephrase": False}


def test_transform_query(pipeline):
    """Test transforming a query."""
    mock_state = cast(MockRAGState, cast(object, {"input": "test question"}))
    mock_config = RunnableConfig()

    pipeline.config.DOC_STORE.DOCUMENT_DESCRIPTION = "test description"

    with (
        patch("rag.core.rag_pipeline.StrOutputParser") as mock_str_parser,
        patch("rag.core.rag_pipeline.ChatPromptTemplate") as mock_prompt,
    ):
        mock_chain = MagicMock()
        mock_chain.invoke = MagicMock(return_value="<query>Better question?</query>")

        mock_prompt_instance = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)

        pipeline.query_rewriter_llm = MagicMock()
        pipeline.query_rewriter_llm.__or__ = MagicMock(return_value=mock_chain)

        mock_str_parser.return_value = MagicMock()
        mock_str_parser.return_value.__or__ = MagicMock(return_value=mock_chain)

        final_chain = MagicMock()
        final_chain.invoke = MagicMock(return_value="<query>Better question?</query>")
        mock_chain.__or__.return_value = final_chain

        result = pipeline.transform_query(mock_state, mock_config)

        assert result == {
            "input": "Better question?",
            "needs_rephrase": False,
            "requires_more_information": False,
        }


def test_grade_hallucination_detected(pipeline):
    """Test grading when hallucination is detected."""
    mock_state = cast(
        MockRAGState,
        cast(
            object,
            {
                "input": "test question",
                "answer": "test answer",
                "context": [Document(page_content="test content", metadata={"filename": "test.txt"})],
            },
        ),
    )
    mock_config = RunnableConfig()

    mock_result = MagicMock()
    mock_result.binary_score = "yes"

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_result

    with patch("rag.core.rag_pipeline.ChatPromptTemplate") as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)

        pipeline.structured_llm_grade_hallucination = mock_chain

        result = pipeline.grade_hallucination(mock_state, mock_config)

        assert isinstance(result, Command)
        assert result.goto == "generate_answer"
        assert result.update == {"hallucination_score": True}


def test_grade_hallucination_not_detected(pipeline):
    """Test grading when no hallucination is detected."""
    mock_state = cast(
        MockRAGState,
        cast(
            object,
            {
                "input": "test question",
                "answer": "test answer",
                "context": [Document(page_content="test content", metadata={"filename": "test.txt"})],
            },
        ),
    )
    mock_config = RunnableConfig()

    mock_result = MagicMock()
    mock_result.binary_score = "no"

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_result

    with patch("rag.core.rag_pipeline.ChatPromptTemplate") as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)

        pipeline.structured_llm_grade_hallucination = mock_chain

        result = pipeline.grade_hallucination(mock_state, mock_config)

        assert isinstance(result, Command)
        assert result.goto == "grade_answer"
        assert result.update == {"hallucination_score": False}


def test_grade_answer_relevant(pipeline):
    """Test grading when answer is relevant."""
    mock_state = cast(MockRAGState, cast(object, {"input": "test question", "answer": "test answer"}))
    mock_config = RunnableConfig()

    mock_result = MagicMock()
    mock_result.binary_score = "yes"

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_result

    with patch("rag.core.rag_pipeline.ChatPromptTemplate") as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)

        pipeline.structured_llm_grade_answer = mock_chain

        result = pipeline.grade_answer(mock_state, mock_config)

        assert isinstance(result, Command)
        assert result.goto == "__end__"
        assert result.update == {"answer_score": True}


def test_grade_answer_not_relevant(pipeline):
    """Test grading when answer is not relevant."""
    mock_state = cast(MockRAGState, cast(object, {"input": "test question", "answer": "test answer"}))
    mock_config = RunnableConfig()

    mock_result = MagicMock()
    mock_result.binary_score = "no"

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_result

    with patch("rag.core.rag_pipeline.ChatPromptTemplate") as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)

        pipeline.structured_llm_grade_answer = mock_chain

        result = pipeline.grade_answer(mock_state, mock_config)

        assert isinstance(result, Command)
        assert result.goto == "transform_query"
        assert result.update == {"answer_score": False}


def test_generate_answer(pipeline):
    """Test generating an answer."""
    doc = Document(page_content="test content", metadata={"filename": "test.txt"})
    mock_state = cast(
        MockRAGState,
        cast(
            object, {"input": "test question", "context": [doc], "messages": [SystemMessage(content="system prompt")]}
        ),
    )
    mock_config = RunnableConfig()

    mock_response = MagicMock()
    mock_response.content = "Generated answer"

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response

    with patch("rag.core.rag_pipeline.ChatPromptTemplate") as mock_prompt:
        mock_prompt_instance = MagicMock()
        mock_prompt.from_messages.return_value = mock_prompt_instance
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_chain)

        pipeline.llm = mock_chain

        result = pipeline.generate_answer(mock_state, mock_config)

        assert "answer" in result
        assert result["answer"] == "Generated answer"
        assert "messages" in result
        assert isinstance(result["messages"], list)
