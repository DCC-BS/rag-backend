"""Tests for the Retrieval-Augmented Generation pipeline using pytest."""

from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain.schema import Document
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, Interrupt

from rag.actions.retrieve_action import RetrieverProtocol
from rag.services.rag_pipeline import SHRAGPipeline
from tests.unit.rag.services.test_mocks import MockRAGState


@pytest.fixture
def pipeline():
    """Create a pipeline instance with mocked dependencies."""
    mock_logger = Mock()
    mock_retriever = Mock(spec=RetrieverProtocol)
    mock_llm = Mock()
    mock_memory = Mock(spec=MemorySaver)
    mock_config = Mock()

    pipeline = SHRAGPipeline(
        llm=mock_llm,
        retriever=mock_retriever,
        memory=mock_memory,
    )

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


def test_user_review_rewritten_query_needs_information(pipeline):
    """Test user review of rewritten query when more information is needed."""
    mock_state = cast(MockRAGState, cast(object, {"input": "rewritten query", "requires_more_information": True}))
    mock_config = RunnableConfig()

    with patch("rag.core.rag_pipeline.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"action": "provide_information", "data": "updated query"}

        result = pipeline.user_review_rewritten_query(mock_state, mock_config)

        mock_interrupt.assert_called_once_with(
            value={
                "type": "needs_information",
                "question": "Um die Frage zu beantworten, benötige ich weitere Informationen. Bitte geben Sie eine detailliertere Frage ein.",
                "rewritten_query": "rewritten query",
            }
        )

        assert isinstance(result, Command)
        assert result.goto == "transform_query"
        assert result.update == {"input": "updated query"}


def test_user_review_rewritten_query_needs_approval_accept(pipeline):
    """Test user review of rewritten query when approval is needed and user accepts."""
    mock_state = cast(MockRAGState, cast(object, {"input": "rewritten query", "requires_more_information": False}))
    mock_config = RunnableConfig()

    with patch("rag.core.rag_pipeline.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"action": "accept"}

        result = pipeline.user_review_rewritten_query(mock_state, mock_config)

        mock_interrupt.assert_called_once_with(
            value={
                "type": "needs_approval",
                "question": "Ist die umgeformte Frage korrekt?",
                "rewritten_query": "rewritten query",
            }
        )

        assert isinstance(result, Command)
        assert result.goto == "retrieve"
        assert result.update == {"input": "rewritten query"}


def test_user_review_rewritten_query_needs_approval_modify(pipeline):
    """Test user review of rewritten query when approval is needed and user modifies."""
    mock_state = cast(MockRAGState, cast(object, {"input": "rewritten query", "requires_more_information": False}))
    mock_config = RunnableConfig()

    with patch("rag.core.rag_pipeline.interrupt") as mock_interrupt:
        mock_interrupt.return_value = {"action": "modify", "data": "modified query"}

        result = pipeline.user_review_rewritten_query(mock_state, mock_config)

        mock_interrupt.assert_called_once_with(
            value={
                "type": "needs_approval",
                "question": "Ist die umgeformte Frage korrekt?",
                "rewritten_query": "rewritten query",
            }
        )

        assert isinstance(result, Command)
        assert result.goto == "retrieve"
        assert result.update == {"input": "modified query"}


def test_setup_retriever(pipeline):
    """Test the _setup_retriever method."""
    with (
        patch("rag.core.rag_pipeline.get_lancedb_doc_store") as mock_get_docstore,
        patch("rag.core.rag_pipeline.BentoMLReranker") as mock_reranker_class,
        patch("rag.core.rag_pipeline.LanceDBRetriever") as mock_retriever_class,
    ):
        # Set up mocks
        mock_table = MagicMock()
        mock_embeddings = MagicMock()
        mock_vector_store = MagicMock()
        mock_vector_store.embeddings = mock_embeddings
        mock_vector_store._text_key = "text"
        mock_vector_store._vector_key = "vector"
        mock_vector_store.get_table.return_value = mock_table

        mock_get_docstore.return_value = mock_vector_store

        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker

        mock_retriever = MagicMock()
        mock_retriever_class.return_value = mock_retriever

        # Configure pipeline
        pipeline.config.DOC_STORE.TABLE_NAME = "test_table"
        pipeline.config.EMBEDDINGS.API_URL = "http://test-url"
        pipeline.config.RETRIEVER.TOP_K = 5
        pipeline.config.RETRIEVER.FETCH_FOR_RERANKING = 10

        # Call the method
        result = pipeline._setup_retriever()

        # Assert the mocks were called correctly
        mock_get_docstore.assert_called_once()
        mock_reranker_class.assert_called_once_with(api_url="http://test-url", column="text")
        mock_retriever_class.assert_called_once()

        # Check the arguments to LanceDBRetriever
        _, kwargs = mock_retriever_class.call_args
        assert kwargs["table"] == mock_table
        assert kwargs["embeddings"] == mock_embeddings
        assert kwargs["reranker"] == mock_reranker
        assert kwargs["config"].vector_col == "vector"
        assert kwargs["config"].fts_col == "text"
        assert kwargs["config"].k == 5
        assert kwargs["config"].docs_before_rerank == 10

        # Check the result
        assert result == mock_retriever


@pytest.mark.asyncio
async def test_astream_query_new_query(pipeline):
    """Test astream_query for a new query."""
    mock_stream_response = MagicMock()
    pipeline.update_handlers = {
        "retrieve": lambda data: mock_stream_response,
        "generate_answer": lambda data: mock_stream_response,
    }

    mock_data = [
        ("updates", {"retrieve": {"context": []}}),
        ("updates", {"generate_answer": {"answer": "test answer"}}),
    ]

    async def mock_async_stream_generator():
        for item in mock_data:
            yield item

    mock_graph = MagicMock()
    async_generator_instance = mock_async_stream_generator()
    mock_graph.astream = MagicMock(return_value=async_generator_instance)

    pipeline.graph = mock_graph

    # Call the method
    responses = []
    async for response in pipeline.astream_query(
        question="test question", user_organization="test org", thread_id="test-thread"
    ):
        responses.append(response)

    # Assert the mocks were called correctly
    assert mock_graph.astream.called
    assert len(responses) == 2
    assert all(response == mock_stream_response for response in responses)

    # Check the call arguments
    args, kwargs = mock_graph.astream.call_args
    assert kwargs["input"] == {"input": "test question", "user_organization": "test org"}
    assert kwargs["stream_mode"] == ["updates"]
    assert kwargs["config"]["configurable"]["thread_id"] == "test-thread"


@pytest.mark.asyncio
async def test_astream_query_resume(pipeline):
    """Test astream_query for resuming an interaction."""
    # Set up mocks
    mock_stream_response = MagicMock()
    pipeline.update_handlers = {"retrieve": lambda data: mock_stream_response}

    # Create an async iterator of the data we want to yield
    mock_data = [
        ("updates", {"retrieve": {"context": []}}),
    ]

    async def mock_async_stream_generator():
        for item in mock_data:
            yield item

    mock_graph = MagicMock()
    async_generator_instance = mock_async_stream_generator()
    mock_graph.astream = MagicMock(return_value=async_generator_instance)

    pipeline.graph = mock_graph

    # Call the method
    responses = []
    async for response in pipeline.astream_query(
        thread_id="test-thread", resume_action="accept", resume_data="modified data"
    ):
        responses.append(response)

    # Assert the mocks were called correctly
    assert mock_graph.astream.called
    assert len(responses) == 1
    assert responses[0] == mock_stream_response

    # Check the call arguments
    args, kwargs = mock_graph.astream.call_args
    assert isinstance(kwargs["input"], Command)
    assert kwargs["input"].resume == {"action": "accept", "data": "modified data"}
    assert kwargs["stream_mode"] == ["updates"]
    assert kwargs["config"]["configurable"]["thread_id"] == "test-thread"


@pytest.mark.asyncio
async def test_astream_query_with_interrupt(pipeline):
    """Test astream_query with an interruption."""
    mock_stream_response = MagicMock()
    mock_interrupt_response = MagicMock()

    pipeline.update_handlers = {"retrieve": lambda data: mock_stream_response}

    interrupt_value = {"question": "Is this correct?", "rewritten_query": "rewritten query", "type": "needs_approval"}
    interrupt_obj = Interrupt(value=interrupt_value)

    # Create an async iterator of the data we want to yield
    mock_data = [
        ("updates", {"retrieve": {"context": []}}),
        ("updates", {"__interrupt__": [interrupt_obj]}),
    ]

    async def mock_async_stream_generator():
        for item in mock_data:
            yield item

    mock_graph = MagicMock()
    async_generator_instance = mock_async_stream_generator()
    mock_graph.astream = MagicMock(return_value=async_generator_instance)

    pipeline.graph = mock_graph

    with patch.object(pipeline, "update_handlers") as mock_handlers:
        mock_handlers.__getitem__.side_effect = lambda key: (
            lambda data: mock_interrupt_response if key == "__interrupt__" else mock_stream_response
        )
        with patch("rag.utils.stream_response.StreamResponse.create_interrupt_response") as mock_create_interrupt:
            mock_create_interrupt.return_value = mock_interrupt_response

            # Call the method
            responses = []
            async for response in pipeline.astream_query(
                question="test question", user_organization="test org", thread_id="test-thread"
            ):
                responses.append(response)

            # Check interrupt handling
            mock_create_interrupt.assert_called_once_with(
                message="Benutzerinteraktion erforderlich", interrupt_data=interrupt_value
            )
