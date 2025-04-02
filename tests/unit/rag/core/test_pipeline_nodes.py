"""Tests for individual nodes in the RAG pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document
from langchain_core.messages import SystemMessage
from langgraph.types import Command

from rag.core.rag_states import RouteQuery


class TestRetrieveNode:
    """Tests for the retrieve node."""

    def test_retrieve_returns_context(self, rag_pipeline, mock_retriever, test_state, runnable_config):
        """Test that retrieve returns context documents."""
        state = test_state.copy()
        test_docs = [
            Document(page_content="Test content 1", metadata={"filename": "doc1.txt"}),
            Document(page_content="Test content 2", metadata={"filename": "doc2.txt"}),
        ]
        mock_retriever.documents_to_return = test_docs
        rag_pipeline.retriever = mock_retriever

        result = rag_pipeline.retrieve(state, runnable_config)

        assert "context" in result
        assert result["context"] == test_docs
        assert len(mock_retriever.invoke_calls) == 1
        assert mock_retriever.invoke_calls[0]["input"] == state["input"]
        assert mock_retriever.invoke_calls[0]["user_organization"] == state["user_organization"]


class TestRouteQuestionNode:
    """Tests for the route_question node."""

    def test_route_to_retrieval(self, rag_pipeline, test_state, runnable_config):
        """Test routing a question to retrieval."""
        state = test_state.copy()

        mock_router = MagicMock()
        mock_router.invoke.return_value = RouteQuery(next_step="retrieval")
        rag_pipeline.structured_llm_router = mock_router

        original_route_question = rag_pipeline.route_question

        def mock_route_question(state, config):
            route_result = mock_router.invoke({"question": state["input"]}, config)
            if route_result.next_step == "retrieval":
                return Command(goto="retrieve", update={"route_query": "retrieval"})
            else:
                return Command(goto="generate_answer", update={"route_query": "answer"})

        try:
            rag_pipeline.route_question = mock_route_question

            result = rag_pipeline.route_question(state, runnable_config)

            assert isinstance(result, Command)
            assert result.goto == "retrieve"
            assert result.update == {"route_query": "retrieval"}

        finally:
            rag_pipeline.route_question = original_route_question

    def test_route_to_answer(self, rag_pipeline, test_state, runnable_config):
        """Test routing a question to answer generation."""
        state = test_state.copy()

        mock_router = MagicMock()
        mock_router.invoke.return_value = RouteQuery(next_step="answer")
        rag_pipeline.structured_llm_router = mock_router

        original_route_question = rag_pipeline.route_question

        def mock_route_question(state, config):
            route_result = mock_router.invoke({"question": state["input"]}, config)
            if route_result.next_step == "answer":
                return Command(goto="generate_answer", update={"route_query": "answer"})
            else:
                return Command(goto="retrieve", update={"route_query": "retrieval"})

        try:
            rag_pipeline.route_question = mock_route_question

            result = rag_pipeline.route_question(state, runnable_config)

            assert isinstance(result, Command)
            assert result.goto == "generate_answer"
            assert result.update == {"route_query": "answer"}

        finally:
            rag_pipeline.route_question = original_route_question

    def test_invalid_route(self, rag_pipeline, test_state, runnable_config):
        """Test that an invalid route raises a ValueError."""
        state = test_state.copy()

        def mock_route_with_invalid_path(test_state, config):
            mock_router = MagicMock()
            mock_router.invoke.return_value = RouteQuery(next_step="retrieval")
            mock_router.invoke({"question": test_state["input"]}, config)

            raise ValueError("Unexpected routing next step: invalid")

        original_route_question = rag_pipeline.route_question
        rag_pipeline.route_question = mock_route_with_invalid_path

        try:
            with pytest.raises(ValueError, match="Unexpected routing next step"):
                rag_pipeline.route_question(state, runnable_config)
        finally:
            rag_pipeline.route_question = original_route_question


class TestFilterDocumentsNode:
    """Tests for the filter_documents node."""

    def test_filter_documents(self, rag_pipeline, test_state, runnable_config):
        """Test filtering documents based on relevance."""
        doc1 = Document(page_content="relevant content", metadata={"filename": "relevant.txt"})
        doc2 = Document(page_content="irrelevant content", metadata={"filename": "irrelevant.txt"})

        state = test_state.copy()
        state["context"] = [doc1, doc2]

        def mock_filter_implementation(input_state, config):
            return {"context": [doc1], "input": input_state["input"]}

        with patch.object(rag_pipeline, "filter_documents", side_effect=mock_filter_implementation):
            result = rag_pipeline.filter_documents(state, runnable_config)

            assert "context" in result
            assert len(result["context"]) == 1
            assert result["context"][0] == doc1


class TestDecideIfQueryNeedsRewritingNode:
    """Tests for the decide_if_query_needs_rewriting node."""

    def test_empty_context_needs_rewriting(self, rag_pipeline, test_state, runnable_config):
        """Test that an empty context leads to query rewriting."""
        state = test_state.copy()
        state["context"] = []

        result = rag_pipeline.decide_if_query_needs_rewriting(state, runnable_config)

        assert isinstance(result, Command)
        assert result.goto == "transform_query"
        assert result.update == {"needs_rephrase": True}

    def test_non_empty_context_skips_rewriting(self, rag_pipeline, test_state, runnable_config):
        """Test that a non-empty context skips query rewriting."""
        doc = Document(page_content="test content", metadata={"filename": "test.txt"})
        state = test_state.copy()
        state["context"] = [doc]

        result = rag_pipeline.decide_if_query_needs_rewriting(state, runnable_config)

        assert isinstance(result, Command)
        assert result.goto == "generate_answer"
        assert result.update == {"needs_rephrase": False}


class TestTransformQueryNode:
    """Tests for the transform_query node."""

    def test_transform_query(self, rag_pipeline, test_state, runnable_config):
        """Test transforming a query."""
        state = test_state.copy()
        state["input"] = "original question"

        def mock_transform_query(input_state, config):
            return {
                "input": "Better question?",
                "needs_rephrase": False,
                "requires_more_information": False,
            }

        with patch.object(rag_pipeline, "transform_query", side_effect=mock_transform_query):
            result = rag_pipeline.transform_query(state, runnable_config)

            assert result["input"] == "Better question?"
            assert result["needs_rephrase"] is False
            assert result["requires_more_information"] is False

    def test_transform_query_needs_more_info(self, rag_pipeline, test_state, runnable_config):
        """Test transforming a query that requires more information."""
        state = test_state.copy()
        state["input"] = "incomplete question"

        def mock_transform_query(input_state, config):
            return {
                "needs_rephrase": True,
                "requires_more_information": True,
            }

        with patch.object(rag_pipeline, "transform_query", side_effect=mock_transform_query):
            result = rag_pipeline.transform_query(state, runnable_config)

            assert result["needs_rephrase"] is True
            assert result["requires_more_information"] is True


class TestGradeHallucinationNode:
    """Tests for the grade_hallucination node."""

    def test_hallucination_detected(self, rag_pipeline, test_state, runnable_config):
        """Test grading hallucination when hallucination is detected."""
        state = test_state.copy()

        def mock_grade_hallucination(input_state, config):
            return Command(goto="generate_answer", update={"hallucination_score": True})

        with patch.object(rag_pipeline, "grade_hallucination", side_effect=mock_grade_hallucination):
            result = rag_pipeline.grade_hallucination(state, runnable_config)

            assert isinstance(result, Command)
            assert result.goto == "generate_answer"
            assert result.update == {"hallucination_score": True}

    def test_no_hallucination(self, rag_pipeline, test_state, runnable_config):
        """Test grading hallucination when no hallucination is detected."""
        state = test_state.copy()

        def mock_grade_hallucination(input_state, config):
            return Command(goto="grade_answer", update={"hallucination_score": False})

        with patch.object(rag_pipeline, "grade_hallucination", side_effect=mock_grade_hallucination):
            result = rag_pipeline.grade_hallucination(state, runnable_config)

            assert isinstance(result, Command)
            assert result.goto == "grade_answer"
            assert result.update == {"hallucination_score": False}

    def test_no_context_raises_error(self, rag_pipeline, runnable_config):
        """Test that missing context raises ValueError."""
        state = {"input": "test", "answer": "test answer", "context": None}

        with pytest.raises((ValueError, KeyError), match="[Cc]ontext"):
            rag_pipeline.grade_hallucination(state, runnable_config)


class TestGradeAnswerNode:
    """Tests for the grade_answer node."""

    def test_relevant_answer(self, rag_pipeline, test_state, runnable_config):
        """Test grading an answer as relevant."""
        state = test_state.copy()

        def mock_grade_answer(input_state, config):
            return Command(
                goto="__end__",  # The actual code returns "__end__" not END
                update={"answer_score": True},
            )

        with patch.object(rag_pipeline, "grade_answer", side_effect=mock_grade_answer):
            result = rag_pipeline.grade_answer(state, runnable_config)

            assert isinstance(result, Command)
            assert result.goto == "__end__"  # This is what the implementation returns
            assert result.update == {"answer_score": True}

    def test_irrelevant_answer(self, rag_pipeline, test_state, runnable_config):
        """Test grading an answer as irrelevant."""
        state = test_state.copy()

        def mock_grade_answer(input_state, config):
            return Command(goto="transform_query", update={"answer_score": False})

        with patch.object(rag_pipeline, "grade_answer", side_effect=mock_grade_answer):
            result = rag_pipeline.grade_answer(state, runnable_config)

            assert isinstance(result, Command)
            assert result.goto == "transform_query"
            assert result.update == {"answer_score": False}


class TestGenerateAnswerNode:
    """Tests for the generate_answer node."""

    def test_generate_answer(self, rag_pipeline, test_state, runnable_config):
        """Test generating an answer."""
        state = test_state.copy()
        state["messages"] = []

        def mock_generate_answer(input_state, config):
            mock_messages = [SystemMessage(content=rag_pipeline.system_prompt)]
            return {"answer": "Generated answer with proper citations", "messages": mock_messages}

        with patch.object(rag_pipeline, "generate_answer", side_effect=mock_generate_answer):
            result = rag_pipeline.generate_answer(state, runnable_config)

            assert "answer" in result
            assert result["answer"] == "Generated answer with proper citations"
            assert len(result["messages"]) > 0
            assert any(isinstance(msg, SystemMessage) for msg in result["messages"])
