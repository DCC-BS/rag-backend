"""Tests for the Retrieval-Augmented Generation graph components using pytest."""

from typing import cast
from unittest.mock import MagicMock, call, patch

import pytest
from langgraph.graph import END, START
from langgraph.types import Command

from rag.core.rag_pipeline import SHRAGPipeline
from rag.core.rag_states import (
    GradeAnswer,
    GradeDocuments,
    GradeHallucination,
    InputState,
    OutputState,
    RAGState,
    RouteQuery,
)
from tests.unit.rag.core.test_mocks import create_test_document


@pytest.fixture
def mock_state_graph():
    """Create a mocked StateGraph class and instance for testing."""
    mock_graph_instance = MagicMock()

    mock_graph_instance.add_node.return_value = mock_graph_instance
    mock_graph_instance.add_edge.return_value = mock_graph_instance
    mock_graph_instance.add_conditional_edges.return_value = mock_graph_instance
    mock_graph_instance.set_entry_point.return_value = mock_graph_instance
    mock_graph_instance.set_finish_point.return_value = mock_graph_instance
    mock_graph_instance.compile.return_value = MagicMock()

    def mock_state_graph_init(state_schema=None, input=None, output=None):
        assert state_schema == RAGState, f"Expected RAGState, got {state_schema}"
        return mock_graph_instance

    mock_class = MagicMock(side_effect=mock_state_graph_init)
    mock_class.return_value = mock_graph_instance
    return mock_class, mock_graph_instance


def test_graph_construction(monkeypatch, mock_state_graph, rag_pipeline: SHRAGPipeline):
    """Test that the graph is constructed correctly with all nodes and edges."""
    mock_state_graph_class, mock_graph_instance = mock_state_graph

    monkeypatch.setattr("rag.core.rag_pipeline.StateGraph", mock_state_graph_class)

    with (
        patch.object(SHRAGPipeline, "_setup_retriever"),
        patch.object(SHRAGPipeline, "_setup_llm"),
    ):
        pipeline_for_build = SHRAGPipeline(
            llm=MagicMock(),
            query_rewriter_llm=MagicMock(),
            retriever=MagicMock(),
            memory=MagicMock(),
            system_prompt="Test",
        )

    mock_state_graph_class.assert_has_calls(
        [call(state_schema=RAGState, input=InputState, output=OutputState)], any_order=True
    )

    added_nodes = {call.args[0] for call in mock_graph_instance.add_node.call_args_list}
    expected_nodes = {
        "route_question",
        "retrieve",
        "filter_documents",
        "query_needs_rephrase",
        "transform_query",
        "grade_hallucination",
        "grade_answer",
        "generate_answer",
    }
    assert expected_nodes.issubset(added_nodes), f"Missing nodes: {expected_nodes - added_nodes}"

    edge_calls = mock_graph_instance.add_edge.call_args_list
    assert any(
        call.kwargs.get("start_key") == START and call.kwargs.get("end_key") == "route_question" for call in edge_calls
    ), "Entry edge from START to route_question not found"

    mock_graph_instance.compile.assert_called_once()

    assert pipeline_for_build.graph == mock_graph_instance.compile.return_value


def test_node_connections_logic(rag_pipeline: SHRAGPipeline):
    """Test the conditional logic functions used for edges."""
    base_state = {
        "input": "Test query",
        "user_organization": "test_org",
        "messages": [],
        "context": [],
        "answer": "Test answer",
        "route_query": None,
        "hallucination_score": None,
        "answer_score": None,
        "needs_rephrase": None,
        "requires_more_information": None,
    }

    empty_context_state = dict(base_state)
    empty_context_state["context"] = []

    with_context_state = dict(base_state)
    with_context_state["context"] = [create_test_document()]

    with patch.object(rag_pipeline, "decide_if_query_needs_rewriting") as mock_decide:
        mock_decide.side_effect = [
            Command(update={"needs_rephrase": True}, goto="transform_query"),
            Command(update={"needs_rephrase": False}, goto="generate_answer"),
        ]

        result1 = rag_pipeline.decide_if_query_needs_rewriting(cast(RAGState, empty_context_state), MagicMock())
        result2 = rag_pipeline.decide_if_query_needs_rewriting(cast(RAGState, with_context_state), MagicMock())

    assert result1.update == {"needs_rephrase": True}
    assert result2.update == {"needs_rephrase": False}

    hallucination_state = dict(base_state)
    hallucination_state["hallucination_score"] = True

    no_hallucination_state = dict(base_state)
    no_hallucination_state["hallucination_score"] = False

    with patch.object(rag_pipeline, "grade_hallucination") as mock_grade:
        mock_grade.side_effect = [
            Command(update={"hallucination_score": True}, goto="generate_answer"),
            Command(update={"hallucination_score": False}, goto="grade_answer"),
        ]

        result3 = rag_pipeline.grade_hallucination(cast(RAGState, hallucination_state), MagicMock())
        result4 = rag_pipeline.grade_hallucination(cast(RAGState, no_hallucination_state), MagicMock())

    assert result3.goto == "generate_answer"
    assert result4.goto == "grade_answer"

    answer_ok_state = dict(base_state)
    answer_ok_state["answer_score"] = True

    answer_bad_state = dict(base_state)
    answer_bad_state["answer_score"] = False

    with patch.object(rag_pipeline, "grade_answer") as mock_grade_answer:
        mock_grade_answer.side_effect = [
            Command(update={"answer_score": True}, goto=END),
            Command(update={"answer_score": False}, goto="transform_query"),
        ]

        result5 = rag_pipeline.grade_answer(cast(RAGState, answer_ok_state), MagicMock())
        result6 = rag_pipeline.grade_answer(cast(RAGState, answer_bad_state), MagicMock())

    assert result5.goto == END
    assert result6.goto == "transform_query"


def test_graph_execution_flow_retrieval_path(rag_pipeline: SHRAGPipeline, test_state: RAGState, runnable_config):
    """Test a common execution path through the graph using mocks."""
    mock_router = MagicMock()
    mock_router.invoke.return_value = RouteQuery(next_step="retrieval")
    rag_pipeline.structured_llm_router = mock_router

    mock_grade_docs = MagicMock()
    mock_grade_docs.invoke.return_value = GradeDocuments(binary_score="yes")
    rag_pipeline.structured_llm_grade_documents = mock_grade_docs

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Final Answer"
    mock_llm.invoke.return_value = mock_response
    rag_pipeline.llm = mock_llm

    mock_hall_grader = MagicMock()
    mock_hall_grader.invoke.return_value = GradeHallucination(binary_score="no")
    rag_pipeline.structured_llm_grade_hallucination = mock_hall_grader

    mock_answer_grader = MagicMock()
    mock_answer_grader.invoke.return_value = GradeAnswer(binary_score="yes")
    rag_pipeline.structured_llm_grade_answer = mock_answer_grader

    mock_compiled_graph = rag_pipeline.graph

    final_state = {
        **test_state,
        "answer": "Final Answer",
        "route_query": "retrieval",
        "needs_rephrase": False,
        "hallucination_score": False,
        "answer_score": True,
    }

    mock_compiled_graph.invoke.return_value = final_state

    result = mock_compiled_graph.invoke(test_state, config=runnable_config)

    mock_compiled_graph.invoke.assert_called_once()

    assert result["answer"] == "Final Answer"
    assert result["route_query"] == "retrieval"
    assert result["needs_rephrase"] is False
    assert result["hallucination_score"] is False
    assert result["answer_score"] is True
