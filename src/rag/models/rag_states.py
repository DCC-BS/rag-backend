from typing import Annotated, Literal

from langchain.schema import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class InputState(BaseModel):
    """Input state for the RAG pipeline."""

    input: str
    user_organizations: list[str]
    document_ids: list[int] | None


class RouteQuery(BaseModel):
    """Route a user query to retrieval or answer generation."""

    next_step: Literal["retrieval", "answer"] = Field(
        description="Given a user question and a conversation history, choose to route it to a vectorstore or a llm.",
    )


class OutputState(BaseModel):
    """Output state for the RAG pipeline."""

    messages: Annotated[list[AnyMessage], add_messages]
    context: list[Document]
    hallucination_score: Literal["yes", "no"] | None
    answer_score: Literal["yes", "no"] | None
    route_query: Literal["retrieval", "answer"] | None
    reason: str | None


class RAGState(InputState, OutputState):
    """Complete state for the RAG pipeline combining input and output states."""

    pass


class GradeHallucination(BaseModel):
    """Binary score for hallucination check on retrieved documents and answer."""

    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the retrieved documents, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score for answer generation."""

    reason: str | None = Field(
        description="Reason for the grade"
    )  # Important to have reason fist to let the LLM "think" and afterwards generate the score
    binary_score: Literal["yes", "no"] = Field(description="Answer is relevant to the question, 'yes' or 'no'")
