from collections.abc import Sequence
from typing import Annotated, Literal

from langchain.schema import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class InputState(TypedDict):
    input: str
    user_organizations: list[str]


class RouteQuery(BaseModel):
    """Route a user query to retrieval or answer generation."""

    next_step: Literal["retrieval", "answer"] = Field(
        default=...,
        description="Given a user question and a conversation history, choose to route it to a vectorstore or a llm.",
    )


class OutputState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: list[Document]
    answer: str | None
    hallucination_score: Literal["yes", "no"] | None
    answer_score: Literal["yes", "no"] | None
    route_query: RouteQuery | None


class RAGState(InputState, OutputState):
    pass


class GradeHallucination(BaseModel):
    """Binary score for hallucination check on retrieved documents and answer."""

    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the retrieved documents, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score for answer generation."""

    binary_score: Literal["yes", "no"] = Field(description="Answer is relevant to the question, 'yes' or 'no'")
