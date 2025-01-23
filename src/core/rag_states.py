import operator
from typing import Annotated, List, Literal, Optional, Sequence

from langchain.schema import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class InputState(TypedDict):
    input: str
    skip_retrieval: bool


class OutputState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: Annotated[Optional[List[Document]], operator.add]


class RAGState(InputState, OutputState):
    pass


class RouteQuery(BaseModel):
    """Route a user query to retrieval or answer generation."""

    next_step: Literal["retrieval", "answer"] = Field(
        ...,
        description="Given a user question and a conversation history, choose to route it to a vectorstore or a llm.",
    )
