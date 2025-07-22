from typing import Any, Literal, override

import structlog
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.types import Command
from pydantic import BaseModel, Field

from rag.actions.action_protocol import ActionProtocol
from rag.models.rag_states import RAGState
from rag.models.stream_response import Sender, StreamResponse


class RouteDecision(BaseModel):
    """Route a user query to retrieval or answer generation."""

    next_step: Literal["retrieval", "answer"] = Field(
        description="Given a user question and a conversation history, choose to route it to a vectorstore or a llm.",
    )


class RouteQuestionAction(ActionProtocol):
    """
    Action to route the user question.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()
        self.llm = llm
        self.structured_llm_router = self.llm.with_structured_output(schema=RouteDecision, method="json_schema")

    @override
    def __call__(self, state: RAGState, config: RunnableConfig) -> Command[Literal["generate_answer", "retrieve"]]:
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            Command: Command object with routing decision
        """

        self.logger.info("---ROUTE QUESTION---")
        writer = get_stream_writer()
        writer("chat.status.routingQuestion")
        if "context" not in state or state.context is None:
            self.logger.info("---ROUTE QUESTION TO RETRIEVAL---")
            return Command(
                update={"route_query": "retrieval"},
                goto="retrieve",
            )

        system = """You are an expert at routing a user question to a vectorstore or llm.
        If you can answer the question based on the conversation history, return "answer".
        If you need more information, return "retrieval"."""

        context_messages: list[tuple[str, str | list[str | dict[Any, Any]]]] = [
            (message.type, message.content) for message in state.messages[1:]
        ]

        route_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            messages=[
                ("system", system),
                *context_messages,  # pyright: ignore[reportArgumentType]
                (
                    "user",
                    "Can you answer the following question based on the conversation history? Question: {question}",
                ),
            ]
        )

        question_router = route_prompt | self.structured_llm_router
        routing_result: RouteDecision = question_router.invoke({"question": state.input}, config)  # pyright: ignore[reportAssignmentType]

        if routing_result.next_step == "answer":
            self.logger.info("---ROUTE QUESTION TO ANSWER GENERATION---")
            return Command(
                update={"route_query": "answer"},
                goto="generate_answer",
            )
        elif routing_result.next_step == "retrieval":
            self.logger.info("---ROUTE QUESTION TO RETRIEVAL---")
            return Command(
                update={"route_query": "retrieval"},
                goto="retrieve",
            )
        else:
            raise ValueError(f"Unexpected routing next step: {routing_result.next_step}")

    @override
    def update_handler(self, data: dict[str, Any]) -> StreamResponse:
        """
        Handles updates from the action and returns a StreamResponse.
        """
        should_retrieve: bool = data.get("route_query") == "retrieval"
        return StreamResponse.create_decision_response(
            sender=Sender.ROUTE_QUESTION_ACTION,
            metadata={"decision": should_retrieve, "reason": "Retrieval" if should_retrieve else "Answer"},
        )
