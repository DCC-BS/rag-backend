from typing import Any, Literal

import structlog
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from rag.actions.action_protocol import ActionProtocol
from rag.models.rag_states import RAGState, RouteQuery
from rag.models.stream_response import StreamResponse


class RouteQuestionAction(ActionProtocol):
    """
    Action to route the user question.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()
        self.llm = llm
        self.structured_llm_router = self.llm.with_structured_output(schema=RouteQuery, method="json_schema")

    def __call__(self, state: RAGState, config: RunnableConfig) -> Command[Literal["generate_answer", "retrieve"]]:
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        self.logger.info("---ROUTE QUESTION---")
        system = """You are an expert at routing a user question to a vectorstore or llm.
        If you can answer the question based on the conversation history, return "answer".
        If you need more information, return "retrieval"."""
        context_messages: list[tuple[str, str | list[str | dict[Any, Any]]]] = [
            (message.type, message.content) for message in state["messages"][1:]
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
        routing_result: RouteQuery = question_router.invoke({"question": state["input"]}, config)  # pyright: ignore[reportAssignmentType]
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

    def update_handler(self, data: dict[str, Any]) -> StreamResponse:
        """
        Handles updates from the action and returns a StreamResponse.
        """
        return StreamResponse.create_status(
            message=f"{"Suche relevante Dokumente" if data.get("route_query") == "retrieval" else "Antworte auf die Frage"}",
        )
