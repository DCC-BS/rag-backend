from typing import Any, Literal, override

import structlog
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from rag.actions.action_protocol import ActionProtocol
from rag.models.rag_states import GradeHallucination, RAGState
from rag.models.stream_response import StreamResponse


class GradeHallucinationAction(ActionProtocol):
    """
    Action to grade hallucination on retrieved documents and answer.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()
        self.llm: ChatOpenAI = llm
        self.structured_llm_grade_hallucination = self.llm.with_structured_output(
            GradeHallucination, method="json_schema"
        )

    @override
    def __call__(self, state: RAGState, config: RunnableConfig) -> Command[Literal["generate_answer", "grade_answer"]]:
        """
        Grade hallucination on retrieved documents and answer.
        """
        self.logger.info("---GRADE HALLUCINATION---")
        system = """You are a hallucination checker. You are given a list of retrieved documents and an answer.
        You are to grade the answer based on the retrieved documents.
        If the answer is "Entschuldigung, ich kann die Antwort nicht in den Dokumenten finden.", return 'no'.
        If the answer is not grounded in the retrieved documents, return 'yes'.
        If the answer is grounded in the retrieved document, return 'no'.
        """
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", "Retrieved documents: {documents} \\n Answer: {answer}"),
        ])
        hallucination_grader = hallucination_prompt | self.structured_llm_grade_hallucination
        if state["context"] is None:
            raise ValueError("Context is None")
        hallucination_result: GradeHallucination = hallucination_grader.invoke(
            {
                "documents": "\n\n".join([doc.page_content for doc in state["context"]]),
                "answer": state["answer"],
            },
            config,
        )  # pyright: ignore[reportAssignmentType]
        if hallucination_result.binary_score == "yes":
            self.logger.info("---HALLUCINATION DETECTED---")
            return Command(
                update={"hallucination_score": True},
            )
        else:
            self.logger.info("---HALLUCINATION NOT DETECTED---")
            return Command(
                update={"hallucination_score": False},
            )

    @override
    def update_handler(self, data: dict[str, Any]) -> StreamResponse:
        """
        Handles updates from the action and returns a StreamResponse.
        """
        return StreamResponse.create_status(
            message="Antwort enthält Halluzinationen",
            sender="GradeHallucinationAction",
            decision="Ja" if data.get("hallucination_score") else "Nein",
        )
