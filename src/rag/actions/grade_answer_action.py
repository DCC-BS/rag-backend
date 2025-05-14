from typing import Any

import structlog
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.types import Command

from rag.actions.action_protocol import ActionProtocol
from rag.models.rag_states import GradeAnswer, RAGState
from rag.models.stream_response import StreamResponse


class GradeAnswerAction(ActionProtocol):
    """
    Action to grade answer generation.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()
        self.llm = llm
        self.structured_llm_grade_answer = self.llm.with_structured_output(GradeAnswer, method="json_schema")

    def __call__(self, state: RAGState, config: RunnableConfig):
        """
        Grade answer generation.
        """
        self.logger.info("---GRADE ANSWER---")
        system = """You are a grader assessing relevance of an answer to a user question. \n
            If the answer is relevant to the question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous answers. \n
            Give a binary score 'yes' or 'no' score to indicate whether the answer is relevant to the question.
            If the answer is relevant, return 'yes'. If the answer is not relevant, return 'no'.
            If the answer is not relevant, provide a reason for the grade."""
        answer_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            messages=[
                ("system", system),
                ("user", "Answer: {answer} \\n User question: {question}"),
            ]
        )
        answer_grader = answer_prompt | self.structured_llm_grade_answer
        answer_result: GradeAnswer = answer_grader.invoke(
            {"answer": state["answer"], "question": state["input"]}, config
        )  # pyright: ignore[reportAssignmentType]
        if answer_result.binary_score == "yes":
            self.logger.info("---ANSWER IS RELEVANT---")

            return Command(
                update={"answer_score": True},
                goto=END,
            )
        else:
            self.logger.info("---ANSWER IS NOT RELEVANT---")
            return Command(
                update={
                    "answer_score": False,
                    "answer": "Entschuldigung, ich kann die Antwort nicht in den Dokumenten finden.",
                },
                goto=END,
            )

    def update_handler(self, data: dict[str, Any]) -> StreamResponse:
        """
        Handles updates from the action and returns a StreamResponse.
        """
        return StreamResponse.create_status(
            message="Antwort ist relevant",
            decision="Ja" if data.get("answer_score") else "Nein",
        )
