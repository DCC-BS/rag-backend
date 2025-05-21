from typing import Any

import structlog
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from rag.actions.action_protocol import ActionProtocol
from rag.models.rag_states import RAGState
from rag.models.stream_response import StreamResponse


class GenerateAnswerAction(ActionProtocol):
    """
    Action to generate an answer based on the context and question.
    """

    system_prompt: str = (
        "You are an subject matter expert at social welfare regulations for the government in Basel, Switzerland. "
        "You are given a question and a context of documents that are relevant to the question. "
        "You are to answer the question based on the context and the conversation history. "
        "If you don't know the answer, say 'Entschuldigung, ich kann die Antwort nicht in den Dokumenten finden.' "
        "Don't try to make up an answer. "
        "Answer in German. "
        "For each statement in your answer, cite the source document id by adding it in brackets at the end of the sentence or paragraph like this: [file_id]"
        "If multiple documents support a statement, include all relevant citations like [file_id_1][file_id_2][file_id_3]"
        "Only cite documents that directly support your statements."
    )

    def __init__(self, llm: ChatOpenAI) -> None:
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()
        self.llm = llm

    def __call__(self, state: RAGState, config: RunnableConfig):
        """
        Generate an answer based on the context and question.

        Args:
            state: The current graph state
            config: The configuration for the graph

        Returns:
            dict: The answer and updated messages
        """
        self.logger.info("---GENERATE ANSWER---")
        if not state["messages"]:
            state["messages"] = [SystemMessage(content=self.system_prompt)]

        # Format context with document metadata for citations
        formatted_context: list[str] = []
        for idx, doc in enumerate(state["context"]):  # pyright: ignore[reportOptionalIterable]
            formatted_context.append(
                f"""
                <document>
                <content>
                {doc.page_content}
                </content>
                <file_id>
                {idx + 1}
                </file_id>
                </document>
                """
            )

        context: str = "\n\n".join(formatted_context)

        template = ChatPromptTemplate([
            ("user", "Context: {context}\n\nQuestion: {input}"),
        ])
        prompt = template.invoke({"context": context, "input": state["input"]}, config)
        messages: list[BaseMessage] = list(state["messages"]) + list(prompt.to_messages())
        response = self.llm.invoke(messages, config)
        return {"messages": [*messages, response], "answer": response.content}

    def update_handler(self, data: dict[str, Any]) -> StreamResponse:
        """
        Handles updates from the action and returns a StreamResponse.
        """
        return StreamResponse.create_answer_response(message="AI Antwort", answer_text=data.get("answer", ""))
