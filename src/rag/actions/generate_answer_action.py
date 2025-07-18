from datetime import datetime
from typing import Any, override

import structlog
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer

from rag.actions.action_protocol import ActionProtocol
from rag.models.rag_states import RAGState
from rag.models.stream_response import Sender, StreamResponse
from rag.tools.calculator import add, divide, multiply, square, subtract


class GenerateAnswerAction(ActionProtocol):
    """
    Action to generate an answer based on the context and question.
    """

    today_date: str = datetime.now().strftime("%Y-%m-%d")

    system_prompt: str = f"""
        "You are an subject matter expert at social welfare regulations for the government in Basel, Switzerland. "
        "You are given a question and a context of documents that are relevant to the question. "
        "You are to answer the question based on the context and the conversation history. \n"
        "If you don't know the answer, say 'Entschuldigung, ich kann die Antwort nicht in den Dokumenten finden.' "
        "If there are no documents available, say 'Entschuldigung, ich konnte keine relevanten Dokumente finden.' "
        "Don't try to make up an answer. "
        "Answer in German. \n"
        "Use markdown formatting for your answer. \n"
        "You can use tools to calculate numbers. \n"
        "For each statement in your answer, cite the source document id by enclosing it in square brackets at the end of the sentence or paragraph like this: [file_id] \n"
        "If multiple documents support a statement, include all relevant citations like [file_id_1, file_id_2, file_id_3]\n"
        "Only cite documents that directly support your statements. \n\n"
        "Today's date is {today_date}. \n"
        "Example:\n"
        "Context:\n"
        "<document>\n"
        "<content>"
        "Die Haftpflichtversicherung ist eine Versicherung, die die Haftpflicht für die Versicherungsnehmer abdeckt.\n"
        "</content>\n"
        "<file_id>"
        "1"
        "</file_id>\n"
        "</document>\n"
        "<document>\n"
        "<content>"
        "In der Schweiz gilt die Versicherungspflicht. Das bedeutet, dass jeder Mensch, der in der Schweiz lebt, eine Krankenversicherung haben muss."
        "</content>\n"
        "<file_id>"
        "2"
        "</file_id>\n"
        "</document>\n\n"
        "Question: Was ist die Haftpflichtversicherung?\n"
        "Answer: Die Haftpflichtversicherung ist eine Versicherung, die die Haftpflicht für die Versicherungsnehmer abdeckt[1]. "
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()
        self.llm = llm.bind_tools([multiply, add, subtract, divide, square])

    @override
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

        context: str = ""
        if state["route_query"] and state["route_query"] == "retrieval":
            formatted_context: list[str] = []
            for idx, doc in enumerate(state["context"]):  # pyright: ignore[reportOptionalIterable]
                formatted_context.append(
                    f"""
                    <document>
                    <content>
                    {doc.page_content}
                    </content>
                    <file_id>
                    [{idx + 1}]
                    </file_id>
                    </document>
                    """
                )

            context = "\n\n".join(formatted_context)

        template = ChatPromptTemplate([
            ("user", "Context: {context}\n\nQuestion: {input} \nothink"),
        ])
        prompt = template.invoke({"context": context, "input": state["input"]}, config)
        messages: list[BaseMessage] = list(state["messages"]) + list(prompt.to_messages())
        writer = get_stream_writer()
        writer("chat.status.generatingAnswer")
        response = self.llm.invoke(messages, config)
        return {"messages": [*messages, response]}

    @override
    def update_handler(self, data: dict[str, Any]) -> StreamResponse:
        """
        Handles updates from the action and returns a StreamResponse.
        """
        return StreamResponse.create_answer_response(sender=Sender.ANSWER_ACTION, answer_text=data.get("answer", ""))
