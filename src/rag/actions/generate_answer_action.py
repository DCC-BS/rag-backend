from datetime import datetime
from typing import Any, override

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
from rag.utils.logger import get_logger


class GenerateAnswerAction(ActionProtocol):
    """
    Action to generate an answer based on the context and question.
    """

    today_date: str = datetime.now().strftime("%Y-%m-%d")

    system_prompt: str = """You are an expert for the government in Basel, Switzerland.
You are given a question and a context of documents that are relevant to the question.
You are to answer the question based on the context and the conversation history.
If you don't know the answer, say 'Entschuldigung, ich kann die Antwort nicht in den Dokumenten finden.'
You can use tools to calculate.
For each statement in your answer, cite the source document id relevant to the statement.
Statements should be in the same language as the question of the user.
Statements may be multiple sentences.
Format your response using markdown.
If multiple documents support a statement, include all relevant citations.
Only cite documents that directly support your statements.
The structure of your answer should follow this schema:
The statement. [document_id, document_id, ...]
Example:
Apples are fruits. [1]
Fruits are incredibly healthy due to their rich content of essential vitamins, minerals, fiber, and beneficial plant compounds called phytochemicals and antioxidants. Here's a breakdown of why they're so good for you: [2,3,4]
1. ** Rich in Vitamins and Minerals: **\n * Fruits are packed with vitamins like Vitamin C (important for immunity and skin health), Vitamin A (for vision and immune function), and B vitamins (for energy and metabolism). \n * They also provide crucial minerals such as potassium (for healthy blood pressure and muscle function), magnesium, and folate. [2,3]"""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.logger = get_logger()
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
        self.logger.debug("---GENERATE ANSWER---")
        if not state.messages:
            state.messages = [SystemMessage(content=self.system_prompt)]

        context: str = ""
        if state.route_query and state.route_query == "retrieval":
            formatted_context: list[str] = []
            for idx, doc in enumerate(state.context):  # pyright: ignore[reportOptionalIterable]
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
        prompt = template.invoke({"context": context, "input": state.input}, config)
        messages: list[BaseMessage] = list(state.messages) + list(prompt.to_messages())
        writer = get_stream_writer()
        writer("chat.status.generatingAnswer")
        response: BaseMessage = self.llm.invoke(messages, config)
        print(response.content)
        return {"messages": [*messages, response]}

    @override
    def update_handler(self, data: dict[str, Any]) -> StreamResponse:
        """
        Handles updates from the action and returns a StreamResponse.
        """
        return StreamResponse.create_answer_response(sender=Sender.ANSWER_ACTION, answer_text=data.get("answer", ""))
