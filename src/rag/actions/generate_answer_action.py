from collections.abc import Sequence
from datetime import datetime
from typing import Any, cast, override

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.types import StreamWriter
from openai import BadRequestError

from rag.actions.action_protocol import ActionProtocol
from rag.models.rag_states import RAGState
from rag.models.stream_response import Sender, StreamResponse
from rag.tools.calculator import add, divide, multiply, square, subtract
from rag.utils.logger import get_logger
from rag.utils.message_summarizer import MessageHistorySummarizer


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
The statement. <ref>document_id</ref> <ref>document_id</ref>
Example:
Apples are fruits. <ref>1</ref>
Fruits are incredibly healthy due to their rich content of essential vitamins, minerals, fiber, and beneficial plant compounds called phytochemicals and antioxidants. Here's a breakdown of why they're so good for you: <ref>2</ref> <ref>3</ref> <ref>4</ref>
1. ** Rich in Vitamins and Minerals: **\n * Fruits are packed with vitamins like Vitamin C (important for immunity and skin health), Vitamin A (for vision and immune function), and B vitamins (for energy and metabolism). \n * They also provide crucial minerals such as potassium (for healthy blood pressure and muscle function), magnesium, and folate. <ref>2</ref> <ref>3</ref>"""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.logger = get_logger()
        self.llm = llm.bind_tools([multiply, add, subtract, divide, square])
        self.summarizer = MessageHistorySummarizer(llm=llm)

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
        self._ensure_system_message(state)

        context = self._build_document_context(state)
        prompt_with_context = self._create_context_prompt(context, state.input, config)
        messages_for_generation = list(state.messages) + list(prompt_with_context.to_messages())

        writer = get_stream_writer()
        writer("chat.status.generatingAnswer")

        response = self._generate_with_retry(state, messages_for_generation, prompt_with_context, config, writer)
        user_message = self._create_user_message(state.input, config)

        return {"messages": [*list(user_message.to_messages()), response]}

    def _ensure_system_message(self, state: RAGState) -> None:
        """Ensure the state has a system message."""
        if not state.messages:
            state.messages = [SystemMessage(content=self.system_prompt)]

    def _build_document_context(self, state: RAGState) -> str:
        """Build formatted document context from state."""
        if not (state.route_query and state.route_query == "retrieval"):
            return ""

        formatted_docs = [
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
            for idx, doc in enumerate(state.context)  # pyright: ignore[reportOptionalIterable]
        ]
        return "\n\n".join(formatted_docs)

    def _create_context_prompt(self, context: str, user_input: str, config: RunnableConfig):
        """Create prompt with context for generation."""
        template = ChatPromptTemplate([("user", "Context: {context}\n\nQuestion: {input} \nothink")])
        return template.invoke({"context": context, "input": user_input}, config)

    def _create_user_message(self, user_input: str, config: RunnableConfig):
        """Create clean user message without context."""
        template = ChatPromptTemplate([("user", "{input}")])
        return template.invoke({"input": user_input}, config)

    def _generate_with_retry(
        self,
        state: RAGState,
        messages_for_generation: Sequence[BaseMessage],
        prompt_with_context,
        config: RunnableConfig,
        writer: StreamWriter,
    ) -> BaseMessage:
        """Generate answer with automatic retry on context length error."""
        try:
            return self.llm.invoke(messages_for_generation, config)
        except BadRequestError as e:
            if self._is_context_length_error(e):
                return self._handle_context_length_error(state, prompt_with_context, config, writer, e)
            raise

    def _is_context_length_error(self, error: BadRequestError) -> bool:
        """Check if error is due to context length exceeded."""
        return "maximum context length" in str(error).lower() and error.status_code == 400

    def _handle_context_length_error(
        self,
        state: RAGState,
        prompt_with_context,
        config: RunnableConfig,
        writer: StreamWriter,
        error: BadRequestError,
    ) -> BaseMessage:
        """Handle context length error by summarizing and retrying."""
        self.logger.warning(
            "Context length exceeded, summarizing message history",
            error=str(error),
            original_message_count=len(state.messages),
        )

        summarized_messages = self._summarize_history(state)
        messages_for_generation = summarized_messages + list(prompt_with_context.to_messages())

        self.logger.info("Retrying with summarized message history")
        writer("chat.status.retryingWithSummarizedHistory")

        response = self._retry_with_summary(messages_for_generation, config)
        state.messages = cast(list[AnyMessage], summarized_messages)

        return response

    def _summarize_history(self, state: RAGState) -> list[BaseMessage]:
        """Summarize message history to reduce token count."""
        return self.summarizer.summarize_messages(
            messages=list(state.messages),
            system_prompt=self.system_prompt,
        )

    def _retry_with_summary(self, messages: Sequence[BaseMessage], config: RunnableConfig) -> BaseMessage:
        """Retry LLM invocation with summarized messages."""
        try:
            return self.llm.invoke(messages, config)
        except BadRequestError as retry_error:
            self.logger.exception("Failed to generate answer even after summarization", error=str(retry_error))
            raise

    @override
    def update_handler(self, data: dict[str, Any]) -> StreamResponse:
        """
        Handles updates from the action and returns a StreamResponse.
        """
        return StreamResponse.create_answer_response(sender=Sender.ANSWER_ACTION, answer_text=data.get("answer", ""))
