import structlog
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class MessageHistorySummarizer:
    """
    Summarizes message history to reduce token count.

    This class handles the summarization of conversation history when the context
    window is approaching its limit. It uses an LLM to create a concise summary
    of the conversation while preserving important context.
    """

    SUMMARIZATION_SYSTEM_PROMPT = """You are an expert at summarizing conversations. Create a concise but comprehensive summary of the following conversation.
The summary should:
- Preserve key information, facts, and context from the conversation
- Maintain the chronological flow of the discussion
- Keep important details that might be referenced later
- Be written in a clear, structured format
- Be significantly shorter than the original conversation
- Use the same language as the conversation

Format your summary as a narrative that captures the essence of the conversation."""

    def __init__(self, llm: ChatOpenAI) -> None:
        """
        Initialize the MessageHistorySummarizer.

        Args:
            llm: The language model client to use for summarization.
        """
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()
        self.llm: ChatOpenAI = llm

    def summarize_messages(
        self, messages: list[AnyMessage], system_prompt: str | None = None, keep_last_n: int = 1
    ) -> list[BaseMessage]:
        """
        Summarize the message history to reduce token count.

        Args:
            messages: The full message history to summarize.
            system_prompt: The system prompt to preserve (not included in summary).

        Returns:
            A new message list with the system prompt, summary, and current message.
        """
        self.logger.info("Starting message history summarization")

        system_message, messages_to_summarize = self._extract_messages(messages)
        # Split into prior history and messages to keep (e.g., the latest user turn)
        keep_last_n = max(0, keep_last_n)
        history, tail = (
            (messages_to_summarize[:-keep_last_n], messages_to_summarize[-keep_last_n:])
            if keep_last_n > 0
            else (messages_to_summarize, [])
        )

        if not history:
            self.logger.warning("No messages to summarize, returning original messages")
            return list(messages)

        summary_text = self._generate_summary(history)
        new_messages = self._build_message_list(system_message, system_prompt, summary_text)
        new_messages.extend(tail)

        self.logger.info(
            "Message history summarized",
            original_message_count=len(messages),
            new_message_count=len(new_messages),
        )

        return new_messages

    def _extract_messages(self, messages: list[AnyMessage]) -> tuple[SystemMessage | None, list[AnyMessage]]:
        """Extract system message and messages to summarize."""
        system_message: SystemMessage | None = None
        messages_to_summarize: list[AnyMessage] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_message = msg
            else:
                messages_to_summarize.append(msg)

        return system_message, messages_to_summarize

    def _generate_summary(self, messages: list[AnyMessage]) -> str:
        """Generate a summary of the conversation."""
        conversation_text = self._format_messages_for_summary(messages)
        summary_prompt = self._create_summarization_prompt()

        self.logger.debug("Generating conversation summary")
        summary_chain = summary_prompt | self.llm
        summary_response: BaseMessage = summary_chain.invoke({"conversation": conversation_text})

        return str(summary_response.content)

    def _create_summarization_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for summarization."""
        return ChatPromptTemplate.from_messages([
            ("system", self.SUMMARIZATION_SYSTEM_PROMPT),
            ("user", "Please summarize the following conversation:\n\n{conversation}"),
        ])

    def _build_message_list(
        self,
        system_message: SystemMessage | None,
        system_prompt: str | None,
        summary_text: str,
    ) -> list[BaseMessage]:
        """Build the new message list with system prompt and summary."""
        new_messages: list[BaseMessage] = []

        if system_message:
            new_messages.append(system_message)
        elif system_prompt:
            new_messages.append(SystemMessage(content=system_prompt))

        summary_content = f"[Conversation Summary - Previous Context]\n\n{summary_text}"
        new_messages.append(SystemMessage(content=summary_content))

        return new_messages

    def _format_messages_for_summary(self, messages: list[AnyMessage]) -> str:
        """Format messages into a readable text format for summarization."""
        formatted_lines = [
            f"{self._get_message_role(msg)}: {content}"
            for msg in messages
            if (content := self._get_message_content(msg))
        ]
        return "\n\n".join(formatted_lines)

    def _get_message_role(self, message: AnyMessage) -> str:
        """Get the role/type of a message."""
        role_mapping = {
            "HumanMessage": "User",
            "AIMessage": "Assistant",
            "SystemMessage": "System",
            "FunctionMessage": "Function",
            "ToolMessage": "Tool",
        }
        return role_mapping.get(message.__class__.__name__, message.__class__.__name__)

    def _get_message_content(self, message: AnyMessage) -> str:
        """Extract content from a message."""
        if not hasattr(message, "content"):
            return ""

        content = message.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return self._extract_text_from_list(content)

        return str(content)

    def _extract_text_from_list(self, content_list: list[str | dict[str, str]]) -> str:
        """Extract text from structured content list."""
        text_parts = [
            str(item["text"]) if isinstance(item, dict) and "text" in item else str(item)
            for item in content_list
            if isinstance(item, dict | str)
        ]
        return " ".join(text_parts)
