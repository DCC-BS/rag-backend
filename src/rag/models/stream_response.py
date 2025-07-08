from enum import Enum

from langchain.schema import Document
from pydantic import BaseModel


class StreamResponseType(str, Enum):
    """Type of stream response"""

    STATUS = "status"
    DOCUMENTS = "documents"
    ANSWER = "answer"
    INTERRUPT = "interrupt"
    DECISION = "decision"


class Sender(str, Enum):
    """Sender of the stream response"""

    RETRIEVE_ACTION = "retrieve_action"
    ROUTE_QUESTION_ACTION = "should_retrieve"
    ANSWER_ACTION = "answer_action"
    GRADE_ACTION = "is_truthful"
    GRADE_HALLUCINATION_ACTION = "is_hallucination"
    BACKOFF = "backoff"
    STATUS = "status"


class StreamResponse(BaseModel):
    """Unified model for stream responses from the RAG pipeline

    This model unifies different types of responses that can be returned by the RAG pipeline:
    - Status updates (simple messages about pipeline progress)
    - Document responses (when documents are retrieved)
    - Answer responses (when the AI generates an answer)
    - Interrupt responses (when user interaction is required)
    - Decision responses (when the AI makes a decision)

    Each response has a type, sender, and optional additional metadata depending on the type.
    """

    type: StreamResponseType
    sender: Sender
    metadata: dict[str, str | list[Document] | bool] | None = None

    @classmethod
    def create_status(cls, translation_key: str) -> "StreamResponse":
        """Create a simple status update response"""
        return cls(type=StreamResponseType.STATUS, sender=Sender.STATUS, metadata={"translation_key": translation_key})

    @classmethod
    def create_document_response(cls, sender: Sender, docs: list[Document]) -> "StreamResponse":
        """Create a response containing retrieved documents"""
        return cls(type=StreamResponseType.DOCUMENTS, sender=sender, metadata={"documents": docs})

    @classmethod
    def create_answer_response(cls, sender: Sender, answer_text: str) -> "StreamResponse":
        """Create a response containing an AI generated answer"""
        return cls(type=StreamResponseType.ANSWER, sender=sender, metadata={"answer": answer_text})

    @classmethod
    def create_interrupt_response(
        cls, sender: Sender, interrupt_data: dict[str, str | list[Document] | bool]
    ) -> "StreamResponse":
        """Create a response for user interaction interrupts"""
        return cls(type=StreamResponseType.INTERRUPT, sender=sender, metadata=interrupt_data)

    @classmethod
    def create_decision_response(
        cls, sender: Sender, metadata: dict[str, str | list[Document] | bool]
    ) -> "StreamResponse":
        """Create a response containing a decision"""
        return cls(type=StreamResponseType.DECISION, sender=sender, metadata=metadata)
