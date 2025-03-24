from enum import Enum

from langchain.schema import Document
from pydantic import BaseModel


class StreamResponseType(str, Enum):
    """Type of stream response"""

    STATUS = "status"
    DOCUMENTS = "documents"
    ANSWER = "answer"
    INTERRUPT = "interrupt"


class StreamResponse(BaseModel):
    """Unified model for stream responses from the RAG pipeline

    This model unifies different types of responses that can be returned by the RAG pipeline:
    - Status updates (simple messages about pipeline progress)
    - Document responses (when documents are retrieved)
    - Answer responses (when the AI generates an answer)
    - Interrupt responses (when user interaction is required)

    Each response has a type, message, and optional additional data depending on the type.
    """

    type: StreamResponseType
    message: str
    decision: str | None = None
    documents: list[Document] | None = None
    answer: str | None = None
    metadata: dict[str, str] | None = None

    @classmethod
    def create_status(cls, message: str, decision: str | None = None) -> "StreamResponse":
        """Create a simple status update response"""
        return cls(type=StreamResponseType.STATUS, message=message, decision=decision)

    @classmethod
    def create_document_response(cls, message: str, docs: list[Document]) -> "StreamResponse":
        """Create a response containing retrieved documents"""
        return cls(type=StreamResponseType.DOCUMENTS, message=message, documents=docs)

    @classmethod
    def create_answer_response(cls, message: str, answer_text: str) -> "StreamResponse":
        """Create a response containing an AI generated answer"""
        return cls(type=StreamResponseType.ANSWER, message=message, answer=answer_text)

    @classmethod
    def create_interrupt_response(cls, message: str, interrupt_data: dict[str, str]) -> "StreamResponse":
        """Create a response for user interaction interrupts"""
        return cls(type=StreamResponseType.INTERRUPT, message=message, metadata=interrupt_data)
