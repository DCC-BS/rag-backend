from typing import Any, Protocol, override

import structlog
from langchain.schema import Document
from langchain_core.runnables import RunnableConfig

from rag.actions.action_protocol import ActionProtocol
from rag.models.rag_states import RAGState
from rag.models.stream_response import Sender, StreamResponse


class RetrieverProtocol(Protocol):
    def invoke(self, **kwargs: Any) -> list[Document]:
        """Invoke the retriever with any parameters and return a list of documents."""
        ...


class RetrieveAction(ActionProtocol):
    """
    Action to retrieve relevant documents.
    """

    def __init__(self, retriever: RetrieverProtocol) -> None:
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()
        self.retriever = retriever

    @override
    def __call__(self, state: RAGState, config: RunnableConfig) -> dict[str, list[Document]]:
        self.logger.info("---RETRIEVE DOCUMENTS---")
        docs: list[Document] = self.retriever.invoke(
            input=state.input,
            user_roles=state.user_organizations,
            document_ids=state.document_ids,
            config=config,
        )

        return {"context": docs}

    @override
    def update_handler(self, data: dict[str, Any]) -> StreamResponse:
        """
        Handles updates from the action and returns a StreamResponse.
        """
        return StreamResponse.create_document_response(sender=Sender.RETRIEVE_ACTION, docs=data.get("context", []))
