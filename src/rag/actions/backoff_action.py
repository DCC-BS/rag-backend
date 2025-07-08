from typing import Any, override

import structlog
from langchain_core.runnables import RunnableConfig

from rag.actions.action_protocol import ActionProtocol
from rag.models.rag_states import RAGState
from rag.models.stream_response import Sender, StreamResponse


class BackoffAction(ActionProtocol):
    """
    Action to handle errors.
    """

    def __init__(self) -> None:
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()

    @override
    def __call__(self, state: RAGState, config: RunnableConfig):
        """
        Handle errors.

        Args:
            state (dict): The current graph state

        Returns:
            dict: The updated graph state with the error message
        """

        self.logger.info("---BACKOFF ACTION---")
        return {"answer": "Es konnten keine relevanten Dokumente gefunden werden."}

    @override
    def update_handler(self, data: dict[str, Any]) -> StreamResponse:
        """
        Handles updates from the action and returns a StreamResponse.
        """
        return StreamResponse.create_answer_response(
            sender=Sender.BACKOFF,
            answer_text=data.get("answer", ""),
        )
