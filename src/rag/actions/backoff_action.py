from typing import Any, override

from langchain_core.runnables import RunnableConfig

from rag.actions.action_protocol import ActionProtocol
from rag.models.rag_states import RAGState
from rag.models.stream_response import Sender, StreamResponse
from rag.utils.logger import get_logger


class BackoffAction(ActionProtocol):
    """
    Action to handle errors.
    """

    def __init__(self) -> None:
        self.logger = get_logger()

    @override
    def __call__(self, state: RAGState, config: RunnableConfig):
        """
        Handle errors.

        Args:
            state (dict): The current graph state

        Returns:
            dict: The updated graph state with the error message
        """

        self.logger.debug("---BACKOFF ACTION---")
        return {"answer": "Es konnten keine relevanten Dokumente gefunden werden."}

    @override
    def update_handler(self, data: dict[str, Any] | None) -> StreamResponse:
        """
        Handles updates from the action and returns a StreamResponse.
        """
        return StreamResponse.create_answer_response(
            sender=Sender.BACKOFF,
            answer_text=data.get("answer", "") if data else "",
        )
