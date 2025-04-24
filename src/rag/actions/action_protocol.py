from typing import Any, Protocol

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import StateDict

from rag.models.stream_response import StreamResponse


class ActionProtocol(Protocol):
    """
    Protocol for RAG actions.
    """

    def __call__(self, state: StateDict, config: RunnableConfig) -> Any:
        """
        Executes the action.

        Args:
            state: The current graph state.
            config: The configuration for the graph.

        Returns:
            The result of the action.
        """
        ...

    def update_handler(self, data: dict[str, Any]) -> StreamResponse:
        """
        Handles updates from the action and returns a StreamResponse.

        Args:
            data: The update data from the action.

        Returns:
            A StreamResponse object.
        """
        ...
