from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import structlog
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag.actions.backoff_action import BackoffAction
from rag.actions.generate_answer_action import GenerateAnswerAction
from rag.actions.grade_answer_action import GradeAnswerAction
from rag.actions.retrieve_action import RetrieveAction, RetrieverProtocol
from rag.actions.route_question_action import RouteQuestionAction
from rag.connectors.pg_retriever import PGRoleRetriever
from rag.models.rag_states import (
    RAGState,
)
from rag.models.stream_response import Sender, StreamResponse
from rag.utils.config import AppConfig, ConfigurationManager
from rag.utils.model_clients import get_llm_client


class SHRAGPipeline:
    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        retriever: RetrieverProtocol | None = None,
        memory: MemorySaver | None = None,
    ) -> None:
        self.logger: structlog.stdlib.BoundLogger = structlog.get_logger()
        self.config: AppConfig = ConfigurationManager.get_config()

        # Setup components
        self.retriever: RetrieverProtocol = retriever or self._setup_retriever()
        self.llm: ChatOpenAI = llm or get_llm_client(self.config).client

        # Instantiate actions
        self.route_question_action: RouteQuestionAction = RouteQuestionAction(llm=self.llm)
        self.retrieve_action: RetrieveAction = RetrieveAction(retriever=self.retriever)
        # self.grade_hallucination_action: GradeHallucinationAction = GradeHallucinationAction(llm=self.llm)
        self.grade_answer_action: GradeAnswerAction = GradeAnswerAction(llm=self.llm)
        self.generate_answer_action: GenerateAnswerAction = GenerateAnswerAction(llm=self.llm)
        self.backoff_action: BackoffAction = BackoffAction()

        # Map node names to their update handlers
        self.update_handlers: dict[str, Callable[[dict[str, Any]], StreamResponse]] = {
            "route_question": self.route_question_action.update_handler,
            "retrieve": self.retrieve_action.update_handler,
            # "grade_hallucination": self.grade_hallucination_action.update_handler,
            "grade_answer": self.grade_answer_action.update_handler,
            "generate_answer": self.generate_answer_action.update_handler,
            "backoff": self.backoff_action.update_handler,
        }

        self.memory: MemorySaver = memory or MemorySaver()

        self.graph: CompiledStateGraph[RAGState] = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph[RAGState]:
        """Create and compile the state graph."""
        # Create StateGraph with proper LangGraph 0.5 syntax
        workflow: StateGraph[RAGState] = StateGraph(RAGState)

        # Add nodes using the instantiated actions
        _ = workflow.add_node("route_question", self.route_question_action)
        _ = workflow.add_node("retrieve", self.retrieve_action)
        # _ = workflow.add_node("grade_hallucination", self.grade_hallucination_action)
        _ = workflow.add_node("grade_answer", self.grade_answer_action)
        _ = workflow.add_node("generate_answer", self.generate_answer_action)
        _ = workflow.add_node("backoff", self.backoff_action)

        # Add conditional edge based on skip_retrieval
        _ = workflow.add_edge(start_key=START, end_key="route_question")
        _ = workflow.add_edge(start_key="generate_answer", end_key="grade_answer")
        _ = workflow.add_edge(start_key="grade_answer", end_key=END)
        _ = workflow.add_edge(start_key="backoff", end_key=END)

        # Add conditional edges for routing and grading
        _ = workflow.add_conditional_edges(
            source="route_question",
            path=lambda state: state.route_query,
            path_map={
                "retrieval": "retrieve",
                "answer": "generate_answer",
            },
        )

        _ = workflow.add_conditional_edges(
            source="retrieve",
            path=lambda state: len(state.context) > 0,
            path_map={
                True: "generate_answer",
                False: "backoff",
            },
        )

        # _ = workflow.add_conditional_edges(
        #     source="grade_hallucination",
        #     path=lambda state: state.hallucination_score,
        #     path_map={
        #         True: "generate_answer",
        #         False: "grade_answer",
        #     },
        # )

        # Compile graph
        compiled_graph: CompiledStateGraph[RAGState] = workflow.compile(checkpointer=self.memory)
        try:
            _ = compiled_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
        except Exception as e:
            self.logger.info(f"Could not draw graph: {e}")

        return compiled_graph

    def _prepare_user_input(
        self,
        message: str | None,
        user_organizations: list[str] | None,
        thread_id: str | None,
        document_ids: list[int] | None,
    ) -> RAGState:
        if message is None or user_organizations is None or user_organizations == [] or thread_id is None:
            raise ValueError("message, user_organizations, and thread_id are required for new queries")

        return RAGState(
            input=message,
            user_organizations=user_organizations,
            document_ids=document_ids,
            messages=[],
            context=[],
            hallucination_score=None,
            answer_score=None,
            route_query=None,
            reason=None,
        )

    def _handle_updates_event(self, content: dict[str, Any]) -> Iterator[StreamResponse]:
        for key, update_content in content.items():
            if key == "generate_answer":
                continue
            if key in self.update_handlers:
                yield self.update_handlers[key](update_content)
            else:
                self.logger.info(f"Unknown update key: {key}, value: {update_content}")

    def _handle_messages_event(self, stream_content: Any) -> StreamResponse | None:
        message_chunk, metadata = stream_content

        if not isinstance(message_chunk, AIMessageChunk):
            return None
        if not isinstance(message_chunk.content, str) or not message_chunk.content:
            return None
        if not isinstance(metadata, dict) or metadata.get("langgraph_node") != "generate_answer":
            return None

        return StreamResponse.create_answer_response(sender=Sender.ANSWER_ACTION, answer_text=message_chunk.content)

    def _setup_retriever(self):
        retriever = PGRoleRetriever(
            reranker_api=self.config.RERANKER.API_URL,
            embedding_api=self.config.EMBEDDINGS.API_URL,
            embedding_instructions=self.config.EMBEDDINGS.EMBEDDING_INSTRUCTIONS,
            bm25_limit=self.config.RETRIEVER.BM25_LIMIT,
            vector_limit=self.config.RETRIEVER.VECTOR_LIMIT,
            top_k=self.config.RETRIEVER.RERANK_TOP_K,
        )
        return retriever

    async def astream_query(
        self,
        message: str | None = None,
        user_organizations: list[str] | None = None,
        thread_id: str | None = None,
        document_ids: list[int] | None = None,
    ) -> AsyncIterator[StreamResponse]:
        if user_organizations is None:
            user_organizations = []
        user_input: RAGState = self._prepare_user_input(message, user_organizations, thread_id, document_ids)
        # thread_id is validated by _prepare_user_input for its necessity.
        # It must be non-None if execution reaches here without an error.
        if thread_id is None:
            raise ValueError("thread_id is required for configuring the stream.")

        recursion_limit = 25
        config = RunnableConfig(recursion_limit=recursion_limit, configurable={"thread_id": thread_id})

        async for kind, stream_content in self.graph.astream(
            input=user_input, stream_mode=["updates", "messages", "custom"], config=config
        ):
            if kind == "custom":
                yield StreamResponse.create_status(translation_key=stream_content)
            if kind == "updates" and isinstance(stream_content, dict):
                for response in self._handle_updates_event(stream_content):
                    yield response
            elif kind == "messages":
                response = self._handle_messages_event(stream_content)
                if response:
                    yield response
