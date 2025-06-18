from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import structlog
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import SecretStr

from rag.actions.generate_answer_action import GenerateAnswerAction
from rag.actions.retrieve_action import RetrieveAction, RetrieverProtocol
from rag.actions.route_question_action import RouteQuestionAction
from rag.connectors.bento_embeddings import BentoMLReranker
from rag.connectors.document_storage import get_lancedb_doc_store
from rag.connectors.lance_retriever import LanceDBRetriever, LanceDBRetrieverConfig
from rag.models.rag_states import (
    InputState,
    OutputState,
    RAGState,
)
from rag.models.stream_response import StreamResponse
from rag.utils.config import AppConfig, ConfigurationManager


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
        self.llm: ChatOpenAI = llm or self._setup_llm()

        # Instantiate actions
        self.route_question_action: RouteQuestionAction = RouteQuestionAction(llm=self.llm)
        self.retrieve_action: RetrieveAction = RetrieveAction(retriever=self.retriever)
        # self.grade_hallucination_action: GradeHallucinationAction = GradeHallucinationAction(llm=self.llm)
        # self.grade_answer_action: GradeAnswerAction = GradeAnswerAction(llm=self.llm)
        self.generate_answer_action: GenerateAnswerAction = GenerateAnswerAction(llm=self.llm)

        # Map node names to their update handlers
        self.update_handlers: dict[str, Callable[[dict[str, Any]], StreamResponse]] = {
            "route_question": self.route_question_action.update_handler,
            "retrieve": self.retrieve_action.update_handler,
            # "grade_hallucination": self.grade_hallucination_action.update_handler,
            # "grade_answer": self.grade_answer_action.update_handler,
            # "generate_answer": self.generate_answer_action.update_handler,
        }

        self.memory: MemorySaver = memory or MemorySaver()

        self.graph: CompiledStateGraph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """Create and compile the state graph."""
        workflow: StateGraph = StateGraph(state_schema=RAGState, input=InputState, output=OutputState)

        # Add nodes using the instantiated actions
        _ = workflow.add_node("route_question", self.route_question_action)
        _ = workflow.add_node("retrieve", self.retrieve_action)
        # _ = workflow.add_node("grade_hallucination", self.grade_hallucination_action)
        # _ = workflow.add_node("grade_answer", self.grade_answer_action)
        _ = workflow.add_node("generate_answer", self.generate_answer_action)

        # Add conditional edge based on skip_retrieval
        _ = workflow.add_edge(start_key=START, end_key="route_question")
        _ = workflow.add_edge(start_key="retrieve", end_key="generate_answer")
        _ = workflow.add_edge(start_key="generate_answer", end_key=END)
        # _ = workflow.add_edge(start_key="generate_answer", end_key="grade_hallucination")
        # _ = workflow.add_edge(start_key="grade_answer", end_key=END)

        # Add conditional edges for routing and grading
        _ = workflow.add_conditional_edges(
            source="route_question",
            path=lambda state: state.get("route_query"),
            path_map={
                "retrieval": "retrieve",
                "answer": "generate_answer",
            },
        )

        # _ = workflow.add_conditional_edges(
        #     source="grade_hallucination",
        #     path=lambda state: state.get("hallucination_score"),
        #     path_map={
        #         True: "generate_answer",
        #         False: "grade_answer",
        #     },
        # )

        # Compile graph
        compiled_graph = workflow.compile(checkpointer=self.memory)
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
        resume_action: str | None,
        resume_data: str | None,
    ) -> dict[str, Any] | Command[Any]:
        if resume_action is not None:
            if thread_id is None:
                raise ValueError("thread_id is required when resuming execution")
            return Command(resume={"action": resume_action, "data": resume_data})

        if message is None or user_organizations is None or user_organizations == [] or thread_id is None:
            raise ValueError("message, user_organizations, and thread_id are required for new queries")
        return {"input": message, "user_organizations": user_organizations}

    def _handle_updates_event(self, content: dict[str, Any]) -> Iterator[StreamResponse]:
        for key, update_content in content.items():
            if key in self.update_handlers and key != "generate_answer":
                yield self.update_handlers[key](update_content)
            elif key != "generate_answer":
                self.logger.info(f"Unknown update key: {key}, value: {update_content}")

    def _handle_messages_event(self, stream_content: Any) -> StreamResponse | None:
        message_chunk, metadata = stream_content

        if not isinstance(message_chunk, AIMessageChunk):
            return None
        if not isinstance(message_chunk.content, str) or not message_chunk.content:
            return None
        if not isinstance(metadata, dict) or metadata.get("langgraph_node") != "generate_answer":
            return None

        return StreamResponse.create_answer_response(
            message="AI Antwort", sender="GenerateAnswerAction", answer_text=message_chunk.content
        )

    def _setup_retriever(self):
        vector_store = get_lancedb_doc_store()
        if vector_store.embeddings is None:
            raise ValueError("Embeddings are None")
        if vector_store._text_key is None:
            raise ValueError("Vector store has no text key")
        if vector_store._vector_key is None:
            raise ValueError("Vector store has no vector key")

        reranker = BentoMLReranker(api_url=self.config.EMBEDDINGS.API_URL, column=vector_store._text_key)
        retriever = LanceDBRetriever(
            name="LanceDBRetriever",
            table=vector_store.get_table(self.config.DOC_STORE.TABLE_NAME),
            embeddings=vector_store.embeddings,
            reranker=reranker,
            config=LanceDBRetrieverConfig(
                vector_col=vector_store._vector_key,
                fts_col=vector_store._text_key,
                k=self.config.RETRIEVER.TOP_K,
                docs_before_rerank=self.config.RETRIEVER.FETCH_FOR_RERANKING,
            ),
        )
        return retriever

    def _setup_llm(self):
        llm = ChatOpenAI(
            model=self.config.LLM.MODEL,
            api_key=SecretStr("None"),
            base_url=self.config.LLM.API_URL,
        )

        return llm

    async def astream_query(
        self,
        message: str | None = None,
        user_organizations: list[str] | None = None,
        thread_id: str | None = None,
        resume_action: str | None = None,
        resume_data: str | None = None,
    ) -> AsyncIterator[StreamResponse]:
        if user_organizations is None:
            user_organizations = []
        user_input = self._prepare_user_input(message, user_organizations, thread_id, resume_action, resume_data)
        self.logger.info(f"User input: {user_input}")
        # thread_id is validated by _prepare_user_input for its necessity.
        # It must be non-None if execution reaches here without an error.
        if thread_id is None:
            # This case should ideally not be hit if _prepare_user_input is comprehensive,
            # but as RunnableConfig requires it, an explicit check guards against misuse.
            raise ValueError("thread_id is required for configuring the stream.")

        recursion_limit = self.config.RETRIEVER.MAX_RECURSION
        config = RunnableConfig(recursion_limit=recursion_limit, configurable={"thread_id": thread_id})

        async for kind, stream_content in self.graph.astream(
            input=user_input, stream_mode=["updates", "messages"], config=config
        ):
            if kind == "updates" and isinstance(stream_content, dict):
                for response in self._handle_updates_event(stream_content):
                    yield response
            elif kind == "messages":
                response = self._handle_messages_event(stream_content)
                if response:
                    yield response
