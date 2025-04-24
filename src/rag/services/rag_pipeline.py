from collections.abc import AsyncIterator

import structlog
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import SecretStr

from rag.actions.generate_answer_action import GenerateAnswerAction
from rag.actions.grade_answer_action import GradeAnswerAction
from rag.actions.grade_hallucination_action import GradeHallucinationAction
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
        self.route_question_action = RouteQuestionAction(llm=self.llm)
        self.retrieve_action = RetrieveAction(retriever=self.retriever)
        self.grade_hallucination_action = GradeHallucinationAction(llm=self.llm)
        self.grade_answer_action = GradeAnswerAction(llm=self.llm)
        self.generate_answer_action = GenerateAnswerAction(llm=self.llm)

        self.memory: MemorySaver = memory or MemorySaver()

        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """Create and compile the state graph."""
        workflow: StateGraph = StateGraph(state_schema=RAGState, input=InputState, output=OutputState)

        # Add nodes using the instantiated actions
        _ = workflow.add_node("route_question", self.route_question_action)
        _ = workflow.add_node("retrieve", self.retrieve_action)
        _ = workflow.add_node("grade_hallucination", self.grade_hallucination_action)
        _ = workflow.add_node("grade_answer", self.grade_answer_action)
        _ = workflow.add_node("generate_answer", self.generate_answer_action)

        # Add conditional edge based on skip_retrieval
        _ = workflow.add_edge(start_key=START, end_key="route_question")
        _ = workflow.add_edge(start_key="retrieve", end_key="generate_answer")
        _ = workflow.add_edge(start_key="generate_answer", end_key="grade_hallucination")

        # Add conditional edges for routing and grading
        _ = workflow.add_conditional_edges(
            start_key="route_question",
            condition=lambda state: state.get("route_query"),
            conditional_edge_mapping={
                "retrieval": "retrieve",
                "answer": "generate_answer",
            },
        )

        _ = workflow.add_conditional_edges(
            start_key="grade_hallucination",
            condition=lambda state: state.get("hallucination_score") is False,
            conditional_edge_mapping={
                True: "generate_answer",
                False: "grade_answer",
            },
        )

        _ = workflow.add_conditional_edges(
            start_key="grade_answer",
            condition=lambda state: state.get("answer_score") is True,
            conditional_edge_mapping={
                True: END,
                False: END,
            },
        )

        # Compile graph
        compiled_graph = workflow.compile(checkpointer=self.memory)
        try:
            _ = compiled_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
        except Exception as e:
            self.logger.info(f"Could not draw graph: {e}")

        return compiled_graph

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

    def _setup_llm(self, temperature: float | None = None):
        temperature = temperature or self.config.LLM.TEMPERATURE
        llm = ChatOpenAI(
            model=self.config.LLM.MODEL,
            temperature=temperature,
            api_key=SecretStr("None"),
            base_url=self.config.LLM.API_URL,
            max_completion_tokens=self.config.LLM.MAX_TOKENS,
        )

        return llm

    async def astream_query(
        self,
        question: str | None = None,
        user_organization: str | None = None,
        thread_id: str | None = None,
        resume_action: str | None = None,
        resume_data: str | None = None,
    ) -> AsyncIterator[StreamResponse]:
        if resume_action is not None:
            # Resume execution case
            user_input = Command(resume={"action": resume_action, "data": resume_data})
            if thread_id is None:
                raise ValueError("thread_id is required when resuming execution")
        else:
            # New query case
            if question is None or user_organization is None or thread_id is None:
                raise ValueError("question, user_organization, and thread_id are required for new queries")
            user_input = {"input": question, "user_organization": user_organization}

        recursion_limit = self.config.RETRIEVER.MAX_RECURSION
        config = RunnableConfig(recursion_limit=recursion_limit, configurable={"thread_id": thread_id})

        async for chunk in self.graph.astream(input=user_input, stream_mode=["updates"], config=config):
            kind, content = chunk
            if kind == "updates" and isinstance(content, dict):
                # Each update may include outputs from one or several nodes.
                for key, update in content.items():
                    if key in self.update_handlers:
                        yield self.update_handlers[key](update)
                    else:
                        self.logger.info(f"Unknown update: {key}")
