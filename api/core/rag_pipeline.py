import asyncio
import re
from collections.abc import Iterator
from typing import AsyncIterator, Literal

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt, interrupt
from omegaconf.listconfig import ListConfig
from omegaconf.omegaconf import DictConfig
from pydantic import SecretStr

from core.bento_embeddings import BentoMLReranker
from core.lance_retriever import LanceDBRetriever, LanceDBRetrieverConfig
from core.rag_states import (
    GradeAnswer,
    GradeDocuments,
    GradeHallucination,
    InputState,
    OutputState,
    RAGState,
    RouteQuery,
)
from data.document_storage import get_lancedb_doc_store
from utils.config import get_config
from utils.logging import setup_logger
from utils.stream_response import StreamResponse


class SHRAGPipeline:
    def __init__(self) -> None:
        self.logger = setup_logger()
        self.config: DictConfig | ListConfig = get_config()
        self.system_prompt: str = (
            "You are an subject matter expert at social welfare regulations for the government in Basel, Switzerland. "
            "You are given a question and a context of documents that are relevant to the question. "
            "You are to answer the question based on the context and the conversation history. "
            "If you don't know the answer, say 'Entschuldigung, ich kann die Antwort nicht in den Dokumenten finden.' "
            "Don't try to make up an answer. "
            "Answer in German. "
            "For each statement in your answer, cite the source document using [filename] format at the end of the relevant sentence or paragraph. "
            "If multiple documents support a statement, include all relevant citations like [doc1][doc2]. "
            "Only cite documents that directly support your statements."
        )

        # Setup components
        self.retriever = self._setup_retriever()
        self.llm = self._setup_llm()
        self.query_rewriter_llm = self._setup_llm(temperature=0.8)
        self.structured_llm_router = self.llm.with_structured_output(
            RouteQuery, method="json_schema"
        )
        self.structured_llm_grade_documents = self.llm.with_structured_output(
            GradeDocuments, method="json_schema"
        )
        self.structured_llm_grade_hallucination = self.llm.with_structured_output(
            GradeHallucination, method="json_schema"
        )
        self.structured_llm_grade_answer = self.llm.with_structured_output(
            GradeAnswer, method="json_schema"
        )
        self.memory: MemorySaver = MemorySaver()

        # Setup update handlers
        self.update_handlers = {
            "retrieve": lambda data: StreamResponse.create_document_response(
                message="Relevante Dokumente gefunden", docs=data.get("context", [])
            ),
            "route_question": lambda data: StreamResponse.create_status(
                message=f"{'Suche relevante Dokumente' if data.get('route_query') == 'retrieval' else 'Antworte auf die Frage'}",
            ),
            "filter_documents": lambda data: StreamResponse.create_status(
                message=f"{len(data.get('context', []))} Dokumente sind relevant."
            ),
            "transform_query": lambda data: StreamResponse.create_status(
                message=f"Frage umformuliert: {data.get('input')}"
            ),
            "query_needs_rephrase": lambda data: StreamResponse.create_status(
                message="Die Frage muss umformuliert werden",
                decision="Ja" if data.get("needs_rephrase") else "Nein",
            ),
            "grade_hallucination": lambda data: StreamResponse.create_status(
                message="Antwort enthält Halluzinationen",
                decision="Ja" if data.get("hallucination_score") else "Nein",
            ),
            "grade_answer": lambda data: StreamResponse.create_status(
                message="Antwort ist relevant",
                decision="Ja" if data.get("answer_score") else "Nein",
            ),
        }

        # Create graph
        workflow: StateGraph = StateGraph(
            state_schema=RAGState, input=InputState, output=OutputState
        )
        # Add nodes
        workflow.add_node("route_question", self.route_question)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("filter_documents", self.filter_documents)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node(
            "user_review_rewritten_query", self.user_review_rewritten_query
        )
        workflow.add_node("query_needs_rephrase", self.decide_if_query_needs_rewriting)
        workflow.add_node("grade_hallucination", self.grade_hallucination)
        workflow.add_node("grade_answer", self.grade_answer)
        workflow.add_node("generate_answer", self.generate_answer)

        # Add conditional edge based on skip_retrieval
        workflow.add_edge(start_key=START, end_key="route_question")
        workflow.add_edge(start_key="retrieve", end_key="filter_documents")
        workflow.add_edge(start_key="filter_documents", end_key="query_needs_rephrase")
        workflow.add_edge(
            start_key="transform_query", end_key="user_review_rewritten_query"
        )
        workflow.add_edge(start_key="generate_answer", end_key="grade_hallucination")
        workflow.add_edge(start_key="grade_hallucination", end_key="grade_answer")

        # Compile graph
        self.graph = workflow.compile(checkpointer=self.memory)
        self.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    def retrieve(
        self, state: RAGState, config: RunnableConfig
    ) -> dict[str, list[Document]]:
        docs: list[Document] = self.retriever.invoke(
            input=state["input"],
            user_organization=state["user_organization"],
            config=config,
        )

        return {"context": docs}

    def route_question(
        self, state: RAGState, config: RunnableConfig
    ) -> Command[Literal["generate_answer", "retrieve"]]:
        """
        Route question to web search or RAG.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        self.logger.info("---ROUTE QUESTION---")
        system = """You are an expert at routing a user question to a vectorstore or llm.
        If you can answer the question based on the conversation history, return "answer".
        If you need more information, return "retrieval"."""
        context_messages = [
            (message.type, message.content) for message in state["messages"][1:]
        ]
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                *context_messages,  # pyright: ignore[reportArgumentType]
                (
                    "user",
                    "Can you answer the following question based on the conversation history? Question: {question}",
                ),
            ]
        )
        question_router = route_prompt | self.structured_llm_router
        routing_result: RouteQuery = question_router.invoke(
            {"question": state["input"]}, config
        )  # pyright: ignore[reportAssignmentType]
        if routing_result.next_step == "answer":
            self.logger.info("---ROUTE QUESTION TO ANSWER GENERATION---")
            return Command(
                update={"route_query": "answer"},
                goto="generate_answer",
            )
        elif routing_result.next_step == "retrieval":
            self.logger.info("---ROUTE QUESTION TO RETRIEVAL---")
            return Command(
                update={"route_query": "retrieval"},
                goto="retrieve",
            )
        else:
            raise ValueError(
                f"Unexpected routing next step: {routing_result.next_step}"
            )

    def grade_hallucination(
        self, state: RAGState, config: RunnableConfig
    ) -> Command[Literal["generate_answer", "grade_answer"]]:
        """
        Grade hallucination on retrieved documents and answer.
        """
        self.logger.info("---GRADE HALLUCINATION---")
        system = """You are a hallucination checker. You are given a list of retrieved documents and an answer.
        You are to grade the answer based on the retrieved documents.
        If the answer is not grounded in the retrieved documents, return 'yes'.
        If the answer is grounded in the retrieved document, return 'no'.
        """
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("user", "Retrieved documents: {documents} \\n Answer: {answer}"),
            ]
        )
        hallucination_grader = (
            hallucination_prompt | self.structured_llm_grade_hallucination
        )
        if state["context"] is None:
            raise ValueError("Context is None")
        hallucination_result: GradeHallucination = hallucination_grader.invoke(
            {
                "documents": "\n\n".join(
                    [doc.page_content for doc in state["context"]]
                ),
                "answer": state["answer"],
            },
            config,
        )  # pyright: ignore[reportAssignmentType]
        if hallucination_result.binary_score == "yes":
            self.logger.info("---HALLUCINATION DETECTED---")
            return Command(
                update={"hallucination_score": True},
                goto="generate_answer",
            )
        else:
            self.logger.info("---HALLUCINATION NOT DETECTED---")
            return Command(
                update={"hallucination_score": False},
                goto="grade_answer",
            )

    def grade_answer(
        self, state: RAGState, config: RunnableConfig
    ) -> Command[Literal["transform_query"]]:  # pyright: ignore[reportInvalidTypeForm]
        """
        Grade answer generation.
        """
        self.logger.info("---GRADE ANSWER---")
        system = """You are a grader assessing relevance of an answer to a user question. \n 
            If the answer is relevant to the question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous answers. \n
            Give a binary score 'yes' or 'no' score to indicate whether the answer is relevant to the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("user", "Answer: {answer} \\n User question: {question}"),
            ]
        )
        answer_grader = answer_prompt | self.structured_llm_grade_answer
        answer_result: GradeAnswer = answer_grader.invoke(
            {"answer": state["answer"], "question": state["input"]}, config
        )  # pyright: ignore[reportAssignmentType]
        if answer_result.binary_score == "yes":
            self.logger.info("---ANSWER IS RELEVANT---")

            return Command(
                update={"answer_score": True},
                goto=END,
            )
        else:
            self.logger.info("---ANSWER IS NOT RELEVANT---")
            return Command(
                update={"answer_score": False},
                goto="transform_query",
            )

    def filter_documents(self, state: RAGState, config: RunnableConfig):
        """
        Filter retrieved documents based on relevance to the user question.

        Args:
            state (dict): The current graph state
            config (dict): The configuration for the graph

        Returns:
            dict: The filtered documents
        """
        self.logger.info("---FILTER DOCUMENTS---")
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "user",
                    "Retrieved document: {document} \\n User question: {question}",
                ),
            ]
        )
        grade_documents_llm = grade_prompt | self.structured_llm_grade_documents
        question: str = state["input"]
        relevant_documents: list[Document] = []
        for doc in state["context"]:  # pyright: ignore[reportOptionalIterable]
            grade_documents_result: GradeDocuments = grade_documents_llm.invoke(
                {"document": doc.page_content, "question": question}, config
            )  # pyright: ignore[reportAssignmentType]
            if grade_documents_result.binary_score == "no":
                self.logger.info(
                    f"Document {doc.metadata['filename']} is not relevant to the question."
                )
            else:
                relevant_documents.append(doc)
        return {"context": relevant_documents, "input": question}

    def decide_if_query_needs_rewriting(
        self, state: RAGState, config: RunnableConfig
    ) -> Command[Literal["transform_query", "generate_answer"]]:
        """
        Decide if the query needs to be re-written.
        """
        self.logger.info("---DECIDE IF QUERY NEEDS REWRITING---")
        filtered_documents: list[Document] = state["context"]  # pyright: ignore[reportOptionalIterable, reportAssignmentType]
        if len(filtered_documents) == 0:
            self.logger.info("---QUERY NEEDS REWRITING---")
            return Command(
                update={"needs_rephrase": True},
                goto="transform_query",
            )

        else:
            self.logger.info("---QUERY DOES NOT NEED REWRITING---")
            return Command(
                update={"needs_rephrase": False},
                goto="generate_answer",
            )

    def user_review_rewritten_query(
        self, state: RAGState, config: RunnableConfig
    ) -> Command[Literal["retrieve", "transform_query"]]:
        """
        User review of rewritten query. There are two cases:
        1. The model needs more information to create a better query
        2. The model has rewritten the query and needs user approval
        """
        self.logger.info("---USER REVIEW REWRITTEN QUERY---")
        rewritten_query: str = state["input"]

        if state["requires_more_information"]:
            human_review = interrupt(
                value={
                    "type": "needs_information",
                    "question": "Um die Frage zu beantworten, benötige ich weitere Informationen. Bitte geben Sie eine detailliertere Frage ein.",
                    "rewritten_query": rewritten_query,
                }
            )
            if human_review["action"] == "provide_information":
                return Command(
                    update={"input": human_review["data"]},
                    goto="transform_query",
                )
            else:
                raise ValueError(f"Unexpected review action: {human_review['action']}")
        else:
            # Case 2: Model has rewritten query and needs approval
            human_review = interrupt(
                value={
                    "type": "needs_approval",
                    "question": "Ist die umgeformte Frage korrekt?",
                    "rewritten_query": rewritten_query,
                }
            )
            if human_review["action"] == "accept":
                return Command(goto="retrieve", update={"input": rewritten_query})
            elif human_review["action"] == "modify":
                return Command(
                    update={"input": human_review["data"]},
                    goto="retrieve",
                )
            else:
                raise ValueError(f"Unexpected review action: {human_review['action']}")

    def transform_query(self, state: RAGState, config: RunnableConfig):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        self.logger.info("---TRANSFORM QUERY---")
        system = """You are an expert question re-writer that converts an input question to a better version that is optimized 
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
            The vectorstore contains documents from this domain: {document_description}.
            Write the re-phrased retrieval query in German as a question enclosed in <query> tags.
            The re-phrased question should be a question that can be answered by the vectorstore.
            The re-phrased question must be relevant to the initial user's question and keep the same meaning and intent.
            If you need more information, return "more_information".

            For example, if the input question is "Kindergeld", the re-phrased query should be 
            <query>Habe ich Anspruch auf Kindergeld?</query>.
            """
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "user",
                    "Here is the initial question: \n\n {question} \n Formulate an improved query enclosed in <query> tags.",
                ),
            ]
        )
        question_rewriter = (
            re_write_prompt | self.query_rewriter_llm | StrOutputParser()
        )
        question = state["input"]
        document_description = self.config.DOC_STORE.DOCUMENT_DESCRIPTION

        # Re-write question
        better_question = question_rewriter.invoke(
            {"question": question, "document_description": document_description}, config
        )
        if better_question == "more_information":
            return {"needs_rephrase": True, "requires_more_information": True}
        # Keep only content between <query> and </query> tags
        match = re.search(r"<query>(.*?)</query>", better_question)
        if match is None:
            raise ValueError("Could not find query in re-written question")
        better_question = match.group(1)

        return {
            "input": better_question,
            "needs_rephrase": False,
            "requires_more_information": False,
        }

    async def generate_answer(self, state: RAGState, config: RunnableConfig):
        if not state["messages"]:
            state["messages"] = [SystemMessage(content=self.system_prompt)]

        # Format context with document metadata for citations
        formatted_context: list[str] = []
        for doc in state["context"]:  # pyright: ignore[reportOptionalIterable]
            formatted_context.append(
                f"Content: {doc.page_content}\n"
                f"Source: {doc.metadata.get('filename', 'unknown')}"
            )

        context: str = "\n\n".join(formatted_context)

        template = ChatPromptTemplate(
            [
                ("user", "Context: {context}\n\nQuestion: {input}"),
            ]
        )
        prompt = await template.ainvoke(
            {"context": context, "input": state["input"]}, config
        )
        messages = state["messages"] + prompt.messages  # pyright: ignore
        response = await self.llm.ainvoke(messages, config)
        return {"messages": messages + [response], "answer": response.content}

    def _setup_retriever(self):
        vector_store = get_lancedb_doc_store()
        reranker = BentoMLReranker(
            api_url=self.config.EMBEDDINGS.API_URL, column=vector_store._text_key
        )
        if vector_store.embeddings is None:
            raise ValueError("Embeddings are None")
        if vector_store._text_key is None:
            raise ValueError("Vector store has no text key")
        if vector_store._vector_key is None:
            raise ValueError("Vector store has no vector key")
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
        latest_message_run_id = None
        if resume_action is not None:
            # Resume execution case
            input = Command(resume={"action": resume_action, "data": resume_data})
            if thread_id is None:
                raise ValueError("thread_id is required when resuming execution")
        else:
            # New query case
            if question is None or user_organization is None or thread_id is None:
                raise ValueError(
                    "question, user_organization, and thread_id are required for new queries"
                )
            input = {"input": question, "user_organization": user_organization}

        recursion_limit = self.config.RETRIEVER.MAX_RECURSION
        config = RunnableConfig(
            recursion_limit=recursion_limit, configurable={"thread_id": thread_id}
        )

        async for chunk in self.graph.astream(
            input=input, stream_mode=["updates", "messages"], config=config
        ):
            kind, content = chunk
            if kind == "updates":
                # Each update may include outputs from one or several nodes.
                if isinstance(content, dict):
                    for key, update in content.items():
                        if key in self.update_handlers:
                            yield self.update_handlers[key](update)
                        elif key == "__interrupt__":
                            self.logger.info(f"Interrupt: {content}")
                            interrupt = update[0]
                            if isinstance(interrupt, Interrupt):
                                question_to_user = interrupt.value["question"]
                                rewritten_query = interrupt.value["rewritten_query"]
                                interrupt_type = interrupt.value["type"]
                                yield StreamResponse.create_interrupt_response(
                                    message="Benutzerinteraktion erforderlich",
                                    question=question_to_user,
                                    rewritten_query=rewritten_query,
                                    type=interrupt_type,
                                )
                            else:
                                raise ValueError("Interrupt is not an Interrupt")
                        else:
                            self.logger.info(f"Unknown update: {key}")
            elif kind == "messages":
                if isinstance(content, tuple):
                    message = content[0]
                    source = content[1]
                    if (
                        isinstance(message, AIMessage)
                        and source["langgraph_node"] == "generate_answer"
                    ):
                        if (
                            latest_message_run_id is None
                            or message.id != latest_message_run_id
                        ):
                            if isinstance(message.content, str):
                                latest_message_run_id = message.id
                                yield StreamResponse.create_answer_response(
                                    message="AI Antwort", answer_text=message.content
                                )
                            else:
                                raise ValueError("Message content is not a string")

    def stream_query(
        self, question: str, user_role: str, thread_id: str
    ) -> Iterator[StreamResponse]:
        """Synchronous wrapper around astream_query"""
        self.logger.info(f"Thread ID: {thread_id}")

        async def run_async():
            async for chunk in self.astream_query(
                question=question, user_organization=user_role, thread_id=thread_id
            ):
                yield chunk

        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            agen = run_async()
            while True:
                try:
                    yield loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    def resume_query(
        self, thread_id: str, action: str, data: str | None = None
    ) -> Iterator[StreamResponse]:
        """Resume a RAG pipeline execution after an interrupt.

        Args:
            thread_id: The ID of the thread to resume
            action: The action to take ('accept', 'modify', or 'provide_information')
            data: Optional data to provide (modified query or additional information)
        """
        self.logger.info(f"Resuming Thread ID: {thread_id}")

        async def run_async():
            async for chunk in self.astream_query(
                thread_id=thread_id, resume_action=action, resume_data=data
            ):
                yield chunk

        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            agen = run_async()
            while True:
                try:
                    yield loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
