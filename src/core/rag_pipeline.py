import asyncio
import re
from typing import Iterator, List, Literal, Tuple, Union

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

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


class SHRAGPipeline:
    def __init__(self, user_roles: List[str]) -> None:
        self.config = get_config()
        self.user_roles = user_roles
        self.system_prompt = (
            "You are an subject matter expert at social welfare regulations for the government in Basel, Switzerland."
            "You are given a question and a context of documents that are relevant to the question."
            "You are to answer the question based on the context and the conversation history."
            "If you don't know the answer, 'Entschuldigung, ich kann die Antwort nicht in den Dokumenten finden.'."
            "Don't try to make up an answer."
            "Answer in German."
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
        self.memory = MemorySaver()

        # Create graph
        workflow = StateGraph(RAGState, input=InputState, output=OutputState)
        # Add nodes
        workflow.add_node("route_question", self.route_question)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("filter_documents", self.filter_documents)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("query_needs_rephrase", self.decide_if_query_needs_rewriting)
        workflow.add_node("grade_hallucination", self.grade_hallucination)
        workflow.add_node("grade_answer", self.grade_answer)
        workflow.add_node("generate_answer", self.generate_answer)

        # Add conditional edge based on skip_retrieval
        workflow.add_edge(START, "route_question")
        workflow.add_edge("retrieve", "filter_documents")
        workflow.add_edge("filter_documents", "query_needs_rephrase")
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate_answer", "grade_hallucination")
        workflow.add_edge("grade_hallucination", "grade_answer")

        # Compile graph
        self.graph = workflow.compile(checkpointer=self.memory)
        self.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    def retrieve(self, state: RAGState, config: RunnableConfig):
        docs = self.retriever.invoke(state["input"], config)
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

        print("---ROUTE QUESTION---")
        system = """You are an expert at routing a user question to a vectorstore or llm.
        If you can answer the question based on the conversation history, return "answer".
        If you need more information, return "retrieval"."""
        context_messages = [(message.type, message.content) for message in state["messages"][1:]]
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                *context_messages,
                (
                    "user",
                    "Can you answer the following question based on the conversation history? Question: {question}",
                ),
            ]
        )
        question_router = route_prompt | self.structured_llm_router
        routing_result = question_router.invoke({"question": state["input"]}, config)
        if routing_result.next_step == "answer":
            print("---ROUTE QUESTION TO ANSWER GENERATION---")
            return Command(
                update={"route_query": "answer"},
                goto="generate_answer",
            )
        elif routing_result.next_step == "retrieval":
            print("---ROUTE QUESTION TO RAG---")
            return Command(
                update={"route_query": "retrieval"},
                goto="retrieve",
            )

    def grade_hallucination(
        self, state: RAGState, config: RunnableConfig
    ) -> Command[Literal["generate_answer", "grade_answer"]]:
        """
        Grade hallucination on retrieved documents and answer.
        """
        print("---GRADE HALLUCINATION---")
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
        hallucination_grader = hallucination_prompt | self.structured_llm_grade_hallucination
        hallucination_result = hallucination_grader.invoke(
            {
                "documents": "\n\n".join([doc.page_content for doc in state["context"]]),
                "answer": state["answer"],
            },
            config,
        )
        if hallucination_result.binary_score == "yes":
            print("---HALLUCINATION DETECTED---")
            return Command(
                update={"hallucination_score": "yes"},
                goto="generate_answer",
            )
        else:
            print("---HALLUCINATION NOT DETECTED---")
            return Command(
                update={"hallucination_score": "no"},
                goto="grade_answer",
            )

    def grade_answer(
        self, state: RAGState, config: RunnableConfig
    ) -> Command[Literal[END, "transform_query"]]:
        """
        Grade answer generation.
        """
        print("---GRADE ANSWER---")
        system = """You are a grader assessing relevance of an answer to a user question. \n 
            If the answer is relevant to the question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous answers    . \n
            Give a binary score 'yes' or 'no' score to indicate whether the answer is relevant to the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("user", "Answer: {answer} \\n User question: {question}"),
            ]
        )
        answer_grader = answer_prompt | self.structured_llm_grade_answer
        answer_result = answer_grader.invoke(
            {"answer": state["answer"], "question": state["input"]}, config
        )
        if answer_result.binary_score == "yes":
            print("---ANSWER IS RELEVANT---")
            return Command(
                update={"answer_score": "yes"},
                goto=END,
            )
        else:
            print("---ANSWER IS NOT RELEVANT---")
            return Command(
                update={"answer_score": "no"},
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
        print("---FILTER DOCUMENTS---")
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("user", "Retrieved document: {document} \\n User question: {question}"),
            ]
        )
        grade_documents_llm = grade_prompt | self.structured_llm_grade_documents
        question = state["input"]
        relevant_documents = []
        for doc in state["context"]:
            grade_documents_result = grade_documents_llm.invoke(
                {"document": doc.page_content, "question": question}, config
            )
            if grade_documents_result.binary_score == "no":
                print(f"Document {doc.metadata['filename']} is not relevant to the question.")
            else:
                relevant_documents.append(doc)
        return {"context": relevant_documents, "input": question}

    def decide_if_query_needs_rewriting(
        self, state: RAGState, config: RunnableConfig
    ) -> Command[Literal["transform_query", "generate_answer"]]:
        """
        Decide if the query needs to be re-written.
        """
        print("---DECIDE IF QUERY NEEDS REWRITING---")
        filtered_documents = state["context"]
        if len(filtered_documents) == 0:
            print("---QUERY NEEDS REWRITING---")
            return Command(
                update={"needs_rephrase": True},
                goto="transform_query",
            )
        else:
            print("---QUERY DOES NOT NEED REWRITING---")
            return Command(
                update={"needs_rephrase": False},
                goto="generate_answer",
            )

    def transform_query(self, state: RAGState, config: RunnableConfig):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        system = """You are an expert question re-writer that converts an input question to a better version that is optimized 
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
            The vectorstore contains documents from this domain: {document_description}.
            Write the re-phrased retrieval query in German as a question enclosed in <query> tags.
            The re-phrased question should be a question that can be answered by the vectorstore.
            The re-phrased question must be relevant to the initial user's question and keep the same meaning and intent.
            
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
        question_rewriter = re_write_prompt | self.query_rewriter_llm | StrOutputParser()
        question = state["input"]
        documents = state["context"]
        document_description = self.config.DOC_STORE.DOCUMENT_DESCRIPTION

        # Re-write question
        better_question = question_rewriter.invoke(
            {"question": question, "document_description": document_description}, config
        )
        # Keep only content between <query> and </query> tags
        better_question = re.search(r"<query>(.*?)</query>", better_question).group(1)

        return {"context": documents, "input": better_question}

    async def generate_answer(self, state: RAGState, config: RunnableConfig):
        if not state["messages"]:
            state["messages"] = [SystemMessage(content=self.system_prompt)]

        template = ChatPromptTemplate(
            [
                ("user", "Context: {context}\n\nQuestion: {input}"),
            ]
        )
        context = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = await template.ainvoke({"context": context, "input": state["input"]}, config)
        messages = state["messages"] + prompt.messages
        response = await self.llm.ainvoke(messages, config)
        return {"messages": messages + [response], "answer": response.content}

    def _setup_retriever(self):
        vector_store = get_lancedb_doc_store()
        reranker = BentoMLReranker(
            api_url=self.config.EMBEDDINGS.API_URL, column=vector_store._text_key
        )
        retriever = LanceDBRetriever(
            table=vector_store.get_table(self.config.DOC_STORE.TABLE_NAME),
            embeddings=vector_store.embeddings,
            reranker=reranker,
            config=LanceDBRetrieverConfig(
                vector_col=vector_store._vector_key,
                fts_col=vector_store._text_key,
                k=self.config.RETRIEVER.TOP_K,
                docs_before_rerank=self.config.RETRIEVER.FETCH_FOR_RERANKING,
                filter=f"metadata.organization IN ('{'\', \''.join(self.user_roles)}')",
            ),
        )
        return retriever

    def _setup_llm(self, temperature: float = None):
        temperature = temperature or self.config.LLM.TEMPERATURE
        llm = ChatOpenAI(
            model_name=self.config.LLM.MODEL,
            temperature=temperature,
            openai_api_key="None",
            openai_api_base=self.config.LLM.API_URL,
            max_tokens=self.config.LLM.MAX_TOKENS,
        )

        return llm

    def query(self, question: str, skip_retrieval: bool = False) -> Tuple[str, List]:
        result = self.graph.invoke({"input": question, "skip_retrieval": skip_retrieval})
        return result["answer"], result["context"]

    async def astream_query(
        self, question: str, thread_id: str
    ) -> Iterator[Tuple[str, Union[List[Document], str]]]:
        input = {"input": question}
        recursion_limit = self.config.RETRIEVER.MAX_RECURSION
        config = {"recursion_limit": recursion_limit, "configurable": {"thread_id": thread_id}}
        async for chunk in self.graph.astream(
            input=input, stream_mode=["updates", "messages"], config=config
        ):
            if "updates" == chunk[0]:
                if "retrieve" in chunk[1]:
                    yield ("Relevante Dokumente gefunden", chunk[1]["retrieve"]["context"])
                elif "route_question" in chunk[1]:
                    yield (
                        "Frage weitergeleitet",
                        f"Frage an {chunk[1]['route_question']['route_query']} weitergeleitet.\n",
                    )
                elif "filter_documents" in chunk[1]:
                    yield (
                        "Dokumente gefiltert",
                        f"{len(chunk[1]['filter_documents']['context'])} Dokumente sind relevant.\n",
                    )
                elif "transform_query" in chunk[1]:
                    yield (
                        "Frage umformuliert",
                        f"Frage umformuliert: {chunk[1]['transform_query']['input']}\n",
                    )
                elif "query_needs_rephrase" in chunk[1]:
                    yield (
                        "Frage analysiert",
                        f"Die Frage muss umformuliert werden: {chunk[1]['query_needs_rephrase']['needs_rephrase']}\n",
                    )
                elif "grade_hallucination" in chunk[1]:
                    yield (
                        "Halluzination überprüft",
                        f"Antwort enthält Halluzinationen: {chunk[1]['grade_hallucination']['hallucination_score']}\n",
                    )
                elif "grade_answer" in chunk[1]:
                    yield (
                        "Antwort bewertet",
                        f"Antwort ist relevant: {chunk[1]['grade_answer']['answer_score']}\n",
                    )
            if "messages" == chunk[0]:
                for message in chunk[1]:
                    if isinstance(message, AIMessage):
                        yield message.content

    def stream_query(self, question: str, thread_id: str) -> Iterator[Union[List[Document], str]]:
        """Synchronous wrapper around astream_query"""
        print("Thread ID: ", thread_id)

        async def run_async():
            async for chunk in self.astream_query(question, thread_id):
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
