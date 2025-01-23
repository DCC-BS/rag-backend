import asyncio
from typing import Iterator, List, Tuple, Union

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from core.bento_embeddings import BentoMLReranker
from core.lance_retriever import LanceDBRetriever, LanceDBRetrieverConfig
from core.rag_states import InputState, OutputState, RAGState, RouteQuery
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
        self.structured_llm_router = self.llm.with_structured_output(
            RouteQuery, method="json_schema"
        )
        self.memory = MemorySaver()

        # Create graph
        workflow = StateGraph(RAGState, input=InputState, output=OutputState)
        # Add nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("answer", self.call_model)

        # Add conditional edge based on skip_retrieval
        workflow.add_conditional_edges(
            START,
            self.route_question,
            {
                "retrieval": "retrieve",
                "answer": "answer",
            },
        )
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)

        # Compile graph
        self.graph = workflow.compile(checkpointer=self.memory)
        self.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    def retrieve(self, state: RAGState, config: RunnableConfig):
        docs = self.retriever.invoke(state["input"], config)
        return {"context": docs}

    def route_question(self, state: RAGState, config: RunnableConfig):
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
            return "answer"
        elif routing_result.next_step == "retrieval":
            print("---ROUTE QUESTION TO RAG---")
            return "retrieval"

    async def call_model(self, state: RAGState, config: RunnableConfig):
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
        return {"messages": messages + [response]}

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

    def _setup_llm(self):
        llm = ChatOpenAI(
            model_name=self.config.LLM.MODEL,
            temperature=self.config.LLM.TEMPERATURE,
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
    ) -> Iterator[Union[List[Document], str]]:
        input = {"input": question}
        config = {"configurable": {"thread_id": thread_id}}
        async for chunk in self.graph.astream(
            input=input, stream_mode=["updates", "messages"], config=config
        ):
            if "updates" == chunk[0]:
                if "retrieve" in chunk[1]:
                    yield chunk[1]["retrieve"]["context"]
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
