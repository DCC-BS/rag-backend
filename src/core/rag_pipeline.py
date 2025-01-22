import asyncio
import operator
from typing import Annotated, Iterator, List, Optional, Sequence, Tuple, Union

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from core.bento_embeddings import BentoMLReranker
from core.lance_retriever import LanceDBRetriever, LanceDBRetrieverConfig
from data.document_storage import get_lancedb_doc_store
from utils.config import get_config


class RAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: Annotated[Optional[List[Document]], operator.add]
    input: str


class SHRAGPipeline:
    def __init__(self, user_roles: List[str]) -> None:
        self.config = get_config()
        self.user_roles = user_roles
        self.system_prompt = (
            "You are an subject matter expert at social welfare regulations for the government in Basel, Switzerland."
            "You are given a question and a context of documents that are relevant to the question."
            "You are to answer the question based on the context."
            "If you don't know the answer, 'Entschuldigung, ich kann die Antwort nicht in den Dokumenten finden.'."
            "Don't try to make up an answer."
            "Answer in German."
        )

        # Setup components
        self.retriever = self._setup_retriever()
        self.llm = self._setup_llm()

        # Create graph
        workflow = StateGraph(RAGState)
        # Add nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("answer", self.call_model)

        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)

        # Compile graph
        self.graph = workflow.compile()

    def retrieve(self, state: RAGState):
        docs = self.retriever.invoke(state["input"])
        return {"context": docs}

    async def call_model(self, state: RAGState, config: RunnableConfig):
        if not state["messages"]:
            state["messages"] = [SystemMessage(content=self.system_prompt)]

        template = ChatPromptTemplate([
            ("user", "Context: {context}\n\nQuestion: {input}"),
        ])
        context = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = await template.ainvoke({"context": context, "input": state["input"]}, config)
        messages = state["messages"] + prompt.messages
        response = await self.llm.ainvoke(messages, config)
        return {"messages": response}

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

    def query(self, question: str) -> Tuple[str, List]:
        result = self.graph.invoke({"input": question})
        return result["answer"], result["context"]

    async def astream_query(self, question: str) -> Iterator[Union[List[Document], str]]:
        input = {
            "input": question,
        }
        async for chunk in self.graph.astream(input=input, stream_mode=["updates", "messages"]):
            if "updates" == chunk[0]:
                if "retrieve" in chunk[1]:
                    yield chunk[1]["retrieve"]["context"]
            if "messages" == chunk[0]:
                for message in chunk[1]:
                    if isinstance(message, AIMessage):
                        yield message.content

    def stream_query(self, question: str) -> Iterator[Union[List[Document], str]]:
        """Synchronous wrapper around astream_query"""
        async def run_async():
            async for chunk in self.astream_query(question):
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
        

