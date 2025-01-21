from typing import Annotated, List, Optional, Sequence, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.messages import BaseMessage, SystemMessage
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
    context: Optional[List[Document]]
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

    def call_model(self, state: RAGState, config):
        template = ChatPromptTemplate([
            ("user", "Context: {context}\n\nQuestion: {input}"),
        ])
        context = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = template.invoke({"context": context, "input": state["input"]})
        messages = SystemMessage(content=self.system_prompt) + state["messages"] + prompt.messages
        response = self.llm.invoke(messages.messages, config)
        return {"messages": [response]}

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
        )

        return llm

    def query(self, question: str) -> Tuple[str, List]:
        result = self.graph.invoke({"input": question})
        return result["answer"], result["context"]

    def stream_query(self, question: str):
        initial_state = {
            "input": question,
            "messages": [],
            "context": None
        }
        for chunk in self.graph.stream(initial_state, stream_mode="updates"):
            if "retrieve" in chunk and "context" in chunk["retrieve"]:
                yield chunk["retrieve"]["context"]
            if "answer" in chunk and "messages" in chunk["answer"]:
                yield chunk["answer"]["messages"][0].content