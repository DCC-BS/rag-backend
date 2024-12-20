import os
from typing import List, Tuple

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from document_storage import get_lancedb_doc_store
from utils import config_loader


class SHRAGPipeline:
    def __init__(self, user_roles: List[str]) -> None:
        self.user_roles = user_roles
        self.config = config_loader("conf/conf.yaml")
        self.vector_store = get_lancedb_doc_store()
        llm = self._setup_llm()
        prompt = self._setup_prompt()
        retriever = self._setup_retriever()

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    def _setup_retriever(self):
        
        return self.vector_store.as_retriever(
            search_type=self.config.RETRIEVER.TYPE,
            search_kwargs={
                "k": self.config.RETRIEVER.TOP_K,
                # "fetch_k": self.config.RETRIEVER.FETCH_K,
                "filter": f"metadata.organization IN ('{'\', \''.join(self.user_roles)}')",
                # "query_type": "hybrid",
                # "name": self.config.DOC_STORE.TABLE_NAME
            },
        )

    def _setup_prompt(self):
        system_prompt = (
            "You are an subject matter expert at social welfare regulations for the government in Basel, Switzerland."
            "Use the following pieces of context to answer the question at the end."
            "If you don't know the answer, 'Entschuldigung, ich kann die Antwort nicht in den Dokumenten finden.', don't try to make up an answer."
            "Keep the answer concise."
            "Answer in German."
            "Context:"
            "\n\n"
            "{context}"
            "\n\n"
        )

        return ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "{input}")]
        )

    def _setup_llm(self):
        return ChatOpenAI(
            model_name=self.config.LLM.MODEL,
            temperature=self.config.LLM.TEMPERATURE,
            openai_api_key="None",
            openai_api_base=os.getenv("API_BASE_URL"),
        )

    def query(self, question: str) -> Tuple[str, List]:
        result = self.rag_chain.invoke({"input": question})
        return result["answer"], result["context"]
    
    def stream_query(self, question: str):
        context = None
        for chunk in self.rag_chain.stream({"input": question}):
            if "answer" in chunk:
                yield chunk["answer"]
            if "context" in chunk:
                context = chunk["context"]
        yield context
