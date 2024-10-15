import os

import httpx
import streamlit as st
from haystack import Pipeline
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator 
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from lancedb_haystack import LanceDBEmbeddingRetriever, LanceDBFTSRetriever
from haystack.dataclasses import ChatMessage, StreamingChunk
from utils import config_loader
from document_storrage import create_inmemory_document_store, create_lancedb_document_store, already_lancedb_existing, get_lancedb_doc_store
from typing import List
from haystack.utils import Secret
language = "German"


class SHRAGPipeline:
    def __init__(self, user_roles: List[str]) -> None:
        self.config = config_loader('conf/conf.yaml')
        self.user_prompt_template = (
            "Context information is below. \n"
            "---------------------\n"
            "{% for doc in documents %}"
            "    {{ doc.content }}"
            "{% endfor %}"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query. \n"
            " If you can not answer the question based on the information provided as context,"
            " say 'Entschuldigung, ich kann die Antwort auf deine Frage in meinen Dokumenten nicht finden.'. \n"
            "Question: {{question}} \n"
            "Answer: "
        )
        
        system_message = ChatMessage.from_system((
            "You are an subject matter expert at welfare regulations for the government in Basel, Switzerland. \n"
            " You act as an assistant for question-answering tasks. \n"
            " For every question you retrieve context information to create a truthful answer. \n"
            " You always answer in {{language}}."
        )
        )
        self.messages = [system_message, ChatMessage.from_user(self.user_prompt_template)]

        retriever = self._get_retriever(user_roles)
        prompt_builder = ChatPromptBuilder(template=self.messages)
        llm = OpenAIChatGenerator(
            api_key=Secret.from_token("None"),
            streaming_callback=self.streamlit_write_streaming_chunk, 
            api_base_url=os.getenv("API_BASE_URL"),
            model=self.config["LLM"]["MODEL"],
            generation_kwargs={
                **self.config["LLM"]["GENERATION_ARGS"]
            },
            # timeout=600
            )
        if len(os.environ.get("PROXY_URL")) > 0:
            llm.client._client = httpx.Client(proxy=os.environ.get("PROXY_URL"))
        answer_builder = AnswerBuilder()

        rag_pipeline = Pipeline()
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", llm)
        rag_pipeline.add_component("answer_builder", answer_builder)
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
        rag_pipeline.connect("llm.replies", "answer_builder.replies")
        rag_pipeline.connect("retriever", "answer_builder.documents")
        self.rag_pipeline = rag_pipeline


    def _get_retriever(self, user_roles):
        organization_filter = self._user_roles_to_filter(user_roles)
        top_k = self.config["RETRIEVER"]["TOP_K"]
        retriever_type = self.config["RETRIEVER"]["TYPE"]
        doc_store_type = self.config["DOC_STORE"]["TYPE"]
        if doc_store_type.lower() == 'inmemory':
            doc_store = create_inmemory_document_store(user_roles)
            if retriever_type.lower() == 'embedding':
                return InMemoryEmbeddingRetriever(document_store=doc_store, top_k=top_k, filters=organization_filter)
            elif retriever_type.lower() == 'bm25':
                return InMemoryBM25Retriever(document_store=doc_store, top_k=top_k, filters=organization_filter)
            else:
                raise ValueError(f"Unsupported retriever {retriever_type} for {doc_store_type} document store.")
        elif doc_store_type.lower() == 'lancedb':
            if already_lancedb_existing():
                doc_store = get_lancedb_doc_store()
            else:
                doc_store = create_lancedb_document_store(user_roles)
            if retriever_type.lower() == 'embedding':
                return LanceDBEmbeddingRetriever(document_store=doc_store, top_k=top_k, filters=organization_filter)
            elif retriever_type.lower() == 'bm25':
                return LanceDBFTSRetriever(document_store=doc_store, top_k=top_k, filters=organization_filter)
            else:
                raise ValueError(f"Unsupported retriever {retriever_type} for {doc_store_type} document store.")
        else:
            raise ValueError(f"Unsupported DocStore type {doc_store_type}.")

    def _user_roles_to_filter(self, user_roles: List[str]):
        return {"field": "meta.organization", "operator": "in", "value": user_roles}

    def query(self, question: str) -> str:
         # Create a new Streamlit container for the AI's response.
        self.placeholder = st.empty()

        # Initialize an empty list for response tokens.
        self.tokens = []
        # TODO: Needs adjustment to work with embeddings (add query embedding)
        response = self.rag_pipeline.run(
            data={
                "retriever": {"query": question, "top_k": self.config["RETRIEVER"]["TOP_K"]},
                "answer_builder": {"query": question},
                "prompt_builder": {
                    "template_variables": {"question": question, "language": language},
                    "template": self.messages,
                },
            }
        )
        if 'answers' in response["answer_builder"]:
            response_content = response["answer_builder"]['answers'][0].data
            relevant_documents = response["answer_builder"]['answers'][0].documents
            # Add the assistant's response to the chat history.
            self.messages.append(ChatMessage.from_assistant(response_content))
            # Prepare new template message for the next user input.
            self.messages.append(ChatMessage.from_user(self.user_prompt_template))
            return response_content, relevant_documents
        else:
            raise Exception('No valid response or unexpected response format.')

    
    def streamlit_write_streaming_chunk(self, chunk: StreamingChunk):
        """
            Streams a response chunk to the Streamlit UI.

        Args:
            chunk (StreamingChunk): The streaming chunk from the language model.
        """
        # Append the latest streaming chunk to the tokens list.
        self.tokens.append(chunk.content)

        # Update the Streamlit container with the current stream of tokens.
        self.placeholder.write("".join(self.tokens))


    def add_message_to_chat_history(self, message: ChatMessage):
        """
            Add a message to the chat history.

        Args:
            message (ChatMessage): The message to add to the chat history.
        """
        self.messages.append(message)
