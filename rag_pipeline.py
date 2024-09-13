import os

import httpx
import streamlit as st
from haystack import Pipeline
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils import Secret

language = "German"


class SHRAGPipeline:
    def __init__(self, document_store) -> None:
        self.user_prompt_template = (
            "Context information is below. \n"
            "---------------------\n"
            "{% for doc in documents %}"
            "    {{ doc.content }}"
            "{% endfor %}"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query. \n"
            "Question: {{question}} \n"
            "Answer: "
        )
        
        system_message = ChatMessage.from_system((
            "You are an subject matter expert at welfare regulations for the government in Basel, Switzerland."
            "You act as an assistant for question-answering tasks."
            "For every question you retrieve context information to create a truthful answer."
            "If you can not find the answer in the context information,"
            "just say 'Entschuldigung, ich kann die Antwort auf deine Frage in meinen Dokumenten nicht finden.'."
            "You always answer in {{language}}."
        )
        )
        self.messages = [system_message, ChatMessage.from_user(self.user_prompt_template)]

        retriever = InMemoryBM25Retriever(document_store=document_store, top_k=5)
        prompt_builder = ChatPromptBuilder(template=self.messages)
        llm = OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), streaming_callback=self.streamlit_write_streaming_chunk)
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

        rag_pipeline.draw("./rag_pipeline.png")

    def query(self, question: str) -> str:
         # Create a new Streamlit container for the AI's response.
        self.placeholder = st.empty()

        # Initialize an empty list for response tokens.
        self.tokens = []

        response = self.rag_pipeline.run(
            data={
                "retriever": {"query": question},
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
