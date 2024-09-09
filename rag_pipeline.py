import os
import streamlit as st
import httpx
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils import Secret

language = "German"


class SHRAGPipeline:
    def __init__(self, document_store) -> None:
        user_prompt_template = ChatMessage.from_user("""
            Given these documents, answer the question.
            Documents:
            {% for doc in documents %}
                {{ doc.content }}
            {% endfor %}
            Question: {{question}}
            Answer:
            """)
        system_message = ChatMessage.from_system(
            "You are an subject matter expert on welfare for the government in Basel. If you can not find the answer in the given documens, then say 'Sorry, I can not find the answer you are looking for.'. You always answer in {{language}}."
        )
        self.messages = [system_message, user_prompt_template]

        retriever = InMemoryBM25Retriever(document_store=document_store)
        prompt_builder = ChatPromptBuilder(template=self.messages)
        llm = OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"), streaming_callback=self.streamlit_write_streaming_chunk)
        llm.client._client = httpx.Client(proxy=os.environ.get("PROXY_URL"))

        rag_pipeline = Pipeline()
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", llm)
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
        self.rag_pipeline = rag_pipeline

    def query(self, question: str) -> str:
         # Create a new Streamlit container for the AI's response.
        self.placeholder = st.empty()

        # Initialize an empty list for response tokens.
        self.tokens = []

        response = self.rag_pipeline.run(
            data={
                "retriever": {"query": question},
                "prompt_builder": {
                    "template_variables": {"question": question, "language": language},
                    "template": self.messages,
                },
            }
        )

        if 'replies' in response["llm"]:
            response_content = response["llm"]['replies'][0].content
            # Add the assistant's response to the chat history.
            self.messages.append(ChatMessage.from_assistant(response_content))
            return response_content
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
