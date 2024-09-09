import os
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import PDFMinerToDocument
import httpx
from dotenv import load_dotenv

load_dotenv()

# Write documents to InMemoryDocumentStore
load_document_pipeline = Pipeline()
document_store = InMemoryDocumentStore()
document_writer = DocumentWriter(
    document_store=document_store, policy=DuplicatePolicy.OVERWRITE
)
load_document_pipeline.add_component("converter", PDFMinerToDocument())
load_document_pipeline.add_component("cleaner", DocumentCleaner())
load_document_pipeline.add_component(
    "splitter", DocumentSplitter(split_by="page", split_length=1)
)
load_document_pipeline.add_component(
    "writer", DocumentWriter(document_store=document_store)
)
load_document_pipeline.connect("converter", "cleaner")
load_document_pipeline.connect("cleaner", "splitter")
load_document_pipeline.connect("splitter", "writer")

load_document_pipeline.run(
    {
        "converter": {
            "sources": ["data/Sozialhilfe-Handbuch_interne Version_Stand 16.04.2024.pdf"]
        }
    }
)

# Build a RAG pipeline
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
messages = [system_message, user_prompt_template]

retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = ChatPromptBuilder(template=messages)
llm = OpenAIChatGenerator(api_key=Secret.from_env_var("OPENAI_API_KEY"))
llm.client._client = httpx.Client(proxy=os.environ.get("PROXY_URL"))

language = "German"
question = "Wie alt sind die Menschen in Basel im Durchschnitt?"
question_2 = "Zu was sind unterstützte Personen verpflichtet?"
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.messages")


results = rag_pipeline.run(
    data={
        "retriever": {"query": question},
        "prompt_builder": {
            "template_variables": {"question": question, "language": language},
            "template": messages
        },
    }
)
print(results)
print(results["llm"]["replies"])

results = rag_pipeline.run(
    data={
        "retriever": {"query": question_2},
        "prompt_builder": {
            "template_variables": {"question": question_2, "language": language},
            "template": messages
        },
    }
)
print(results)
print(results["llm"]["replies"])