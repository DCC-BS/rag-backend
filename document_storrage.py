from typing import List

import pyarrow as pa
from haystack import Pipeline
from haystack.components.converters import DOCXToDocument, PDFMinerToDocument
from haystack.components.joiners.document_joiner import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from lancedb_haystack import LanceDBDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.utils import Secret
# from embedding import get_document_embedder
from utils import config_loader, find_files

config = config_loader("conf/conf.yaml")


def get_sources_for_user_roles(user_roles: List[str]):
    pdf_sources = []
    word_sources = []

    def _add_files(base_folder: str, pdf_sources, word_sources):
        docs = find_files(base_folder)
        pdfs = docs.get(".pdf", None)
        if pdfs:
            pdf_sources += pdfs
        docxs = docs.get(".docx", None)
        if docxs:
            word_sources += docxs
        return pdf_sources, word_sources

    if "Sozialhilfe" in user_roles:
        pdf_sources, word_sources = _add_files(
            config["DOC_SOURCES"]["SH"], pdf_sources, word_sources
        )
    if "Ergänzungsleistungen" in user_roles:
        pdf_sources, word_sources = _add_files(
            config["DOC_SOURCES"]["EL"], pdf_sources, word_sources
        )
    return {
        "pdf_converter": {"sources": pdf_sources},
        "word_converter": {"sources": word_sources},
    }


def create_inmemory_document_store(user_roles: List[str]) -> InMemoryDocumentStore:
    sources = get_sources_for_user_roles(user_roles)
    indexing_pipeline = Pipeline()
    document_store = InMemoryDocumentStore()
    document_writer = DocumentWriter(
        document_store=document_store, policy=DuplicatePolicy.OVERWRITE
    )
    indexing_pipeline.add_component("pdf_converter", PDFMinerToDocument())
    indexing_pipeline.add_component("word_converter", DOCXToDocument())
    indexing_pipeline.add_component("joiner", DocumentJoiner())
    indexing_pipeline.add_component("cleaner", DocumentCleaner())
    indexing_pipeline.add_component(
        "splitter", DocumentSplitter(split_by="page", split_length=1)
    )
    indexing_pipeline.add_component("writer", document_writer)
    indexing_pipeline.connect("pdf_converter", "joiner")
    indexing_pipeline.connect("word_converter", "joiner")
    indexing_pipeline.connect("joiner", "cleaner")
    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "writer")

    indexing_pipeline.run(sources)
    return document_store


def create_lancedb_document_store(user_roles: List[str]):
    metadata_schema = pa.struct([("date_added", pa.string())])
    document_store = LanceDBDocumentStore(
        database=config["DOC_STORE"]["PATH"],
        table_name=config["DOC_STORE"]["TABLE_NAME"],
        metadata_schema=metadata_schema,
        embedding_dims=config["EMBEDDINGS"]["DIM"],
    )
    sources = get_sources_for_user_roles(user_roles)
    indexing_pipeline = Pipeline()
    document_writer = DocumentWriter(
        document_store=document_store, policy=DuplicatePolicy.OVERWRITE
    )
    indexing_pipeline.add_component("pdf_converter", PDFMinerToDocument())
    indexing_pipeline.add_component("word_converter", DOCXToDocument())
    indexing_pipeline.add_component("joiner", DocumentJoiner())
    indexing_pipeline.add_component("cleaner", DocumentCleaner())
    indexing_pipeline.add_component(
        "splitter", DocumentSplitter(split_by="page", split_length=1)
    )
    indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model="BAAI/bge-m3", token=Secret.from_env_var("HF_API_TOKEN")))
    indexing_pipeline.add_component("writer", document_writer)
    indexing_pipeline.connect("pdf_converter", "joiner")
    indexing_pipeline.connect("word_converter", "joiner")
    indexing_pipeline.connect("joiner", "cleaner")
    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    indexing_pipeline.run(sources)

    return document_store

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    create_lancedb_document_store("Sozialhilfe")