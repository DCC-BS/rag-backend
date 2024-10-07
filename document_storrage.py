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
import lancedb
# from embedding import get_document_embedder
from utils import config_loader, find_files

config = config_loader("conf/conf.yaml")


def get_sources_for_user_roles(user_roles: List[str]):
    sources_per_organization = []

    def _add_files(base_folder: str):
        docs = find_files(base_folder)
        pdfs = docs.get(".pdf", [])
        docxs = docs.get(".docx", [])
        return pdfs, docxs

    if "Sozialhilfe" in user_roles:
        pdf_sources, word_sources = _add_files(config["DOC_SOURCES"]["SH"])
        sources_per_organization.append({
            "pdf_converter": {"sources": pdf_sources, 'meta':{'organization': 'Sozialhilfe'}},
            "word_converter": {"sources": word_sources, 'meta':{'organization': 'Sozialhilfe'}}
        })
    if "Ergänzungsleistungen" in user_roles:
        pdf_sources, word_sources = _add_files(config["DOC_SOURCES"]["EL"])
        sources_per_organization.append({
            "pdf_converter": {"sources": pdf_sources, 'meta':{'organization': 'Ergänzungsleistungen'}},
            "word_converter": {"sources": word_sources, 'meta':{'organization': 'Ergänzungsleistungen'}}
        })
    return sources_per_organization

def create_inmemory_document_store(user_roles: List[str]) -> InMemoryDocumentStore:
    sources_per_organization = get_sources_for_user_roles(user_roles)
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

    for sources in sources_per_organization:
        indexing_pipeline.run(sources)
    return document_store

def already_lancedb_existing():
    uri = config["DOC_STORE"]["PATH"]
    table = config["DOC_STORE"]["TABLE_NAME"]
    db = lancedb.connect(uri)
    if table in db.table_names():
        # Document store already existing
        return True
    return False

def get_lancedb_doc_store():
    uri = config["DOC_STORE"]["PATH"]
    table = config["DOC_STORE"]["TABLE_NAME"]
    metadata_schema = pa.struct([("date_added", pa.string())])
    document_store = LanceDBDocumentStore(
        database=uri,
        table_name=table,
        metadata_schema=metadata_schema,
        embedding_dims=config["EMBEDDINGS"]["DIM"],
    )
    return document_store

def create_lancedb_document_store(user_roles: List[str]):
    document_store = get_lancedb_doc_store()
    sources_per_organization = get_sources_for_user_roles(user_roles)
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

    for sources in sources_per_organization:
        indexing_pipeline.run(sources)

    return document_store

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    create_lancedb_document_store("Sozialhilfe")