from typing import Dict, List, Tuple

import pyarrow as pa
from haystack import Pipeline
from haystack.components.converters import DOCXToDocument, PDFMinerToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.joiners.document_joiner import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from lancedb_haystack import LanceDBDocumentStore

import lancedb
from utils import config_loader, find_files

config = config_loader("conf/conf.yaml")

ROLE_SOZIALHILFE = "Sozialhilfe"
ROLE_ERGANZUNGSLEISTUNGEN = "Ergänzungsleistungen"
DOC_TYPE_PDF = ".pdf"
DOC_TYPE_DOCX = ".docx"


def get_sources_for_user_roles(user_roles: List[str]) -> Dict[str, Dict]:
    """
    Retrieve document sources based on user roles.

    Args:
        user_roles (List[str]): List of user roles.

    Returns:
        Dict[str, Dict]: Dictionary containing PDF and Word document sources.
    """
    pdf_docs: Dict[str, List] = {"sources": [], "meta": []}
    word_docs: Dict[str, List] = {"sources": [], "meta": []}

    role_config = {
        ROLE_SOZIALHILFE: config["DOC_SOURCES"]["SH"],
        ROLE_ERGANZUNGSLEISTUNGEN: config["DOC_SOURCES"]["EL"],
    }

    for role, folder in role_config.items():
        if role in user_roles:
            _process_docs_for_role(role, folder, pdf_docs, word_docs)

    return {"pdf_converter": pdf_docs, "word_converter": word_docs}


def _add_files(base_folder: str) -> Tuple[List[str], List[str]]:
    docs = find_files(base_folder)
    return docs.get(DOC_TYPE_PDF, []), docs.get(DOC_TYPE_DOCX, [])


def _process_docs_for_role(
    role: str, base_folder: str, pdf_docs: dict, word_docs: dict
):
    pdf_sources, word_sources = _add_files(base_folder)
    if pdf_sources:
        pdf_docs["sources"].extend(pdf_sources)
        pdf_docs["meta"].extend(
            [{"organization": role, "file_path": path} for path in pdf_sources]
        )
    if word_sources:
        word_docs["sources"].extend(word_sources)
        word_docs["meta"].extend(
            [{"organization": role, "file_path": path} for path in word_sources]
        )


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
    metadata_schema = pa.struct(
        [
            ("organization", pa.string()),
            ("file_path", pa.string()),
            ("page_number", pa.string()),
        ]
    )
    document_store = LanceDBDocumentStore(
        database=uri,
        table_name=table,
        metadata_schema=metadata_schema,
        embedding_dims=config["EMBEDDINGS"]["DIM"],
    )
    return document_store


def create_lancedb_document_store(user_roles: List[str]):
    document_store = get_lancedb_doc_store()
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
    indexing_pipeline.add_component(
        "embedder",
        SentenceTransformersDocumentEmbedder(
            model="BAAI/bge-m3", token=Secret.from_env_var("HF_API_TOKEN")
        ),
    )
    indexing_pipeline.add_component("writer", document_writer)
    indexing_pipeline.connect("pdf_converter", "joiner")
    indexing_pipeline.connect("word_converter", "joiner")
    indexing_pipeline.connect("joiner", "cleaner")
    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    indexing_pipeline.run(sources)

    return document_store


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    create_lancedb_document_store("Sozialhilfe")
