from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores.lancedb import LanceDB
from langchain_core.documents.base import Document

import lancedb
from lancedb.db import DBConnection
from rag.core.bento_embeddings import BentoEmbeddings
from rag.documents.docling_loader import DoclingLoader
from rag.utils.config import AppConfig, ConfigurationManager

config: AppConfig = ConfigurationManager().get_config()


def create_lancedb_document_store() -> LanceDB:
    sources: dict[str, str] = config.DOC_SOURCES

    documents: list[Document] = []
    extensions: list[str] = ["pdf", "docx", "pptx", "html"]  # , "xlsx"]

    for role, folder in sources.items():
        for extension in extensions:
            doc_loader: DirectoryLoader = DirectoryLoader(
                path=folder,
                glob=f"**/*.{extension}",
                loader_cls=DoclingLoader,  # pyright: ignore[reportArgumentType]
                show_progress=True,
                loader_kwargs={"organization": role},
            )

            docs: list[Document] = doc_loader.load()
            documents.extend(docs)

    embeddings: BentoEmbeddings = BentoEmbeddings(api_url=config.EMBEDDINGS.API_URL)

    db: DBConnection = lancedb.connect(uri=config.DOC_STORE.PATH)
    table: str = config.DOC_STORE.TABLE_NAME

    vector_store: LanceDB = LanceDB.from_documents(
        documents=documents,
        embedding=embeddings,
        connection=db,
        table_name=table,
    )

    vector_store.get_table().create_fts_index(  # pyright: ignore[reportAny]
        "text", use_tantivy=True, language="German", stem=True, remove_stop_words=True
    )

    return vector_store


def get_lancedb_doc_store() -> LanceDB:
    db: DBConnection = lancedb.connect(uri=config.DOC_STORE.PATH)
    table: str = config.DOC_STORE.TABLE_NAME

    embeddings: BentoEmbeddings = BentoEmbeddings(api_url=config.EMBEDDINGS.API_URL)

    return LanceDB(connection=db, table_name=table, embedding=embeddings)
