from typing import List

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings

import lancedb
from config import load_config
from docling_loader import DoclingLoader
import torch

config = load_config()

def create_lancedb_document_store(user_roles: List[str]):
    sources = config.DOC_SOURCES

    documents = []
    extensions = ["pdf", "docx", "pptx", "html", "xlsx"]

    for role, folder in sources.items():
        for extension in extensions:
            doc_loader = DirectoryLoader(
                folder, glob=f"**/*.{extension}", loader_cls=DoclingLoader, show_progress=True, loader_kwargs={"organization": role}
            )

            docs = doc_loader.load()
            documents.extend(docs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDINGS.MODEL,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"task":"text-matching"}
    )
    
    db = lancedb.connect(config.DOC_STORE.PATH)
    table = config.DOC_STORE.TABLE_NAME

    vector_store = LanceDB.from_documents(
        documents,
        embeddings,
        connection=db,
        table_name=table,
    )

    return vector_store


def get_lancedb_doc_store():
    db = lancedb.connect(config.DOC_STORE.PATH)
    table = config.DOC_STORE.TABLE_NAME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDINGS.MODEL,
        model_kwargs={"device": device, "trust_remote_code": True},
    )

    return LanceDB(connection=db, table_name=table, embedding=embeddings)

