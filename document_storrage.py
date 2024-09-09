from haystack.components.converters import PDFMinerToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


def create_inmemory_document_store() -> InMemoryDocumentStore:
    indexing_pipeline = Pipeline()
    document_store = InMemoryDocumentStore()
    document_writer = DocumentWriter(
        document_store=document_store, policy=DuplicatePolicy.OVERWRITE
    )
    indexing_pipeline.add_component("converter", PDFMinerToDocument())
    indexing_pipeline.add_component("cleaner", DocumentCleaner())
    indexing_pipeline.add_component(
        "splitter", DocumentSplitter(split_by="page", split_length=1)
    )
    indexing_pipeline.add_component("writer", document_writer)
    indexing_pipeline.connect("converter", "cleaner")
    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "writer")

    indexing_pipeline.run(
        {
            "converter": {
                "sources": [
                    "data/Sozialhilfe-Handbuch_interne Version_Stand 16.04.2024.pdf"
                ]
            }
        }
    )
    return document_store
