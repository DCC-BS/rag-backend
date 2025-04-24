import os
from pathlib import Path

import structlog

from rag.documents.document_storage import create_lancedb_document_store
from rag.utils.config import AppConfig, ConfigurationManager

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"


def main():
    """
    Main function to create and populate the document store.
    """
    # Load configuration
    config: AppConfig = ConfigurationManager.get_config()

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Create data directory if it doesn't exist
    os.makedirs(config.DOC_STORE.PATH, exist_ok=True)  # type: ignore[attr-defined]

    # Create and populate document store
    logger.info("Creating document store...")
    _ = create_lancedb_document_store()
    logger.info("Document store created successfully")


if __name__ == "__main__":
    main()
