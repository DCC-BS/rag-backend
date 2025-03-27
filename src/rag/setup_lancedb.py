import os
from pathlib import Path

import structlog
from rich.traceback import install

from utils.config import load_config

logger = structlog.get_logger()

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"


from data.document_storage import create_lancedb_document_store

# Configure rich traceback for unhandled exceptions
install(show_locals=True)


def main():
    """
    Main function to create and populate the document store.
    """
    # Load configuration
    config = load_config()

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Create data directory if it doesn't exist
    os.makedirs(config.DOC_STORE.PATH, exist_ok=True)

    # Create and populate document store
    logger.info("Creating document store...")
    create_lancedb_document_store()
    logger.info("Document store created successfully")


if __name__ == "__main__":
    main()
