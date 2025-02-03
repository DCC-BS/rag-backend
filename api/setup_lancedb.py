import os
import sys
from pathlib import Path

from rich.traceback import install

from utils.config import load_config

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"


from data.document_storage import create_lancedb_document_store
from utils.logging import setup_logger

# Configure rich traceback for unhandled exceptions
install(show_locals=True)

logger = setup_logger()


def handle_exception(exc_type, exc_value, exc_traceback):
    """Handles unhandled exceptions, logs them using the logger."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


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
    vector_store = create_lancedb_document_store()
    logger.info("Document store created successfully")


if __name__ == "__main__":
    main()
