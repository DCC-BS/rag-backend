from config import load_config
import sys
from rich.traceback import install
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.9"


from document_storage import create_lancedb_document_store
from log_config import setup_logger
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

if __name__ == "__main__":
    load_config()
    create_lancedb_document_store()