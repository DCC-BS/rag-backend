import logging
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler


def setup_logger(log_file_path="logs/app.log"):
    """
    Sets up the application logger.

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger("RAG-Bot")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=1024 * 1024 * 5, backupCount=3
    )  # 5MB per file, keep 3 backups

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = RichHandler(
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    console_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
