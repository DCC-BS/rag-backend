#!/usr/bin/env python3
"""CLI entry point for the document ingestion service."""

import asyncio
import sys

import structlog

from rag.services.document_ingestion import DocumentIngestionService


async def main() -> None:
    """Main entry point for the document ingestion service."""
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger()

    try:
        # Create and run the ingestion service
        service = DocumentIngestionService()
        await service.run()
    except KeyboardInterrupt:
        logger.info("Document ingestion service stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Document ingestion service failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
