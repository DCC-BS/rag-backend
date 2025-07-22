#!/usr/bin/env python3
"""CLI entry point for the document ingestion service."""

import argparse
import asyncio
import sys

import structlog

from rag.services.document_ingestion import S3DocumentIngestionService
from rag.utils.config import ConfigurationManager


async def main() -> None:
    """Main entry point for the document ingestion service."""
    parser = argparse.ArgumentParser(
        description="Document ingestion service for S3/MinIO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run with default settings
  %(prog)s --scan-only               # Run initial scan only (no watch mode)
  %(prog)s --config custom.yaml      # Use custom configuration file
  %(prog)s --verbose                 # Use console output instead of JSON logging

The service will:
1. Perform an initial scan of all configured S3 buckets
2. Process any new or updated documents
3. Clean up orphaned documents
4. Start watching for changes (unless --scan-only is used)
        """,
    )

    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Run initial scan only without starting watch mode",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file (optional)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (debug level)",
    )

    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    logger = structlog.get_logger()

    try:
        # Load configuration
        config = ConfigurationManager.get_config()

        logger.info("Starting document ingestion service")
        if args.config:
            logger.warning("Custom config file parameter provided but not supported yet")
        logger.info("Configuration loaded from default sources")

        if args.scan_only:
            logger.info("Running in scan-only mode (watch disabled)")

        # Create the ingestion service
        service = S3DocumentIngestionService(config)

        if args.scan_only:
            # Run initial scan only
            service.initial_scan()
            logger.info("Initial scan completed, exiting (scan-only mode)")
        else:
            # Run the full service (initial scan + watch mode)
            await run_service_with_cleanup(service)

    except KeyboardInterrupt:
        logger.info("Document ingestion service stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Document ingestion service failed", error=str(e))
        sys.exit(1)


async def run_service_with_cleanup(service: S3DocumentIngestionService) -> None:
    """Run the ingestion service with optional cleanup during initial scan."""
    # Perform initial scan with optional cleanup
    service.initial_scan()

    # Start change monitoring if enaböled
    if service.config.INGESTION.WATCH_ENABLED:
        service.logger.info("Document ingestion service is running. Press Ctrl+C to stop.")
        try:
            await service.watch_s3_changes()
        except KeyboardInterrupt:
            service.logger.info("Received shutdown signal")
    else:
        service.logger.info("Watch mode disabled, exiting after initial scan")


if __name__ == "__main__":
    asyncio.run(main())
