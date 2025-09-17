#!/usr/bin/env python3
"""CLI utility to clean up orphaned documents from the database.

This script checks all documents stored in the PostgreSQL database and verifies
that their corresponding S3 objects still exist. Documents whose S3 objects
have been deleted are removed from the database.
"""

import argparse
import sys

from rag.services.document_ingestion import S3DocumentIngestionService
from rag.utils.config import ConfigurationManager
from rag.utils.logger import get_logger


def main() -> None:
    """Main entry point for the orphaned document cleanup utility."""
    logger = get_logger()

    parser = argparse.ArgumentParser(
        description="Clean up orphaned documents from the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run cleanup with default configuration
  %(prog)s --dry-run          # Preview what would be deleted without actually deleting
  %(prog)s --config custom.yaml  # Use custom configuration file

This script will:
1. Query all documents from the PostgreSQL database
2. Check if each document's corresponding S3 object still exists
3. Remove documents from the database if their S3 objects are missing
4. Log detailed statistics about the cleanup process
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what documents would be deleted without actually deleting them",
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

    try:
        # Load configuration
        config = ConfigurationManager.get_config()

        logger.info("Starting orphaned document cleanup")
        if args.config:
            logger.warning("Custom config file parameter provided but not supported yet")
        logger.info("Configuration loaded from default sources")

        if args.dry_run:
            logger.info("DRY RUN MODE: No documents will actually be deleted")

        # Initialize the ingestion service
        ingestion_service = S3DocumentIngestionService(config)

        if args.dry_run:
            # For dry run, we'll modify the cleanup method behavior
            # We'll create a custom version that doesn't actually delete
            stats = run_dry_run_cleanup(ingestion_service)
        else:
            # Run the actual cleanup
            stats = ingestion_service.cleanup_orphaned_documents()

        # Display summary
        logger.info("Cleanup Summary:")
        logger.info(f"  Total documents checked: {stats["total_documents"]}")
        logger.info(f"  Orphaned documents found: {stats["orphaned_documents"]}")
        logger.info(f"  Invalid document paths: {stats["invalid_paths"]}")
        logger.info(f"  S3 check errors: {stats["s3_check_errors"]}")

        if args.dry_run:
            logger.info(f"  Documents that would be deleted: {stats["orphaned_documents"]}")
            logger.info("No documents were actually deleted (dry run mode)")
        else:
            logger.info(f"  Documents successfully deleted: {stats["deleted_documents"]}")

        if stats["orphaned_documents"] > 0:
            if args.dry_run:
                logger.warning(f"Found {stats["orphaned_documents"]} orphaned documents that would be deleted")
                logger.info("Run without --dry-run to actually delete them")
            else:
                logger.info(f"Successfully cleaned up {stats["deleted_documents"]} orphaned documents")
        else:
            logger.info("No orphaned documents found - database is clean!")

    except KeyboardInterrupt:
        logger.info("Cleanup interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Failed to complete orphaned document cleanup", error=str(e))
        sys.exit(1)


def run_dry_run_cleanup(ingestion_service: S3DocumentIngestionService) -> dict[str, int]:
    """Run cleanup in dry-run mode (check only, no deletions).

    Args:
        ingestion_service: The S3 document ingestion service instance

    Returns:
        Dictionary with cleanup statistics
    """
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from rag.models.db_document import Document

    logger = get_logger()
    logger.info("Starting dry-run cleanup of orphaned documents")

    cleanup_stats = {
        "total_documents": 0,
        "orphaned_documents": 0,
        "invalid_paths": 0,
        "s3_check_errors": 0,
        "deleted_documents": 0,  # Will remain 0 in dry run
    }

    try:
        with Session(ingestion_service.engine) as session:
            # Get all documents from database
            statement = select(Document)
            documents = session.execute(statement).scalars().all()

            cleanup_stats["total_documents"] = len(documents)
            logger.info(f"Found {len(documents)} documents in database to validate")

            for document in documents:
                # Parse document path
                parsed_path = ingestion_service.s3_utils.parse_document_path(document.document_path)
                if not parsed_path:
                    cleanup_stats["invalid_paths"] += 1
                    continue

                bucket_name, object_key = parsed_path

                try:
                    # Check if S3 object exists
                    if not ingestion_service.s3_utils.object_exists(bucket_name, object_key):
                        logger.info(
                            f"[DRY RUN] Would delete orphaned document: {document.document_path}",
                            document_id=document.id,
                            file_name=document.file_name,
                        )
                        cleanup_stats["orphaned_documents"] += 1
                    else:
                        logger.debug(f"Document valid - S3 object exists: {document.document_path}")

                except Exception as e:
                    logger.warning(
                        f"Error checking S3 object existence for {document.document_path}",
                        error=str(e),
                        document_id=document.id,
                    )
                    cleanup_stats["s3_check_errors"] += 1
                    continue

    except Exception as e:
        logger.exception("Failed to run dry-run cleanup", error=str(e))
        raise

    # Log final statistics
    logger.info(
        "Dry-run orphaned document cleanup completed",
        total_documents=cleanup_stats["total_documents"],
        orphaned_documents=cleanup_stats["orphaned_documents"],
        invalid_paths=cleanup_stats["invalid_paths"],
        s3_check_errors=cleanup_stats["s3_check_errors"],
    )

    return cleanup_stats


if __name__ == "__main__":
    main()
