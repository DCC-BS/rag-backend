#!/usr/bin/env python3
"""CLI utility to drop application-managed tables and data from the database.

By default, only tables managed by our Alembic/ORM (e.g., documents, document_chunks)
are dropped. Optionally, the `alembic_version` table can also be dropped.

An optional flag is provided to drop the entire `public` schema with CASCADE,
but this is not the default and should be used with caution.
"""

import argparse
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from rag.models.db_document import Base
from rag.utils.db import get_db_url
from rag.utils.logger import get_logger

logger = get_logger()


def drop_app_tables(engine: Engine) -> None:
    """Drop only application-managed tables using ORM metadata (safe by default)."""

    with engine.begin() as connection:
        logger.warning("Dropping application tables via ORM metadata")
        Base.metadata.drop_all(bind=connection)
    logger.info("Application tables dropped successfully")


def drop_alembic_version(engine: Engine) -> None:
    """Drop the alembic_version table if present."""
    with engine.begin() as connection:
        logger.warning("Dropping alembic_version table if it exists")
        connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
    logger.info("alembic_version handled")


def drop_schema_cascade(engine: Engine, schema: str = "public") -> None:
    """Drop and recreate the given schema using CASCADE (PostgreSQL)."""
    with engine.begin() as connection:
        logger.warning("Dropping schema with CASCADE", schema=schema)
        connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    logger.info("Schema dropped and recreated successfully", schema=schema)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop all tables and data from the configured PostgreSQL database",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Confirm destructive action without interactive prompt",
    )
    parser.add_argument(
        "--drop-schema",
        action="store_true",
        help="Drop and recreate public schema with CASCADE (DANGEROUS)",
    )
    parser.add_argument(
        "--keep-alembic-version",
        action="store_true",
        help="Do not drop alembic_version table",
    )

    args = parser.parse_args()

    if not args.yes:
        logger.error("Confirmation required. Re-run with --yes to proceed.")
        sys.exit(1)

    try:
        db_url: str = get_db_url()
        engine: Engine = create_engine(url=db_url, pool_pre_ping=True)

        if args.drop_schema:
            drop_schema_cascade(engine)
        else:
            # Default: only app tables, optionally alembic_version
            drop_app_tables(engine)
            if not args.keep_alembic_version:
                drop_alembic_version(engine)

        logger.info("Database drop completed")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Failed to drop database", error=str(e))
        sys.exit(1)
    finally:
        engine.dispose() if engine else None  # pyright: ignore[reportPossiblyUnboundVariable]


if __name__ == "__main__":
    main()
