#!/bin/bash

set -e

MODE="${RAG_MODE:-api}"

if [ "$MODE" = "api" ]; then
echo "Mode: api - waiting for PostgreSQL and running migrations..."
echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -q; do
  sleep 1
done
echo "PostgreSQL is ready."

echo "Creating database if it doesn't exist..."
PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tc "SELECT 1 FROM pg_database WHERE datname = '$POSTGRES_DB'" | grep -q 1 || PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE \"$POSTGRES_DB\""
echo "Database is ready."

echo "Running database migrations..."
uv run alembic upgrade head

echo "Starting FastAPI application..."
exec uv run uvicorn src.rag.app:app --port "${BACKEND_PORT}" --host "${BACKEND_HOST}"

elif [ "$MODE" = "ingestion" ]; then
echo "Mode: ingestion - starting ingestion service (no migrations)..."
exec uv run src/rag/cli/run_ingestion.py

else
echo "Unknown RAG_MODE: $MODE. Expected 'api' or 'ingestion'."
exit 1
fi
