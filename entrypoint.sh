#!/bin/bash

set -e

# Wait for the database to be ready
# A simple loop to check for the database connection
echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -q; do
  sleep 1
done
echo "PostgreSQL is ready."

# Create database if it doesn't exist
echo "Creating database if it doesn't exist..."
PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tc "SELECT 1 FROM pg_database WHERE datname = '$POSTGRES_DB'" | grep -q 1 || PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE \"$POSTGRES_DB\""
echo "Database is ready."

# Run Alembic migrations
echo "Running database migrations..."
uv run alembic upgrade head

# Start the FastAPI application
# The "$@" part means "execute any commands passed to this script"
# This allows you to still pass arguments like --host or --port if needed.
echo "Starting FastAPI application..."
exec "$@"
