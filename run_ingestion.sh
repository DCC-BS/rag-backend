#!/bin/bash
# Script to run the document ingestion service

echo "Starting RAG Document Ingestion Service..."

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir -p data/{EL,SH,EL2}
    echo "Created data directory with subdirectories: EL, SH, EL2"
fi

# Run the ingestion service
uv run src/rag/cli/run_ingestion.py
