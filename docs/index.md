# RAG Application

[![Release](https://img.shields.io/github/v/release/DCC-BS/rag-application)](https://img.shields.io/github/v/release/DCC-BS/rag-application)
[![Build status](https://img.shields.io/github/actions/workflow/status/DCC-BS/rag-application/main.yml?branch=main)](https://github.com/DCC-BS/rag-application/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/DCC-BS/rag-application)](https://img.shields.io/github/commit-activity/m/DCC-BS/rag-application)
[![License](https://img.shields.io/github/license/DCC-BS/rag-application)](https://img.shields.io/github/license/DCC-BS/rag-application)

A comprehensive Retrieval-Augmented Generation (RAG) application that provides intelligent document processing, embedding generation, and semantic search capabilities.

## Overview

This RAG application combines document ingestion, vector embeddings, and intelligent retrieval to enable semantic search and question-answering over document collections. The system supports various document formats and provides role-based access control for different user groups.

## Key Features

- **Automated Document Processing**: Intelligent extraction and processing of various document formats
- **Vector Embeddings**: Advanced embedding generation for semantic search capabilities
- **Role-Based Access Control**: Directory-based access management for different user roles
- **Real-time File Monitoring**: Automatic detection and processing of new documents
- **PostgreSQL Integration**: Robust database storage with vector search capabilities
- **Docker Support**: Complete containerized deployment solution

## Getting Started

### Quick Start with Docker

```bash
# Clone the repository
git clone <repository-url>
cd rag-backend

# Start all services
docker-compose up
```

### Development Setup

```bash
# Install dependencies
uv install

# Set up database
# Configure your environment
# Run the application
```

## Documentation

### Core Services

- **[Document Ingestion Service](ingestion.md)** - Comprehensive guide to the document ingestion pipeline, including file monitoring, processing, and embedding generation

### Configuration

The application uses YAML-based configuration files located in `src/rag/conf/` for different components:

- `conf.yaml` - Main application configuration
- `ingestion.yaml` - Document ingestion settings
- `chat.yaml` - Chat interface configuration

## Architecture

The RAG application follows a modular architecture with separate services for:

- Document ingestion and processing
- Embedding generation
- Vector storage and retrieval
- API endpoints
- Role-based access management

## Support

For detailed information about specific components, refer to the individual documentation pages linked above. The codebase includes comprehensive tests and examples to help you get started quickly.
