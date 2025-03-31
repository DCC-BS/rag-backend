# SH-RAG-Chat Tests

This directory contains tests for the SH-RAG-Chat project.

## Test Structure

```
tests/
├── unit/               # Unit tests that test individual components in isolation
│   └── rag/            # Tests for the rag package
│       └── core/       # Tests for the rag.core package
│           └── test_bento_embeddings.py  # Tests for BentoEmbeddings class
├── integration/        # Integration tests that test components working together
│   └── rag/            # Tests for the rag package
│       └── core/       # Tests for the rag.core package
│           └── test_bento_embeddings_integration.py  # Integration tests for BentoEmbeddings
└── conftest.py         # Shared pytest fixtures and configuration
```

## Running Tests

The project uses `uv` to run tests and pytest configuration is defined in `pyproject.toml`.

### All Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=src
```

### Specific Tests

```bash
# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests
uv run pytest tests/integration/

# Run specific test file
uv run pytest tests/unit/rag/core/test_bento_embeddings.py

# Run specific test function
uv run pytest tests/unit/rag/core/test_bento_embeddings.py::TestBentoEmbeddings::test_embed_documents
```

### Running Async Tests

The async tests use pytest-asyncio. To run them:

```bash
# Run all async tests
uv run pytest -v -m asyncio
```

## Environment Variables

Some integration tests require environment variables:

- `BENTO_API_URL`: URL of the BentoML API (default: http://localhost:3000)

Set these before running the tests:

```bash
# Unix
export BENTO_API_URL=http://your-api-url

# Windows
set BENTO_API_URL=http://your-api-url
```

## Test Dependencies

The required test dependencies are already included in the `dev` dependency group in `pyproject.toml`:

- pytest: Testing framework
- pytest-asyncio: For testing async code
- pytest-cov: For code coverage reports

These should be installed in the dev dependencies section of your project.
