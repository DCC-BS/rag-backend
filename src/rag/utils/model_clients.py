"""Utility functions for creating model clients (LLM, embedding, reranker) with automatic model detection."""

import os
from typing import NamedTuple

import cohere
import structlog
from langchain_openai import ChatOpenAI
from openai import Client
from pydantic import SecretStr

from rag.utils.config import AppConfig, ConfigurationManager


class EmbeddingClient(NamedTuple):
    """Container for embedding client and model."""

    client: Client
    model: str


class RerankerClient(NamedTuple):
    """Container for reranker client and model."""

    client: cohere.ClientV2
    model: str


class LLMClient(NamedTuple):
    """Container for LLM client and model."""

    client: ChatOpenAI
    model: str


def get_embedding_client(config: AppConfig | None = None) -> EmbeddingClient:
    """Get embedding client with automatically detected model.

    Args:
        config: Application configuration, will load default if None

    Returns:
        EmbeddingClient with client and model name

    Raises:
        Exception: If client creation fails
    """
    config = config or ConfigurationManager.get_config()
    logger = structlog.get_logger()

    client = Client(base_url=config.EMBEDDINGS.API_URL, api_key=os.getenv("OPENAI_API_KEY", "none"))

    # Try to get model from service, fall back to default
    try:
        model_data = client.models.list().data
        model = model_data[0].id if model_data else "Qwen/Qwen3-Embedding-0.6B"
    except Exception:
        logger.exception("Could not determine embedding model from client, falling back to default.")
        model = "Qwen/Qwen3-Embedding-0.6B"

    return EmbeddingClient(client=client, model=model)


def get_reranker_client(config: AppConfig | None = None) -> RerankerClient:
    """Get reranker client with automatically detected model.

    Args:
        config: Application configuration, will load default if None

    Returns:
        RerankerClient with client and model name

    Raises:
        Exception: If client creation fails
    """
    config = config or ConfigurationManager.get_config()
    logger = structlog.get_logger()

    client = cohere.ClientV2(base_url=config.RERANKER.API_URL, api_key=os.getenv("OPENAI_API_KEY", "none"))

    # Try to get model from service, fall back to default
    try:
        models_response = client.models.list()
        model_data = models_response.data  # pyright: ignore[reportAttributeAccess]
        model = model_data[0]["id"] if model_data else "Qwen/Qwen3-Reranker-0.6B"
    except Exception:
        logger.exception("Could not determine reranker model from client, falling back to default.")
        model = "Qwen/Qwen3-Reranker-0.6B"

    return RerankerClient(client=client, model=model)


def get_llm_client(config: AppConfig | None = None) -> LLMClient:
    """Get LLM client with automatically detected model.

    Args:
        config: Application configuration, will load default if None

    Returns:
        LLMClient with client and model name

    Raises:
        Exception: If client creation fails
    """
    config = config or ConfigurationManager.get_config()
    logger = structlog.get_logger()

    # Create OpenAI-compatible client first to get available models
    temp_client = Client(base_url=config.LLM.API_URL, api_key=os.getenv("OPENAI_API_KEY", "none"))

    # Try to get model from service, fall back to default
    try:
        model_data = temp_client.models.list().data
        model = model_data[0].id if model_data else "Qwen/Qwen2.5-72B-Instruct"
    except Exception:
        logger.exception("Could not determine LLM model from client, falling back to default.")
        model = "Qwen/Qwen2.5-72B-Instruct"

    # Create the actual LangChain ChatOpenAI client
    llm_client = ChatOpenAI(
        model=model,
        api_key=SecretStr("none"),
        base_url=config.LLM.API_URL,
    )

    return LLMClient(client=llm_client, model=model)
