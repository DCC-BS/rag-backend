from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from omegaconf import OmegaConf


@dataclass
class ChatConfig:
    EXAMPLE_QUERIES: list[str]
    DEFAULT_PROMPT: str


@dataclass
class DocStoreConfig:
    TYPE: str
    PATH: str
    TABLE_NAME: str
    MAX_CHUNK_SIZE: int
    MIN_CHUNK_SIZE: int
    SPLIT_OVERLAP: int
    DOCUMENT_DESCRIPTION: str


@dataclass
class RetrieverConfig:
    BM25_LIMIT: int
    VECTOR_LIMIT: int
    RERANK_TOP_K: int


@dataclass
class EmbeddingsConfig:
    API_URL: str
    EMBEDDING_INSTRUCTIONS: str


@dataclass
class LLMConfig:
    API_URL: str


@dataclass
class RerankerConfig:
    API_URL: str


@dataclass
class DoclingConfig:
    API_URL: str
    MAX_TOKENS: int
    MIN_TOKENS: int
    PAGE_BREAK_PLACEHOLDER: str


@dataclass
class IngestionConfig:
    STORAGE_TYPE: str
    S3_ENDPOINT: str
    S3_ACCESS_KEY: str
    S3_SECRET_KEY: str
    BUCKET_PREFIX: str = "documents"
    WATCH_ENABLED: bool = True
    BATCH_SIZE: int = 10
    SCAN_INTERVAL: int = 3600  # seconds
    PROCESSED_TAG: str = "ingestion_processed"
    UNPROCESSABLE_TAG: str = "unprocessable"
    NO_CHUNKS_TAG: str = "no_chunks"
    NOT_SUPPORTED_FILE_TAG: str = "not_supported_file"


@dataclass
class DataSource:
    name: str
    path: str


@dataclass
class AppConfig:
    APP_NAME: str
    VERSION: str
    DESCRIPTION: str
    RETRIEVER: RetrieverConfig
    EMBEDDINGS: EmbeddingsConfig
    LLM: LLMConfig
    RERANKER: RerankerConfig
    DOCLING: DoclingConfig
    INGESTION: IngestionConfig
    CHAT: ChatConfig
    ROLES: list[str]
    CORS_ORIGINS: list[str]
    AZURE_CLIENT_ID: str
    AZURE_TENANT_ID: str
    SCOPE_DESCRIPTION: str
    HMAC_SECRET: str


class ConfigurationManager:
    _instance: "ConfigurationManager | None" = None
    _config: AppConfig | None = None

    def __new__(cls) -> "ConfigurationManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def _raise_config_type_error(config_obj: object) -> NoReturn:
        """Raise a TypeError when config object is of wrong type."""
        raise TypeError("Failed to load configuration with correct type")

    @staticmethod
    def _raise_config_load_error(conf_file: Path, err: Exception) -> NoReturn:
        """Raise a ValueError when config file loading fails."""
        raise ValueError(f"Error loading config from {conf_file}") from err

    @staticmethod
    def _raise_config_validation_error(err: Exception) -> NoReturn:
        """Raise a ValueError when config validation fails."""
        error_msg = str(err).lower()
        if "missing" in error_msg:
            raise ValueError(f"Configuration validation failed: missing required fields - {err}")
        elif any(term in error_msg for term in ["type", "expected"]):
            raise ValueError(f"Configuration validation failed: type validation error - {err}")
        else:
            raise ValueError(f"Configuration validation failed: {err}")

    @classmethod
    def _load_config(cls) -> AppConfig:
        """Loads and merges configuration files using OmegaConf."""
        if cls._config is None:
            OmegaConf.clear_resolvers()

            # Create base config with schema and set to strict mode
            schema = OmegaConf.structured(AppConfig)

            # Create empty config
            config = OmegaConf.create()

            # Recursively find all yaml files in conf directory
            conf_dir = Path("src/rag/conf")
            yaml_files = sorted([f for f in conf_dir.rglob("*.yaml") if f.is_file()], key=lambda x: x.as_posix())

            if not yaml_files:
                raise ValueError(f"No configuration files found in {conf_dir}")

            # Merge all configs
            for conf_file in yaml_files:
                try:
                    with open(conf_file, encoding="utf-8") as f:
                        file_conf = OmegaConf.load(f)
                    config = OmegaConf.merge(config, file_conf)
                except Exception as err:
                    cls._raise_config_load_error(conf_file, err)

            try:
                # Merge schema with loaded config
                merged_config = OmegaConf.merge(schema, config)

                # Set struct mode after merging to allow dynamic dictionary keys
                OmegaConf.set_struct(merged_config, True)

                # Additional validation to ensure all required fields are present
                OmegaConf.resolve(merged_config)

                # Convert to object with validation
                config_obj = OmegaConf.to_object(merged_config)
                if not isinstance(config_obj, AppConfig):
                    cls._raise_config_type_error(config_obj)
                cls._config = config_obj
            except Exception as err:
                cls._raise_config_validation_error(err)

        return cls._config

    @classmethod
    def get_config(cls) -> AppConfig:
        """
        Retrieves the loaded configuration.

        Returns:
            AppConfig: The loaded configuration with strong typing.

        Raises:
            ValueError: If configuration has not been loaded yet or validation fails.
            TypeError: If configuration object is of wrong type.
        """
        if cls._config is None:
            return cls._load_config()
        return cls._config
