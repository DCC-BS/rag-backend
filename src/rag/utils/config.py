from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from omegaconf import MISSING, OmegaConf


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
    TYPE: str
    FETCH_FOR_RERANKING: int
    TOP_K: int
    MAX_RECURSION: int


@dataclass
class EmbeddingsConfig:
    API_URL: str


@dataclass
class LLMConfig:
    MODEL: str
    API_URL: str
    TEMPERATURE: float
    MAX_TOKENS: int


@dataclass
class DoclingConfig:
    NUM_THREADS: int
    USE_GPU: bool


@dataclass
class LoginCredentials:
    email: str
    failed_login_attempts: int
    name: str
    password: str
    logged_in: bool
    roles: list[str]


@dataclass
class LoginCookieConfig:
    expiry_days: int
    key: str
    name: str


@dataclass
class DataSource:
    name: str
    path: str


@dataclass
class Credentials:
    usernames: dict[str, LoginCredentials]


@dataclass
class LoginConfig:
    credentials: Credentials
    cookie: LoginCookieConfig


@dataclass
class AppConfig:
    APP_NAME: str
    VERSION: str
    DESCRIPTION: str
    DOC_STORE: DocStoreConfig
    RETRIEVER: RetrieverConfig
    EMBEDDINGS: EmbeddingsConfig
    LLM: LLMConfig
    DOCLING: DoclingConfig
    CHAT: ChatConfig
    ROLES: list[str]
    DATA_DIR: str
    LOGIN_CONFIG: LoginConfig
    DOC_SOURCES: dict[str, str] = MISSING


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
        raise ValueError("Configuration validation failed") from err

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
