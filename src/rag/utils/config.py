from dataclasses import dataclass
from pathlib import Path

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
class LoginConfig:
    credentials: dict[str, dict[str, LoginCredentials]]
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
    DOC_SOURCES: dict[str, str]
    LOGIN_CONFIG: LoginConfig


class ConfigurationManager:
    _instance: "ConfigurationManager | None" = None
    _config: AppConfig | None = None

    def __new__(cls) -> "ConfigurationManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def _load_config(cls) -> AppConfig:
        """Loads and merges configuration files using OmegaConf."""
        if cls._config is None:
            OmegaConf.clear_resolvers()

            # Create base config with schema
            schema = OmegaConf.structured(AppConfig)  # pyright: ignore[reportAny]
            config = OmegaConf.create()

            # Recursively find all yaml files in conf directory
            conf_dir = Path("src/rag/conf")
            yaml_files = sorted([f for f in conf_dir.rglob("*.yaml") if f.is_file()], key=lambda x: x.as_posix())

            # Merge all configs with schema validation
            for conf_file in yaml_files:
                try:
                    file_conf = OmegaConf.load(file_=conf_file)
                    config = OmegaConf.merge(config, file_conf)
                except Exception as e:
                    print(f"Error loading config from {conf_file}: {e}")

            # Merge schema with loaded config and convert to object
            merged_config = OmegaConf.merge(schema, config)  # pyright: ignore[reportAny]
            config_obj = OmegaConf.to_object(merged_config)
            if not isinstance(config_obj, AppConfig):
                raise ValueError("Failed to load configuration with correct type")
            cls._config = config_obj

        return cls._config

    @classmethod
    def get_config(cls) -> AppConfig:
        """
        Retrieves the loaded configuration.

        Returns:
            AppConfig: The loaded configuration with strong typing.

        Raises:
            ValueError: If configuration has not been loaded yet.
        """
        if cls._config is None:
            return cls._load_config()
        return cls._config
