from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import ValidationError

from src.rag.utils.config import (
    AppConfig,
    ConfigurationManager,
    LoginCredentials,
)


@pytest.fixture
def sample_yaml_config():
    return """
APP_NAME: "Test App"
VERSION: "1.0.0"
DESCRIPTION: "Test Description"
DOC_STORE:
  TYPE: "lance"
  PATH: "/path/to/store"
  TABLE_NAME: "documents"
  MAX_CHUNK_SIZE: 1000
  MIN_CHUNK_SIZE: 100
  SPLIT_OVERLAP: 50
  DOCUMENT_DESCRIPTION: "Test docs"
RETRIEVER:
  TYPE: "semantic"
  FETCH_FOR_RERANKING: 10
  TOP_K: 5
  MAX_RECURSION: 3
EMBEDDINGS:
  API_URL: "http://localhost:8000"
LLM:
  MODEL: "test-model"
  API_URL: "http://localhost:8001"
  TEMPERATURE: 0.7
  MAX_TOKENS: 512
DOCLING:
  NUM_THREADS: 4
  USE_GPU: false
CHAT:
  EXAMPLE_QUERIES: ["How are you?", "What is this?"]
  DEFAULT_PROMPT: "Hello!"
ROLES: ["admin", "user"]
DATA_DIR: "/data"
DOC_SOURCES:
  Sozialhilfe: "${DATA_DIR}/SH"
  Ergänzungsleistungen: "${DATA_DIR}/EL"
  Ergänzungsleistungen2: "${DATA_DIR}/EL2"
LOGIN_CONFIG:
  credentials:
    usernames:
      admin:
        email: "admin@test.com"
        failed_login_attempts: 0
        name: "Admin"
        password: "password123"
        logged_in: false
        roles: ["admin"]
  cookie:
    expiry_days: 30
    key: "some_key"
    name: "session"
"""


def test_singleton_pattern():
    """Test that ConfigurationManager follows singleton pattern."""
    config_manager1 = ConfigurationManager()
    config_manager2 = ConfigurationManager()
    assert config_manager1 is config_manager2


@pytest.fixture
def mock_conf_dir(tmp_path):
    """Create a temporary conf directory with test config files."""
    conf_dir = tmp_path / "src" / "rag" / "conf"
    conf_dir.mkdir(parents=True)
    return conf_dir


def test_config_loading(mock_conf_dir, sample_yaml_config):
    """Test configuration loading from yaml files."""
    config_file = mock_conf_dir / "config.yaml"
    config_file.write_text(sample_yaml_config, encoding="utf-8")

    with patch("src.rag.utils.config.Path") as mock_path:
        mock_path.return_value = mock_conf_dir.parent.parent.parent

        # Reset singleton state
        ConfigurationManager._instance = None
        ConfigurationManager._config = None

        config = ConfigurationManager.get_config()

        assert isinstance(config, AppConfig)
        assert config.APP_NAME == "Test App"
        assert config.VERSION == "1.0.0"
        assert config.DOC_STORE.TYPE == "lance"
        assert config.RETRIEVER.TOP_K == 5
        assert len(config.CHAT.EXAMPLE_QUERIES) == 2
        # Test data sources with environment variable interpolation
        assert len(config.DOC_SOURCES) == 3
        assert config.DOC_SOURCES["Sozialhilfe"] == "/data/SH"
        assert config.DOC_SOURCES["Ergänzungsleistungen"] == "/data/EL"
        assert config.DOC_SOURCES["Ergänzungsleistungen2"] == "/data/EL2"
        # Test credentials
        admin_creds = config.LOGIN_CONFIG.credentials.usernames["admin"]
        assert isinstance(admin_creds, LoginCredentials)
        assert admin_creds.email == "admin@test.com"


def test_config_caching():
    """Test that configuration is properly cached."""
    # Reset singleton state
    ConfigurationManager._instance = None
    ConfigurationManager._config = None

    with patch("src.rag.utils.config.Path") as mock_path:
        mock_path.return_value = Path("src/rag/conf")

        config1 = ConfigurationManager.get_config()
        config2 = ConfigurationManager.get_config()

        assert config1 is config2


def test_invalid_config(mock_conf_dir):
    """Test handling of invalid configuration."""
    invalid_yaml = """
APP_NAME: "Test App"
VERSION: "1.0.0"
DESCRIPTION: "Test Description"
DOC_STORE:
    TYPE: 123  # Invalid: TYPE should be a string
    PATH: true  # Invalid: PATH should be a string
    TABLE_NAME: ["invalid"]  # Invalid: TABLE_NAME should be a string
    MAX_CHUNK_SIZE: "not a number"  # Invalid: should be an integer
DOC_SOURCES:
    123: "/path"  # Invalid: key should be a string
    "test": ["invalid"]  # Invalid: value should be a string
"""

    config_file = mock_conf_dir / "invalid.yaml"
    config_file.write_text(invalid_yaml)

    with patch("src.rag.utils.config.Path") as mock_path:
        mock_path.return_value = mock_conf_dir.parent.parent.parent

        # Reset singleton state
        ConfigurationManager._instance = None
        ConfigurationManager._config = None

        with pytest.raises((ValueError, ValidationError)) as exc_info:
            ConfigurationManager.get_config()

        # Verify the error message contains type validation information
        assert any(["type" in str(exc_info.value).lower(), "validation" in str(exc_info.value).lower()])


def test_invalid_config_missing_fields(mock_conf_dir):
    """Test handling of configuration with missing required fields."""
    incomplete_yaml = """
APP_NAME: "Test App"
VERSION: "1.0.0"
# Missing DESCRIPTION and other required fields
DOC_STORE:
    TYPE: "lance"
    # Missing required PATH field
    TABLE_NAME: "documents"
    MAX_CHUNK_SIZE: 1000
"""

    config_file = mock_conf_dir / "incomplete.yaml"
    config_file.write_text(incomplete_yaml, encoding="utf-8")

    with patch("src.rag.utils.config.Path") as mock_path:
        mock_path.return_value = mock_conf_dir.parent.parent.parent

        # Reset singleton state
        ConfigurationManager._instance = None
        ConfigurationManager._config = None

        with pytest.raises(ValueError) as exc_info:
            ConfigurationManager.get_config()

        assert "missing" in str(exc_info.value).lower()


def test_invalid_config_wrong_types(mock_conf_dir):
    """Test handling of configuration with wrong data types."""
    invalid_types_yaml = """
APP_NAME: "Test App"
VERSION: 1.0  # Should be string
DESCRIPTION: "Test Description"
DOC_STORE:
    TYPE: ["invalid"]  # Should be string
    PATH: 123  # Should be string
    TABLE_NAME: true  # Should be string
    MAX_CHUNK_SIZE: "1000"  # Should be integer
    MIN_CHUNK_SIZE: false  # Should be integer
    SPLIT_OVERLAP: 50.5  # Should be integer
    DOCUMENT_DESCRIPTION: null  # Should be string
RETRIEVER:
    TYPE: "semantic"
    FETCH_FOR_RERANKING: "10"  # Should be integer
    TOP_K: 5.5  # Should be integer
    MAX_RECURSION: true  # Should be integer
EMBEDDINGS:
    API_URL: ["invalid"]  # Should be string
LLM:
    MODEL: 123  # Should be string
    API_URL: null  # Should be string
    TEMPERATURE: "0.7"  # Should be float
    MAX_TOKENS: "512"  # Should be integer
DOCLING:
    NUM_THREADS: "4"  # Should be integer
    USE_GPU: "false"  # Should be boolean
CHAT:
    EXAMPLE_QUERIES: "not a list"  # Should be list
    DEFAULT_PROMPT: ["not", "a", "string"]  # Should be string
ROLES: "not a list"  # Should be list
DATA_DIR: 123  # Should be string
DOC_SOURCES:
    test1: 123  # Should be string
    test2: true  # Should be string
LOGIN_CONFIG:
    credentials:
        usernames:
            admin:
                email: 123  # Should be string
                failed_login_attempts: "0"  # Should be integer
                name: ["invalid"]  # Should be string
                password: null  # Should be string
                logged_in: "true"  # Should be boolean
                roles: "admin"  # Should be list
    cookie:
        expiry_days: "30"  # Should be integer
        key: true  # Should be string
        name: 123  # Should be string
"""

    config_file = mock_conf_dir / "invalid_types.yaml"
    config_file.write_text(invalid_types_yaml, encoding="utf-8")

    with patch("src.rag.utils.config.Path") as mock_path:
        mock_path.return_value = mock_conf_dir.parent.parent.parent

        # Reset singleton state
        ConfigurationManager._instance = None
        ConfigurationManager._config = None

        with pytest.raises((ValueError, ValidationError)) as exc_info:
            ConfigurationManager.get_config()

        error_msg = str(exc_info.value).lower()
        assert any(["type" in error_msg, "expected" in error_msg])


def test_duplicate_entries(mock_conf_dir):
    """Test handling of duplicate entries in multiple config files."""
    base_config = """
APP_NAME: "Base App"
VERSION: "1.0.0"
DESCRIPTION: "Base Description"
DOC_STORE:
    TYPE: "lance"
    PATH: "/path/to/store"
    TABLE_NAME: "documents"
    MAX_CHUNK_SIZE: 1000
    MIN_CHUNK_SIZE: 100
    SPLIT_OVERLAP: 50
    DOCUMENT_DESCRIPTION: "Test docs"
RETRIEVER:
    TYPE: "semantic"
    FETCH_FOR_RERANKING: 10
    TOP_K: 5
    MAX_RECURSION: 3
EMBEDDINGS:
    API_URL: "http://localhost:8000"
LLM:
    MODEL: "test-model"
    API_URL: "http://localhost:8001"
    TEMPERATURE: 0.7
    MAX_TOKENS: 512
DOCLING:
    NUM_THREADS: 4
    USE_GPU: false
CHAT:
    EXAMPLE_QUERIES: ["How are you?", "What is this?"]
    DEFAULT_PROMPT: "Hello!"
ROLES: ["admin", "user"]
DATA_DIR: "/data"
DOC_SOURCES:
    source1: "/path1"
    source2: "/path2"
LOGIN_CONFIG:
    credentials:
        usernames:
            admin:
                email: "admin@test.com"
                failed_login_attempts: 0
                name: "Admin"
                password: "password123"
                logged_in: false
                roles: ["admin"]
    cookie:
        expiry_days: 30
        key: "some_key"
        name: "session"
"""

    override1_config = """
DOC_SOURCES:
    source2: "/different/path2"  # Duplicate key with different value
    source3: "/path3"
"""

    override2_config = """
DOC_SOURCES:
    source2: "/yet/another/path2"  # Another duplicate with different value
    source4: "/path4"
"""

    (mock_conf_dir / "01_base.yaml").write_text(base_config, encoding="utf-8")
    (mock_conf_dir / "02_override1.yaml").write_text(override1_config, encoding="utf-8")
    (mock_conf_dir / "03_override2.yaml").write_text(override2_config, encoding="utf-8")

    with patch("src.rag.utils.config.Path") as mock_path:
        mock_path.return_value = mock_conf_dir.parent.parent.parent

        # Reset singleton state
        ConfigurationManager._instance = None
        ConfigurationManager._config = None

        # Should not raise an error - last value should win
        config = ConfigurationManager.get_config()

        # Verify that the last value for duplicate key wins
        assert config.DOC_SOURCES["source2"] == "/yet/another/path2"
        # Verify other keys are preserved
        assert config.DOC_SOURCES["source1"] == "/path1"
        assert config.DOC_SOURCES["source3"] == "/path3"
        assert config.DOC_SOURCES["source4"] == "/path4"
        assert len(config.DOC_SOURCES) == 4


def test_invalid_yaml_syntax(mock_conf_dir):
    """Test handling of YAML files with invalid syntax."""
    invalid_syntax_yaml = """
APP_NAME: "Test App"
VERSION: "1.0.0"
DESCRIPTION: "Test Description"
DOC_STORE:
    TYPE: "lance"
    PATH: /path/to/store  # Missing quotes
    TABLE_NAME: documents  # Missing quotes
    - invalid list item  # Invalid syntax
    MAX_CHUNK_SIZE: 1000
    MIN_CHUNK_SIZE: : 100  # Invalid syntax (double colon)
    SPLIT_OVERLAP: 50
    *invalid_anchor  # Invalid YAML syntax
"""

    config_file = mock_conf_dir / "invalid_syntax.yaml"
    config_file.write_text(invalid_syntax_yaml, encoding="utf-8")

    with patch("src.rag.utils.config.Path") as mock_path:
        mock_path.return_value = mock_conf_dir.parent.parent.parent

        # Reset singleton state
        ConfigurationManager._instance = None
        ConfigurationManager._config = None

        with pytest.raises(ValueError) as exc_info:
            ConfigurationManager.get_config()

        assert "error loading config" in str(exc_info.value).lower()


def test_multiple_config_files(mock_conf_dir):
    """Test merging of multiple configuration files."""
    base_config = """
APP_NAME: "Base App"
VERSION: "1.0.0"
DESCRIPTION: "Base Description"
DOC_STORE:
  TYPE: "lance"
  PATH: "/path/to/store"
  TABLE_NAME: "documents"
  MAX_CHUNK_SIZE: 1000
  MIN_CHUNK_SIZE: 100
  SPLIT_OVERLAP: 50
  DOCUMENT_DESCRIPTION: "Test docs"
RETRIEVER:
  TYPE: "semantic"
  FETCH_FOR_RERANKING: 10
  TOP_K: 5
  MAX_RECURSION: 3
EMBEDDINGS:
  API_URL: "http://localhost:8000"
LLM:
  MODEL: "test-model"
  API_URL: "http://localhost:8001"
  TEMPERATURE: 0.7
  MAX_TOKENS: 512
DOCLING:
  NUM_THREADS: 4
  USE_GPU: false
CHAT:
  EXAMPLE_QUERIES: ["How are you?", "What is this?"]
  DEFAULT_PROMPT: "Hello!"
ROLES: ["admin", "user"]
DATA_DIR: "/data"
DOC_SOURCES:
  Sozialhilfe: "${DATA_DIR}/SH"
  Ergänzungsleistungen: "${DATA_DIR}/EL"
LOGIN_CONFIG:
  credentials:
    usernames:
      admin:
        email: "admin@test.com"
        failed_login_attempts: 0
        name: "Admin"
        password: "password123"
        logged_in: false
        roles: ["admin"]
  cookie:
    expiry_days: 30
    key: "some_key"
    name: "session"
"""

    override_config = """
APP_NAME: "Override App"
DOC_SOURCES:
  Ergänzungsleistungen2: "${DATA_DIR}/EL2"
"""

    (mock_conf_dir / "01_base.yaml").write_text(base_config, encoding="utf-8")
    (mock_conf_dir / "02_override.yaml").write_text(override_config, encoding="utf-8")

    with patch("src.rag.utils.config.Path") as mock_path:
        mock_path.return_value = mock_conf_dir.parent.parent.parent

        # Reset singleton state
        ConfigurationManager._instance = None
        ConfigurationManager._config = None

        config = ConfigurationManager.get_config()
        assert config.APP_NAME == "Override App"
        assert config.DESCRIPTION == "Base Description"  # Should retain base config values
        # Test merged DOC_SOURCES
        assert len(config.DOC_SOURCES) == 3
        assert config.DOC_SOURCES["Sozialhilfe"] == "/data/SH"
        assert config.DOC_SOURCES["Ergänzungsleistungen"] == "/data/EL"
        assert config.DOC_SOURCES["Ergänzungsleistungen2"] == "/data/EL2"
