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

        ConfigurationManager._instance = None
        ConfigurationManager._config = None

        config = ConfigurationManager.get_config()

        assert isinstance(config, AppConfig)
        assert config.APP_NAME == "Test App"
        assert config.VERSION == "1.0.0"
        assert config.DOC_STORE.TYPE == "lance"
        assert config.RETRIEVER.TOP_K == 5
        assert len(config.CHAT.EXAMPLE_QUERIES) == 2
        assert len(config.DOC_SOURCES) == 3
        assert config.DOC_SOURCES["Sozialhilfe"] == "/data/SH"
        assert config.DOC_SOURCES["Ergänzungsleistungen"] == "/data/EL"
        assert config.DOC_SOURCES["Ergänzungsleistungen2"] == "/data/EL2"
        admin_creds = config.LOGIN_CONFIG.credentials.usernames["admin"]
        assert isinstance(admin_creds, LoginCredentials)
        assert admin_creds.email == "admin@test.com"


def test_config_caching():
    """Test that configuration is properly cached."""
    ConfigurationManager._instance = None
    ConfigurationManager._config = None

    with patch("src.rag.utils.config.Path") as mock_path:
        mock_path.return_value = Path("src/rag/conf")

        config1 = ConfigurationManager.get_config()
        config2 = ConfigurationManager.get_config()

        assert config1 is config2


def test_invalid_config(mock_conf_dir, invalid_yaml_config):
    """Test handling of invalid configuration."""
    config_file = mock_conf_dir / "invalid.yaml"
    config_file.write_text(invalid_yaml_config)

    with patch("src.rag.utils.config.Path") as mock_path:
        mock_path.return_value = mock_conf_dir.parent.parent.parent

        ConfigurationManager._instance = None
        ConfigurationManager._config = None

        with pytest.raises((ValueError, ValidationError)) as exc_info:
            ConfigurationManager.get_config()

        assert any(["type" in str(exc_info.value).lower(), "validation" in str(exc_info.value).lower()])


def test_invalid_config_missing_fields(mock_conf_dir, incomplete_yaml_config):
    """Test handling of configuration with missing required fields."""
    config_file = mock_conf_dir / "incomplete.yaml"
    config_file.write_text(incomplete_yaml_config, encoding="utf-8")

    with patch("src.rag.utils.config.Path") as mock_path:
        mock_path.return_value = mock_conf_dir.parent.parent.parent

        ConfigurationManager._instance = None
        ConfigurationManager._config = None

        with pytest.raises(ValueError) as exc_info:
            ConfigurationManager.get_config()

        assert "missing" in str(exc_info.value).lower()


def test_invalid_config_wrong_types(mock_conf_dir, invalid_yaml_config):
    """Test handling of configuration with wrong data types."""
    config_file = mock_conf_dir / "invalid_types.yaml"
    config_file.write_text(invalid_yaml_config, encoding="utf-8")

    with patch("src.rag.utils.config.Path") as mock_path:
        mock_path.return_value = mock_conf_dir.parent.parent.parent

        ConfigurationManager._instance = None
        ConfigurationManager._config = None

        with pytest.raises((ValueError, ValidationError)) as exc_info:
            ConfigurationManager.get_config()

        error_msg = str(exc_info.value).lower()
        assert any(["type" in error_msg, "expected" in error_msg])


def test_duplicate_entries(mock_conf_dir, base_config_for_duplicates):
    """Test handling of duplicate entries in multiple config files."""
    base_config = base_config_for_duplicates

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

        ConfigurationManager._instance = None
        ConfigurationManager._config = None

        config = ConfigurationManager.get_config()

        assert config.DOC_SOURCES["source2"] == "/yet/another/path2"
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

        ConfigurationManager._instance = None
        ConfigurationManager._config = None

        config = ConfigurationManager.get_config()
        assert config.APP_NAME == "Override App"
        assert config.DESCRIPTION == "Base Description"
        assert len(config.DOC_SOURCES) == 3
        assert config.DOC_SOURCES["Sozialhilfe"] == "/data/SH"
        assert config.DOC_SOURCES["Ergänzungsleistungen"] == "/data/EL"
        assert config.DOC_SOURCES["Ergänzungsleistungen2"] == "/data/EL2"
