# tests/test_config.py

import pytest
import os
import sys
import yaml

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config_loader import load_config

# =================================================================
# Tests for Configuration Loader (utils/config_loader.py)
# =================================================================

@pytest.fixture
def mock_config_file(tmp_path):
    """Creates a temporary mock config.yaml file for testing."""
    config_content = {
        "llm_providers": {
            "openai": {"api_key_env": "OPENAI_API_KEY", "default_model": "gpt-3.5-turbo"},
            "groq": {"api_key_env": "GROQ_API_KEY", "default_model": "llama3-8b-8192"},
        },
        "active_llm_provider": "openai",
        "embedding_model": "text-embedding-ada-002"
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_content, f)
    return str(config_path)

def test_config_loading_success(mock_config_file):
    """Tests that load_config successfully loads a valid config file."""
    config = load_config(config_path=mock_config_file)
    assert config["active_llm_provider"] == "openai"
    assert config["embedding_model"] == "text-embedding-ada-002"

def test_config_retrieves_model_config(mock_config_file):
    """Tests that the correct configuration is retrieved for a specific LLM provider."""
    config = load_config(config_path=mock_config_file)
    openai_config = config["llm_providers"]["openai"]
    assert openai_config["default_model"] == "gpt-3.5-turbo"
    assert openai_config["api_key_env"] == "OPENAI_API_KEY"

def test_config_file_not_found():
    """Tests that a FileNotFoundError is raised if the config file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config(config_path="non_existent_config.yaml")

def test_config_malformed_yaml(tmp_path):
    """Tests that a YAMLError is raised if the config file is not valid YAML."""
    malformed_path = tmp_path / "malformed.yaml"
    malformed_path.write_text("llm_provider: openai\n  bad-indent: here")
    
    with pytest.raises(yaml.YAMLError):
        load_config(config_path=str(malformed_path))
