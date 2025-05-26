"""
Unit tests for the configuration module.
"""
import os
import pytest
from unittest.mock import patch

from app.core.config import Settings


def test_default_settings():
    """Test that default settings are loaded correctly."""
    settings = Settings()
    
    # Check default values
    assert settings.LOG_LEVEL == "INFO"
    assert "BAAI/bge-m3" in settings.SUPPORTED_MODELS
    assert "sentence-transformers/all-MiniLM-L6-v2" in settings.SUPPORTED_MODELS


@patch.dict(os.environ, {"SUPPORTED_MODELS": "model1,model2,model3", "LOG_LEVEL": "DEBUG"})
def test_settings_from_env_vars():
    """Test that settings are loaded correctly from environment variables."""
    settings = Settings()
    
    # Check values from environment variables
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.SUPPORTED_MODELS == ["model1", "model2", "model3"]


@patch.dict(os.environ, {"SUPPORTED_MODELS": "model1, model2, model3"})
def test_settings_with_spaces():
    """Test that settings are loaded correctly from environment variables with spaces."""
    settings = Settings()
    
    # Check that spaces are stripped
    assert settings.SUPPORTED_MODELS == ["model1", "model2", "model3"]


@patch.dict(os.environ, {"SUPPORTED_MODELS": ""})
def test_empty_models_list():
    """Test behavior with empty SUPPORTED_MODELS."""
    settings = Settings()
    
    # Should fall back to defaults
    assert "BAAI/bge-m3" in settings.SUPPORTED_MODELS
    assert "sentence-transformers/all-MiniLM-L6-v2" in settings.SUPPORTED_MODELS
