"""
Configuration module for the Memementor Embedding Service.

This module loads and validates configuration settings from environment variables
with sensible defaults.
"""
import os
import logging
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Settings:
    """
    Settings for the application, loaded from environment variables.
    """
    def __init__(self):
        # Load SUPPORTED_MODELS
        models_str = os.getenv('SUPPORTED_MODELS', 'BAAI/bge-m3,sentence-transformers/all-MiniLM-L6-v2')
        self.SUPPORTED_MODELS = [model.strip() for model in models_str.split(',') if model.strip()]
        
        # Use default models if no models were specified or if the list is empty
        if not self.SUPPORTED_MODELS:
            self.SUPPORTED_MODELS = ["BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"]
            logger.warning("No models specified in SUPPORTED_MODELS, using defaults")
            
        logger.debug(f"Loaded SUPPORTED_MODELS: {self.SUPPORTED_MODELS}")
        
        # Load LOG_LEVEL
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        
        # Load HF_TOKEN
        self.HF_TOKEN = os.getenv('HF_TOKEN')
        
        # Load DEVICE
        self.DEVICE = os.getenv('DEVICE', 'cpu')
        
        # Load API_KEY for Bearer token authentication
        self.API_KEY = os.getenv('API_KEY_SECRET')
        if not self.API_KEY:
            logger.warning("API_KEY_SECRET not set. API will be accessible without authentication.")
        else:
            logger.info("API authentication enabled with Bearer token.")
            
        # Enable/disable authentication
        self.REQUIRE_AUTH = os.getenv('REQUIRE_AUTH', 'false').lower() == 'true'


# Create a global settings instance
settings = Settings()
