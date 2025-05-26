"""
Model loader module for the Memementor Embedding Service.

This is where we load and manage all the embedding models from Hugging Face.
The heavy lifting happens here - downloading models, setting up the right
device (CPU/GPU), and making them available to the API.
"""
# Third-party imports
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
import logging
import time  # for tracking load times
from typing import Dict, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MODEL_LOAD_TIMEOUT = 300  # seconds - some big models take a while to load!


async def load_models_on_startup(app: FastAPI, settings) -> None:
    """
    Load all the models specified in settings during application startup.
    
    This function runs during the FastAPI lifespan startup event. It tries to load
    each model specified in the settings, keeping track of successes and failures.
    
    Args:
        app: FastAPI application instance where we'll store the models
        settings: Application settings containing model configuration
    """
    # Initialize our model storage
    app.state.loaded_models = {}
    success_count = 0
    failed_models = []
    
    # Let everyone know what we're about to do
    logger.info(f"🔄 Loading {len(settings.SUPPORTED_MODELS)} models: {', '.join(settings.SUPPORTED_MODELS)}")
    
    # Try to load each model one by one
    for model_name in settings.SUPPORTED_MODELS:
        start_time = time.time()
        try:
            logger.info(f"⏳ Loading model: {model_name}")
            
            # This is where the magic happens! But it's synchronous and can block...
            # Might be worth looking into async loading in the future?
            model = SentenceTransformer(
                model_name,
                use_auth_token=settings.HF_TOKEN,  # For private/gated models
                device=settings.DEVICE  # CPU/GPU selection
            )
            
            # Store the model for later use
            app.state.loaded_models[model_name] = model
            success_count += 1
            
            # Calculate and log the load time
            load_time = time.time() - start_time
            logger.info(f"✅ Loaded model: {model_name} in {load_time:.2f}s")
            
            # Log some info about the model
            embedding_dim = model.get_sentence_embedding_dimension()
            logger.info(f"   Model info - Embedding dimension: {embedding_dim}")
            
        except Exception as e:
            failed_models.append(model_name)
            logger.error(f"❌ Failed to load model {model_name}: {str(e)}")
            logger.debug(f"Detailed error:", exc_info=True)
    
    # Warn if we couldn't load any models
    if success_count == 0 and settings.SUPPORTED_MODELS:
        logger.warning("⚠️ No models were successfully loaded! The service won't work properly.")
    elif failed_models:
        logger.warning(f"⚠️ Some models failed to load: {', '.join(failed_models)}")
    else:
        logger.info(f"🎉 All models loaded successfully!")
    
    # Final summary
    logger.info(f"📊 Model loading summary: {success_count}/{len(settings.SUPPORTED_MODELS)} models loaded")
    logger.info(f"🧠 Available models: {', '.join(app.state.loaded_models.keys())}")
    
    # TODO: Add model caching to disk to speed up future loads?


def get_model(model_name: str, app_state) -> Optional[SentenceTransformer]:
    """
    Get a loaded model by name.
    
    This is a simple helper function that safely retrieves a model from the app state.
    It returns None if the model doesn't exist rather than raising an exception.
    
    Args:
        model_name: Name of the model to retrieve (e.g., 'BAAI/bge-m3')
        app_state: FastAPI application state where we stored our loaded models
        
    Returns:
        The SentenceTransformer model if found, None if not available
    """
    # Simple dictionary get with None as default
    if not hasattr(app_state, "loaded_models"):
        # This shouldn't happen in normal operation, but just in case
        logger.warning("No loaded_models found in app_state!")
        return None
        
    return app_state.loaded_models.get(model_name)


def get_available_models(app_state) -> Dict[str, SentenceTransformer]:
    """
    Get all available loaded models.
    
    Useful for endpoints that need to list available models or for
    internal functions that need to operate on all models.
    
    Args:
        app_state: FastAPI application state containing loaded models
        
    Returns:
        Dictionary mapping model names to their SentenceTransformer instances
    """
    # If we don't have any models loaded yet, return an empty dict
    if not hasattr(app_state, "loaded_models"):
        return {}
        
    return app_state.loaded_models  # Simple but effective!
