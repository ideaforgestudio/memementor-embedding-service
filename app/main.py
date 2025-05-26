"""
Main application module for the Memementor Embedding Service.

This is where all the magic happens! This module sets up the FastAPI app,
handles model loading during startup, and wires up all the routes.
"""
# Standard library imports
from contextlib import asynccontextmanager
import logging

# Third-party imports
from fastapi import FastAPI

# Local imports - keep these organized!
from app.core.model_loader import load_models_on_startup
from app.core.config import settings
from app.api.v1.endpoints import embeddings as v1_embeddings
from app.utils.error_handlers import install_error_handlers

# TODO: Add metrics collection for monitoring (maybe Prometheus?)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    
    This handles the startup/shutdown lifecycle. We load all the models
    when the app starts up so they're ready to go for inference requests.
    
    Args:
        app: FastAPI application instance
    """
    # Let's get this party started!
    logger.info("🚀 Starting up Memementor Embedding Service")
    
    try:
        # Load all the embedding models - this might take a while for larger models
        await load_models_on_startup(app, settings)
        logger.info(f"Successfully loaded {len(app.state.loaded_models)} models")
    except Exception as e:
        # Catch any unexpected errors during startup
        logger.error(f"Error during startup: {e}", exc_info=True)
        # Re-raise so FastAPI knows something went wrong
        raise
    
    # This is where FastAPI takes over and handles requests
    yield
    
    # Time to clean up and go home
    logger.info("👋 Shutting down Memementor Embedding Service")
    # We could add explicit cleanup here if needed in the future


# Set up our FastAPI app with all the bells and whistles
app = FastAPI(
    lifespan=lifespan,  # This is where our model loading happens
    title="Memementor Embedding Service",
    description="A blazing fast service for generating text embeddings using various models from Hugging Face",
    version="0.1.0",  # Remember to bump this when we make significant changes
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc (nicer looking docs)
    # openapi_url=None,  # Uncomment to disable OpenAPI schema
)

# Make sure we catch and format errors nicely
install_error_handlers(app)

# Wire up our API endpoints
app.include_router(
    v1_embeddings.router,
    prefix="/v1",  # Version prefix for API versioning
    tags=["Embeddings V1"]
)


@app.get("/", tags=["Health"])
async def read_root():
    """
    Root endpoint for health checks and basic service info.
    
    This is useful for monitoring and making sure everything is up and running.
    It also shows which models are currently loaded and available for use.
    
    Returns:
        Dict with service status and available models
    """
    # Check if we have any models loaded
    available_models = []
    if hasattr(app.state, "loaded_models"):
        available_models = list(app.state.loaded_models.keys())
    
    # Return a nice health status response
    return {
        "service": "Memementor Embedding Service",
        "status": "healthy",  # We could add more complex health checking logic here
        "version": "0.1.0",  # Should match the version in the FastAPI app definition
        "available_models": available_models,
        # "uptime": "1d 3h 45m"  # TODO: Add uptime tracking
    }
