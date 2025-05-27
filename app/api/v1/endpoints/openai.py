"""
OpenAI-compatible API endpoints for the Memementor Embedding Service.

This module provides API endpoints that match OpenAI's API format,
allowing for drop-in compatibility with clients using OpenAI's API.
"""
# FastAPI imports
from fastapi import APIRouter, HTTPException, Request, status, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Local imports
from app.schemas.openai_schemas import (
    OpenAIEmbeddingRequest, 
    OpenAIEmbeddingResponse, 
    OpenAIEmbeddingData, 
    OpenAIUsage,
    map_model_name
)
from app.core.model_loader import get_model
from app.core.config import settings

# Standard library
import logging
from typing import List, Dict, Any
import time

# Third-party libraries
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Create router - this gets attached to the main app
router = APIRouter()

# Set up Bearer token authentication
security = HTTPBearer(auto_error=False)


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify the API key if authentication is required.
    
    This follows OpenAI's authentication pattern using Bearer tokens.
    """
    if not settings.REQUIRE_AUTH:
        return True
        
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Authentication required", "type": "authentication_error"}},
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Bearer authentication required", "type": "authentication_error"}},
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != settings.API_KEY_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Invalid API key", "type": "invalid_api_key"}},
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True


@router.post(
    "/embeddings",
    response_model=OpenAIEmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Create embeddings (OpenAI-compatible)",
    description="Creates embeddings for the provided text using the specified model. Compatible with OpenAI's API."
)
async def create_embeddings(
    payload: OpenAIEmbeddingRequest, 
    request: Request,
    authenticated: bool = Depends(verify_api_key)
):
    """
    OpenAI-compatible endpoint for generating text embeddings.
    
    This function provides a drop-in replacement for OpenAI's embeddings API.
    It accepts the same request format and returns the same response format.
    
    Args:
        payload: The request body following OpenAI's API format
        request: FastAPI request object
        authenticated: Dependency injection for authentication
        
    Returns:
        An OpenAI-compatible response containing the embedding vectors
        
    Raises:
        HTTPException: For various error conditions (model not found, processing errors, etc.)
    """
    # Start timing the request
    start_time = time.time()
    
    # Map the model name from OpenAI format to our format if needed
    mapped_model = map_model_name(payload.model)
    
    # Log the incoming request
    is_batch = isinstance(payload.input, list)
    batch_size = len(payload.input) if is_batch else 1
    logger.info(f"📥 OpenAI-compatible embedding request: model={payload.model} (mapped to {mapped_model}), batch_size={batch_size}")
    
    # Grab the model from our loaded models
    model_instance = get_model(mapped_model, request.app.state)
    
    # Make sure the requested model exists
    if not model_instance:
        # Get the list of models we do have available
        available_models = list(request.app.state.loaded_models.keys()) if hasattr(request.app.state, "loaded_models") else []
        
        # Log the error
        logger.warning(f"❌ Model '{mapped_model}' not found! Available models: {', '.join(available_models)}")
        
        # Return error in OpenAI format
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model '{payload.model}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            }
        )
    
    try:
        # Normalize input to always be a list of strings
        input_texts: List[str] = [payload.input] if isinstance(payload.input, str) else payload.input
        
        # Log what we're about to do
        logger.debug(f"🔄 Processing {len(input_texts)} text(s) with model {mapped_model}")
        
        # Generate embeddings using our model
        embeddings_list_np = await run_in_threadpool(model_instance.encode, input_texts)
        
        # Process the results and build our response
        response_data = []
        for i, embedding_np in enumerate(embeddings_list_np):
            # Convert numpy array to a regular Python list for JSON serialization
            if isinstance(embedding_np, np.ndarray):
                embedding_list = embedding_np.tolist()
            else:
                embedding_list = list(embedding_np)
                
            # Build the response object for this embedding
            response_data.append(
                OpenAIEmbeddingData(
                    object="embedding",
                    embedding=embedding_list,
                    index=i
                )
            )
        
        # Calculate how long this took
        processing_time = time.time() - start_time
        
        # Log success with timing info
        logger.info(f"✅ Generated {len(response_data)} OpenAI-compatible embeddings in {processing_time:.3f}s")
        
        # Estimate token count (very rough estimation)
        # In a production system, you'd want to use a proper tokenizer
        total_chars = sum(len(text) for text in input_texts)
        estimated_tokens = total_chars // 4  # Very rough approximation
        
        # Build and return the final response in OpenAI format
        return OpenAIEmbeddingResponse(
            object="list",
            data=response_data,
            model=payload.model,  # Return the original model name for compatibility
            usage=OpenAIUsage(
                prompt_tokens=estimated_tokens,
                total_tokens=estimated_tokens
            )
        )
        
    except HTTPException:
        # Just pass through any HTTP exceptions we've already created
        raise
        
    except ValueError as e:
        # Handle validation errors in OpenAI format
        logger.warning(f"⚠️ Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "param": "input"
                }
            }
        )
        
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"💥 Error generating embeddings: {str(e)}")
        logger.debug("Detailed error:", exc_info=True)
        
        # Return a sanitized error message in OpenAI format
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "An error occurred while generating embeddings",
                    "type": "server_error"
                }
            }
        )
