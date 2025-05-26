"""
Embeddings endpoint for the Memementor Embedding Service.

This is the main API endpoint that handles embedding generation requests.
It takes text input(s) and a model name, then returns vector embeddings
that can be used for semantic search, clustering, or other NLP tasks.
"""
# FastAPI imports
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

# Local imports
from app.schemas.embedding_schemas import EmbeddingRequest, EmbeddingResponse, EmbeddingData, UsageInfo
from app.core.model_loader import get_model

# Standard library
import logging
from typing import List, Union, Dict, Any
import time  # for timing requests

# Third-party libraries
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Create router - this gets attached to the main app
router = APIRouter()

# TODO: Add caching for frequently requested embeddings?


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate embeddings for text",
    description="Generate vector embeddings for text using the specified model. Works with both single strings and batches."
)
async def create_embeddings(payload: EmbeddingRequest, request: Request):
    """
    The main endpoint for generating text embeddings.
    
    This function takes text input(s) and a model name, then returns vector embeddings
    that represent the semantic meaning of the text. These vectors can be used for
    similarity search, clustering, classification, or other NLP tasks.
    
    Args:
        payload: The request body containing 'input' (text or list of texts) and 'model' name
        request: FastAPI request object (gives us access to app state with loaded models)
        
    Returns:
        A structured response containing the embedding vectors
        
    Raises:
        HTTPException: For various error conditions (model not found, processing errors, etc.)
    """
    # Start timing the request
    start_time = time.time()
    
    # Log the incoming request - useful for debugging and monitoring
    is_batch = isinstance(payload.input, list)
    batch_size = len(payload.input) if is_batch else 1
    logger.info(f"📥 Embedding request: model={payload.model}, batch_size={batch_size}")
    
    # Grab the model from our loaded models
    model_instance = get_model(payload.model, request.app.state)
    
    # Make sure the requested model exists
    if not model_instance:
        # Get the list of models we do have available
        available_models = list(request.app.state.loaded_models.keys()) if hasattr(request.app.state, "loaded_models") else []
        
        # Log the error and return a helpful message
        logger.warning(f"❌ Model '{payload.model}' not found! Available models: {', '.join(available_models)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "model_not_found",
                "message": f"Model '{payload.model}' not available", 
                "available_models": available_models
            }
        )
    
    try:
        # Normalize input to always be a list of strings
        input_texts: List[str] = [payload.input] if isinstance(payload.input, str) else payload.input
        
        # Some basic validation
        if not input_texts:
            raise ValueError("Input texts cannot be empty")
            
        if any(not isinstance(text, str) for text in input_texts):
            raise ValueError("All input items must be strings")
        
        # Log what we're about to do
        logger.debug(f"🔄 Processing {len(input_texts)} text(s) with model {payload.model}")
        
        # This is where the magic happens! But it's CPU-intensive, so we run it in a thread
        # to avoid blocking the event loop. This keeps our API responsive.
        embeddings_list_np = await run_in_threadpool(model_instance.encode, input_texts)
        
        # Process the results and build our response
        response_data = []
        for i, embedding_np in enumerate(embeddings_list_np):
            # Different models might return different numpy array shapes
            # so we need to handle that gracefully
            if isinstance(embedding_np, np.ndarray):
                # Convert numpy array to a regular Python list for JSON serialization
                embedding_list = embedding_np.tolist()
            else:
                # Just in case we get something unexpected
                embedding_list = list(embedding_np)
                
            # Build the response object for this embedding
            response_data.append(
                EmbeddingData(
                    object="embedding",
                    embedding=embedding_list,
                    index=i
                )
            )
        
        # Calculate how long this took
        processing_time = time.time() - start_time
        
        # Log success with timing info
        logger.info(f"✅ Generated {len(response_data)} embeddings in {processing_time:.3f}s")
        
        # Build and return the final response
        # Note: We don't actually count tokens yet, but we could add that in the future
        return EmbeddingResponse(
            object="list",
            data=response_data,
            model=payload.model,
            usage=UsageInfo(prompt_tokens=0, total_tokens=0)  # TODO: Implement actual token counting
        )
        
    except HTTPException:
        # Just pass through any HTTP exceptions we've already created
        raise
        
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"⚠️ Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "validation_error", "message": str(e)}
        )
        
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"💥 Error generating embeddings: {str(e)}")
        logger.debug("Detailed error:", exc_info=True)
        
        # Return a sanitized error message to the client
        # (we don't want to leak internal details in production)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "embedding_generation_failed", "message": "Failed to generate embeddings"}
        )
