"""
Unit tests for the embeddings functionality.
"""
import pytest
from unittest.mock import Mock, patch
import numpy as np
from fastapi import HTTPException

from app.schemas.embedding_schemas import EmbeddingRequest, EmbeddingResponse
from app.api.v1.endpoints.embeddings import create_embeddings


class MockSentenceTransformer:
    """Mock for SentenceTransformer class."""
    
    def __init__(self, model_name):
        self.model_name = model_name
    
    def encode(self, texts):
        """Mock encode method that returns random embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        # Return a random embedding of size 384 for each text
        return np.random.rand(len(texts), 384)


@pytest.fixture
def mock_request():
    """Fixture for a mock FastAPI request."""
    request = Mock()
    request.app = Mock()
    request.app.state = Mock()
    request.app.state.loaded_models = {
        "sentence-transformers/all-MiniLM-L6-v2": MockSentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
        "BAAI/bge-m3": MockSentenceTransformer("BAAI/bge-m3")
    }
    return request


@pytest.mark.asyncio
async def test_create_embeddings_single_text(mock_request):
    """Test embedding generation for a single text input."""
    # Create a request payload with a single text
    payload = EmbeddingRequest(
        input="This is a test sentence.",
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Mock run_in_threadpool to directly call the function
    with patch("app.api.v1.endpoints.embeddings.run_in_threadpool", side_effect=lambda f, *args: f(*args)):
        response = await create_embeddings(payload, mock_request)
    
    # Verify the response structure
    assert isinstance(response, EmbeddingResponse)
    assert response.object == "list"
    assert response.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(response.data) == 1
    assert response.data[0].object == "embedding"
    assert response.data[0].index == 0
    assert len(response.data[0].embedding) == 384  # Check embedding dimension


@pytest.mark.asyncio
async def test_create_embeddings_batch_text(mock_request):
    """Test embedding generation for a batch of text inputs."""
    # Create a request payload with multiple texts
    payload = EmbeddingRequest(
        input=["First test sentence.", "Second test sentence.", "Third test sentence."],
        model="BAAI/bge-m3"
    )
    
    # Mock run_in_threadpool to directly call the function
    with patch("app.api.v1.endpoints.embeddings.run_in_threadpool", side_effect=lambda f, *args: f(*args)):
        response = await create_embeddings(payload, mock_request)
    
    # Verify the response structure
    assert isinstance(response, EmbeddingResponse)
    assert response.object == "list"
    assert response.model == "BAAI/bge-m3"
    assert len(response.data) == 3
    
    # Check each embedding in the response
    for i, embedding_data in enumerate(response.data):
        assert embedding_data.object == "embedding"
        assert embedding_data.index == i
        assert len(embedding_data.embedding) == 384  # Check embedding dimension


@pytest.mark.asyncio
async def test_create_embeddings_model_not_found(mock_request):
    """Test error handling when the requested model is not found."""
    # Create a request payload with a non-existent model
    payload = EmbeddingRequest(
        input="This is a test sentence.",
        model="non-existent-model"
    )
    
    # Verify that an HTTPException is raised
    with pytest.raises(HTTPException) as excinfo:
        await create_embeddings(payload, mock_request)
    
    # Check the exception details
    assert excinfo.value.status_code == 400
    # Check for the new error response format
    detail = excinfo.value.detail
    assert isinstance(detail, dict)
    assert "error" in detail
    assert detail["error"] == "model_not_found"
    assert "message" in detail
    assert "Model 'non-existent-model' not available" in detail["message"]
    assert "available_models" in detail


@pytest.mark.asyncio
async def test_create_embeddings_model_error(mock_request):
    """Test error handling when the model encoding fails."""
    # Create a request payload
    payload = EmbeddingRequest(
        input="This is a test sentence.",
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Mock the model to raise an exception
    mock_request.app.state.loaded_models["sentence-transformers/all-MiniLM-L6-v2"].encode = Mock(side_effect=Exception("Model error"))
    
    # Mock run_in_threadpool to propagate the exception
    with patch("app.api.v1.endpoints.embeddings.run_in_threadpool", side_effect=lambda f, *args: f(*args)):
        with pytest.raises(HTTPException) as excinfo:
            await create_embeddings(payload, mock_request)
    
    # Check the exception details
    assert excinfo.value.status_code == 500
    # Check for the new error response format
    detail = excinfo.value.detail
    assert isinstance(detail, dict)
    assert "error" in detail
    assert detail["error"] == "embedding_generation_failed"
    assert "message" in detail
    assert "Failed to generate embeddings" in detail["message"]
