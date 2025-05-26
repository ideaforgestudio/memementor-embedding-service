"""
Integration tests for the embeddings API.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from app.main import app


class MockSentenceTransformer:
    """Mock for SentenceTransformer class for integration tests."""
    
    def __init__(self, model_name):
        self.model_name = model_name
    
    def encode(self, texts):
        """Mock encode method that returns random embeddings with correct dimensions for each model."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Use different dimensions based on the model
        if self.model_name == "BAAI/bge-m3":
            # BAAI/bge-m3 produces 1024-dimensional embeddings
            return np.random.rand(len(texts), 1024)
        else:
            # all-MiniLM-L6-v2 produces 384-dimensional embeddings
            return np.random.rand(len(texts), 384)


@pytest.fixture
def client():
    """
    Create a TestClient for the FastAPI app with mocked models.
    """
    # Mock the load_models_on_startup function to use our mock models
    with patch("app.core.model_loader.load_models_on_startup") as mock_load:
        # Setup the mock to add models to app.state
        async def mock_load_models(app_instance, settings):
            app_instance.state.loaded_models = {
                "sentence-transformers/all-MiniLM-L6-v2": MockSentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
                "BAAI/bge-m3": MockSentenceTransformer("BAAI/bge-m3")
            }
        
        mock_load.side_effect = mock_load_models
        
        # Create a test client
        with TestClient(app) as client:
            yield client


def test_read_root(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "status" in data
    assert data["status"] == "healthy"
    assert "available_models" in data


def test_create_embeddings_single_text(client):
    """Test embedding generation for a single text input."""
    # Create a request payload with a single text
    payload = {
        "input": "This is a test sentence.",
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    # Make the request
    response = client.post("/v1/embeddings", json=payload)
    
    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(data["data"]) == 1
    assert data["data"][0]["object"] == "embedding"
    assert data["data"][0]["index"] == 0
    assert len(data["data"][0]["embedding"]) == 384  # Check embedding dimension


def test_create_embeddings_batch_text(client):
    """Test embedding generation for a batch of text inputs."""
    # Create a request payload with multiple texts
    payload = {
        "input": ["First test sentence.", "Second test sentence.", "Third test sentence."],
        "model": "BAAI/bge-m3"
    }
    
    # Make the request
    response = client.post("/v1/embeddings", json=payload)
    
    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert data["model"] == "BAAI/bge-m3"
    assert len(data["data"]) == 3
    
    # Check each embedding in the response
    for i, embedding_data in enumerate(data["data"]):
        assert embedding_data["object"] == "embedding"
        assert embedding_data["index"] == i
        assert len(embedding_data["embedding"]) == 1024  # BAAI/bge-m3 produces 1024-dimensional embeddings


def test_create_embeddings_model_not_found(client):
    """Test error handling when the requested model is not found."""
    # Create a request payload with a non-existent model
    payload = {
        "input": "This is a test sentence.",
        "model": "non-existent-model"
    }
    
    # Make the request
    response = client.post("/v1/embeddings", json=payload)
    
    # Verify the response
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    # Check for the new error response format
    assert "error" in data["detail"]
    assert data["detail"]["error"] == "model_not_found"
    assert "Model 'non-existent-model' not available" in data["detail"]["message"]
    assert "available_models" in data["detail"]


def test_create_embeddings_invalid_request(client):
    """Test error handling for invalid request data."""
    # Test missing required field
    payload = {
        "input": "This is a test sentence."
        # Missing 'model' field
    }
    
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 422
    
    # Test empty input
    payload = {
        "input": "",
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 422
    
    # Test empty input list
    payload = {
        "input": [],
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 422
