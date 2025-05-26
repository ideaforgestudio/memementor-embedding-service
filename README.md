# Memementor Embedding Service

A production-ready, multi-model FastAPI-based API service to generate sentence embeddings using various models from Hugging Face.

## Overview

This service provides a RESTful API for generating text embeddings using different models from Hugging Face's sentence-transformers library. It supports:

- Multiple embedding models loaded at startup
- Single text and batch text embedding generation
- Configurable model selection via environment variables
- Comprehensive error handling and logging

## About Our Organization

Memementor is dedicated to advancing AI-powered knowledge management solutions. Our embedding service is a core component of our knowledge infrastructure, enabling semantic search, content recommendation, and intelligent document processing.

We believe in building robust, scalable, and accessible AI services that can be integrated into various applications. This embedding service represents our commitment to providing high-quality AI tools for developers and organizations.

For more information about our projects and services, visit [mykb.online](https://mykb.online).

## Setup

### Prerequisites

- Python 3.9+
- pip (Python package manager)

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd memementor-embedding-service
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file from the example:

```bash
cp .env.example .env
```

5. Edit the `.env` file to configure your supported models and other settings:

```
# Comma-separated list of Hugging Face model names
SUPPORTED_MODELS="BAAI/bge-m3,sentence-transformers/all-MiniLM-L6-v2"

# Logging level
LOG_LEVEL="INFO"

# Hugging Face API token for downloading models (required for private/gated models)
# HF_TOKEN="your_huggingface_token_here"

# Device for model inference (cpu, cuda, mps)
DEVICE="cpu"

# API authentication key (for the Bearer token)
# API_KEY_SECRET="your_secret_key_here"

# Whether to require authentication (true/false)
# REQUIRE_AUTH="false"
```

## Running the Service

### Development Mode

```bash
uvicorn app.main:app --reload
```

The service will be available at http://localhost:8000.

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Using Docker

1. Build the Docker image:

```bash
docker build -t memementor-embedding-service .
```

2. Run the container:

```bash
docker run -p 8000:8000 --env-file .env memementor-embedding-service
```

## API Endpoints

### GET /

Returns the service health status and available models.

**Response Example:**

```json
{
  "service": "Memementor Embedding Service",
  "status": "healthy",
  "version": "0.1.0",
  "available_models": ["BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"]
}
```

### POST /v1/embeddings

Generate embeddings for text using a specified model.

**Request Body:**

```json
{
  "input": ["The quick brown fox jumps over the lazy dog."],
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

**Response Example:**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ..., -0.05],
      "index": 0
    }
  ],
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
```

## API Documentation
```

### API Reference

#### Health Check

```
GET /
```

Returns the service health status and available models.

**Response:**

```json
{
  "service": "Memementor Embedding Service",
  "status": "healthy",
  "version": "0.1.0",
  "available_models": ["BAAI/bge-m3", "sentence-transformers/all-MiniLM-L6-v2"]
}
```

#### Generate Embeddings

```
POST /v1/embeddings
```

Generate embeddings for text using a specified model.

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| input | string or array of strings | Yes | Text(s) to embed |
| model | string | Yes | Model name to use for embedding generation |

**Example Request (Single Text):**

```json
{
  "input": "The quick brown fox jumps over the lazy dog.",
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

**Example Request (Multiple Texts):**

```json
{
  "input": [
    "First sentence to embed.",
    "Second sentence to embed.",
    "Third sentence to embed."
  ],
  "model": "BAAI/bge-m3"
}
```

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ..., -0.05],
      "index": 0
    },
    // Additional embeddings if input was an array
  ],
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
```

**Error Responses:**

- 400 Bad Request: Invalid model name or other request error
- 422 Unprocessable Entity: Invalid input format
- 500 Internal Server Error: Server-side error during embedding generation

## Testing

### Unit and Integration Tests

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=app
```

### Load Testing

A k6 load testing script is provided to test the API under load. The script is located at `k6_test.js`.

1. Install k6 from https://k6.io/docs/get-started/installation/

2. Edit the script to include your API token:
   ```javascript
   const API_TOKEN = 'YOUR_BEARER_TOKEN_HERE'; // Replace with your actual token
   ```

3. Run the load test:
   ```bash
   k6 run k6_test.js
   ```

The script tests both single text and batch text embedding generation with a gradually increasing load, and provides metrics on response times and error rates.

## Troubleshooting

### Common Issues

1. **Model loading failures**: Check that the model names in your `.env` file are valid Hugging Face model identifiers.

2. **Memory issues**: Large models may require significant memory. Consider using smaller models or increasing the available memory.

3. **Slow response times**: Embedding generation can be CPU-intensive. Consider using a machine with more CPU resources or using smaller/faster models.

## License

[MIT License](LICENSE)
