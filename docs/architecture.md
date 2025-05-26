# Memementor Embedding Service Architecture

## High-Level Architecture

```
Client -> FastAPI (Uvicorn) -> API Router -> Embedding Endpoint Logic
                                    |
                                    -> Model Loader/Cache (In-memory dict)
                                    |
                                    -> SentenceTransformer Model(s)
                                    |
                                    -> Pydantic (Validation & Serialization)
                                    |
                                    -> Configuration Module
                                    |
                                    -> Logging Module
```

## Components

### FastAPI Application (main.py)
Main entry point for the application. Handles startup/shutdown events (lifespan), includes API routers, and configures the application.

### API Router (app/api/v1/)
Defines routes and forwards requests to endpoint handlers. Organized by API version for future extensibility.

### Embedding Endpoint (embeddings.py)
Contains logic for `/v1/embeddings` endpoint. Handles request validation, model selection, embedding generation, and response formatting.

### Model Loader (model_loader.py)
Responsible for loading specified sentence-transformers models from Hugging Face Hub into an in-memory dictionary during application startup. Provides access to loaded models.

### Configuration Module (config.py)
Loads settings (e.g., supported models, log level) from environment variables with sensible defaults.

### Pydantic Schemas (embedding_schemas.py)
Defines data structures for request and response validation and serialization.

### Error Handling (error_handlers.py)
Centralized custom exception handlers for FastAPI.

### Uvicorn
ASGI server to run the FastAPI application.

## API Design (/v1/embeddings)

### Endpoint
- **Path**: POST /v1/embeddings
- **Purpose**: Generate embeddings for text using a specified model

### Request Body (JSON)
```json
{
  "input": ["The quick brown fox jumps over the lazy dog."],
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

- **input**: Union[str, List[str]] (Required) - Text(s) to embed.
- **model**: str (Required) - Name of the Hugging Face model to use (e.g., "BAAI/bge-m3"). Must be one of the configured and loaded models.

### Response Body (JSON - Success 200 OK)
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ..., -0.05],
      "index": 0
    }
    // ... more objects if input was a list
  ],
  "model": "BAAI/bge-m3",
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
```

### Error Responses
- **400 Bad Request**: Malformed JSON, missing required fields, or requested model not available
- **422 Unprocessable Entity**: Validation errors (e.g., invalid model name, incorrect input type)
- **500 Internal Server Error**: Unexpected server-side errors (e.g., model inference failure)

### Error Response Body
```json
{
  "detail": "Error message or list of validation errors"
}
```

## Model Management Strategy

- Models are defined in `SUPPORTED_MODELS` environment variable (comma-separated)
- Models are loaded at application startup using FastAPI's lifespan context manager
- Loaded models are stored in `app.state.loaded_models` dictionary
  - Keys: model names (e.g., "BAAI/bge-m3")
  - Values: loaded SentenceTransformer instances
- If a model fails to load, an error is logged and the model is skipped
- The API returns an error if a client requests a model that failed to load or is not configured

## Concurrency Strategy

- FastAPI endpoints are defined as `async def` to support asynchronous processing
- The `model.encode()` method from sentence-transformers is CPU-bound
- To prevent blocking the asyncio event loop for long-running encoding tasks, these calls are wrapped with `fastapi.concurrency.run_in_threadpool`
- This allows the server to handle multiple requests concurrently even when processing is CPU-intensive

## Configuration Management

- Uses python-dotenv to load environment variables from a `.env` file during development
- `app/core/config.py` defines Pydantic settings models to parse and validate environment variables, providing defaults
- Key configurations: `SUPPORTED_MODELS`, `LOG_LEVEL`

## Logging Strategy

- Standard Python logging module
- Configurable log level via `LOG_LEVEL` environment variable
- Structured logging with timestamp, level, and message
- Logs important events:
  - Application startup/shutdown
  - Model loading (success/failure)
  - Request received
  - Errors and exceptions
