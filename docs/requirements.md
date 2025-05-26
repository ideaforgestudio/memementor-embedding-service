# Memementor Embedding Service Requirements

## Functional Requirements

### FR1: RESTful API Endpoint
The service MUST provide a RESTful API endpoint to generate embeddings for input text.

### FR2: Single Text Support
The service MUST support generating embeddings for a single text string.

### FR3: Batch Text Support
The service MUST support generating embeddings for a batch (list) of text strings in a single request.

### FR4: Model Selection
The service MUST allow clients to specify which supported embedding model to use for generation.

### FR5: Structured Response Format
The service MUST return embeddings in a structured JSON format, including the embedding vector, input index, and model used.

### FR6: Model Loading and Caching
The service MUST load specified embedding models upon startup and keep them in memory for efficient request processing.

### FR7: Request Validation
The service MUST validate incoming requests for correct format and required parameters.

### FR8: Error Handling
The service MUST handle errors gracefully and return appropriate HTTP status codes and error messages.

### FR9: Configurable Models
The list of supported models MUST be configurable (e.g., via environment variables).

## Non-Functional Requirements

### NFR1: Performance

#### NFR1.1: Efficient Model Loading
Model loading at startup should be efficient.

#### NFR1.2: Low Latency
Embedding generation latency for a single short text (e.g., < 100 words) using all-MiniLM-L6-v2 should ideally be under 100ms on standard hardware (excluding network latency).

#### NFR1.3: Concurrency
The service should handle at least 10 concurrent requests without significant degradation, depending on available hardware resources.

### NFR2: Scalability

#### NFR2.1: Horizontal Scaling
The service architecture SHOULD support horizontal scaling (running multiple instances).

#### NFR2.2: Statelessness
The service SHOULD be stateless to facilitate scaling.

### NFR3: Reliability

#### NFR3.1: High Availability
The service should aim for high availability (e.g., 99.9%).

#### NFR3.2: Request Isolation
Errors in processing one request should not affect other requests.

#### NFR3.3: Graceful Degradation
If a configured model fails to load, the service should log the error and continue running with successfully loaded models, if any.

### NFR4: Maintainability

#### NFR4.1: Code Quality
Code MUST be well-commented and follow Python best practices (PEP 8).

#### NFR4.2: Project Structure
The project structure MUST be logical and modular.

#### NFR4.3: Configuration Management
Configuration SHOULD be externalized (e.g., environment variables).

### NFR5: Security (Basic)

#### NFR5.1: Input Validation
Input validation MUST be implemented to prevent common vulnerabilities.

#### NFR5.2: Future Authentication
The API should be securable via API keys or other authentication mechanisms (placeholder for future implementation).

### NFR6: Usability (Developer Experience)

#### NFR6.1: Documentation
The API MUST be well-documented.

#### NFR6.2: Easy Setup
Setting up and running the service locally SHOULD be straightforward.

## Supported Models (Initial)

- BAAI/bge-m3
- sentence-transformers/all-MiniLM-L6-v2
- (Mechanism to add more models easily)
