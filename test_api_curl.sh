#!/bin/bash
# Test script for the Memementor Embedding Service API using curl

echo "===== Testing Original API ====="
echo "1. Testing health endpoint"
curl -s http://localhost:8000/ | jq

echo -e "\n2. Testing embeddings endpoint with single text"
curl -s -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The quick brown fox jumps over the lazy dog.",
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }' | jq '.data[0].embedding | length'

echo -e "\n3. Testing embeddings endpoint with batch text"
curl -s -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["First sentence to embed.", "Second sentence to embed."],
    "model": "BAAI/bge-m3"
  }' | jq '.data | length'

echo -e "\n\n===== Testing OpenAI-Compatible API ====="
echo "1. Testing OpenAI-compatible embeddings endpoint with single text"
curl -s -X POST http://localhost:8000/v1/chat/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The quick brown fox jumps over the lazy dog.",
    "model": "text-embedding-ada-002",
    "encoding_format": "float"
  }' | jq '.data[0].embedding | length'

echo -e "\n2. Testing OpenAI-compatible embeddings endpoint with OpenAI model name"
curl -s -X POST http://localhost:8000/v1/chat/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The quick brown fox jumps over the lazy dog.",
    "model": "text-embedding-3-small"
  }' | jq '.data[0].embedding | length'

echo -e "\n3. Testing OpenAI-compatible embeddings endpoint with batch text"
curl -s -X POST http://localhost:8000/v1/chat/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["First sentence to embed.", "Second sentence to embed."],
    "model": "text-embedding-ada-002"
  }' | jq '.data | length'

echo -e "\n4. Testing OpenAI-compatible embeddings endpoint with direct Hugging Face model name"
curl -s -X POST http://localhost:8000/v1/chat/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The quick brown fox jumps over the lazy dog.",
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }' | jq '.data[0].embedding | length'

echo -e "\n5. Testing OpenAI-compatible embeddings endpoint with authentication (if enabled)"
curl -s -X POST http://localhost:8000/v1/chat/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d '{
    "input": "The quick brown fox jumps over the lazy dog.",
    "model": "text-embedding-ada-002"
  }' | jq

echo -e "\n6. Testing error handling with non-existent model"
curl -s -X POST http://localhost:8000/v1/chat/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The quick brown fox jumps over the lazy dog.",
    "model": "non-existent-model"
  }' | jq
