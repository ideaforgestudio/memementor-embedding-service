"""
OpenAI-compatible API schemas for the Memementor Embedding Service.

This module defines Pydantic schemas that match OpenAI's API format,
allowing for drop-in compatibility with clients using OpenAI's API.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Union, Literal, Optional, Dict, Any


class OpenAIEmbeddingRequest(BaseModel):
    """
    Schema for OpenAI-compatible embedding request.
    
    Based on: https://platform.openai.com/docs/api-reference/embeddings/create
    """
    model: str = Field(
        ...,  # Required field
        description="ID of the model to use. Can be a Hugging Face model ID or an OpenAI model ID."
    )
    input: Union[str, List[str]] = Field(
        ...,  # Required field
        description="Input text to embed, can be a string or array of strings."
    )
    encoding_format: Optional[Literal["float", "base64"]] = Field(
        "float",
        description="The format to return the embeddings in. Only 'float' is currently supported."
    )
    user: Optional[str] = Field(
        None,
        description="A unique identifier representing your end-user. Not used but included for compatibility."
    )
    
    @field_validator('encoding_format')
    def validate_encoding_format(cls, v):
        """Validate that encoding_format is supported."""
        if v != "float":
            raise ValueError("Only 'float' encoding_format is currently supported")
        return v
    
    @field_validator('input')
    def validate_input_not_empty(cls, v):
        """Validate that input is not empty."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("Input string cannot be empty")
        if isinstance(v, list):
            if not v:
                raise ValueError("Input list cannot be empty")
            if any(not isinstance(item, str) for item in v):
                raise ValueError("All items in input list must be strings")
            if any(not item.strip() for item in v):
                raise ValueError("Input list cannot contain empty strings")
        return v


class OpenAIEmbeddingData(BaseModel):
    """
    Schema for a single embedding result in OpenAI format.
    """
    object: Literal["embedding"] = "embedding"
    embedding: List[float] = Field(
        ...,
        description="The embedding vector"
    )
    index: int = Field(
        ...,
        description="The index of this embedding in the input array"
    )


class OpenAIUsage(BaseModel):
    """
    Schema for token usage information in OpenAI format.
    """
    prompt_tokens: int = Field(
        ...,
        description="Number of tokens in the prompt"
    )
    total_tokens: int = Field(
        ...,
        description="Total number of tokens used"
    )


class OpenAIEmbeddingResponse(BaseModel):
    """
    Schema for embedding response in OpenAI format.
    """
    object: Literal["list"] = "list"
    data: List[OpenAIEmbeddingData] = Field(
        ...,
        description="List of embedding objects"
    )
    model: str = Field(
        ...,
        description="ID of the model used"
    )
    usage: OpenAIUsage = Field(
        ...,
        description="Usage statistics"
    )


# Model mapping from OpenAI models to Hugging Face models
OPENAI_TO_HF_MODEL_MAPPING = {
    # OpenAI embedding models
    "text-embedding-ada-002": "sentence-transformers/all-MiniLM-L6-v2",  # 1536 dimensions in OpenAI, 384 in this model
    "text-embedding-3-small": "BAAI/bge-m3",  # 1536 dimensions in OpenAI, 1024 in this model
    "text-embedding-3-large": "BAAI/bge-m3",  # 3072 dimensions in OpenAI, using 1024 dim model as fallback
    
    # Add more mappings as needed
}

# Function to map OpenAI model names to Hugging Face model names
def map_model_name(model_name: str) -> str:
    """
    Maps an OpenAI model name to a Hugging Face model name.
    If the model name is not recognized as an OpenAI model, returns it unchanged
    (assuming it's already a Hugging Face model name).
    """
    return OPENAI_TO_HF_MODEL_MAPPING.get(model_name, model_name)
