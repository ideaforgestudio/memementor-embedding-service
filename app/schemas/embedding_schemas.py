"""
Pydantic schemas for the Memementor Embedding Service.

This module defines the data structures for request and response validation
and serialization.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Union, Literal


class EmbeddingRequest(BaseModel):
    """
    Schema for embedding generation request.
    """
    input: Union[str, List[str]] = Field(
        ...,  # Required field
        description="Text(s) to embed. Can be a single string or a list of strings.",
        example=["The quick brown fox jumps over the lazy dog."]
    )
    model: str = Field(
        ...,  # Required field
        description="Name of the model to use for embedding generation. Must be one of the supported models.",
        example="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    @validator('input')
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


class EmbeddingData(BaseModel):
    """
    Schema for a single embedding result.
    """
    object: Literal["embedding"] = "embedding"
    embedding: List[float] = Field(
        ...,
        description="The embedding vector representation of the input text."
    )
    index: int = Field(
        ...,
        description="The index of this embedding in the request. Starts at 0 for the first embedding.",
        example=0
    )


class UsageInfo(BaseModel):
    """
    Schema for token usage information.
    """
    prompt_tokens: int = Field(
        0,
        description="Number of tokens in the prompt. May be estimated or 0 if not applicable.",
        example=0
    )
    total_tokens: int = Field(
        0,
        description="Total number of tokens used. May be estimated or 0 if not applicable.",
        example=0
    )


class EmbeddingResponse(BaseModel):
    """
    Schema for embedding generation response.
    """
    object: Literal["list"] = "list"
    data: List[EmbeddingData] = Field(
        ...,
        description="List of embedding data objects, one for each input text."
    )
    model: str = Field(
        ...,
        description="Name of the model used for embedding generation.",
        example="sentence-transformers/all-MiniLM-L6-v2"
    )
    usage: UsageInfo = Field(
        default_factory=UsageInfo,
        description="Token usage information for this request."
    )
