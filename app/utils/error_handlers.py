"""
Error handlers for the Memementor Embedding Service.

This module defines custom exception handlers for FastAPI.
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc):
    """
    Handle HTTPException and return a consistent JSON response.
    
    Args:
        request: FastAPI request object
        exc: Exception instance
        
    Returns:
        JSONResponse with error details
    """
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle RequestValidationError and return a consistent JSON response.
    
    Args:
        request: FastAPI request object
        exc: RequestValidationError instance
        
    Returns:
        JSONResponse with validation error details
    """
    errors = exc.errors()
    error_messages = []
    
    for error in errors:
        loc = " -> ".join([str(l) for l in error["loc"]])
        msg = f"{loc}: {error['msg']}"
        error_messages.append(msg)
    
    logger.error(f"Validation error: {error_messages}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": error_messages}
    )


async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions and return a consistent JSON response.
    
    Args:
        request: FastAPI request object
        exc: Exception instance
        
    Returns:
        JSONResponse with error details
    """
    logger.exception(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


def install_error_handlers(app: FastAPI):
    """
    Install all error handlers on the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    from fastapi.exceptions import HTTPException
    
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
