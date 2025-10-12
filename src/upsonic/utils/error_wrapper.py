"""
Error wrapper module for Upsonic framework.
"""

import functools
import asyncio
from typing import Any, Callable, Union, Optional
from upsonic.utils.package.exception import (
    UupsonicError,
    AgentExecutionError,
    ModelConnectionError,
    TaskProcessingError,
    ConfigurationError,
    RetryExhaustedError,
    NoAPIKeyException,
    CallErrorException
)
from upsonic.utils.printing import error_message


def map_pydantic_error_to_upsonic(error: Exception) -> UupsonicError:
    """
    Maps third-party errors to Upsonic-specific errors.
    
    Args:
        error: The original error from third-party sources
        
    Returns:
        UupsonicError: A wrapped Upsonic-specific error
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # API Key related errors
    if any(keyword in error_str for keyword in ['api key', 'apikey', 'authentication', 'unauthorized', '401']):
        return NoAPIKeyException(
            f"API key error: {str(error)}"
        )
    
    # Connection and network errors
    if any(keyword in error_str for keyword in ['connection', 'network', 'timeout', 'refused', 'unreachable']):
        return ModelConnectionError(
            message=f"Failed to connect to model service: {str(error)}",
            error_code="CONNECTION_ERROR",
            original_error=error
        )
    
    # Rate limiting and quota errors
    if any(keyword in error_str for keyword in ['rate limit', 'quota', 'billing', 'usage limit']):
        return ModelConnectionError(
            message=f"Model service quota or rate limit exceeded: {str(error)}",
            error_code="QUOTA_EXCEEDED",
            original_error=error
        )
    
    # Model or validation errors
    if any(keyword in error_str for keyword in ['validation', 'invalid input', 'bad request', '400']):
        return TaskProcessingError(
            message=f"Invalid task or input format: {str(error)}",
            error_code="VALIDATION_ERROR",
            original_error=error
        )
    
    # Configuration errors
    if any(keyword in error_str for keyword in ['configuration', 'config', 'setup', 'missing']):
        return ConfigurationError(
            message=f"Configuration error: {str(error)}",
            error_code="CONFIG_ERROR",
            original_error=error
        )
    
    # Server errors
    if any(keyword in error_str for keyword in ['500', 'server error', 'internal error', 'service unavailable']):
        return ModelConnectionError(
            message=f"Model service error: {str(error)}",
            error_code="SERVER_ERROR",
            original_error=error
        )
    
    # Pydantic specific errors
    if 'pydantic' in error_type.lower() or 'pydantic' in error_str:
        return AgentExecutionError(
            message=f"Agent execution failed: {str(error)}",
            error_code="AGENT_ERROR",
            original_error=error
        )
    
    # Default case - generic agent execution error
    return AgentExecutionError(
        message=f"Unexpected error during agent execution: {str(error)}",
        error_code="UNKNOWN_ERROR",
        original_error=error
    )


def upsonic_error_handler(
    max_retries: int = 0,
    show_error_details: bool = True,
    return_none_on_error: bool = False
):
    """
    Decorator that wraps functions to handle and convert errors to Upsonic-specific errors.
    
    Args:
        max_retries: Number of retries for transient errors (default: 0)
        show_error_details: Whether to display error details to user (default: True)
        return_none_on_error: Whether to return None instead of raising on error (default: False)
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                last_error = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except UupsonicError:
                        # Already a Upsonic error, re-raise
                        raise
                    except Exception as e:
                        last_error = e
                        upsonic_error = map_pydantic_error_to_upsonic(e)
                        
                        # If this is the last attempt or not a retryable error, handle it
                        if attempt == max_retries or not _is_retryable_error(upsonic_error):
                            if show_error_details:
                                _display_error(upsonic_error)
                            
                            if return_none_on_error:
                                return None
                            else:
                                raise upsonic_error
                        
                        # Wait before retry (exponential backoff)
                        if attempt < max_retries:
                            await asyncio.sleep(2 ** attempt)
                
                # This should never be reached, but just in case
                if return_none_on_error:
                    return None
                else:
                    raise RetryExhaustedError(
                        message=f"All {max_retries + 1} attempts failed. Last error: {str(last_error)}",
                        error_code="RETRY_EXHAUSTED",
                        original_error=last_error
                    )
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                last_error = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except UupsonicError:
                        # Already a Upsonic error, re-raise
                        raise
                    except Exception as e:
                        last_error = e
                        upsonic_error = map_pydantic_error_to_upsonic(e)
                        
                        # If this is the last attempt or not a retryable error, handle it
                        if attempt == max_retries or not _is_retryable_error(upsonic_error):
                            if show_error_details:
                                _display_error(upsonic_error)
                            
                            if return_none_on_error:
                                return None
                            else:
                                raise upsonic_error
                        
                        # Wait before retry
                        if attempt < max_retries:
                            import time
                            time.sleep(2 ** attempt)
                
                # This should never be reached, but just in case
                if return_none_on_error:
                    return None
                else:
                    raise RetryExhaustedError(
                        message=f"All {max_retries + 1} attempts failed. Last error: {str(last_error)}",
                        error_code="RETRY_EXHAUSTED",
                        original_error=last_error
                    )
            
            return sync_wrapper
    
    return decorator


def _is_retryable_error(error: UupsonicError) -> bool:
    """
    Determines if an error is retryable.
    
    Args:
        error: The Upsonic error to check
        
    Returns:
        bool: True if the error is retryable, False otherwise
    """
    retryable_codes = {
        "CONNECTION_ERROR",
        "SERVER_ERROR",
        "TIMEOUT_ERROR"
    }
    
    return (
        isinstance(error, ModelConnectionError) and 
        error.error_code in retryable_codes
    )


def _display_error(error: Union[UupsonicError, Exception]) -> None:
    """
    Displays error information to the user using the existing error_message function.
    
    Args:
        error: The Upsonic error to display
    """
    error_type_map = {
        NoAPIKeyException: "API Key Error",
        ModelConnectionError: "Connection Error", 
        TaskProcessingError: "Task Processing Error",
        ConfigurationError: "Configuration Error",
        AgentExecutionError: "Agent Execution Error",
        RetryExhaustedError: "Retry Exhausted Error"
    }
    
    error_type_name = error_type_map.get(type(error), "Upsonic Error")
    error_code = getattr(error, 'error_code', None)
    
    # Get error message based on error type
    if hasattr(error, 'message'):
        detail = error.message
    else:
        # For simple Exception classes, use str(error)
        detail = str(error)
    
    # Convert error code to HTTP-like status for display
    status_code = _get_status_code_from_error_code(error_code) if error_code else None
    
    error_message(
        error_type=error_type_name,
        detail=detail,
        error_code=status_code
    )


def _get_status_code_from_error_code(error_code: str) -> Optional[int]:
    """
    Maps error codes to HTTP-like status codes for display.
    
    Args:
        error_code: The error code to map
        
    Returns:
        Optional[int]: HTTP-like status code or None
    """
    code_map = {
        "CONNECTION_ERROR": 503,
        "SERVER_ERROR": 500,
        "QUOTA_EXCEEDED": 429,
        "VALIDATION_ERROR": 400,
        "CONFIG_ERROR": 422,
        "AGENT_ERROR": 500,
        "RETRY_EXHAUSTED": 503,
        "UNKNOWN_ERROR": 500
    }
    
    return code_map.get(error_code) 