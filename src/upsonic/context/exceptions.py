"""
Custom exceptions for context processing.
"""


class ContextProcessingError(Exception):
    """Base exception for context processing errors."""

    pass


class ContextValidationError(ContextProcessingError):
    """Exception raised when context validation fails."""

    pass


class ContextStrategyError(ContextProcessingError):
    """Exception raised when no suitable strategy is found."""

    pass


class ContextSerializationError(ContextProcessingError):
    """Exception raised during context serialization."""

    pass
