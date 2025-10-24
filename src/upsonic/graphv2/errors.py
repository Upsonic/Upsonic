"""
Error types for GraphV2.

This module defines custom exceptions used throughout the GraphV2 system.
"""


class GraphRecursionError(RecursionError):
    """Raised when graph execution exceeds the recursion limit.
    
    The recursion limit controls how many supersteps (node executions)
    a graph can perform before raising this error. This prevents
    infinite loops and runaway execution.
    """
    
    def __init__(self, message: str = "Graph execution exceeded recursion limit"):
        """Initialize the error.
        
        Args:
            message: Error message
        """
        super().__init__(message)
        self.message = message


class GraphInterruptError(Exception):
    """Raised when a graph is interrupted and cannot continue.
    
    This is different from InterruptException which is used internally
    for human-in-the-loop workflows. This error indicates a problem
    with interrupt handling.
    """
    
    def __init__(self, message: str):
        """Initialize the error.
        
        Args:
            message: Error message
        """
        super().__init__(message)
        self.message = message


class GraphValidationError(ValueError):
    """Raised when graph structure or configuration is invalid."""
    
    def __init__(self, message: str):
        """Initialize the error.
        
        Args:
            message: Error message
        """
        super().__init__(message)
        self.message = message

