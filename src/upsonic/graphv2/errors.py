"""
Error types for GraphV2.

This module defines custom exceptions used throughout the GraphV2 system.
"""

from copy import deepcopy
from typing import Any


class GraphExecutionUncertainError(RuntimeError):
    """A logical node invocation may have executed and cannot safely be replayed.

    Dispatch records prove possible execution, not provider success. Inspect
    tasks and reconcile the external outcome before choosing a new invocation.
    """

    def __init__(self, tasks: list[dict[str, Any]]) -> None:
        self.tasks = deepcopy(tasks)
        self.pending_tasks = self.tasks
        names = ", ".join(task["node"] for task in tasks)
        super().__init__(f"Graph execution outcome is uncertain for: {names}")


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

