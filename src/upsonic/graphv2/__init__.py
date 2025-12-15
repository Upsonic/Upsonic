from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .state_graph import StateGraph, START, END
    from .checkpoint import (
        BaseCheckpointer,
        MemorySaver,
        SqliteCheckpointer,
        StateSnapshot,
        Checkpoint,
    )
    from .primitives import Command, interrupt, Send
    from .store import BaseStore, InMemoryStore
    from .cache import BaseCache, InMemoryCache, SqliteCache, CachePolicy
    from .task import task, RetryPolicy, TaskFunction
    from .errors import GraphRecursionError, GraphValidationError, GraphInterruptError

def _get_graphv2_core_classes():
    """Lazy import of core graphv2 classes."""
    from .state_graph import StateGraph, START, END
    
    return {
        'StateGraph': StateGraph,
        'START': START,
        'END': END,
    }

def _get_graphv2_checkpoint_classes():
    """Lazy import of checkpoint classes."""
    from .checkpoint import (
        BaseCheckpointer,
        MemorySaver,
        SqliteCheckpointer,
        StateSnapshot,
        Checkpoint,
    )
    
    return {
        'BaseCheckpointer': BaseCheckpointer,
        'MemorySaver': MemorySaver,
        'SqliteCheckpointer': SqliteCheckpointer,
        'StateSnapshot': StateSnapshot,
        'Checkpoint': Checkpoint,
    }

def _get_graphv2_primitives():
    """Lazy import of primitives."""
    from .primitives import Command, interrupt, Send
    
    return {
        'Command': Command,
        'interrupt': interrupt,
        'Send': Send,
    }

def _get_graphv2_store_classes():
    """Lazy import of store classes."""
    from .store import BaseStore, InMemoryStore
    
    return {
        'BaseStore': BaseStore,
        'InMemoryStore': InMemoryStore,
    }

def _get_graphv2_cache_classes():
    """Lazy import of cache classes."""
    from .cache import BaseCache, InMemoryCache, SqliteCache, CachePolicy
    
    return {
        'BaseCache': BaseCache,
        'InMemoryCache': InMemoryCache,
        'SqliteCache': SqliteCache,
        'CachePolicy': CachePolicy,
    }

def _get_graphv2_task_classes():
    """Lazy import of task classes."""
    from .task import task, RetryPolicy, TaskFunction
    
    return {
        'task': task,
        'RetryPolicy': RetryPolicy,
        'TaskFunction': TaskFunction,
    }

def _get_graphv2_error_classes():
    """Lazy import of error classes."""
    from .errors import GraphRecursionError, GraphValidationError, GraphInterruptError
    
    return {
        'GraphRecursionError': GraphRecursionError,
        'GraphValidationError': GraphValidationError,
        'GraphInterruptError': GraphInterruptError,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    # Core classes
    core_classes = _get_graphv2_core_classes()
    if name in core_classes:
        return core_classes[name]
    
    # Checkpoint classes
    checkpoint_classes = _get_graphv2_checkpoint_classes()
    if name in checkpoint_classes:
        return checkpoint_classes[name]
    
    # Primitives
    primitives = _get_graphv2_primitives()
    if name in primitives:
        return primitives[name]
    
    # Store classes
    store_classes = _get_graphv2_store_classes()
    if name in store_classes:
        return store_classes[name]
    
    # Cache classes
    cache_classes = _get_graphv2_cache_classes()
    if name in cache_classes:
        return cache_classes[name]
    
    # Task classes
    task_classes = _get_graphv2_task_classes()
    if name in task_classes:
        return task_classes[name]
    
    # Error classes
    error_classes = _get_graphv2_error_classes()
    if name in error_classes:
        return error_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    # Core graph components
    "StateGraph",
    "START",
    "END",
    
    # Checkpointing
    "BaseCheckpointer",
    "MemorySaver",
    "SqliteCheckpointer",
    "StateSnapshot",
    "Checkpoint",
    
    # Primitives
    "Command",
    "interrupt",
    "Send",
    
    # Store (cross-thread memory)
    "BaseStore",
    "InMemoryStore",
    
    # Cache
    "BaseCache",
    "InMemoryCache",
    "SqliteCache",
    "CachePolicy",
    
    # Task decorator
    "task",
    "RetryPolicy",
    "TaskFunction",
    
    # Errors
    "GraphRecursionError",
    "GraphValidationError",
    "GraphInterruptError",
]

