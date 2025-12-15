from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backends import (
        BackendProtocol,
        StateBackend,
        MemoryBackend,
        CompositeBackend,
    )
    from .tools import (
        FilesystemToolKit,
        PlanningToolKit,
        Todo,
        TodoList,
        SubagentToolKit,
    )
    from .deepagent import DeepAgent

def _get_backend_classes():
    """Lazy import of backend classes."""
    from .backends import (
        BackendProtocol,
        StateBackend,
        MemoryBackend,
        CompositeBackend,
    )
    
    return {
        'BackendProtocol': BackendProtocol,
        'StateBackend': StateBackend,
        'MemoryBackend': MemoryBackend,
        'CompositeBackend': CompositeBackend,
    }

def _get_tool_classes():
    """Lazy import of tool classes."""
    from .tools import (
        FilesystemToolKit,
        PlanningToolKit,
        Todo,
        TodoList,
        SubagentToolKit,
    )
    
    return {
        'FilesystemToolKit': FilesystemToolKit,
        'PlanningToolKit': PlanningToolKit,
        'Todo': Todo,
        'TodoList': TodoList,
        'SubagentToolKit': SubagentToolKit,
    }

def _get_deepagent_class():
    """Lazy import of DeepAgent class."""
    from .deepagent import DeepAgent
    return {'DeepAgent': DeepAgent}

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    # Backend classes
    backend_classes = _get_backend_classes()
    if name in backend_classes:
        return backend_classes[name]
    
    # Tool classes
    tool_classes = _get_tool_classes()
    if name in tool_classes:
        return tool_classes[name]
    
    # DeepAgent class
    deepagent_class = _get_deepagent_class()
    if name in deepagent_class:
        return deepagent_class[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    # Backends
    "BackendProtocol",
    "StateBackend",
    "MemoryBackend",
    "CompositeBackend",
    # Tools
    "FilesystemToolKit",
    "PlanningToolKit",
    "Todo",
    "TodoList",
    "SubagentToolKit",
    # Main Class
    "DeepAgent",
]

