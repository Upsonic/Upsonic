"""
Agent Run Module

This module provides agent-specific run data classes for managing
input, context, and output of agent runs.
"""

import importlib
from typing import Any

_lazy_imports = {}


def _lazy_import(module_name: str, class_name: str = None):
    """Lazy import function to defer heavy imports until actually needed."""
    def _import():
        if module_name not in _lazy_imports:
            _lazy_imports[module_name] = importlib.import_module(module_name)
        
        if class_name:
            return getattr(_lazy_imports[module_name], class_name)
        return _lazy_imports[module_name]
    
    return _import


def _get_AgentRunInput():
    return _lazy_import("upsonic.run.agent.input", "AgentRunInput")()


def _get_AgentRunContext():
    return _lazy_import("upsonic.run.agent.context", "AgentRunContext")()


def _get_DurableCheckpoint():
    return _lazy_import("upsonic.run.agent.context", "DurableCheckpoint")()


def _get_AgentRunOutput():
    return _lazy_import("upsonic.run.agent.output", "AgentRunOutput")()


def __getattr__(name: str) -> Any:
    """Lazy loading of agent run module components."""
    
    if name == "AgentRunInput":
        return _get_AgentRunInput()
    elif name == "AgentRunContext":
        return _get_AgentRunContext()
    elif name == "DurableCheckpoint":
        return _get_DurableCheckpoint()
    elif name == "AgentRunOutput":
        return _get_AgentRunOutput()
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
    )


__all__ = [
    "AgentRunInput",
    "AgentRunContext",
    "DurableCheckpoint",
    "AgentRunOutput",
]



