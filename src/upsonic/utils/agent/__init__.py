"""
Agent Utilities Module

This module provides utility functions for agent operations,
including event yielding methods.
"""

import importlib
from typing import Any

_lazy_imports = {}


def _lazy_import(module_name: str, function_name: str = None):
    """Lazy import function to defer heavy imports until actually needed."""
    def _import():
        if module_name not in _lazy_imports:
            _lazy_imports[module_name] = importlib.import_module(module_name)
        
        if function_name:
            return getattr(_lazy_imports[module_name], function_name)
        return _lazy_imports[module_name]
    
    return _import


def __getattr__(name: str) -> Any:
    """Lazy loading of agent utility components."""
    
    # Try to import from events module
    try:
        events_module = _lazy_import("upsonic.utils.agent.events")()
        return getattr(events_module, name)
    except (ImportError, AttributeError):
        pass
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
    )


__all__ = []


