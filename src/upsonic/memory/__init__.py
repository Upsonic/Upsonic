from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory import save_agent_memory, get_agent_memory, reset_agent_memory
    from ..storage.memory.memory import Memory

def _get_memory_functions():
    """Lazy import of memory functions."""
    from .memory import save_agent_memory, get_agent_memory, reset_agent_memory
    from ..storage.memory.memory import Memory
    
    return {
        'save_agent_memory': save_agent_memory,
        'get_agent_memory': get_agent_memory,
        'reset_agent_memory': reset_agent_memory,
        'Memory': Memory,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    memory_functions = _get_memory_functions()
    if name in memory_functions:
        return memory_functions[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = ['save_agent_memory', 'get_agent_memory', 'reset_agent_memory', 'Memory']
