"""
Backend Architecture for DeepAgent Filesystem

Provides flexible storage backends for the filesystem abstraction:
- BackendProtocol: Abstract interface for all backends
- StateBackend: Ephemeral in-memory storage
- MemoryBackend: Persistent storage via Upsonic Storage
- CompositeBackend: Route operations to different backends by path
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocol import BackendProtocol
    from .state_backend import StateBackend
    from .memory_backend import MemoryBackend
    from .composite_backend import CompositeBackend

def _get_backend_classes():
    """Lazy import of backend classes."""
    from .protocol import BackendProtocol
    from .state_backend import StateBackend
    from .memory_backend import MemoryBackend
    from .composite_backend import CompositeBackend
    
    return {
        'BackendProtocol': BackendProtocol,
        'StateBackend': StateBackend,
        'MemoryBackend': MemoryBackend,
        'CompositeBackend': CompositeBackend,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    backend_classes = _get_backend_classes()
    if name in backend_classes:
        return backend_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "BackendProtocol",
    "StateBackend",
    "MemoryBackend",
    "CompositeBackend",
]

