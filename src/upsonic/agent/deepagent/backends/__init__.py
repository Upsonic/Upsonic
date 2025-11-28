"""
Backend Architecture for DeepAgent Filesystem

Provides flexible storage backends for the filesystem abstraction:
- BackendProtocol: Abstract interface for all backends
- StateBackend: Ephemeral in-memory storage
- MemoryBackend: Persistent storage via Upsonic Storage
- CompositeBackend: Route operations to different backends by path
"""

from .protocol import BackendProtocol
from .state_backend import StateBackend
from .memory_backend import MemoryBackend
from .composite_backend import CompositeBackend

__all__ = [
    "BackendProtocol",
    "StateBackend",
    "MemoryBackend",
    "CompositeBackend",
]

