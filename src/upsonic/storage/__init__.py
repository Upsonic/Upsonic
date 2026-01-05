from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import Storage
    from .providers import (
        InMemoryStorage,
        JSONStorage,
        Mem0Storage,
        PostgresStorage,
        RedisStorage,
        SqliteStorage,
        MongoStorage,
    )
    from .memory import Memory

def _get_base_classes():
    """Lazy import of base classes."""
    from .base import Storage
    
    return {
        'Storage': Storage,
    }

def _get_provider_classes():
    """Lazy import of provider classes."""
    from .providers import (
        InMemoryStorage,
        JSONStorage,
        Mem0Storage,
        PostgresStorage,
        RedisStorage,
        SqliteStorage,
        MongoStorage,
    )
    
    return {
        'InMemoryStorage': InMemoryStorage,
        'JSONStorage': JSONStorage,
        'Mem0Storage': Mem0Storage,
        'PostgresStorage': PostgresStorage,
        'RedisStorage': RedisStorage,
        'SqliteStorage': SqliteStorage,
        'MongoStorage': MongoStorage,
    }

def _get_memory_classes():
    """Lazy import of memory classes."""
    from .memory import Memory
    
    return {
        'Memory': Memory,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    base_classes = _get_base_classes()
    if name in base_classes:
        return base_classes[name]
    
    provider_classes = _get_provider_classes()
    if name in provider_classes:
        return provider_classes[name]
    
    memory_classes = _get_memory_classes()
    if name in memory_classes:
        return memory_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "Storage",
    "InMemoryStorage",
    "JSONStorage",
    "Mem0Storage",
    "PostgresStorage",
    "RedisStorage",
    "SqliteStorage",
    "MongoStorage",
    "Memory", 
]
