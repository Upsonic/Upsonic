from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .base import Storage
from .types import SessionId, UserId

if TYPE_CHECKING:
    from .providers import (
        InMemoryStorage,
        JSONStorage,
        Mem0Storage,
        PostgresStorage,
        RedisStorage,
        SqliteStorage,
    )
    from .session import (
        InteractionSession,
        UserProfile
    )
    from .memory import Memory

def _get_provider_classes():
    """Lazy import of provider classes."""
    from .providers import (
        InMemoryStorage,
        JSONStorage,
        Mem0Storage,
        PostgresStorage,
        RedisStorage,
        SqliteStorage,
    )
    
    return {
        'InMemoryStorage': InMemoryStorage,
        'JSONStorage': JSONStorage,
        'Mem0Storage': Mem0Storage,
        'PostgresStorage': PostgresStorage,
        'RedisStorage': RedisStorage,
        'SqliteStorage': SqliteStorage,
    }

def _get_session_classes():
    """Lazy import of session classes."""
    from .session import (
        InteractionSession,
        UserProfile
    )
    
    return {
        'InteractionSession': InteractionSession,
        'UserProfile': UserProfile,
    }

def _get_memory_classes():
    """Lazy import of memory classes."""
    from .memory import Memory
    
    return {
        'Memory': Memory,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    provider_classes = _get_provider_classes()
    if name in provider_classes:
        return provider_classes[name]
    
    session_classes = _get_session_classes()
    if name in session_classes:
        return session_classes[name]
    
    memory_classes = _get_memory_classes()
    if name in memory_classes:
        return memory_classes[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
__all__ = [
    "Storage",

    "SessionId",
    "UserId",

    "InteractionSession",
    "UserProfile",

    "InMemoryStorage",
    "JSONStorage",
    "Mem0Storage",
    "PostgresStorage",
    "RedisStorage",
    "SqliteStorage",

    "Memory", 
]