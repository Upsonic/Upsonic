from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .in_memory import InMemoryStorage
    from .json import JSONStorage
    from .mem0 import Mem0Storage
    from .postgres import PostgresStorage
    from .redis import RedisStorage
    from .sqlite import SqliteStorage
    from .mongo import MongoStorage

def _get_storage_provider_classes():
    """Lazy import of storage provider classes."""
    from .in_memory import InMemoryStorage
    from .json import JSONStorage
    from .mem0 import Mem0Storage
    from .postgres import PostgresStorage
    from .redis import RedisStorage
    from .sqlite import SqliteStorage
    from .mongo import MongoStorage
    
    return {
        'InMemoryStorage': InMemoryStorage,
        'JSONStorage': JSONStorage,
        'Mem0Storage': Mem0Storage,
        'PostgresStorage': PostgresStorage,
        'RedisStorage': RedisStorage,
        'SqliteStorage': SqliteStorage,
        'MongoStorage': MongoStorage,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    provider_classes = _get_storage_provider_classes()
    if name in provider_classes:
        return provider_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "InMemoryStorage",
    "JSONStorage",
    "Mem0Storage",
    "PostgresStorage",
    "RedisStorage",
    "SqliteStorage",
    "MongoStorage",
]