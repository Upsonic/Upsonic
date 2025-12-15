from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .database import (
        DatabaseBase,
        SqliteDatabase,
        PostgresDatabase,
        MongoDatabase,
        RedisDatabase,
        InMemoryDatabase,
        JSONDatabase,
        Mem0Database,
    )

def _get_database_classes():
    """Lazy import of database classes."""
    from .database import (
        DatabaseBase,
        SqliteDatabase,
        PostgresDatabase,
        MongoDatabase,
        RedisDatabase,
        InMemoryDatabase,
        JSONDatabase,
        Mem0Database,
    )
    
    return {
        'DatabaseBase': DatabaseBase,
        'SqliteDatabase': SqliteDatabase,
        'PostgresDatabase': PostgresDatabase,
        'MongoDatabase': MongoDatabase,
        'RedisDatabase': RedisDatabase,
        'InMemoryDatabase': InMemoryDatabase,
        'JSONDatabase': JSONDatabase,
        'Mem0Database': Mem0Database,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    database_classes = _get_database_classes()
    if name in database_classes:
        return database_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "DatabaseBase",
    "SqliteDatabase",
    "PostgresDatabase",
    "MongoDatabase",
    "RedisDatabase",
    "InMemoryDatabase",
    "JSONDatabase",
    "Mem0Database",
]
