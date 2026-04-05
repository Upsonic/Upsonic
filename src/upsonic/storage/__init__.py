"""Storage module for Upsonic agent framework.

This module provides storage backends for persisting agent sessions
and user memory data.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import Storage, AsyncStorage
    from .json import JSONStorage
    from .in_memory import (
        InMemoryStorage,
        apply_pagination,
        apply_sorting,
        deep_copy_record,
        deep_copy_records,
        get_sort_value,
    )
    from .mem0 import AsyncMem0Storage, Mem0Storage
    from .postgres import (
        AsyncPostgresStorage,
        PostgresStorage,
        SESSION_TABLE_SCHEMA,
        USER_MEMORY_TABLE_SCHEMA,
        get_table_schema_definition,
    )
    from .redis import RedisStorage, SESSION_SCHEMA, USER_MEMORY_SCHEMA
    from .mongo import AsyncMongoStorage, MongoStorage
    from .sqlite import AsyncSqliteStorage, SqliteStorage
    from .memory import (
        Memory,
        SessionMemoryFactory,
        BaseMemoryStrategy,
        BaseSessionMemory,
        PreparedSessionInputs,
        AgentSessionMemory,
        BaseUserMemory,
        UserMemory,
    )


def _get_base_classes() -> dict[str, Any]:
    """Lazy import of base classes."""
    from .base import Storage, AsyncStorage
    
    return {
        "Storage": Storage,
        "AsyncStorage": AsyncStorage,
    }


def _get_sqlite_classes() -> dict[str, Any]:
    """Lazy import of SQLite storage classes."""
    from .sqlite import AsyncSqliteStorage, SqliteStorage
    
    return {
        "AsyncSqliteStorage": AsyncSqliteStorage,
        "SqliteStorage": SqliteStorage,
    }


def _get_postgres_classes() -> dict[str, Any]:
    """Lazy import of PostgreSQL storage classes and schemas."""
    from .postgres import (
        AsyncPostgresStorage,
        PostgresStorage,
        SESSION_TABLE_SCHEMA,
        USER_MEMORY_TABLE_SCHEMA,
        get_table_schema_definition,
    )
    
    return {
        "AsyncPostgresStorage": AsyncPostgresStorage,
        "PostgresStorage": PostgresStorage,
        "SESSION_TABLE_SCHEMA": SESSION_TABLE_SCHEMA,
        "USER_MEMORY_TABLE_SCHEMA": USER_MEMORY_TABLE_SCHEMA,
        "get_table_schema_definition": get_table_schema_definition,
    }


def _get_memory_classes() -> dict[str, Any]:
    """Lazy import of memory classes."""
    from .memory import (
        Memory,
        SessionMemoryFactory,
        BaseMemoryStrategy,
        BaseSessionMemory,
        PreparedSessionInputs,
        AgentSessionMemory,
        BaseUserMemory,
        UserMemory,
    )
    
    return {
        "Memory": Memory,
        "SessionMemoryFactory": SessionMemoryFactory,
        "BaseMemoryStrategy": BaseMemoryStrategy,
        "BaseSessionMemory": BaseSessionMemory,
        "PreparedSessionInputs": PreparedSessionInputs,
        "AgentSessionMemory": AgentSessionMemory,
        "BaseUserMemory": BaseUserMemory,
        "UserMemory": UserMemory,
    }


def _get_redis_classes() -> dict[str, Any]:
    """Lazy import of Redis storage classes and schemas."""
    from .redis import RedisStorage, SESSION_SCHEMA, USER_MEMORY_SCHEMA
    
    return {
        "RedisStorage": RedisStorage,
        "SESSION_SCHEMA": SESSION_SCHEMA,
        "USER_MEMORY_SCHEMA": USER_MEMORY_SCHEMA,
    }


def _get_json_classes() -> dict[str, Any]:
    """Lazy import of JSON storage classes."""
    from .json import JSONStorage
    
    return {
        "JSONStorage": JSONStorage,
    }


def _get_in_memory_classes() -> dict[str, Any]:
    """Lazy import of in-memory storage classes and utilities."""
    from .in_memory import (
        InMemoryStorage,
        apply_pagination,
        apply_sorting,
        deep_copy_record,
        deep_copy_records,
        get_sort_value,
    )
    
    return {
        "InMemoryStorage": InMemoryStorage,
        "apply_pagination": apply_pagination,
        "apply_sorting": apply_sorting,
        "deep_copy_record": deep_copy_record,
        "deep_copy_records": deep_copy_records,
        "get_sort_value": get_sort_value,
    }


def _get_mem0_classes() -> dict[str, Any]:
    """Lazy import of Mem0 storage classes."""
    from .mem0 import AsyncMem0Storage, Mem0Storage
    
    return {
        "Mem0Storage": Mem0Storage,
        "AsyncMem0Storage": AsyncMem0Storage,
    }


def _get_mongo_classes() -> dict[str, Any]:
    """Lazy import of MongoDB storage classes."""
    from .mongo import AsyncMongoStorage, MongoStorage
    
    return {
        "AsyncMongoStorage": AsyncMongoStorage,
        "MongoStorage": MongoStorage,
    }


def _safe_get(loader: Any) -> dict[str, Any]:
    """Call a lazy-loader function, returning an empty dict when its optional dependency is missing."""
    try:
        return loader()
    except (ImportError, ModuleNotFoundError):
        return {}


# Ordered list of lazy loaders.
# Core loaders (base, in_memory, json, memory) should never fail because they
# have no third-party dependencies.  Optional-backend loaders (sqlite,
# postgres, redis, mem0, mongo) are guarded by _safe_get so a missing
# dependency doesn't prevent importing unrelated classes from the same package.
_LOADERS: list[tuple[bool, Any]] = [
    (False, _get_base_classes),
    (False, _get_json_classes),
    (False, _get_in_memory_classes),
    (False, _get_memory_classes),
    (True, _get_sqlite_classes),
    (True, _get_postgres_classes),
    (True, _get_redis_classes),
    (True, _get_mem0_classes),
    (True, _get_mongo_classes),
]


def __getattr__(name: str) -> Any:
    """Lazy loading of storage modules and classes."""
    for is_optional, loader in _LOADERS:
        classes: dict[str, Any] = _safe_get(loader) if is_optional else loader()
        if name in classes:
            return classes[name]

    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )


__all__ = [
    # Base classes
    "Storage",
    "AsyncStorage",
    # Storage classes
    "InMemoryStorage",
    "JSONStorage",
    "Mem0Storage",
    "AsyncMem0Storage",
    "PostgresStorage",
    "AsyncPostgresStorage",
    "RedisStorage",
    "MongoStorage",
    "AsyncMongoStorage",
    "SqliteStorage",
    "AsyncSqliteStorage",
    # Memory classes
    "Memory",
    "SessionMemoryFactory",
    "BaseMemoryStrategy",
    "BaseSessionMemory",
    "PreparedSessionInputs",
    "AgentSessionMemory",
    "BaseUserMemory",
    "UserMemory",
    # PostgreSQL schemas
    "SESSION_TABLE_SCHEMA",
    "USER_MEMORY_TABLE_SCHEMA",
    "get_table_schema_definition",
    # Redis schemas
    "SESSION_SCHEMA",
    "USER_MEMORY_SCHEMA",
    # In-memory utilities
    "apply_pagination",
    "apply_sorting",
    "deep_copy_record",
    "deep_copy_records",
    "get_sort_value",
]
