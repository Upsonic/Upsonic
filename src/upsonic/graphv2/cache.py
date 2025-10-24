"""
Cache system for node-level caching.

This module provides caching infrastructure for expensive node operations,
supporting TTL-based expiration and custom cache key generation.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass
class CachePolicy:
    """Configuration for caching nodes.
    
    Attributes:
        key_func: Function to generate a cache key from the node's input
        ttl: Time to live for the cache entry in seconds (None = never expires)
    """
    
    key_func: Callable[[Any], str] = None
    """Function to generate a cache key from the node's input."""
    
    ttl: Optional[int] = None
    """Time to live for the cache entry in seconds. If None, the entry never expires."""
    
    def __post_init__(self):
        """Set default key function if not provided."""
        if self.key_func is None:
            self.key_func = default_cache_key


def default_cache_key(state: Dict[str, Any]) -> str:
    """Default cache key generator using pickle and hash.
    
    Args:
        state: The state dict to generate a key for
        
    Returns:
        Hex digest of the state hash
    """
    try:
        # Serialize state and hash it
        state_bytes = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(state_bytes).hexdigest()
    except Exception:
        # If pickling fails, use string representation
        return hashlib.sha256(str(state).encode()).hexdigest()


@dataclass
class CacheEntry:
    """A single cache entry.
    
    Attributes:
        value: The cached value
        timestamp: When this entry was created
        ttl: Time to live in seconds (None = never expires)
    """
    
    value: Any
    timestamp: float
    ttl: Optional[int]
    
    def is_expired(self) -> bool:
        """Check if this cache entry has expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.ttl is None:
            return False
        
        return (time.time() - self.timestamp) > self.ttl


class BaseCache(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    def get(self, namespace: Tuple[str, ...], key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            namespace: Cache namespace (e.g., node name)
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        raise NotImplementedError()
    
    @abstractmethod
    def put(self, namespace: Tuple[str, ...], key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value in the cache.
        
        Args:
            namespace: Cache namespace (e.g., node name)
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        raise NotImplementedError()
    
    @abstractmethod
    def clear(self, namespace: Optional[Tuple[str, ...]] = None) -> None:
        """Clear cache entries.
        
        Args:
            namespace: Optional namespace to clear. If None, clears all.
        """
        raise NotImplementedError()


class InMemoryCache(BaseCache):
    """In-memory cache implementation.
    
    Data is lost when the program ends. Useful for development and testing.
    """
    
    def __init__(self):
        """Initialize the in-memory cache."""
        self._storage: Dict[Tuple[str, ...], Dict[str, CacheEntry]] = {}
    
    def get(self, namespace: Tuple[str, ...], key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        if namespace not in self._storage:
            return None
        
        entry = self._storage[namespace].get(key)
        if entry is None:
            return None
        
        if entry.is_expired():
            # Remove expired entry
            del self._storage[namespace][key]
            return None
        
        return entry.value
    
    def put(self, namespace: Tuple[str, ...], key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value in the cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        if namespace not in self._storage:
            self._storage[namespace] = {}
        
        self._storage[namespace][key] = CacheEntry(
            value=value,
            timestamp=time.time(),
            ttl=ttl
        )
    
    def clear(self, namespace: Optional[Tuple[str, ...]] = None) -> None:
        """Clear cache entries.
        
        Args:
            namespace: Optional namespace to clear. If None, clears all.
        """
        if namespace is None:
            self._storage.clear()
        elif namespace in self._storage:
            del self._storage[namespace]


class SqliteCache(BaseCache):
    """SQLite-based cache implementation.
    
    Stores cache entries in a SQLite database for persistence across
    program runs. Suitable for local development and single-machine deployments.
    """
    
    def __init__(self, connection):
        """Initialize the SQLite cache.
        
        Args:
            connection: sqlite3 connection object
        """
        self.connection = connection
        self._create_tables()
    
    def _create_tables(self):
        """Create the cache table if it doesn't exist."""
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value BLOB NOT NULL,
                timestamp REAL NOT NULL,
                ttl INTEGER,
                PRIMARY KEY (namespace, key)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON cache_entries(timestamp)
        """)
        self.connection.commit()
    
    def get(self, namespace: Tuple[str, ...], key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        cursor = self.connection.cursor()
        namespace_str = "|".join(namespace)
        
        cursor.execute("""
            SELECT value, timestamp, ttl
            FROM cache_entries
            WHERE namespace = ? AND key = ?
        """, (namespace_str, key))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        value_blob, timestamp, ttl = row
        
        # Check expiration
        if ttl is not None and (time.time() - timestamp) > ttl:
            # Remove expired entry
            cursor.execute("""
                DELETE FROM cache_entries
                WHERE namespace = ? AND key = ?
            """, (namespace_str, key))
            self.connection.commit()
            return None
        
        return pickle.loads(value_blob)
    
    def put(self, namespace: Tuple[str, ...], key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value in the cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        cursor = self.connection.cursor()
        namespace_str = "|".join(namespace)
        value_blob = pickle.dumps(value)
        
        cursor.execute("""
            INSERT OR REPLACE INTO cache_entries 
            (namespace, key, value, timestamp, ttl)
            VALUES (?, ?, ?, ?, ?)
        """, (namespace_str, key, value_blob, time.time(), ttl))
        
        self.connection.commit()
    
    def clear(self, namespace: Optional[Tuple[str, ...]] = None) -> None:
        """Clear cache entries.
        
        Args:
            namespace: Optional namespace to clear. If None, clears all.
        """
        cursor = self.connection.cursor()
        
        if namespace is None:
            cursor.execute("DELETE FROM cache_entries")
        else:
            namespace_str = "|".join(namespace)
            cursor.execute("""
                DELETE FROM cache_entries
                WHERE namespace = ?
            """, (namespace_str,))
        
        self.connection.commit()

