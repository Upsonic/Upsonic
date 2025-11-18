"""
Tests for cache functionality in GraphV2.
"""

import pytest
import sqlite3
import time
from upsonic.graphv2.cache import (
    InMemoryCache,
    SqliteCache,
    CachePolicy,
    CacheEntry,
    default_cache_key,
)


class TestInMemoryCache:
    """Test InMemoryCache."""

    def test_in_memory_cache(self):
        """Test InMemoryCache."""
        cache = InMemoryCache()

        cache.put(("node1",), "key1", "value1")

        result = cache.get(("node1",), "key1")
        assert result == "value1"

    def test_cache_get(self):
        """Test cache get operation."""
        cache = InMemoryCache()

        cache.put(("node1",), "key1", "value1")
        cache.put(("node1",), "key2", "value2")

        result1 = cache.get(("node1",), "key1")
        result2 = cache.get(("node1",), "key2")

        assert result1 == "value1"
        assert result2 == "value2"

    def test_cache_get_nonexistent(self):
        """Test getting nonexistent cache entry."""
        cache = InMemoryCache()

        result = cache.get(("node1",), "nonexistent")
        assert result is None

    def test_cache_set(self):
        """Test cache set operation."""
        cache = InMemoryCache()

        cache.put(("node1",), "key1", "value1")
        cache.put(("node1",), "key1", "value2")  # Update

        result = cache.get(("node1",), "key1")
        assert result == "value2"

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache = InMemoryCache()

        cache.put(("node1",), "key1", "value1", ttl=0.1)  # 100ms TTL

        result1 = cache.get(("node1",), "key1")
        assert result1 == "value1"

        time.sleep(0.15)  # Wait for expiration

        result2 = cache.get(("node1",), "key1")
        assert result2 is None

    def test_cache_no_ttl(self):
        """Test cache without TTL (never expires)."""
        cache = InMemoryCache()

        cache.put(("node1",), "key1", "value1", ttl=None)

        time.sleep(0.1)

        result = cache.get(("node1",), "key1")
        assert result == "value1"

    def test_cache_clear_namespace(self):
        """Test clearing specific namespace."""
        cache = InMemoryCache()

        cache.put(("node1",), "key1", "value1")
        cache.put(("node2",), "key1", "value2")

        cache.clear(("node1",))

        result1 = cache.get(("node1",), "key1")
        result2 = cache.get(("node2",), "key1")

        assert result1 is None
        assert result2 == "value2"

    def test_cache_clear_all(self):
        """Test clearing all cache."""
        cache = InMemoryCache()

        cache.put(("node1",), "key1", "value1")
        cache.put(("node2",), "key1", "value2")

        cache.clear()

        result1 = cache.get(("node1",), "key1")
        result2 = cache.get(("node2",), "key1")

        assert result1 is None
        assert result2 is None

    def test_cache_multiple_namespaces(self):
        """Test cache with multiple namespaces."""
        cache = InMemoryCache()

        cache.put(("node1",), "key1", "value1")
        cache.put(("node2",), "key1", "value2")
        cache.put(("node3",), "key1", "value3")

        result1 = cache.get(("node1",), "key1")
        result2 = cache.get(("node2",), "key1")
        result3 = cache.get(("node3",), "key1")

        assert result1 == "value1"
        assert result2 == "value2"
        assert result3 == "value3"


class TestSqliteCache:
    """Test SqliteCache."""

    def test_sqlite_cache(self):
        """Test SqliteCache."""
        conn = sqlite3.connect(":memory:")
        cache = SqliteCache(conn)

        cache.put(("node1",), "key1", "value1")

        result = cache.get(("node1",), "key1")
        assert result == "value1"

        conn.close()

    def test_sqlite_cache_ttl(self):
        """Test SQLite cache with TTL."""
        conn = sqlite3.connect(":memory:")
        cache = SqliteCache(conn)

        cache.put(("node1",), "key1", "value1", ttl=0.1)

        result1 = cache.get(("node1",), "key1")
        assert result1 == "value1"

        time.sleep(0.15)

        result2 = cache.get(("node1",), "key1")
        assert result2 is None

        conn.close()

    def test_sqlite_cache_clear(self):
        """Test clearing SQLite cache."""
        conn = sqlite3.connect(":memory:")
        cache = SqliteCache(conn)

        cache.put(("node1",), "key1", "value1")
        cache.put(("node2",), "key1", "value2")

        cache.clear(("node1",))

        result1 = cache.get(("node1",), "key1")
        result2 = cache.get(("node2",), "key1")

        assert result1 is None
        assert result2 == "value2"

        conn.close()

    def test_sqlite_cache_clear_all(self):
        """Test clearing all SQLite cache."""
        conn = sqlite3.connect(":memory:")
        cache = SqliteCache(conn)

        cache.put(("node1",), "key1", "value1")
        cache.put(("node2",), "key1", "value2")

        cache.clear()

        result1 = cache.get(("node1",), "key1")
        result2 = cache.get(("node2",), "key1")

        assert result1 is None
        assert result2 is None

        conn.close()


class TestCachePolicy:
    """Test cache policy."""

    def test_cache_policy(self):
        """Test cache policy."""

        def custom_key_func(state):
            return str(state.get("count", 0))

        policy = CachePolicy(key_func=custom_key_func, ttl=60)

        assert policy.key_func == custom_key_func
        assert policy.ttl == 60

    def test_cache_policy_default_key_func(self):
        """Test cache policy with default key function."""
        policy = CachePolicy(ttl=60)

        assert policy.key_func is not None
        assert callable(policy.key_func)

    def test_cache_policy_no_ttl(self):
        """Test cache policy without TTL."""
        policy = CachePolicy(ttl=None)

        assert policy.ttl is None

    def test_default_cache_key(self):
        """Test default cache key generation."""
        state1 = {"count": 1, "message": "test"}
        state2 = {"count": 1, "message": "test"}
        state3 = {"count": 2, "message": "test"}

        key1 = default_cache_key(state1)
        key2 = default_cache_key(state2)
        key3 = default_cache_key(state3)

        assert key1 == key2  # Same state = same key
        assert key1 != key3  # Different state = different key
        assert len(key1) > 0


class TestCacheEntry:
    """Test CacheEntry."""

    def test_cache_entry(self):
        """Test CacheEntry."""
        entry = CacheEntry(value="test", timestamp=time.time(), ttl=60)

        assert entry.value == "test"
        assert entry.ttl == 60
        assert not entry.is_expired()

    def test_cache_entry_expired(self):
        """Test expired cache entry."""
        entry = CacheEntry(
            value="test",
            timestamp=time.time() - 100,  # 100 seconds ago
            ttl=60,  # Expires after 60 seconds
        )

        assert entry.is_expired()

    def test_cache_entry_no_ttl(self):
        """Test cache entry without TTL."""
        entry = CacheEntry(
            value="test",
            timestamp=time.time() - 1000,  # Very old
            ttl=None,  # Never expires
        )

        assert not entry.is_expired()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
