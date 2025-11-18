"""
Tests for store functionality in GraphV2.
"""

import pytest
from upsonic.graphv2.store import InMemoryStore
from datetime import datetime


class TestInMemoryStore:
    """Test InMemoryStore."""

    def test_in_memory_store(self):
        """Test InMemoryStore."""
        store = InMemoryStore()

        store.put(("user", "preferences"), "theme", "dark")

        item = store.get(("user", "preferences"), "theme")
        assert item is not None
        assert item.value == "dark"
        assert item.namespace == ("user", "preferences")
        assert item.key == "theme"

    def test_store_get(self):
        """Test store get operation."""
        store = InMemoryStore()

        store.put(("user", "preferences"), "theme", "dark")
        store.put(("user", "preferences"), "language", "en")

        item1 = store.get(("user", "preferences"), "theme")
        item2 = store.get(("user", "preferences"), "language")

        assert item1.value == "dark"
        assert item2.value == "en"

    def test_store_get_nonexistent(self):
        """Test getting nonexistent item."""
        store = InMemoryStore()

        item = store.get(("user", "preferences"), "nonexistent")
        assert item is None

    def test_store_set(self):
        """Test store set operation."""
        store = InMemoryStore()

        store.put(("user", "preferences"), "theme", "dark")
        store.put(("user", "preferences"), "theme", "light")  # Update

        item = store.get(("user", "preferences"), "theme")
        assert item.value == "light"

    def test_store_delete(self):
        """Test store delete operation."""
        store = InMemoryStore()

        store.put(("user", "preferences"), "theme", "dark")

        deleted = store.delete(("user", "preferences"), "theme")
        assert deleted is True

        item = store.get(("user", "preferences"), "theme")
        assert item is None

    def test_store_delete_nonexistent(self):
        """Test deleting nonexistent item."""
        store = InMemoryStore()

        deleted = store.delete(("user", "preferences"), "nonexistent")
        assert deleted is False

    def test_store_search(self):
        """Test store search operation."""
        store = InMemoryStore()

        store.put(("user", "preferences"), "theme", "dark")
        store.put(("user", "preferences"), "language", "en")
        store.put(("user", "preferences"), "timezone", "UTC")

        results = store.search(("user", "preferences"))
        assert len(results) == 3

    def test_store_search_limit(self):
        """Test store search with limit."""
        store = InMemoryStore()

        for i in range(10):
            store.put(("user", "preferences"), f"key{i}", f"value{i}")

        results = store.search(("user", "preferences"), limit=5)
        assert len(results) == 5

    def test_store_search_empty_namespace(self):
        """Test searching empty namespace."""
        store = InMemoryStore()

        results = store.search(("user", "preferences"))
        assert len(results) == 0

    def test_store_multiple_namespaces(self):
        """Test store with multiple namespaces."""
        store = InMemoryStore()

        store.put(("user", "preferences"), "theme", "dark")
        store.put(("user", "settings"), "notifications", True)
        store.put(("admin", "config"), "debug", False)

        item1 = store.get(("user", "preferences"), "theme")
        item2 = store.get(("user", "settings"), "notifications")
        item3 = store.get(("admin", "config"), "debug")

        assert item1.value == "dark"
        assert item2.value is True
        assert item3.value is False

    def test_store_timestamp(self):
        """Test that items have timestamps."""
        store = InMemoryStore()

        store.put(("user", "preferences"), "theme", "dark")

        item = store.get(("user", "preferences"), "theme")
        assert isinstance(item.timestamp, datetime)

    def test_store_update_timestamp(self):
        """Test that updates change timestamp."""
        store = InMemoryStore()

        store.put(("user", "preferences"), "theme", "dark")
        item1 = store.get(("user", "preferences"), "theme")
        timestamp1 = item1.timestamp

        import time

        time.sleep(0.01)  # Small delay to ensure different timestamp

        store.put(("user", "preferences"), "theme", "light")
        item2 = store.get(("user", "preferences"), "theme")
        timestamp2 = item2.timestamp

        assert timestamp2 > timestamp1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
