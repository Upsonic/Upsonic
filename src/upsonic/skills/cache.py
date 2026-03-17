"""Lightweight in-memory TTL cache for the skills module."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class SkillCache:
    """Simple in-memory TTL cache for skill data.

    Args:
        ttl_seconds: Time-to-live for cache entries in seconds.
    """

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or ``None`` if missing/expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[key]
            return None
        return entry.value

    def set(self, key: str, value: Any) -> None:
        """Store a value with the configured TTL."""
        self._store[key] = _CacheEntry(
            value=value,
            expires_at=time.monotonic() + self.ttl_seconds,
        )

    def invalidate(self, key: Optional[str] = None) -> None:
        """Clear a specific key or all entries if *key* is ``None``."""
        if key is None:
            self._store.clear()
        else:
            self._store.pop(key, None)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None
