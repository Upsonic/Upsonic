"""
Store system for cross-thread persistent memory.

The store provides a way to save and retrieve information that persists
across different threads, unlike checkpointed state which is thread-specific.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from upsonic._utils import now_utc


@dataclass
class MemoryItem:
    """A single item stored in the memory store.
    
    Attributes:
        namespace: Tuple identifying the scope (e.g., (user_id, "preferences"))
        key: Unique key within the namespace
        value: The stored data
        timestamp: When this item was created/updated
    """
    
    namespace: Tuple[str, ...]
    key: str
    value: Any
    timestamp: datetime = field(default_factory=now_utc)


class BaseStore(ABC):
    """Abstract base class for cross-thread memory storage.
    
    Stores provide persistence across threads, unlike checkpointed state
    which is isolated per thread. This is useful for user preferences,
    learned behaviors, and other cross-session information.
    """
    
    @abstractmethod
    def put(self, namespace: Tuple[str, ...], key: str, value: Any) -> None:
        """Store a value in the given namespace.
        
        Args:
            namespace: Tuple identifying the scope (e.g., (user_id, "preferences"))
            key: Unique key within the namespace
            value: Data to store (should be JSON-serializable)
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get(self, namespace: Tuple[str, ...], key: str) -> Optional[MemoryItem]:
        """Retrieve a specific item from the store.
        
        Args:
            namespace: Tuple identifying the scope
            key: Key to retrieve
            
        Returns:
            The memory item if found, None otherwise
        """
        raise NotImplementedError()
    
    @abstractmethod
    def search(
        self,
        namespace: Tuple[str, ...],
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryItem]:
        """Search for items in a namespace.
        
        Args:
            namespace: Tuple identifying the scope to search
            query: Optional search query (semantic if embeddings configured)
            limit: Maximum number of results
            
        Returns:
            List of matching memory items
        """
        raise NotImplementedError()
    
    @abstractmethod
    def delete(self, namespace: Tuple[str, ...], key: str) -> bool:
        """Delete an item from the store.
        
        Args:
            namespace: Tuple identifying the scope
            key: Key to delete
            
        Returns:
            True if item was deleted, False if not found
        """
        raise NotImplementedError()


class InMemoryStore(BaseStore):
    """In-memory implementation of the store.
    
    Data is lost when the program ends. Useful for development and testing.
    """
    
    def __init__(self):
        """Initialize the in-memory store."""
        self._storage: Dict[Tuple[str, ...], Dict[str, MemoryItem]] = {}
    
    def put(self, namespace: Tuple[str, ...], key: str, value: Any) -> None:
        """Store a value in memory.
        
        Args:
            namespace: Tuple identifying the scope
            key: Unique key within the namespace
            value: Data to store
        """
        if namespace not in self._storage:
            self._storage[namespace] = {}
        
        self._storage[namespace][key] = MemoryItem(
            namespace=namespace,
            key=key,
            value=value,
            timestamp=now_utc()
        )
    
    def get(self, namespace: Tuple[str, ...], key: str) -> Optional[MemoryItem]:
        """Retrieve a specific item.
        
        Args:
            namespace: Tuple identifying the scope
            key: Key to retrieve
            
        Returns:
            The memory item if found, None otherwise
        """
        if namespace not in self._storage:
            return None
        
        return self._storage[namespace].get(key)
    
    def search(
        self,
        namespace: Tuple[str, ...],
        query: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryItem]:
        """Search for items in a namespace.
        
        Args:
            namespace: Tuple identifying the scope to search
            query: Optional search query (currently ignored, returns all)
            limit: Maximum number of results
            
        Returns:
            List of matching memory items
        """
        if namespace not in self._storage:
            return []
        
        items = list(self._storage[namespace].values())
        
        # Sort by timestamp (newest first)
        items.sort(key=lambda x: x.timestamp, reverse=True)
        
        return items[:limit]
    
    def delete(self, namespace: Tuple[str, ...], key: str) -> bool:
        """Delete an item from the store.
        
        Args:
            namespace: Tuple identifying the scope
            key: Key to delete
            
        Returns:
            True if item was deleted, False if not found
        """
        if namespace not in self._storage:
            return False
        
        if key in self._storage[namespace]:
            del self._storage[namespace][key]
            return True
        
        return False

