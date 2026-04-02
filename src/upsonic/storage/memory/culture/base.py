"""Base abstract class for culture memory implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from upsonic.storage.base import Storage
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.models import Model

from upsonic.storage.memory.strategy.base import BaseMemoryStrategy


class BaseCultureMemory(BaseMemoryStrategy, ABC):
    """Abstract base class for culture memory implementations.
    
    Culture memory manages cultural knowledge storage and retrieval.
    Unlike user memory (per-user), culture is shared across all agents
    and provides universal principles, guidelines, and best practices.
    
    Subclasses implement async and sync pairs: ``aget`` / ``get``, ``asave`` / ``save``,
    ``aget_all`` / ``get_all``, ``adelete`` / ``delete`` for both storage backend kinds.
    """
    
    def __init__(
        self,
        storage: "Storage",
        enabled: bool = True,
        model: Optional[Union["Model", str]] = None,
        debug: bool = False,
        debug_level: int = 1,
    ) -> None:
        """
        Initialize the culture memory.
        
        Args:
            storage: Storage backend for persistence
            enabled: Whether culture memory is enabled
            model: Model for culture extraction (required if enabled)
            debug: Enable debug logging
            debug_level: Debug verbosity level (1-3)
        """
        super().__init__(
            storage=storage,
            enabled=enabled,
            model=model,
            debug=debug,
            debug_level=debug_level,
        )
    
    @abstractmethod
    async def aget_all(
        self,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List["CulturalKnowledge"]:
        """
        Get all cultural knowledge entries from storage.
        
        Args:
            agent_id: Filter by agent ID
            team_id: Filter by team ID
            categories: Filter by categories
            limit: Maximum number of entries to return
            
        Returns:
            List of CulturalKnowledge instances
        """
        ...
    
    @abstractmethod
    async def adelete(
        self,
        culture_id: str,
    ) -> bool:
        """
        Delete cultural knowledge from storage.
        
        Args:
            culture_id: ID of the cultural knowledge to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        ...
    
    @abstractmethod
    def get_all(
        self,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List["CulturalKnowledge"]:
        """List entries (sync API; supports sync and async storage)."""
        ...
    
    @abstractmethod
    def delete(
        self,
        culture_id: str,
    ) -> bool:
        """Delete by id (sync API; supports sync and async storage)."""
        ...
