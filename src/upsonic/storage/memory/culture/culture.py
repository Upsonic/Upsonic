"""Culture memory implementation for Upsonic agent framework."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from upsonic.storage.base import Storage
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.models import Model

from upsonic.storage.memory.culture.base import BaseCultureMemory
from upsonic.storage.memory.storage_dispatch import (
    is_async_storage_backend,
    run_awaitable_sync,
)


class CultureMemory(BaseCultureMemory):
    """Culture memory manager for storing and retrieving cultural knowledge.
    
    This implementation:
    - Loads cultural knowledge from storage
    - Formats cultural knowledge for system prompt injection
    - Saves cultural knowledge to storage
    - Supports filtering by agent_id, team_id, categories
    
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
            model: Model for culture operations
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

    async def _aquery_all_cultural(
        self,
        name: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Union[List["CulturalKnowledge"], Tuple[List[Dict[str, Any]], int]]:
        return await self.storage.aget_all_cultural_knowledge(
            name=name,
            limit=limit,
            page=page,
            sort_by=sort_by,
            sort_order=sort_order,
            agent_id=agent_id,
            team_id=team_id,
            deserialize=deserialize,
        )

    def _query_all_cultural(
        self,
        name: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        deserialize: bool = True,
    ) -> Union[List["CulturalKnowledge"], Tuple[List[Dict[str, Any]], int]]:
        if is_async_storage_backend(self.storage):
            return run_awaitable_sync(
                self._aquery_all_cultural(
                    name=name,
                    limit=limit,
                    page=page,
                    sort_by=sort_by,
                    sort_order=sort_order,
                    agent_id=agent_id,
                    team_id=team_id,
                    deserialize=deserialize,
                )
            )
        return self.storage.get_all_cultural_knowledge(
            name=name,
            limit=limit,
            page=page,
            sort_by=sort_by,
            sort_order=sort_order,
            agent_id=agent_id,
            team_id=team_id,
            deserialize=deserialize,
        )

    def get(
        self,
        culture_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Optional["CulturalKnowledge"]:
        """Load cultural knowledge (async or sync storage)."""
        from upsonic.utils.printing import info_log

        if is_async_storage_backend(self.storage):
            return run_awaitable_sync(self.aget(culture_id, agent_id, team_id))

        if not self.enabled:
            return None

        try:
            if culture_id:
                result = self.storage.get_cultural_knowledge(culture_id)
            else:
                results = self.storage.get_all_cultural_knowledge(
                    agent_id=agent_id,
                    team_id=team_id,
                    limit=1,
                    deserialize=True,
                )
                result = results[0] if isinstance(results, list) and results else None

            if result and self.debug:
                info_log(f"Loaded cultural knowledge: {result.name}", "CultureMemory")

            return result
        except Exception as e:
            if self.debug:
                info_log(f"Could not load cultural knowledge: {e}", "CultureMemory")
            return None

    def save(
        self,
        cultural_knowledge: "CulturalKnowledge",
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Optional["CulturalKnowledge"]:
        """Persist cultural knowledge (async or sync storage)."""
        from upsonic.utils.printing import info_log, warning_log

        if is_async_storage_backend(self.storage):
            return run_awaitable_sync(self.asave(cultural_knowledge, agent_id, team_id))

        if not self.enabled:
            return None

        try:
            if cultural_knowledge.id is None:
                cultural_knowledge.id = str(uuid.uuid4())
            if agent_id and cultural_knowledge.agent_id is None:
                cultural_knowledge.agent_id = agent_id
            if team_id and cultural_knowledge.team_id is None:
                cultural_knowledge.team_id = team_id
            cultural_knowledge.bump_updated_at()

            result = self.storage.upsert_cultural_knowledge(cultural_knowledge)

            if self.debug:
                info_log(
                    f"Saved cultural knowledge: {cultural_knowledge.name} (id={cultural_knowledge.id})",
                    "CultureMemory",
                )

            return result
        except Exception as e:
            warning_log(f"Failed to save cultural knowledge: {e}", "CultureMemory")
            return None

    def get_all(
        self,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List["CulturalKnowledge"]:
        """List cultural knowledge (async or sync storage)."""
        from upsonic.utils.printing import info_log

        if not self.enabled:
            return []

        try:
            raw = self._query_all_cultural(
                agent_id=agent_id,
                team_id=team_id,
                limit=limit,
                deserialize=True,
            )
            if not isinstance(raw, list):
                return []

            if categories:
                want = set(categories)
                raw = [
                    k
                    for k in raw
                    if getattr(k, "categories", None)
                    and want.intersection(set(k.categories or []))
                ]

            if raw and self.debug:
                info_log(f"Loaded {len(raw)} cultural knowledge entries", "CultureMemory")

            return raw
        except Exception as e:
            if self.debug:
                info_log(f"Could not load cultural knowledge list: {e}", "CultureMemory")
            return []

    def delete(self, culture_id: str) -> bool:
        """Delete cultural knowledge (async or sync storage)."""
        from upsonic.utils.printing import info_log, warning_log

        if is_async_storage_backend(self.storage):
            return run_awaitable_sync(self.adelete(culture_id))

        if not self.enabled:
            return False

        try:
            self.storage.delete_cultural_knowledge(culture_id)
            if self.debug:
                info_log(f"Deleted cultural knowledge: {culture_id}", "CultureMemory")
            return True
        except Exception as e:
            warning_log(f"Failed to delete cultural knowledge: {e}", "CultureMemory")
            return False

    async def aget(
        self,
        culture_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Optional["CulturalKnowledge"]:
        """Get cultural knowledge from storage."""
        from upsonic.utils.printing import info_log

        if not self.enabled:
            return None

        try:
            if is_async_storage_backend(self.storage):
                if culture_id:
                    result = await self.storage.aget_cultural_knowledge(culture_id)
                else:
                    results = await self.storage.aget_all_cultural_knowledge(
                        agent_id=agent_id,
                        team_id=team_id,
                        limit=1,
                    )
                    result = results[0] if results else None
                if result and self.debug:
                    info_log(f"Loaded cultural knowledge: {result.name}", "CultureMemory")
                return result

            return self.get(culture_id, agent_id, team_id)
        except Exception as e:
            if self.debug:
                info_log(f"Could not load cultural knowledge: {e}", "CultureMemory")
            return None
    
    async def aget_all(
        self,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
        name: Optional[str] = None,
        page: Optional[int] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        deserialize: bool = True,
    ) -> Union[List["CulturalKnowledge"], Tuple[List[Dict[str, Any]], int]]:
        """Get all cultural knowledge entries from storage.
        
        Args:
            name: Filter by name.
            limit: Maximum number of records to return.
            page: Page number (1-indexed).
            sort_by: Column to sort by.
            sort_order: Sort order ('asc' or 'desc').
            agent_id: Filter by agent ID.
            team_id: Filter by team ID.
            categories: When results are deserialized objects, filter by overlapping tags.
            deserialize: If True, return list of CulturalKnowledge objects.
                        If False, return tuple of (list of dicts, total count).
        
        Returns:
            List of CulturalKnowledge objects or tuple of (list of dicts, total count).
        """
        from upsonic.utils.printing import info_log
        
        if not self.enabled:
            return [] if deserialize else ([], 0)
        
        try:
            if is_async_storage_backend(self.storage):
                results = await self.storage.aget_all_cultural_knowledge(
                    name=name,
                    limit=limit,
                    page=page,
                    sort_by=sort_by,
                    sort_order=sort_order,
                    agent_id=agent_id,
                    team_id=team_id,
                    deserialize=deserialize,
                )
            else:
                results = self._query_all_cultural(
                    name=name,
                    limit=limit,
                    page=page,
                    sort_by=sort_by,
                    sort_order=sort_order,
                    agent_id=agent_id,
                    team_id=team_id,
                    deserialize=deserialize,
                )
            
            if results and self.debug:
                if isinstance(results, list):
                    info_log(
                        f"Loaded {len(results)} cultural knowledge entries",
                        "CultureMemory"
                    )
                elif isinstance(results, tuple):
                    info_log(
                        f"Loaded {len(results[0])} cultural knowledge entries (total: {results[1]})",
                        "CultureMemory"
                    )
            
            if not deserialize:
                # Return tuple of (list of dicts, total count)
                if isinstance(results, tuple):
                    return results
                # Fallback if storage returned list instead of tuple
                return (results if isinstance(results, list) else [], 0)
            
            # Return list of CulturalKnowledge objects
            out = results if isinstance(results, list) and results else []
            if categories and out:
                want = set(categories)
                out = [
                    k
                    for k in out
                    if getattr(k, "categories", None)
                    and want.intersection(set(k.categories or []))
                ]
            return out

        except Exception as e:
            if self.debug:
                info_log(f"Could not load cultural knowledge list: {e}", "CultureMemory")
            return [] if deserialize else ([], 0)
    
    async def asave(
        self,
        cultural_knowledge: "CulturalKnowledge",
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Optional["CulturalKnowledge"]:
        """Save cultural knowledge to storage."""
        from upsonic.utils.printing import info_log, warning_log
        
        if not self.enabled:
            return None

        if not is_async_storage_backend(self.storage):
            return self.save(cultural_knowledge, agent_id=agent_id, team_id=team_id)

        try:
            if cultural_knowledge.id is None:
                cultural_knowledge.id = str(uuid.uuid4())
            if agent_id and cultural_knowledge.agent_id is None:
                cultural_knowledge.agent_id = agent_id
            if team_id and cultural_knowledge.team_id is None:
                cultural_knowledge.team_id = team_id
            cultural_knowledge.bump_updated_at()

            result = await self.storage.aupsert_cultural_knowledge(cultural_knowledge)

            if self.debug:
                info_log(
                    f"Saved cultural knowledge: {cultural_knowledge.name} (id={cultural_knowledge.id})",
                    "CultureMemory",
                )

            return result

        except Exception as e:
            warning_log(f"Failed to save cultural knowledge: {e}", "CultureMemory")
            return None
    
    async def adelete(
        self,
        culture_id: str,
    ) -> bool:
        """Delete cultural knowledge from storage."""
        from upsonic.utils.printing import info_log, warning_log
        
        if not self.enabled:
            return False

        if not is_async_storage_backend(self.storage):
            return self.delete(culture_id)

        try:
            await self.storage.adelete_cultural_knowledge(culture_id)

            if self.debug:
                info_log(f"Deleted cultural knowledge: {culture_id}", "CultureMemory")

            return True

        except Exception as e:
            warning_log(f"Failed to delete cultural knowledge: {e}", "CultureMemory")
            return False
    
    def format_for_context(
        self,
        cultural_knowledge_list: List["CulturalKnowledge"],
        max_length: int = 2000,
    ) -> Optional[str]:
        """
        Format cultural knowledge for system prompt injection.
        
        Args:
            cultural_knowledge_list: List of CulturalKnowledge instances
            max_length: Maximum length of formatted output
            
        Returns:
            Formatted string for system prompt, or None if empty
        """
        if not cultural_knowledge_list:
            return None
        
        parts: List[str] = []
        current_length = 0
        
        for knowledge in cultural_knowledge_list:
            preview = knowledge.preview()
            
            entry_parts: List[str] = []
            if preview.get("name"):
                entry_parts.append(f"**{preview['name']}**")
            if preview.get("summary"):
                entry_parts.append(f"  Summary: {preview['summary']}")
            if preview.get("content"):
                entry_parts.append(f"  Content: {preview['content']}")
            if preview.get("categories"):
                entry_parts.append(f"  Categories: {', '.join(preview['categories'])}")
            
            entry = "\n".join(entry_parts)
            entry_length = len(entry)
            
            if current_length + entry_length > max_length:
                break
            
            parts.append(entry)
            current_length += entry_length + 2  # +2 for newlines
        
        if not parts:
            return None
        
        return "\n\n".join(parts)
