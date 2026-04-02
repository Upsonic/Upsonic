"""Base abstract class for user memory implementations."""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Literal, Optional, Type, Union

if TYPE_CHECKING:
    from pydantic import BaseModel
    from upsonic.storage.base import Storage
    from upsonic.models import Model

from upsonic.storage.memory.strategy.base import BaseMemoryStrategy


class BaseUserMemory(BaseMemoryStrategy, ABC):
    """Abstract base class for user memory (user profile/traits) implementations.
    
    User memory is tied to user_id and is the same across all session types
    (Agent, Team, Workflow). It stores user profile information extracted
    from interactions.
    
    **Save vs Load flag separation:**
    
    - ``enabled`` is a **save** flag – controls whether user profile analysis
      is performed and persisted to storage.
    - ``load_enabled`` is a **load** flag – controls whether the persisted
      user profile is injected into subsequent runs as a system prompt.
    
    By default ``load_enabled`` mirrors ``enabled`` for backward compatibility.
    """
    
    def __init__(
        self,
        storage: "Storage",
        user_id: str,
        enabled: bool = True,
        load_enabled: Optional[bool] = None,
        profile_schema: Optional[Type["BaseModel"]] = None,
        dynamic_profile: bool = False,
        update_mode: Literal['update', 'replace'] = 'update',
        model: Optional[Union["Model", str]] = None,
        debug: bool = False,
        debug_level: int = 1,
    ) -> None:
        """
        Initialize the user memory.
        
        Args:
            storage: Storage backend for persistence
            user_id: Unique identifier for the user
            enabled: Save flag – analyze and persist user profile
            load_enabled: Load flag – inject user profile into runs (defaults to ``enabled``)
            profile_schema: Pydantic model for user profile structure
            dynamic_profile: If True, generate schema dynamically from conversation
            update_mode: How to handle profile updates ('update' merges, 'replace' overwrites)
            model: Model for trait analysis (required if enabled)
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
        self.user_id: str = user_id
        self.load_enabled: bool = load_enabled if load_enabled is not None else enabled
        self.profile_schema: Optional[Type["BaseModel"]] = profile_schema
        self.dynamic_profile: bool = dynamic_profile
        self.update_mode: Literal['update', 'replace'] = update_mode
