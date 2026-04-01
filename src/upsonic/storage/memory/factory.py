"""Factory for creating session memory instances based on SessionType."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

if TYPE_CHECKING:
    from upsonic.session.base import SessionType
    from upsonic.storage.base import Storage
    from upsonic.storage.memory.session.base import BaseSessionMemory
    from upsonic.models import Model


class SessionMemoryFactory:
    """Factory for runtime session memory creation based on SessionType.
    
    This factory uses a registry pattern to map SessionType values to
    their corresponding session memory implementations. Implementations
    are registered at module import time.
    
    Usage:
        # Get session memory for a specific type
        memory = SessionMemoryFactory.create(
            session_type=SessionType.AGENT,
            storage=storage,
            session_id="abc123",
            ...
        )
    
    Extensibility:
        To add support for new session types:
        1. Create a new class extending BaseSessionMemory
        2. Register it: SessionMemoryFactory.register(SessionType.TEAM, TeamSessionMemory)
    """
    
    _registry: Dict["SessionType", Type["BaseSessionMemory"]] = {}
    _initialized: bool = False
    
    @classmethod
    def register(
        cls,
        session_type: "SessionType",
        memory_class: Type["BaseSessionMemory"],
    ) -> None:
        """
        Register a session memory class for a session type.
        
        Args:
            session_type: The SessionType this class handles
            memory_class: The class to instantiate for this type
        """
        cls._registry[session_type] = memory_class
    
    @classmethod
    def unregister(cls, session_type: "SessionType") -> bool:
        """
        Unregister a session memory class.
        
        Args:
            session_type: The SessionType to unregister
            
        Returns:
            True if was registered, False otherwise
        """
        if session_type in cls._registry:
            del cls._registry[session_type]
            return True
        return False
    
    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure default implementations are registered."""
        if cls._initialized:
            return
        
        cls._initialized = True
        
        # Register AgentSessionMemory for SessionType.AGENT
        try:
            from upsonic.session.base import SessionType
            from upsonic.storage.memory.session.agent import AgentSessionMemory
            cls.register(SessionType.AGENT, AgentSessionMemory)
        except ImportError:
            pass
    
    @classmethod
    def create(
        cls,
        session_type: "SessionType",
        storage: "Storage",
        session_id: str,
        enabled: bool = True,
        summary_enabled: bool = False,
        load_enabled: Optional[bool] = None,
        load_summary_enabled: Optional[bool] = None,
        num_last_messages: Optional[int] = None,
        feed_tool_call_results: bool = False,
        model: Optional[Union["Model", str]] = None,
        debug: bool = False,
        debug_level: int = 1,
    ) -> "BaseSessionMemory":
        """
        Create a session memory instance for the given session type.
        
        This is the main factory method called at runtime when
        Agent/Team/Workflow needs to interact with session storage.
        
        Args:
            session_type: The type of session (AGENT, TEAM, WORKFLOW)
            storage: Storage backend for persistence
            session_id: Unique identifier for the session
            enabled: Save flag – persist chat history
            summary_enabled: Save flag – generate/persist summaries
            load_enabled: Load flag – inject chat history into runs (defaults to ``enabled``)
            load_summary_enabled: Load flag – inject summary into runs (defaults to ``summary_enabled``)
            num_last_messages: Limit on message turns
            feed_tool_call_results: Include tool results in history
            model: Model for summary generation
            debug: Enable debug logging
            debug_level: Debug verbosity
            
        Returns:
            Appropriate BaseSessionMemory subclass instance
            
        Raises:
            ValueError: If no implementation registered for session_type
        """
        cls._ensure_initialized()
        
        if session_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"No session memory registered for {session_type}. "
                f"Available types: {available}"
            )
        
        memory_class = cls._registry[session_type]
        return memory_class(
            storage=storage,
            session_id=session_id,
            enabled=enabled,
            summary_enabled=summary_enabled,
            load_enabled=load_enabled,
            load_summary_enabled=load_summary_enabled,
            num_last_messages=num_last_messages,
            feed_tool_call_results=feed_tool_call_results,
            model=model,
            debug=debug,
            debug_level=debug_level,
        )
    
    @classmethod
    def get_supported_types(cls) -> List["SessionType"]:
        """Get list of supported session types."""
        cls._ensure_initialized()
        return list(cls._registry.keys())
    
    @classmethod
    def is_supported(cls, session_type: "SessionType") -> bool:
        """Check if a session type is supported."""
        cls._ensure_initialized()
        return session_type in cls._registry
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered implementations. Mainly for testing."""
        cls._registry.clear()
        cls._initialized = False

