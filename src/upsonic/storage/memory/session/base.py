"""Base abstract class for session memory implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:
    from upsonic.session.base import SessionType
    from upsonic.storage.base import Storage
    from upsonic.models import Model

from upsonic.storage.memory.strategy.base import BaseMemoryStrategy


@dataclass
class PreparedSessionInputs:
    """Structured output from session memory get operations.
    
    Contains all the prepared data needed for task execution:
    - message_history: Chat history messages for LLM context
    - context_injection: Session summary or other context (injected into user prompt)
    - metadata_injection: Session/agent metadata (injected into user prompt)
    - session: The raw session object (optional, for further processing)
    """
    message_history: List[Any] = field(default_factory=list)
    context_injection: str = ""
    metadata_injection: str = ""
    session: Optional[Any] = None


class BaseSessionMemory(BaseMemoryStrategy, ABC):
    """Abstract base class for session memory implementations.
    
    Each session type (Agent, Team, Workflow) has its own implementation
    that handles the specific session class and its data format.
    
    **Save vs Load flag separation:**
    
    - ``enabled`` / ``summary_enabled`` are **save** flags – they control
      whether data is persisted to storage (chat history, summary generation).
    - ``load_enabled`` / ``load_summary_enabled`` are **load** flags – they
      control whether persisted data is injected into subsequent runs.
    
    By default the load flags mirror the save flags so existing behaviour is
    preserved.  Users can override them independently, e.g. save everything
    but only inject summaries into runs to save tokens.
    
    Subclasses must define:
    - session_type: Class attribute identifying which SessionType this handles
    - ``aget`` / ``asave`` / ``get`` / ``save``: See
      :class:`~upsonic.storage.memory.strategy.base.BaseMemoryStrategy`
    - ``aload_resumable_run`` / ``aload_run``: async loaders for HITL / inspection
    - ``load_resumable_run`` / ``load_run``: sync-callable loaders (async storage bridged when needed)
    """
    
    # Subclasses MUST define their session type as a class attribute
    session_type: "SessionType"
    
    def __init__(
        self,
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
    ) -> None:
        """
        Initialize the session memory.
        
        Args:
            storage: Storage backend for persistence
            session_id: Unique identifier for the session
            enabled: Save flag – persist chat history to storage
            summary_enabled: Save flag – generate and persist session summaries
            load_enabled: Load flag – inject chat history into runs (defaults to ``enabled``)
            load_summary_enabled: Load flag – inject summary into runs (defaults to ``summary_enabled``)
            num_last_messages: Limit on number of message turns to keep
            feed_tool_call_results: Whether to include tool call results in history
            model: Model for summary generation (required if summary_enabled)
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
        self.session_id: str = session_id
        self.summary_enabled: bool = summary_enabled
        self.load_enabled: bool = load_enabled if load_enabled is not None else enabled
        self.load_summary_enabled: bool = load_summary_enabled if load_summary_enabled is not None else summary_enabled
        self.num_last_messages: Optional[int] = num_last_messages
        self.feed_tool_call_results: bool = feed_tool_call_results
    
    @abstractmethod
    async def aload_resumable_run(
        self,
        run_id: str,
        agent_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Load a resumable run from storage.
        
        Resumable runs include:
        - paused: External tool execution pause
        - error: Durable execution (error recovery)
        - cancelled: Cancel run resumption
        
        Args:
            run_id: The run ID to search for
            agent_id: Optional agent_id to search across sessions
            
        Returns:
            RunData if found and resumable, None otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    async def aload_run(
        self,
        run_id: str,
        agent_id: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Load a run from storage by run_id (regardless of status).
        
        Args:
            run_id: The run ID to search for
            agent_id: Optional agent_id to search across sessions
            
        Returns:
            RunData if found, None otherwise
        """
        raise NotImplementedError
    
    @abstractmethod
    def load_resumable_run(
        self,
        run_id: str,
        agent_id: Optional[str] = None,
    ) -> Optional[Any]:
        """Load a resumable run (sync API; supports sync and async storage)."""
        ...

    @abstractmethod
    def load_run(
        self,
        run_id: str,
        agent_id: Optional[str] = None,
    ) -> Optional[Any]:
        """Load a run by id (sync API; supports sync and async storage)."""
        ...

