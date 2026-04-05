"""Memory orchestrator for Upsonic agent framework.

This module provides the Memory class which orchestrates session and user memory
operations with runtime session type selection.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Type, Union

if TYPE_CHECKING:
    from pydantic import BaseModel
    from upsonic.storage.base import Storage
    from upsonic.session.base import SessionType, Session
    from upsonic.models import Model
    from upsonic.storage.memory.session.base import BaseSessionMemory
    from upsonic.storage.memory.user.base import BaseUserMemory
    from upsonic.session.agent import RunData
    from upsonic.run.agent.output import AgentRunOutput

from upsonic.storage.memory.storage_dispatch import (
    is_async_storage_backend,
    run_awaitable_sync,
)


class Memory:
    """Orchestrator for session and user memory with runtime session type selection.
    
    This class serves as the central coordinator for memory operations:
    - Session memory: Chat history, summaries, session metadata
    - User memory: User profiles and traits extracted from conversations
    
    **Save vs Load flag separation:**
    
    Save flags control what is persisted to storage:
    - ``full_session_memory``: persist chat history
    - ``summary_memory``: generate and persist session summaries
    - ``user_analysis_memory``: analyze and persist user profiles
    
    Load flags control what is injected into subsequent runs:
    - ``load_full_session_memory``: inject chat history as message context
    - ``load_summary_memory``: inject session summary as context
    - ``load_user_analysis_memory``: inject user profile into system prompt
    
    Load flags default to their corresponding save flags for backward
    compatibility.  Users can override them independently, e.g. save
    everything but only inject summaries to conserve tokens.
    
    Usage:
        # Save everything, but only inject summary and user profile
        memory = Memory(
            storage=storage,
            session_id="session_001",
            user_id="user_123",
            full_session_memory=True,
            summary_memory=True,
            user_analysis_memory=True,
            load_full_session_memory=False,
            load_summary_memory=True,
            load_user_analysis_memory=True,
            model="openai/gpt-4o"
        )
    
    The Memory class caches session memory instances per SessionType for efficiency.
    """
    
    def __init__(
        self,
        storage: "Storage",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        load_full_session_memory: Optional[bool] = None,
        load_summary_memory: Optional[bool] = None,
        load_user_analysis_memory: Optional[bool] = None,
        user_profile_schema: Optional[Type["BaseModel"]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union["Model", str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update',
    ) -> None:
        """
        Initialize the Memory orchestrator.
        
        Args:
            storage: Storage backend for persistence
            session_id: Unique session identifier (auto-generated if None)
            user_id: Unique user identifier (auto-generated if None)
            full_session_memory: Save flag – persist chat history to storage
            summary_memory: Save flag – generate and persist session summaries
            user_analysis_memory: Save flag – analyze and persist user profiles
            load_full_session_memory: Load flag – inject chat history into runs
                (defaults to ``full_session_memory``)
            load_summary_memory: Load flag – inject session summary into runs
                (defaults to ``summary_memory``)
            load_user_analysis_memory: Load flag – inject user profile into runs
                (defaults to ``user_analysis_memory``)
            user_profile_schema: Pydantic model for user profile structure
            dynamic_user_profile: Generate profile schema dynamically
            num_last_messages: Limit on message turns to keep in history
            model: Model for summary/profile generation (required if enabled)
            debug: Enable debug logging
            debug_level: Debug verbosity level (1-3)
            feed_tool_call_results: Include tool call results in history
            user_memory_mode: How to update user profile ('update' or 'replace')
        """
        from upsonic.utils.printing import info_log
        
        self.storage: "Storage" = storage
        
        # Save flags – control what is persisted to storage
        self.full_session_memory_enabled: bool = full_session_memory
        self.summary_memory_enabled: bool = summary_memory
        self.user_analysis_memory_enabled: bool = user_analysis_memory
        
        # Load flags – control what is injected into runs (default to save flags)
        self.load_full_session_memory_enabled: bool = (
            load_full_session_memory if load_full_session_memory is not None else full_session_memory
        )
        self.load_summary_memory_enabled: bool = (
            load_summary_memory if load_summary_memory is not None else summary_memory
        )
        self.load_user_analysis_memory_enabled: bool = (
            load_user_analysis_memory if load_user_analysis_memory is not None else user_analysis_memory
        )
        
        self.num_last_messages: Optional[int] = num_last_messages
        self.model: Optional[Union["Model", str]] = model
        self.debug: bool = debug
        self.debug_level: int = debug_level if debug else 1
        self.feed_tool_call_results: bool = feed_tool_call_results
        
        # User memory configuration
        self.user_profile_schema: Optional[Type["BaseModel"]] = user_profile_schema
        self.dynamic_user_profile: bool = dynamic_user_profile
        self.user_memory_mode: Literal['update', 'replace'] = user_memory_mode
        
        # For backward compatibility - expose these attributes
        self.is_profile_dynamic: bool = dynamic_user_profile
        
        # Auto-generate session_id if not provided
        if session_id:
            self.session_id: str = session_id
        else:
            self.session_id = str(uuid.uuid4())
            if self.debug:
                info_log(f"Auto-generated session_id: {self.session_id}", "Memory")
        
        # Auto-generate user_id if not provided
        if user_id:
            self.user_id: str = user_id
        else:
            self.user_id = str(uuid.uuid4())
            if self.debug:
                info_log(f"Auto-generated user_id: {self.user_id}", "Memory")
        
        # Cache of session memory instances (created on demand per SessionType)
        self._session_memory_cache: Dict["SessionType", "BaseSessionMemory"] = {}
        
        # User memory – created when EITHER save or load is enabled
        self._user_memory: Optional["BaseUserMemory"] = None
        if user_analysis_memory or self.load_user_analysis_memory_enabled:
            self._user_memory = self._create_user_memory()
        
        if self.debug:
            info_log("Memory initialized with configuration:", "Memory")
            info_log(f"  - Full Session Memory (save): {self.full_session_memory_enabled}", "Memory")
            info_log(f"  - Summary Memory (save): {self.summary_memory_enabled}", "Memory")
            info_log(f"  - User Analysis Memory (save): {self.user_analysis_memory_enabled}", "Memory")
            info_log(f"  - Load Session History: {self.load_full_session_memory_enabled}", "Memory")
            info_log(f"  - Load Summary: {self.load_summary_memory_enabled}", "Memory")
            info_log(f"  - Load User Profile: {self.load_user_analysis_memory_enabled}", "Memory")
            info_log(f"  - Session ID: {self.session_id}", "Memory")
            info_log(f"  - User ID: {self.user_id}", "Memory")
            info_log(f"  - Max Messages: {self.num_last_messages}", "Memory")
            info_log(f"  - Feed Tool Results: {self.feed_tool_call_results}", "Memory")
            info_log(f"  - User Memory Mode: {self.user_memory_mode}", "Memory")
            info_log(f"  - Dynamic Profile: {self.dynamic_user_profile}", "Memory")
            info_log(f"  - Model: {self.model}", "Memory")
    
    def _create_user_memory(self) -> "BaseUserMemory":
        """Create user memory instance with separate save/load flags."""
        from upsonic.storage.memory.user.user import UserMemory
        
        return UserMemory(
            storage=self.storage,
            user_id=self.user_id,
            enabled=self.user_analysis_memory_enabled,
            load_enabled=self.load_user_analysis_memory_enabled,
            profile_schema=self.user_profile_schema,
            dynamic_profile=self.dynamic_user_profile,
            update_mode=self.user_memory_mode,
            model=self.model,
            debug=self.debug,
            debug_level=self.debug_level,
        )
    
    @property
    def user_memory(self) -> Optional["BaseUserMemory"]:
        """Get user memory instance."""
        return self._user_memory
    
    def get_session_memory(self, session_type: "SessionType") -> Optional["BaseSessionMemory"]:
        """
        Get or create session memory for the given session type.
        
        This is the RUNTIME selection - called when Agent/Team/Workflow
        invokes save or get operations.
        
        IMPORTANT: Session memory is ALWAYS created if storage is available.
        This is required for HITL (Human-in-the-Loop) checkpointing to work.
        Even when full_session_memory and summary_memory are disabled,
        incomplete runs (paused, error, cancelled) MUST be saved to storage
        to enable cross-process resumption.
        
        Args:
            session_type: The type of session (AGENT, TEAM, WORKFLOW)
            
        Returns:
            The appropriate session memory instance, or None if no storage available
        """
        # CRITICAL: Always create session memory if storage is available.
        # HITL checkpoints (paused, error, cancelled runs) must ALWAYS be saved
        # to enable cross-process resumption, regardless of memory settings.
        if self.storage is None:
            return None
        
        # Return cached instance if exists
        if session_type in self._session_memory_cache:
            return self._session_memory_cache[session_type]
        
        # Create new instance using factory
        from upsonic.storage.memory.factory import SessionMemoryFactory
        
        session_memory = SessionMemoryFactory.create(
            session_type=session_type,
            storage=self.storage,
            session_id=self.session_id,
            enabled=self.full_session_memory_enabled,
            summary_enabled=self.summary_memory_enabled,
            load_enabled=self.load_full_session_memory_enabled,
            load_summary_enabled=self.load_summary_memory_enabled,
            num_last_messages=self.num_last_messages,
            feed_tool_call_results=self.feed_tool_call_results,
            model=self.model,
            debug=self.debug,
            debug_level=self.debug_level,
        )
        
        # Cache for future use
        self._session_memory_cache[session_type] = session_memory
        return session_memory
    
    async def prepare_inputs_for_task(
        self,
        session_type: Optional["SessionType"] = None,
        agent_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Gather all relevant memory data before task execution.
        
        This method prepares:
        - Message history (from session memory)
        - Context injection (session summary)
        - System prompt injection (user profile)
        - Metadata injection (session + agent metadata)
        
        Args:
            session_type: The session type (defaults to AGENT)
            agent_metadata: Optional metadata from the caller to inject
            
        Returns:
            Dictionary with prepared memory inputs
        """
        from upsonic.session.base import SessionType
        from upsonic.utils.printing import info_log, debug_log_level2
        
        if session_type is None:
            session_type = SessionType.AGENT
        
        if self.debug:
            info_log("Preparing memory inputs for task...", "Memory")
        
        prepared_data: Dict[str, Any] = {
            "message_history": [],
            "context_injection": "",
            "system_prompt_injection": "",
            "metadata_injection": "",
        }
        
        # Get session memory inputs
        session_memory = self.get_session_memory(session_type)
        if session_memory:
            try:
                session_inputs = await session_memory.aget()
                prepared_data["message_history"] = session_inputs.message_history
                prepared_data["context_injection"] = session_inputs.context_injection
                prepared_data["metadata_injection"] = session_inputs.metadata_injection
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Failed to get session memory inputs: {e}", "Memory")
        
        # Get user memory (system prompt injection)
        if self._user_memory:
            try:
                profile_str = await self._user_memory.aget()
                if profile_str:
                    prepared_data["system_prompt_injection"] = profile_str
            except Exception as e:
                from upsonic.utils.printing import warning_log
                warning_log(f"Failed to get user memory: {e}", "Memory")
        
        # Merge agent metadata
        if agent_metadata:
            agent_meta_parts = []
            for key, value in agent_metadata.items():
                agent_meta_parts.append(f"  {key}: {value}")
            if agent_meta_parts:
                agent_meta_str = "<AgentMetadata>\n" + "\n".join(agent_meta_parts) + "\n</AgentMetadata>"
                if prepared_data["metadata_injection"]:
                    prepared_data["metadata_injection"] = agent_meta_str + "\n\n" + prepared_data["metadata_injection"]
                else:
                    prepared_data["metadata_injection"] = agent_meta_str
                if self.debug:
                    info_log(f"Added agent metadata with {len(agent_metadata)} keys", "Memory")
        
        if self.debug:
            info_log(
                f"Prepared memory inputs: {len(prepared_data['message_history'])} messages, "
                f"summary={bool(prepared_data['context_injection'])}, "
                f"profile={bool(prepared_data['system_prompt_injection'])}, "
                f"metadata={bool(prepared_data['metadata_injection'])}",
                "Memory"
            )
            
            if self.debug_level >= 2:
                message_preview = []
                for msg in prepared_data['message_history'][-3:]:
                    if hasattr(msg, 'parts'):
                        msg_str = str([str(p)[:100] for p in msg.parts[:2]])[:200]
                        message_preview.append(msg_str)
                
                debug_log_level2(
                    "Memory inputs prepared",
                    "Memory",
                    debug=self.debug,
                    debug_level=self.debug_level,
                    message_count=len(prepared_data['message_history']),
                    message_preview=message_preview,
                    has_summary=bool(prepared_data['context_injection']),
                    has_profile=bool(prepared_data['system_prompt_injection']),
                    session_id=self.session_id,
                    user_id=self.user_id,
                )
        
        return prepared_data
    
    async def run_memory_agents_async(
        self,
        output: "AgentRunOutput",
        session_type: Optional["SessionType"] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """Run memory sub-agents (user analysis, summary) WITHOUT persisting.

        Sub-agents make LLM calls that contribute ``model_execution_time``
        to the parent task's usage via ``incr()``.  Call this BEFORE
        ``task.task_end()`` so the timer keeps running while sub-agents work,
        then call ``persist_session_async`` AFTER ``task.task_end()`` so
        storage receives finalized metrics.

        Args:
            output: The run output.
            session_type: The session type (defaults to AGENT).
            agent_id: Optional agent identifier.
        """
        from upsonic.session.base import SessionType
        from upsonic.run.base import RunStatus
        from upsonic.utils.printing import warning_log

        if output is None:
            return

        if session_type is None:
            session_type = SessionType.AGENT

        is_completed: bool = output.status == RunStatus.completed

        if self._user_memory and is_completed:
            try:
                agent_id = output.agent_id
                await self._user_memory.asave(output, agent_id=agent_id)
            except Exception as e:
                if self.debug:
                    warning_log(f"Failed to analyze/update user memory: {e}", "Memory")

        session_memory = self.get_session_memory(session_type)
        if session_memory and hasattr(session_memory, "arun_agents"):
            try:
                await session_memory.arun_agents(output, is_completed)
            except Exception as e:
                if self.debug:
                    warning_log(f"Failed to run memory agents: {e}", "Memory")

    async def persist_session_async(
        self,
        output: "AgentRunOutput",
        session_type: Optional["SessionType"] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """Persist the prepared session to storage.

        Call AFTER ``task.task_end()`` so that ``output.usage.duration`` is
        finalized before writing.

        Args:
            output: The run output (carries final usage metrics).
            session_type: The session type (defaults to AGENT).
            agent_id: Optional agent identifier.
        """
        from upsonic.session.base import SessionType
        from upsonic.run.base import RunStatus
        from upsonic.utils.printing import warning_log, info_log

        if output is None:
            return

        if session_type is None:
            session_type = SessionType.AGENT

        is_completed: bool = output.status == RunStatus.completed

        session_memory = self.get_session_memory(session_type)
        if session_memory and hasattr(session_memory, "apersist"):
            try:
                await session_memory.apersist(output, is_completed)
            except Exception as e:
                if self.debug:
                    warning_log(f"Failed to persist session: {e}", "Memory")

        if self.debug:
            status_str = "completed" if is_completed else output.status.value
            info_log(f"Session saved for run {output.run_id} (status: {status_str})", "Memory")

    async def save_session_async(
        self,
        output: "AgentRunOutput",
        session_type: Optional["SessionType"] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """
        Save session to storage (backward-compatible single-call entry point).
        
        This is the centralized method for ALL session saving operations:
        
        For INCOMPLETE runs (paused, error, cancelled):
        - Saves checkpoint state for HITL resumption
        - Does NOT process memory features (summary, user profile)
        
        For COMPLETED runs:
        - Saves the completed run output
        - Processes memory features if enabled:
          - Generates session summary (if summary_memory enabled)
          - Analyzes user profile (if user_analysis_memory enabled)

        Pipeline steps that need to insert ``task.task_end()`` between
        sub-agent execution and storage persistence should use
        ``run_memory_agents_async`` + ``persist_session_async`` instead.

        Args:
            output: The run output (AgentRunOutput, TeamRunOutput, etc.)
            session_type: The session type (defaults to AGENT)
            agent_id: Optional agent identifier
        """
        await self.run_memory_agents_async(output, session_type, agent_id)
        await self.persist_session_async(output, session_type, agent_id)
    
    
    async def get_session_async(self) -> Optional["Session"]:
        """Get the current session from storage."""
        from upsonic.session.base import SessionType

        if is_async_storage_backend(self.storage):
            return await self.storage.aget_session(
                session_id=self.session_id,
                session_type=SessionType.AGENT,
                deserialize=True
            )
        return self.storage.get_session(
            session_id=self.session_id,
            session_type=SessionType.AGENT,
            deserialize=True
        )
    
    def get_session(self) -> Optional["Session"]:
        """Get the current session from storage (sync version)."""
        return run_awaitable_sync(self.get_session_async())
    
    async def get_messages_async(self) -> list:
        """Get messages from the current session."""
        session = await self.get_session_async()
        if session and hasattr(session, 'messages'):
            return session.messages or []
        return []
    
    def get_messages(self) -> list:
        """Get messages from the current session (sync version)."""
        session = self.get_session()
        if session and hasattr(session, 'messages'):
            return session.messages or []
        return []
    
    async def set_metadata_async(self, metadata: Dict[str, Any]) -> None:
        """Set metadata on the current session."""
        session = await self.get_session_async()
        if session:
            if not session.metadata:
                session.metadata = {}
            session.metadata.update(metadata)
            if is_async_storage_backend(self.storage):
                await self.storage.aupsert_session(session, deserialize=True)
            else:
                self.storage.upsert_session(session, deserialize=True)
    
    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set metadata on the current session (sync version)."""
        run_awaitable_sync(self.set_metadata_async(metadata))
    
    async def get_metadata_async(self) -> Optional[Dict[str, Any]]:
        """Get metadata from the current session."""
        session = await self.get_session_async()
        if session and hasattr(session, 'metadata'):
            return session.metadata
        return None
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata from the current session (sync version)."""
        session = self.get_session()
        if session and hasattr(session, 'metadata'):
            return session.metadata
        return None
    
    async def list_sessions_async(self, user_id: Optional[str] = None) -> list:
        """List sessions, optionally filtered by user_id."""
        from upsonic.session.base import SessionType

        if is_async_storage_backend(self.storage):
            sessions = await self.storage.aget_sessions(
                user_id=user_id or self.user_id,
                session_type=SessionType.AGENT,
                deserialize=True
            )
        else:
            sessions = self.storage.get_sessions(
                user_id=user_id or self.user_id,
                session_type=SessionType.AGENT,
                deserialize=True
            )
        if isinstance(sessions, list):
            return sessions
        return []
    
    def list_sessions(self, user_id: Optional[str] = None) -> list:
        """List sessions, optionally filtered by user_id (sync version)."""
        return run_awaitable_sync(self.list_sessions_async(user_id))
    
    async def find_session_async(self, session_id: Optional[str] = None) -> Optional["Session"]:
        """Find a specific session by session_id."""
        from upsonic.session.base import SessionType

        if is_async_storage_backend(self.storage):
            return await self.storage.aget_session(
                session_id=session_id or self.session_id,
                session_type=SessionType.AGENT,
                deserialize=True
            )
        return self.storage.get_session(
            session_id=session_id or self.session_id,
            session_type=SessionType.AGENT,
            deserialize=True
        )
    
    def find_session(self, session_id: Optional[str] = None) -> Optional["Session"]:
        """Find a specific session by session_id (sync version)."""
        return run_awaitable_sync(self.find_session_async(session_id))
    
    async def delete_session_async(self, session_id: Optional[str] = None) -> bool:
        """Delete the current or specified session."""
        if is_async_storage_backend(self.storage):
            return await self.storage.adelete_session(session_id or self.session_id)
        return self.storage.delete_session(session_id or self.session_id)
    
    def delete_session(self, session_id: Optional[str] = None) -> bool:
        """Delete the current or specified session (sync version)."""
        return run_awaitable_sync(self.delete_session_async(session_id))
    
    async def load_resumable_run_async(
        self,
        run_id: str,
        session_type: Optional["SessionType"] = None,
        agent_id: Optional[str] = None,
    ) -> Optional["RunData"]:
        """
        Load a resumable run from storage by run_id.
        
        Resumable runs include:
        - paused: External tool execution pause
        - error: Durable execution (error recovery)
        - cancelled: Cancel run resumption
        
        Args:
            run_id: The run ID to search for
            session_type: The session type (defaults to AGENT)
            agent_id: Optional agent_id to search across sessions
            
        Returns:
            RunData if found and resumable, None otherwise
        """
        from upsonic.session.base import SessionType
        
        if session_type is None:
            session_type = SessionType.AGENT
        
        session_memory = self.get_session_memory(session_type)
        if session_memory:
            return await session_memory.aload_resumable_run(run_id, agent_id)
        return None
    
    def load_resumable_run(
        self,
        run_id: str,
        session_type: Optional["SessionType"] = None,
        agent_id: Optional[str] = None,
    ) -> Optional["RunData"]:
        """
        Load a resumable run from storage by run_id (sync version).
        """
        from upsonic.session.base import SessionType
        
        if session_type is None:
            session_type = SessionType.AGENT
        
        session_memory = self.get_session_memory(session_type)
        if session_memory:
            return session_memory.load_resumable_run(run_id, agent_id)
        return None
    
    async def load_run_async(
        self,
        run_id: str,
        session_type: Optional["SessionType"] = None,
        agent_id: Optional[str] = None,
    ) -> Optional["RunData"]:
        """
        Load a run from storage by run_id (regardless of status).
        
        Unlike load_resumable_run_async, this returns any run regardless of status.
        Used for checking if a run is completed before attempting to continue.
        
        Args:
            run_id: The run ID to search for
            session_type: The session type (defaults to AGENT)
            agent_id: Optional agent_id to search across sessions
            
        Returns:
            RunData if found, None otherwise
        """
        from upsonic.session.base import SessionType
        
        if session_type is None:
            session_type = SessionType.AGENT
        
        session_memory = self.get_session_memory(session_type)
        if session_memory:
            return await session_memory.aload_run(run_id, agent_id)
        return None
    
    def load_run(
        self,
        run_id: str,
        session_type: Optional["SessionType"] = None,
        agent_id: Optional[str] = None,
    ) -> Optional["RunData"]:
        """
        Load a run from storage by run_id (regardless of status) - sync version.
        """
        from upsonic.session.base import SessionType
        
        if session_type is None:
            session_type = SessionType.AGENT
        
        session_memory = self.get_session_memory(session_type)
        if session_memory:
            return session_memory.load_run(run_id, agent_id)
        return None
