"""Agent session memory implementation for Upsonic agent framework."""
from __future__ import annotations

import copy
import json
import time
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from upsonic.session.agent import AgentSession, RunData
    from upsonic.run.agent.output import AgentRunOutput
    from upsonic.session.base import SessionType

from upsonic.storage.memory.session.base import BaseSessionMemory, PreparedSessionInputs
from upsonic.storage.memory.storage_dispatch import (
    is_async_storage_backend,
    run_awaitable_sync,
)


class AgentSessionMemory(BaseSessionMemory):
    """Session memory implementation for AgentSession.
    
    Handles all session-related operations for Agent class:
    - Loading and preparing session data for task execution
    - Saving completed and incomplete runs
    - Managing message history with limiting/filtering
    - Generating session summaries
    - Loading resumable runs for HITL
    
    Supports both sync (Storage) and async (AsyncStorage) storage backends.
    """
    
    @property
    def session_type(self) -> "SessionType":
        from upsonic.session.base import SessionType
        return SessionType.AGENT
    
    def _build_prepared_inputs_from_session(
        self,
        session: Optional[Any],
    ) -> PreparedSessionInputs:
        """Build :class:`PreparedSessionInputs` from a loaded session.

        Uses **load** flags (``load_enabled``, ``load_summary_enabled``) to
        decide which data to include in the prepared inputs.  Save flags are
        irrelevant here – they only govern persistence.
        """
        from upsonic.utils.printing import info_log, warning_log

        result = PreparedSessionInputs()
        if not session:
            if self.debug:
                info_log("No session found in storage", "AgentSessionMemory")
            return result

        result.session = session

        if self.load_summary_enabled and session.summary:
            result.context_injection = f"<SessionSummary>\n{session.summary}\n</SessionSummary>"
            if self.debug:
                info_log(f"Loaded session summary ({len(session.summary)} chars)", "AgentSessionMemory")

        if self.load_enabled:
            try:
                if session.messages:
                    messages = self._limit_message_history(session.messages)
                    if not self.feed_tool_call_results:
                        messages = self._filter_tool_messages(messages)
                    result.message_history = messages
                    if self.debug:
                        info_log(f"Loaded {len(messages)} messages from session", "AgentSessionMemory")
            except Exception as e:
                warning_log(f"Could not load messages from session: {e}", "AgentSessionMemory")

        if session.metadata:
            metadata_parts = []
            for key, value in session.metadata.items():
                metadata_parts.append(f"  {key}: {value}")
            if metadata_parts:
                result.metadata_injection = (
                    "<SessionMetadata>\n" + "\n".join(metadata_parts) + "\n</SessionMetadata>"
                )
                if self.debug:
                    info_log(f"Loaded session metadata with {len(session.metadata)} keys", "AgentSessionMemory")

        return result

    async def _aget_prepared_async_storage(self) -> PreparedSessionInputs:
        """Load session via async storage and build prepared inputs (for sync ``get`` bridge)."""
        from upsonic.session.base import SessionType

        if not self.load_enabled and not self.load_summary_enabled:
            return PreparedSessionInputs()
        session = await self.storage.aget_session(
            session_id=self.session_id,
            session_type=SessionType.AGENT,
            deserialize=True,
        )
        return self._build_prepared_inputs_from_session(session)

    async def aget(self) -> PreparedSessionInputs:
        """Get AgentSession and prepare inputs for Agent class (async).

        Respects **load** flags to decide what to include.
        """
        from upsonic.session.base import SessionType

        if not self.load_enabled and not self.load_summary_enabled:
            return PreparedSessionInputs()

        if is_async_storage_backend(self.storage):
            session = await self.storage.aget_session(
                session_id=self.session_id,
                session_type=SessionType.AGENT,
                deserialize=True,
            )
            return self._build_prepared_inputs_from_session(session)

        return self.get()

    def get(self) -> PreparedSessionInputs:
        """Get AgentSession and prepare inputs for Agent class (sync).

        Respects **load** flags to decide what to include.
        """
        from upsonic.session.base import SessionType

        if is_async_storage_backend(self.storage):
            return run_awaitable_sync(self._aget_prepared_async_storage())

        if not self.load_enabled and not self.load_summary_enabled:
            return PreparedSessionInputs()

        session = self.storage.get_session(
            session_id=self.session_id,
            session_type=SessionType.AGENT,
            deserialize=True,
        )
        return self._build_prepared_inputs_from_session(session)
    
    async def _load_or_create_session(
        self,
        output: "AgentRunOutput",
    ) -> "AgentSession":
        """Load existing session from storage or create a new one.

        Args:
            output: The run output used to populate a new session.

        Returns:
            The loaded or newly created AgentSession.
        """
        from upsonic.session.agent import AgentSession
        from upsonic.session.base import SessionType
        from upsonic.utils.printing import info_log

        if is_async_storage_backend(self.storage):
            session = await self.storage.aget_session(
                session_id=self.session_id,
                session_type=SessionType.AGENT,
                deserialize=True,
            )
        else:
            session = self.storage.get_session(
                session_id=self.session_id,
                session_type=SessionType.AGENT,
                deserialize=True,
            )

        if not session:
            session = AgentSession(
                session_id=self.session_id,
                agent_id=output.agent_id,
                user_id=output.user_id,
                created_at=int(time.time()),
                metadata={},
                runs={},
            )
            if self.debug:
                info_log(f"Created new session: {self.session_id}", "AgentSessionMemory")

        return session

    async def arun_agents(self, output: "AgentRunOutput", is_completed: bool) -> None:
        """Run memory sub-agents (summary, etc.) WITHOUT persisting to storage.

        This method loads/creates the session, runs sub-agents that contribute
        ``model_execution_time`` via ``usage.incr()``, and caches the prepared
        session on ``self._pending_session`` so that a subsequent ``apersist``
        call can write it with finalized metrics.

        For incomplete runs the session is prepared but no sub-agents run
        (summary generation only happens on completed runs).

        Args:
            output: The run output.
            is_completed: Whether the run completed successfully.
        """
        from upsonic.utils.printing import warning_log

        if output is None:
            return

        try:
            session = await self._load_or_create_session(output)

            run_previously_saved: bool = bool(
                session and session.runs and output.run_id in session.runs
            )

            if is_completed and not self.enabled and not self.summary_enabled and not run_previously_saved:
                self._pending_session = None
                return

            if is_completed:
                if run_previously_saved and not self.enabled and not self.summary_enabled:
                    session.upsert_run(output)
                else:
                    await self._save_completed_run(session, output)
            else:
                await self._save_incomplete_run(session, output)

            self._pending_session = session
        except Exception as e:
            self._pending_session = None
            if self.debug:
                import traceback
                error_trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                warning_log(
                    f"Failed to run memory agents: {e}\n{error_trace[-500:]}",
                    "AgentSessionMemory",
                )

    async def apersist(self, output: "AgentRunOutput", is_completed: bool) -> None:
        """Persist the prepared session to storage.

        Must be called AFTER ``arun_agents`` (and after ``task.task_end()``)
        so that ``output.usage`` carries the finalized ``duration``.

        Args:
            output: The run output (carries final usage metrics).
            is_completed: Whether the run completed successfully.
        """
        from upsonic.utils.printing import info_log, warning_log

        if output is None:
            return

        session: Optional["AgentSession"] = getattr(self, "_pending_session", None)
        if session is None:
            return

        try:
            if output.usage:
                session.update_usage_from_run(output)

            session.updated_at = int(time.time())

            if is_async_storage_backend(self.storage):
                await self.storage.aupsert_session(session, deserialize=True)
            else:
                self.storage.upsert_session(session, deserialize=True)

            if self.debug:
                status_str = "completed" if is_completed else "incomplete"
                info_log(
                    f"Session saved for run {output.run_id} ({status_str})",
                    "AgentSessionMemory",
                )
        except Exception as e:
            if self.debug:
                import traceback
                error_trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                warning_log(
                    f"Failed to persist session: {e}\n{error_trace[-500:]}",
                    "AgentSessionMemory",
                )
        finally:
            self._pending_session = None

    async def asave(self, output: "AgentRunOutput", is_completed: bool) -> None:
        """Save AgentSession to storage (async).
        
        IMPORTANT: HITL (Human-in-the-Loop) checkpoints for incomplete runs 
        (paused, error, cancelled) are ALWAYS saved regardless of the 'enabled' setting.
        This is required for cross-process resumption to work.
        
        For completed runs:
        - If the run was previously saved (e.g., transitioning from paused to completed),
          we ALWAYS update the run status to reflect completion.
        - Full history is only saved when 'enabled' is True.

        This is the backward-compatible single-call entry point. Pipeline steps
        that need to insert ``task.task_end()`` between sub-agent execution and
        storage persistence should use ``arun_agents`` + ``apersist`` instead.
        """
        await self.arun_agents(output, is_completed)
        await self.apersist(output, is_completed)
    
    def save(self, output: "AgentRunOutput", is_completed: bool) -> None:
        """Save AgentSession to storage (sync).
        
        IMPORTANT: HITL (Human-in-the-Loop) checkpoints for incomplete runs 
        (paused, error, cancelled) are ALWAYS saved regardless of the 'enabled' setting.
        This is required for cross-process resumption to work.
        
        For completed runs:
        - If the run was previously saved (e.g., transitioning from paused to completed),
          we ALWAYS update the run status to reflect completion.
        - Full history is only saved when 'enabled' is True.
        """
        from upsonic.session.agent import AgentSession
        from upsonic.session.base import SessionType
        from upsonic.utils.printing import info_log, warning_log
        
        if output is None:
            return

        if is_async_storage_backend(self.storage):
            run_awaitable_sync(self.asave(output, is_completed))
            return

        try:
            # Load or create session using sync method
            session = self.storage.get_session(
                session_id=self.session_id,
                session_type=SessionType.AGENT,
                deserialize=True
            )
            
            # CRITICAL: For HITL to work, incomplete runs (paused, error, cancelled) 
            # MUST ALWAYS be saved, regardless of 'enabled' setting.
            # For completed runs: 
            # - If run was previously saved (exists in session.runs), update it
            # - If not previously saved and enabled=False, skip full save
            run_previously_saved = session and session.runs and output.run_id in session.runs
            
            if is_completed and not self.enabled and not self.summary_enabled and not run_previously_saved:
                # Skip saving completed runs that were never saved before when disabled
                return
            
            if not session:
                session = AgentSession(
                    session_id=self.session_id,
                    agent_id=output.agent_id,
                    user_id=output.user_id,
                    created_at=int(time.time()),
                    metadata={},
                    runs={},
                )
                if self.debug:
                    info_log(f"Created new session: {self.session_id}", "AgentSessionMemory")
            
            # Handle based on completion status - use sync internal methods
            if is_completed:
                if run_previously_saved and not self.enabled and not self.summary_enabled:
                    # Just update the run status, don't do full save with messages
                    session.upsert_run(output)
                else:
                    self._save_completed_run_sync(session, output)
            else:
                self._save_incomplete_run_sync(session, output)
            
            # Update session-level usage by aggregating this run's usage
            # This is done AFTER upserting the run so we have the latest usage data
            if output.usage:
                session.update_usage_from_run(output)
            
            # Always upsert to storage using sync method
            session.updated_at = int(time.time())
            self.storage.upsert_session(session, deserialize=True)
            
            if self.debug:
                status_str = "completed" if is_completed else "incomplete"
                info_log(f"Session saved for run {output.run_id} ({status_str})", "AgentSessionMemory")
                
        except Exception as e:
            if self.debug:
                import traceback
                error_trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                warning_log(f"Failed to save session: {e}\n{error_trace[-500:]}", "AgentSessionMemory")
    
    async def _save_completed_run(
        self,
        session: "AgentSession",
        output: "AgentRunOutput",
    ) -> None:
        """Save completed run with memory features.

        ``self.enabled`` (save flag) controls whether messages are persisted.
        ``self.summary_enabled`` (save flag) controls whether a summary is
        generated.  Summary generation works even when ``enabled=False``
        because it uses the previous summary + current turn from the run
        output, not stored messages.
        """
        from upsonic.utils.printing import info_log, warning_log
        
        if self.debug:
            info_log("Saving completed run...", "AgentSessionMemory")
        
        session.populate_from_run_output(output)
        session.upsert_run(output)
        
        if self.debug:
            info_log(f"Added run output to session (total runs: {len(session.runs or {})})", "AgentSessionMemory")
        
        if self.summary_enabled:
            if not self.model:
                warning_log(
                    "Summary memory enabled but no model configured. Skipping summary.",
                    "AgentSessionMemory"
                )
            else:
                try:
                    if self.debug:
                        info_log("Generating session summary...", "AgentSessionMemory")
                    session.summary = await self._generate_summary(session, output)
                    if self.debug:
                        info_log(f"Summary generated ({len(session.summary or '')} chars)", "AgentSessionMemory")
                except Exception as e:
                    warning_log(f"Failed to generate summary: {e}", "AgentSessionMemory")
        
        summary_llm_usage = getattr(self, '_last_llm_usage', None)
        if summary_llm_usage is not None and output is not None:
            usage = output._ensure_usage()
            usage.incr(summary_llm_usage)
            self._last_llm_usage = None
        
        if self.enabled:
            new_message_count = session.append_new_messages_from_run_output(output)
            if self.debug:
                info_log(
                    f"Appended {new_message_count} new messages (total: {len(session.messages or [])})",
                    "AgentSessionMemory"
                )
        elif self.debug:
            info_log("Skipping message persistence (full_session_memory disabled)", "AgentSessionMemory")
    
    def _save_completed_run_sync(
        self,
        session: "AgentSession",
        output: "AgentRunOutput",
    ) -> None:
        """Save completed run with memory features (sync version - no summary generation)."""
        from upsonic.utils.printing import info_log, warning_log
        
        if self.debug:
            info_log("Saving completed run (sync)...", "AgentSessionMemory")
        
        session.populate_from_run_output(output)
        session.upsert_run(output)
        
        if self.debug:
            info_log(f"Added run output to session (total runs: {len(session.runs or {})})", "AgentSessionMemory")
        
        if self.summary_enabled:
            warning_log(
                "Summary memory requires async execution. Use asave() instead.",
                "AgentSessionMemory"
            )
        
        if self.enabled:
            new_message_count = session.append_new_messages_from_run_output(output)
            if self.debug:
                info_log(
                    f"Appended {new_message_count} new messages (total: {len(session.messages or [])})",
                    "AgentSessionMemory"
                )
        elif self.debug:
            info_log("Skipping message persistence (full_session_memory disabled)", "AgentSessionMemory")
    
    async def _save_incomplete_run(
        self,
        session: "AgentSession",
        output: "AgentRunOutput",
    ) -> None:
        """Save checkpoint for incomplete run (HITL)."""
        from upsonic.utils.printing import info_log
        
        # Populate session data
        session.populate_from_run_output(output)
        
        # Upsert run with checkpoint state
        session.upsert_run(output)
        
        # Append messages for potential resumption
        new_message_count = session.append_new_messages_from_run_output(output)
        
        if self.debug:
            step_info = ""
            if output.requirements:
                unresolved = [r for r in output.requirements if not r.is_resolved]
                if unresolved:
                    paused_step = output.get_paused_step()
                    if paused_step:
                        step_info = f" at step {paused_step.step_number} ({paused_step.name})"
            info_log(f"Checkpoint saved for run {output.run_id}{step_info} ({new_message_count} messages)", "AgentSessionMemory")
    
    def _save_incomplete_run_sync(
        self,
        session: "AgentSession",
        output: "AgentRunOutput",
    ) -> None:
        """Save checkpoint for incomplete run (HITL) - sync version."""
        from upsonic.utils.printing import info_log
        
        # Populate session data
        session.populate_from_run_output(output)
        
        # Upsert run with checkpoint state
        session.upsert_run(output)
        
        # Append messages for potential resumption
        new_message_count = session.append_new_messages_from_run_output(output)
        
        if self.debug:
            step_info = ""
            if output.requirements:
                unresolved = [r for r in output.requirements if not r.is_resolved]
                if unresolved:
                    paused_step = output.get_paused_step()
                    if paused_step:
                        step_info = f" at step {paused_step.step_number} ({paused_step.name})"
            info_log(f"Checkpoint saved for run {output.run_id}{step_info} ({new_message_count} messages)", "AgentSessionMemory")
    
    async def aload_resumable_run(
        self,
        run_id: str,
        agent_id: Optional[str] = None,
    ) -> Optional["RunData"]:
        """Load a resumable run for HITL continuation (async)."""
        from upsonic.run.base import RunStatus
        from upsonic.session.base import SessionType
        from upsonic.utils.printing import debug_log_level2
        
        resumable_statuses = {RunStatus.paused, RunStatus.error, RunStatus.cancelled}
        
        if self.debug:
            debug_log_level2(
                f"Searching for resumable run {run_id}",
                "AgentSessionMemory.aload_resumable_run",
                debug=self.debug,
                debug_level=self.debug_level,
                session_id=self.session_id,
                agent_id=agent_id
            )
        
        # Try current session first
        if self.session_id:
            if is_async_storage_backend(self.storage):
                session = await self.storage.aget_session(
                    session_id=self.session_id,
                    session_type=SessionType.AGENT,
                    deserialize=True
                )
            else:
                session = self.storage.get_session(
                    session_id=self.session_id,
                    session_type=SessionType.AGENT,
                    deserialize=True
                )
            
            if session and session.runs:
                if self.debug:
                    debug_log_level2(
                        f"Found session with {len(session.runs)} runs",
                        "AgentSessionMemory.aload_resumable_run",
                        debug=self.debug,
                        debug_level=self.debug_level,
                        run_ids=list(session.runs.keys())
                    )
                if run_id in session.runs:
                    run_data = session.runs[run_id]
                    if run_data.output and run_data.output.status in resumable_statuses:
                        return run_data
        
        # Search all sessions for agent
        if agent_id:
            if is_async_storage_backend(self.storage):
                sessions = await self.storage.aget_sessions(
                    agent_id=agent_id,
                    session_type=SessionType.AGENT,
                    deserialize=True
                )
            else:
                sessions = self.storage.get_sessions(
                    agent_id=agent_id,
                    session_type=SessionType.AGENT,
                    deserialize=True
                )
            
            if isinstance(sessions, list):
                for session in sessions:
                    if session.runs and run_id in session.runs:
                        run_data = session.runs[run_id]
                        if run_data.output and run_data.output.status in resumable_statuses:
                            return run_data
        
        return None
    
    def load_resumable_run(
        self,
        run_id: str,
        agent_id: Optional[str] = None,
    ) -> Optional["RunData"]:
        """Load a resumable run for HITL continuation (sync)."""
        from upsonic.run.base import RunStatus
        from upsonic.session.base import SessionType
        from upsonic.utils.printing import debug_log_level2

        if is_async_storage_backend(self.storage):
            return run_awaitable_sync(self.aload_resumable_run(run_id, agent_id))

        resumable_statuses = {RunStatus.paused, RunStatus.error, RunStatus.cancelled}
        
        if self.debug:
            debug_log_level2(
                f"Searching for resumable run {run_id}",
                "AgentSessionMemory.load_resumable_run",
                debug=self.debug,
                debug_level=self.debug_level,
                session_id=self.session_id,
                agent_id=agent_id
            )
        
        # Try current session first
        if self.session_id:
            session = self.storage.get_session(
                session_id=self.session_id,
                session_type=SessionType.AGENT,
                deserialize=True
            )
            if session and session.runs:
                if run_id in session.runs:
                    run_data = session.runs[run_id]
                    if run_data.output and run_data.output.status in resumable_statuses:
                        return run_data
        
        # Search all sessions for agent
        if agent_id:
            sessions = self.storage.get_sessions(
                agent_id=agent_id,
                session_type=SessionType.AGENT,
                deserialize=True
            )
            if isinstance(sessions, list):
                for session in sessions:
                    if session.runs and run_id in session.runs:
                        run_data = session.runs[run_id]
                        if run_data.output and run_data.output.status in resumable_statuses:
                            return run_data
        
        return None
    
    async def aload_run(
        self,
        run_id: str,
        agent_id: Optional[str] = None,
    ) -> Optional["RunData"]:
        """Load a run by run_id (regardless of status) - async."""
        from upsonic.session.base import SessionType
        from upsonic.utils.printing import debug_log_level2
        
        if self.debug:
            debug_log_level2(
                f"Loading run {run_id}",
                "AgentSessionMemory.aload_run",
                debug=self.debug,
                debug_level=self.debug_level,
                session_id=self.session_id,
                agent_id=agent_id
            )
        
        # Try current session first
        if self.session_id:
            if is_async_storage_backend(self.storage):
                session = await self.storage.aget_session(
                    session_id=self.session_id,
                    session_type=SessionType.AGENT,
                    deserialize=True
                )
            else:
                session = self.storage.get_session(
                    session_id=self.session_id,
                    session_type=SessionType.AGENT,
                    deserialize=True
                )
            
            if session and session.runs and run_id in session.runs:
                return session.runs[run_id]
        
        # Search all sessions for agent
        if agent_id:
            if is_async_storage_backend(self.storage):
                sessions = await self.storage.aget_sessions(
                    agent_id=agent_id,
                    session_type=SessionType.AGENT,
                    deserialize=True
                )
            else:
                sessions = self.storage.get_sessions(
                    agent_id=agent_id,
                    session_type=SessionType.AGENT,
                    deserialize=True
                )
            
            if isinstance(sessions, list):
                for session in sessions:
                    if session.runs and run_id in session.runs:
                        return session.runs[run_id]
        
        return None
    
    def load_run(
        self,
        run_id: str,
        agent_id: Optional[str] = None,
    ) -> Optional["RunData"]:
        """Load a run by run_id (regardless of status) - sync."""
        from upsonic.session.base import SessionType
        from upsonic.utils.printing import debug_log_level2

        if is_async_storage_backend(self.storage):
            return run_awaitable_sync(self.aload_run(run_id, agent_id))

        if self.debug:
            debug_log_level2(
                f"Loading run {run_id}",
                "AgentSessionMemory.load_run",
                debug=self.debug,
                debug_level=self.debug_level,
                session_id=self.session_id,
                agent_id=agent_id
            )
        
        # Try current session first
        if self.session_id:
            session = self.storage.get_session(
                session_id=self.session_id,
                session_type=SessionType.AGENT,
                deserialize=True
            )
            if session and session.runs and run_id in session.runs:
                return session.runs[run_id]
        
        # Search all sessions for agent
        if agent_id:
            sessions = self.storage.get_sessions(
                agent_id=agent_id,
                session_type=SessionType.AGENT,
                deserialize=True
            )
            if isinstance(sessions, list):
                for session in sessions:
                    if session.runs and run_id in session.runs:
                        return session.runs[run_id]
        
        return None
    
    def _limit_message_history(self, message_history: List[Any]) -> List[Any]:
        """Limit conversation history to the last N runs."""
        from upsonic.messages.messages import ModelRequest, ModelResponse, SystemPromptPart, UserPromptPart
        from upsonic.utils.printing import info_log, warning_log
        
        if not self.num_last_messages or self.num_last_messages <= 0:
            return message_history
        
        if not message_history:
            return []
        
        # Group messages into runs (request-response pairs)
        all_runs = []
        for i in range(0, len(message_history) - 1, 2):
            request = message_history[i]
            response = message_history[i + 1]
            if isinstance(request, ModelRequest) and isinstance(response, ModelResponse):
                all_runs.append((request, response))
        
        if len(all_runs) <= self.num_last_messages:
            if self.debug:
                info_log(
                    f"History has {len(all_runs)} runs, within limit of {self.num_last_messages}",
                    "AgentSessionMemory"
                )
            return message_history
        
        kept_runs = all_runs[-self.num_last_messages:]
        
        if self.debug:
            info_log(
                f"Limiting from {len(all_runs)} runs to last {self.num_last_messages}",
                "AgentSessionMemory"
            )
        
        if not kept_runs:
            return []
        
        # Find original system prompt
        original_system_prompt = None
        if message_history:
            for part in message_history[0].parts:
                if isinstance(part, SystemPromptPart):
                    original_system_prompt = part
                    break
        
        if not original_system_prompt:
            warning_log("Could not find original SystemPromptPart", "AgentSessionMemory")
            return [message for run in kept_runs for message in run]
        
        # Find user prompt in first kept run
        first_request = kept_runs[0][0]
        new_user_prompt = None
        for part in first_request.parts:
            if isinstance(part, UserPromptPart):
                new_user_prompt = part
                break
        
        if not new_user_prompt:
            warning_log("Could not find UserPromptPart in first message", "AgentSessionMemory")
            return [message for run in kept_runs for message in run]
        
        # Reconstruct with system prompt
        modified_first_request = copy.deepcopy(first_request)
        modified_first_request.parts = [original_system_prompt, new_user_prompt]
        
        final_history = []
        final_history.append(modified_first_request)
        final_history.append(kept_runs[0][1])
        
        for run in kept_runs[1:]:
            final_history.extend(run)
        
        if self.debug:
            info_log(
                f"Limited to {len(final_history)} messages from {self.num_last_messages} runs",
                "AgentSessionMemory"
            )
        
        return final_history
    
    def _filter_tool_messages(self, messages: List[Any]) -> List[Any]:
        """Filter out tool-related messages if feed_tool_call_results is False."""
        from upsonic.messages.messages import ModelRequest, ModelResponse
        from upsonic.utils.printing import info_log
        
        filtered = []
        tool_messages_removed = 0
        
        for msg in messages:
            should_filter = False
            
            # Filter tool-return messages
            if isinstance(msg, ModelRequest):
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if hasattr(part, 'part_kind') and part.part_kind == 'tool-return':
                            should_filter = True
                            tool_messages_removed += 1
                            break
            
            # Filter tool-call messages
            elif isinstance(msg, ModelResponse):
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if hasattr(part, 'part_kind') and part.part_kind == 'tool-call':
                            should_filter = True
                            tool_messages_removed += 1
                            break
            
            if not should_filter:
                filtered.append(msg)
        
        if self.debug and tool_messages_removed > 0:
            info_log(f"Filtered out {tool_messages_removed} tool-related messages", "AgentSessionMemory")
        
        return filtered
    
    async def _generate_summary(
        self,
        session: "AgentSession",
        output: "AgentRunOutput",
    ) -> str:
        """Generate session summary."""
        from upsonic.agent.agent import Agent
        from upsonic.tasks.tasks import Task
        from upsonic.messages.messages import ModelMessagesTypeAdapter
        from upsonic.session.agent import AgentSession
        from upsonic.utils.printing import info_log
        
        if not self.model:
            raise ValueError("Model must be configured for summary generation")
        
        if self.debug:
            info_log("Starting summary generation...", "AgentSessionMemory")
        
        # Get new turn messages
        last_turn = []
        if output is not None and hasattr(output, 'new_messages'):
            try:
                new_msgs = output.new_messages()
                if new_msgs:
                    last_turn = ModelMessagesTypeAdapter.dump_python(new_msgs, mode='json')
            except Exception as e:
                if self.debug:
                    info_log(f"Could not get new_messages: {e}", "AgentSessionMemory")
        
        if self.debug:
            info_log(f"Previous summary: {len(session.summary or '')} chars", "AgentSessionMemory")
            info_log(f"New turn messages: {len(last_turn)}", "AgentSessionMemory")
        
        # Get recent messages for context
        recent_messages = await AgentSession.get_all_messages_for_session_id_async(
            storage=self.storage,
            session_id=session.session_id
        )
        
        recent_messages_str = json.dumps([str(m) for m in recent_messages], indent=2) if recent_messages else 'None'
        
        if self.debug:
            info_log(f"Recent messages: {len(recent_messages)} from session", "AgentSessionMemory")
        
        # Skip if no messages
        if not last_turn and not recent_messages:
            if self.debug:
                info_log("No messages for summary, skipping", "AgentSessionMemory")
            return session.summary or ""
        
        summarizer = Agent(name="Summarizer", model=self.model, debug=self.debug)
        
        previous_summary = session.summary or 'None (first interaction)'
        new_turn_str = json.dumps(last_turn, indent=2) if last_turn else 'None (using history only)'
        
        prompt = f"""Update the conversation summary based on the new interaction.

Previous Summary: {previous_summary}

New Conversation Turn:
{new_turn_str}

Recent Chat History:
{recent_messages_str}

YOUR TASK: Create a concise summary capturing key points of the conversation.
Focus on important information, user preferences, and topics discussed.
"""
        task = Task(description=prompt, response_format=str)
        
        summary_output = await summarizer.do_async(task, return_output=True)
        summary_text = str(summary_output.output)
        
        # Store sub-agent usage for parent context aggregation
        if hasattr(summary_output, 'usage') and summary_output.usage:
            if not hasattr(self, '_last_llm_usage'):
                self._last_llm_usage = None
            from upsonic.usage import RunUsage
            if self._last_llm_usage is None:
                self._last_llm_usage = RunUsage()
            self._last_llm_usage.incr(summary_output.usage)
        
        if self.debug:
            info_log(f"Summary generated: {len(summary_text)} chars", "AgentSessionMemory")
        
        return summary_text
