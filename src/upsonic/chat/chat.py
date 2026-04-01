"""High-level Chat interface for conversational AI sessions.

This module provides the Chat class - a comprehensive interface for managing
conversational sessions with storage binding, memory integration, and cost tracking.
"""
from __future__ import annotations

import asyncio
import time
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    List,
    Literal,
    Optional,
    Union,
    overload,
)

from dataclasses import dataclass

from upsonic.tasks.tasks import Task
from upsonic.storage.memory.memory import Memory
from upsonic.storage.base import Storage
from upsonic.storage.in_memory.in_memory import InMemoryStorage
from .session_manager import SessionManager, SessionState
from .message import ChatMessage
from .schemas import InvokeResult

if TYPE_CHECKING:
    from upsonic.agent.agent import Agent
    from upsonic.run.events.events import AgentEvent
else:
    Agent = "Agent"
    AgentEvent = "AgentEvent"


class Chat:
    """
    A comprehensive Chat interface for managing conversational AI sessions.
    
    The Chat class serves as a stateful session orchestrator that provides:
    - Session lifecycle management with storage binding
    - Memory integration and persistence
    - Cost and token tracking from session usage
    - Both blocking and streaming interfaces
    - Error handling and retry mechanisms
    
    Key Architecture:
    - Storage is the single source of truth for session data
    - Agent's Memory handles run persistence and message history
    - SessionManager provides developer-friendly access to data
    - ChatMessage is only for display (converted from ModelMessage on access)
    
    Usage:
        Basic usage:
        ```python
        from upsonic import Chat, Agent
        
        agent = Agent("openai/gpt-4o")
        chat = Chat(session_id="user123_session1", user_id="user123", agent=agent)
        
        # Send a message
        response = await chat.invoke("Hello, how are you?")
        print(response)
        
        # Access chat history (converted to ChatMessage on demand)
        print(chat.all_messages)
        print(f"Total cost: ${chat.total_cost}")
        ```
        
        Advanced usage with streaming:
        ```python
        async for chunk in chat.invoke("Tell me a story", stream=True):
            print(chunk, end='', flush=True)
        ```
        
        With custom storage:
        ```python
        from upsonic.storage import SqliteStorage
        
        storage = SqliteStorage("chat.db")
        chat = Chat(
            session_id="session1",
            user_id="user1", 
            agent=agent,
            storage=storage,
            full_session_memory=True,
            summary_memory=True,
        )
        ```
    """
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        agent: "Agent",
        *,
        storage: Optional[Storage] = None,
        # Memory configuration (save flags)
        full_session_memory: bool = True,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        # Memory configuration (load flags)
        load_full_session_memory: bool = True,
        load_summary_memory: Optional[bool] = None,
        load_user_analysis_memory: Optional[bool] = None,
        user_profile_schema: Optional[type] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update',
        # Chat configuration
        debug: bool = False,
        debug_level: int = 1,
        max_concurrent_invocations: int = 1,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize a Chat session.
        
        Args:
            session_id: Unique identifier for this chat session
            user_id: Unique identifier for the user
            agent: The Agent instance to handle conversations
            storage: Storage backend (defaults to InMemoryStorage)
            full_session_memory: Enable full conversation history storage
            summary_memory: Enable conversation summarization
            user_analysis_memory: Enable user profile analysis
            user_profile_schema: Custom user profile schema
            dynamic_user_profile: Enable dynamic profile schema generation
            num_last_messages: Limit conversation history to last N messages
            feed_tool_call_results: Include tool calls in memory
            user_memory_mode: How to update user profiles ('update' or 'replace')
            debug: Enable debug logging
            debug_level: Debug level (1 = standard, 2 = detailed)
            max_concurrent_invocations: Maximum concurrent invoke calls
            retry_attempts: Number of retry attempts for failed calls
            retry_delay: Delay between retry attempts
        """
        # Input validation
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if agent is None:
            raise ValueError("agent cannot be None")
        if max_concurrent_invocations < 1:
            raise ValueError("max_concurrent_invocations must be at least 1")
        if retry_attempts < 0:
            raise ValueError("retry_attempts must be non-negative")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if num_last_messages is not None and num_last_messages < 1:
            raise ValueError("num_last_messages must be at least 1 if specified")
        
        self.session_id = session_id.strip()
        self.user_id = user_id.strip()
        self.agent = agent
        self.debug = debug
        self.debug_level = debug_level if debug else 1
        
        # Initialize storage (single source of truth)
        self._storage = storage or InMemoryStorage()
        
        # Initialize memory with all configuration
        self._memory = Memory(
            storage=self._storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            load_full_session_memory=load_full_session_memory,
            load_summary_memory=load_summary_memory,
            load_user_analysis_memory=load_user_analysis_memory,
            user_profile_schema=user_profile_schema,
            dynamic_user_profile=dynamic_user_profile,
            num_last_messages=num_last_messages,
            model=agent.model,
            debug=debug,
            debug_level=debug_level,
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        # Update agent's memory to use our configured memory
        self.agent.memory = self._memory
        
        # Initialize session manager with storage binding
        self._session_manager = SessionManager(
            session_id=session_id,
            user_id=user_id,
            storage=self._storage,
            debug=debug,
            debug_level=debug_level,
            max_concurrent_invocations=max_concurrent_invocations
        )
        
        # Retry configuration
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
        self._max_concurrent_invocations = max_concurrent_invocations
        
        if self.debug:
            from upsonic.utils.printing import debug_log, debug_log_level2
            debug_log(
                f"Chat initialized: session_id={session_id}, user_id={user_id}",
                "Chat",
                debug=self.debug,
                debug_level=self.debug_level
            )
            if self.debug_level >= 2:
                debug_log_level2(
                    "Chat detailed initialization",
                    "Chat",
                    debug=self.debug,
                    debug_level=self.debug_level,
                    session_id=session_id,
                    user_id=user_id,
                    agent_name=getattr(agent, 'name', 'Unknown'),
                    full_session_memory=full_session_memory,
                    summary_memory=summary_memory,
                    user_analysis_memory=user_analysis_memory,
                    max_concurrent_invocations=max_concurrent_invocations
                )
    
    @property
    def state(self) -> SessionState:
        """Current state of the chat session (runtime only)."""
        return self._session_manager.state
    
    @property
    def all_messages(self) -> List[ChatMessage]:
        """
        Get all messages in the current chat session.
        
        Retrieves messages from storage and converts to ChatMessage.
        
        Returns:
            List of ChatMessage objects representing the conversation history
        """
        return self._session_manager.all_messages
    
    @property
    def input_tokens(self) -> int:
        """Total input tokens used in this chat session (from storage)."""
        return self._session_manager.input_tokens
    
    @property
    def output_tokens(self) -> int:
        """Total output tokens used in this chat session (from storage)."""
        return self._session_manager.output_tokens
    
    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) used in this chat session."""
        return self._session_manager.total_tokens
    
    @property
    def total_cost(self) -> float:
        """Total cost of this chat session in USD (from storage)."""
        return self._session_manager.total_cost
    
    @property
    def total_requests(self) -> int:
        """Total API requests made in this chat session (from storage)."""
        return self._session_manager.total_requests
    
    @property
    def total_tool_calls(self) -> int:
        """Total tool calls made in this chat session (from storage)."""
        return self._session_manager.total_tool_calls
    
    @property
    def run_duration(self) -> Optional[float]:
        """Total run duration in seconds (from storage RunUsage)."""
        return self._session_manager.run_duration
    
    @property
    def time_to_first_token(self) -> Optional[float]:
        """Time to first token in seconds (from storage RunUsage)."""
        return self._session_manager.time_to_first_token
    
    @property
    def start_time(self) -> float:
        """Session start time (Unix timestamp)."""
        return self._session_manager.start_time
    
    @property
    def end_time(self) -> Optional[float]:
        """Session end time (Unix timestamp, None if still active)."""
        return self._session_manager.end_time
    
    @property
    def duration(self) -> float:
        """
        Duration of the chat session in seconds.
        
        If closed, returns the fixed duration from start to end.
        If active, returns duration from start to now.
        """
        return self._session_manager.duration
    
    @property
    def last_activity(self) -> float:
        """Time since last activity in seconds."""
        return self._session_manager.last_activity
    
    @property
    def last_activity_time(self) -> float:
        """Last activity timestamp (Unix timestamp)."""
        return self._session_manager.last_activity_time
    
    @property
    def is_closed(self) -> bool:
        """Check if the session has been closed."""
        return self._session_manager.is_closed
    
    def get_usage(self) -> Any:
        """Get the full RunUsage object from session storage."""
        return self._session_manager.get_usage()
    
    def get_session_metrics(self) -> Any:
        """Get comprehensive session metrics."""
        return self._session_manager.get_session_metrics()
    
    def get_session_summary(self) -> str:
        """Get a human-readable session summary."""
        return self._session_manager.get_session_summary()
    
    def get_recent_messages(self, count: int = 10) -> List[ChatMessage]:
        """Get the most recent messages as ChatMessage objects."""
        return self._session_manager.get_recent_messages(count)
    
    def clear_history(self) -> None:
        """
        Clear the chat history from storage.
        
        This clears session.messages while preserving other session data.
        """
        self._session_manager.clear_history()
    
    async def aclear_history(self) -> None:
        """Clear the chat history from storage (async)."""
        await self._session_manager.aclear_history()
    
    def reset_session(self) -> None:
        """
        Reset the chat session to initial state.
        
        This deletes the session from storage and resets runtime state.
        """
        self._session_manager.reset_session()
    
    async def areset_session(self) -> None:
        """Reset the chat session to initial state (async)."""
        await self._session_manager.areset_session()
    
    def reopen(self) -> None:
        """
        Reopen a closed session.
        
        This allows resuming a session that was previously closed via close().
        The session duration continues from where it left off (cumulative).
        All message history and usage data are preserved.
        
        Example:
            chat = Chat(session_id="my_session", user_id="user1", agent=agent)
            await chat.invoke("Hello")
            chat.close()  # Session closed, duration frozen
            
            # Later, reopen to continue
            chat.reopen()
            await chat.invoke("Continue where we left off")
        """
        self._session_manager.reopen_session()
    
    async def areopen(self) -> None:
        """Reopen a closed session (async)."""
        await self._session_manager.areopen_session()
    
    # ========================================================================
    # History Manipulation Methods
    # ========================================================================
    
    def get_raw_messages(self) -> List[Any]:
        """
        Get raw ModelMessage list from storage for direct manipulation.
        
        Use this when you need to directly access and modify the message history.
        
        Returns:
            List of ModelMessage objects
        """
        return self._session_manager.get_raw_messages()
    
    async def aget_raw_messages(self) -> List[Any]:
        """Get raw ModelMessage list from storage (async)."""
        return await self._session_manager.aget_raw_messages()
    
    def set_messages(self, messages: List[Any]) -> None:
        """
        Set the message list in storage.
        
        Use this after manipulating messages (e.g., removing attachments).
        
        Args:
            messages: List of ModelMessage objects to set
        """
        self._session_manager.set_messages(messages)
    
    async def aset_messages(self, messages: List[Any]) -> None:
        """Set the message list in storage (async)."""
        await self._session_manager.aset_messages(messages)
    
    def delete_message(self, message_index: int) -> bool:
        """
        Delete a message from the chat history by index.
        
        Args:
            message_index: The index of the message to delete (from all_messages)
            
        Returns:
            True if deletion was successful, False otherwise
            
        Example:
            # Delete the first message
            success = chat.delete_message(0)
            
            # Delete based on ChatMessage.message_index
            for msg in chat.all_messages:
                if "delete me" in msg.content:
                    chat.delete_message(msg.message_index)
        """
        return self._session_manager.delete_message(message_index)
    
    async def adelete_message(self, message_index: int) -> bool:
        """Delete a message from the chat history by index (async)."""
        return await self._session_manager.adelete_message(message_index)
    
    def remove_attachment(
        self,
        message_index: int,
        attachment_index: int
    ) -> bool:
        """
        Remove an attachment from a specific message.
        
        This allows you to selectively remove attachments (images, PDFs, etc.)
        from a message while keeping the text content and other attachments.
        
        Args:
            message_index: The index of the message containing the attachment
                           (from ChatMessage.message_index)
            attachment_index: The index of the attachment within the message
                              (from ChatAttachment.index)
            
        Returns:
            True if removal was successful, False otherwise
            
        Example:
            # Send message with multiple attachments
            await chat.invoke("Analyze these", context=["image1.png", "document.pdf"])
            
            # Get the message and its attachments
            for msg in chat.all_messages:
                if msg.attachments:
                    for att in msg.attachments:
                        if att.type == "document":  # Remove PDFs but keep images
                            chat.remove_attachment(msg.message_index, att.index)
        """
        return self._session_manager.remove_attachment_from_message(
            message_index,
            attachment_index
        )
    
    async def aremove_attachment(
        self,
        message_index: int,
        attachment_index: int
    ) -> bool:
        """Remove an attachment from a specific message (async)."""
        return await self._session_manager.aremove_attachment_from_message(
            message_index,
            attachment_index
        )
    
    def remove_attachment_by_path(self, path: str) -> int:
        """
        Remove all attachments matching the given path from all messages.
        
        This is the simplest way to remove attachments - just pass the path
        and all occurrences are found and removed across all messages.
        
        Args:
            path: The file path to remove (e.g., "/path/to/file.pdf")
                  Matches if path is contained in or equal to attachment identifier.
            
        Returns:
            int: Number of attachments removed
            
        Example:
            # Send messages with various attachments
            await chat.invoke("Check this", context=["image.png", "doc.pdf"])
            await chat.invoke("And this", context=["doc.pdf", "other.txt"])
            
            # Remove all occurrences of doc.pdf from all messages
            removed = chat.remove_attachment_by_path("doc.pdf")
            print(f"Removed {removed} attachments")  # Output: Removed 2 attachments
        """
        return self._session_manager.remove_attachment_by_path(path)
    
    async def aremove_attachment_by_path(self, path: str) -> int:
        """Remove all attachments matching the given path (async)."""
        return await self._session_manager.aremove_attachment_by_path(path)
    
    def _transition_state(self, new_state: SessionState) -> None:
        """Safely transition to a new state."""
        self._session_manager.transition_state(new_state)
    
    def _normalize_input(
        self,
        input_data: Union[str, Task],
        context: Optional[List[str]] = None
    ) -> Task:
        """Normalize various input types into a Task object."""
        if input_data is None:
            raise ValueError("Input data cannot be None")
        
        if isinstance(input_data, str):
            if not input_data.strip():
                raise ValueError("Input string cannot be empty or whitespace only")
            return Task(description=input_data.strip(), context=context)
        elif isinstance(input_data, Task):
            if not input_data.description or not input_data.description.strip():
                raise ValueError("Task description cannot be empty or whitespace only")
            return input_data
        else:
            raise TypeError(
                f"Unsupported input type: {type(input_data)}. "
                f"Expected str or Task, got {type(input_data)}"
            )
    
    # ========================================================================
    # Retry Logic
    # ========================================================================
    
    async def _execute_with_retry(self, coro_func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute a coroutine with retry logic."""
        last_exception = None
        
        for attempt in range(self._retry_attempts + 1):
            try:
                return await coro_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if not self._is_retryable_error(e):
                    if self.debug:
                        from upsonic.utils.printing import debug_log
                        debug_log(f"Non-retryable error: {e}", "Chat")
                    self._transition_state(SessionState.ERROR)
                    raise e
                
                if attempt < self._retry_attempts:
                    if self.debug:
                        from upsonic.utils.printing import debug_log
                        debug_log(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {self._retry_delay}s...",
                            "Chat"
                        )
                    await asyncio.sleep(self._retry_delay * (2 ** attempt))
                else:
                    if self.debug:
                        from upsonic.utils.printing import debug_log
                        debug_log(
                            f"All {self._retry_attempts + 1} attempts failed. "
                            f"Last error: {e}",
                            "Chat"
                        )
        
        self._transition_state(SessionState.ERROR)
        raise last_exception
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError,
        )
        
        if isinstance(error, retryable_errors):
            return True
        
        error_msg = str(error).lower()
        retryable_patterns = [
            'timeout',
            'connection',
            'network',
            'rate limit',
            'temporary',
            'service unavailable',
            'internal server error',
            'bad gateway',
            'gateway timeout'
        ]
        
        return any(pattern in error_msg for pattern in retryable_patterns)
    
    
    @overload
    async def invoke(
        self,
        input_data: Union[str, Task],
        *,
        context: Optional[List[str]] = None,
        stream: Literal[False] = False,
        events: Literal[False] = False,
        return_run_output: Literal[True],
        **kwargs: Any
    ) -> InvokeResult: ...

    @overload
    async def invoke(
        self,
        input_data: Union[str, Task],
        *,
        context: Optional[List[str]] = None,
        stream: Literal[False] = False,
        events: Literal[False] = False,
        **kwargs: Any
    ) -> str: ...

    @overload
    async def invoke(
        self,
        input_data: Union[str, Task],
        *,
        context: Optional[List[str]] = None,
        stream: Literal[True],
        events: Literal[False] = False,
        **kwargs: Any
    ) -> AsyncIterator[str]: ...

    @overload
    async def invoke(
        self,
        input_data: Union[str, Task],
        *,
        context: Optional[List[str]] = None,
        stream: Literal[True],
        events: Literal[True],
        **kwargs: Any
    ) -> AsyncIterator["AgentEvent"]: ...

    @overload
    async def invoke(
        self,
        input_data: Union[str, Task],
        *,
        context: Optional[List[str]] = None,
        stream: bool = False,
        events: Literal[True] = ...,
        **kwargs: Any
    ) -> AsyncIterator["AgentEvent"]: ...

    async def invoke(
        self,
        input_data: Union[str, Task],
        *,
        context: Optional[List[str]] = None,
        stream: bool = False,
        events: bool = False,
        return_run_output: bool = False,
        **kwargs: Any
    ) -> Union[str, InvokeResult, AsyncIterator[str], AsyncIterator["AgentEvent"]]:
        """
        Send a message to the chat and get a response.
        
        This is the primary method for interacting with the chat. It handles:
        - Input normalization and validation
        - State management and concurrency control
        - Agent execution (blocking or streaming)
        - Response processing and cost tracking (via agent's memory)
        
        Args:
            input_data: The message content (string) or Task object
            context: Optional list of file paths to attach
            stream: Whether to stream the response
            events: If True, yield AgentEvent objects instead of text chunks (requires stream=True)
            return_run_output: If True (and stream=False), return InvokeResult with optional run_output for HITL
            **kwargs: Additional arguments passed to the agent
            
        Returns:
            If stream=False and events=False and return_run_output=False: The response string
            If stream=False and return_run_output=True: InvokeResult(text, run_output) for HITL handling
            If stream=True and events=False: AsyncIterator yielding text chunks
            If stream=True and events=True: AsyncIterator yielding AgentEvent objects
            
        Raises:
            RuntimeError: If chat is in an invalid state
            ValueError: If input validation fails
            Exception: If agent execution fails after retries
            
        Example:
            # Stream text chunks
            async for chunk in chat.invoke("Hello", stream=True):
                print(chunk, end='', flush=True)
            
            # Stream events
            async for event in chat.invoke("Hello", stream=True, events=True):
                print(event)
        """
        # If events=True, force stream=True
        if events and not stream:
            stream = True
        
        # return_run_output only applies to blocking (non-streaming) invocation
        if stream and return_run_output:
            return_run_output = False
        
        # State and concurrency checks
        if not self._session_manager.can_accept_invocation():
            if self._session_manager.state == SessionState.ERROR:
                raise RuntimeError(
                    "Chat is in error state. Reset or create a new chat session."
                )
            else:
                current = self._session_manager._concurrent_invocations
                max_allowed = self._session_manager._max_concurrent_invocations
                raise RuntimeError(
                    f"Maximum concurrent invocations exceeded. "
                    f"Current: {current}, Max allowed: {max_allowed}. "
                    f"Wait for current operations to complete."
                )
        
        # Normalize input
        task = self._normalize_input(input_data, context)
        
        # Update state and activity
        self._session_manager.start_invocation()
        self._transition_state(SessionState.STREAMING if stream else SessionState.AWAITING_RESPONSE)
        
        # Start response timer
        response_start_time = self._session_manager.start_response_timer()
        
        if stream:
            if events:
                return self._invoke_streaming_events(task, response_start_time, **kwargs)
            else:
                return self._invoke_streaming(task, response_start_time, **kwargs)
        else:
            return await self._invoke_blocking_async(task, response_start_time, return_run_output=return_run_output, **kwargs)
    
    async def _invoke_blocking_async(
        self,
        task: Task,
        response_start_time: float,
        *,
        return_run_output: bool = False,
        **kwargs: Any
    ) -> Union[str, InvokeResult]:
        """Handle blocking invocation."""
        from upsonic.run.agent.output import AgentRunOutput as AgentRunOutputConcrete

        async def _execute() -> Union[str, InvokeResult]:
            if self.debug and self.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                debug_log_level2(
                    "Chat invocation starting",
                    "Chat",
                    debug=self.debug,
                    debug_level=self.debug_level,
                    session_id=self.session_id,
                    user_id=self.user_id,
                    task_description=task.description[:300] if task.description else None,
                )
            
            if return_run_output:
                result = await self.agent.do_async(task, debug=self.debug, return_output=True, **kwargs)
                if not isinstance(result, AgentRunOutputConcrete):
                    response_text = str(result)
                    return InvokeResult(text=response_text, run_output=None)
                response_text = str(result.output) if result.output else (str(result) if hasattr(result, "__str__") else "")
                if result.is_paused:
                    return InvokeResult(text=response_text, run_output=result)
                return InvokeResult(text=response_text, run_output=None)
            
            result = await self.agent.do_async(task, debug=self.debug, **kwargs)
            response_text = str(result)
            
            if self.debug and self.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                execution_time = time.time() - response_start_time
                debug_log_level2(
                    "Chat invocation completed",
                    "Chat",
                    debug=self.debug,
                    debug_level=self.debug_level,
                    session_id=self.session_id,
                    execution_time=execution_time,
                    response_preview=response_text[:500],
                )
            
            return response_text
        
        try:
            result = await self._execute_with_retry(_execute)
            self._session_manager.end_response_timer(response_start_time)
            return result
        except Exception:
            self._session_manager.end_response_timer(response_start_time)
            raise
        finally:
            self._session_manager.end_invocation()
            self._transition_state(SessionState.IDLE)
    
    def _invoke_streaming(
        self,
        task: Task,
        response_start_time: float,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Handle streaming invocation (text chunks)."""
        async def _execute_streaming() -> AsyncIterator[str]:
            async for chunk in self.agent.astream(task, debug=self.debug, events=False, **kwargs):
                if isinstance(chunk, str):
                    yield chunk
        
        async def _stream_with_retry() -> AsyncIterator[str]:
            last_exception = None
            stream_generator = None
            
            try:
                for attempt in range(self._retry_attempts + 1):
                    try:
                        stream_generator = _execute_streaming()
                        async for chunk in stream_generator:
                            yield chunk
                        return
                    except Exception as exc:
                        last_exception = exc
                        if stream_generator:
                            try:
                                await stream_generator.aclose()
                            except Exception:
                                pass
                            stream_generator = None
                        
                        if "context manager is already active" in str(exc):
                            if self.debug:
                                from upsonic.utils.printing import debug_log
                                debug_log(
                                    f"Streaming context conflict on attempt {attempt + 1}",
                                    "Chat"
                                )
                            await asyncio.sleep(self._retry_delay * (3 ** attempt))
                        elif attempt < self._retry_attempts:
                            if self.debug:
                                from upsonic.utils.printing import debug_log
                                debug_log(
                                    f"Streaming attempt {attempt + 1} failed: {last_exception}",
                                    "Chat"
                                )
                            await asyncio.sleep(self._retry_delay * (2 ** attempt))
                        else:
                            raise last_exception
            finally:
                if stream_generator:
                    try:
                        await stream_generator.aclose()
                    except Exception:
                        pass
                
                self._session_manager.end_response_timer(response_start_time)
                self._session_manager.end_invocation()
                self._transition_state(SessionState.IDLE)
        
        return _stream_with_retry()
    
    def _invoke_streaming_events(
        self,
        task: Task,
        response_start_time: float,
        **kwargs: Any
    ) -> AsyncIterator["AgentEvent"]:
        """Handle streaming invocation with events (yields AgentEvent objects)."""
        from upsonic.run.events.events import AgentEvent
        
        async def _execute_streaming_events() -> AsyncIterator[AgentEvent]:
            async for event in self.agent.astream(task, debug=self.debug, events=True, **kwargs):
                yield event
        
        async def _stream_events_with_retry() -> AsyncIterator[AgentEvent]:
            last_exception = None
            stream_generator = None
            
            try:
                for attempt in range(self._retry_attempts + 1):
                    try:
                        stream_generator = _execute_streaming_events()
                        async for event in stream_generator:
                            yield event
                        return
                    except Exception as exc:
                        last_exception = exc
                        if stream_generator:
                            try:
                                await stream_generator.aclose()
                            except Exception:
                                pass
                            stream_generator = None
                        
                        if "context manager is already active" in str(exc):
                            if self.debug:
                                from upsonic.utils.printing import debug_log
                                debug_log(
                                    f"Event streaming context conflict on attempt {attempt + 1}",
                                    "Chat"
                                )
                            await asyncio.sleep(self._retry_delay * (3 ** attempt))
                        elif attempt < self._retry_attempts:
                            if self.debug:
                                from upsonic.utils.printing import debug_log
                                debug_log(
                                    f"Event streaming attempt {attempt + 1} failed: {last_exception}",
                                    "Chat"
                                )
                            await asyncio.sleep(self._retry_delay * (2 ** attempt))
                        else:
                            raise last_exception
            finally:
                if stream_generator:
                    try:
                        await stream_generator.aclose()
                    except Exception:
                        pass
                
                self._session_manager.end_response_timer(response_start_time)
                self._session_manager.end_invocation()
                self._transition_state(SessionState.IDLE)
        
        return _stream_events_with_retry()
    
    @overload
    def stream(
        self,
        input_data: Union[str, Task],
        *,
        context: Optional[List[str]] = None,
        events: Literal[False] = False,
        **kwargs: Any
    ) -> AsyncIterator[str]: ...

    @overload
    def stream(
        self,
        input_data: Union[str, Task],
        *,
        context: Optional[List[str]] = None,
        events: Literal[True],
        **kwargs: Any
    ) -> AsyncIterator["AgentEvent"]: ...

    def stream(
        self,
        input_data: Union[str, Task],
        *,
        context: Optional[List[str]] = None,
        events: bool = False,
        **kwargs: Any
    ) -> Union[AsyncIterator[str], AsyncIterator["AgentEvent"]]:
        """
        Stream a response from the chat.
        
        This is a dedicated streaming method that returns an async iterator.
        
        Args:
            input_data: The message content (string) or Task object
            context: Optional list of file paths to attach
            events: If True, yield AgentEvent objects instead of text chunks
            **kwargs: Additional arguments passed to the agent
            
        Returns:
            If events=False: AsyncIterator yielding text chunks
            If events=True: AsyncIterator yielding AgentEvent objects
            
        Example:
            # Stream text chunks
            async for chunk in chat.stream("Tell me a story"):
                print(chunk, end='', flush=True)
            
            # Stream events (tool calls, text deltas, etc.)
            async for event in chat.stream("Calculate 5+3", events=True):
                print(event)
        """
        task = self._normalize_input(input_data, context)
        
        self._session_manager.start_invocation()
        self._transition_state(SessionState.STREAMING)
        
        response_start_time = self._session_manager.start_response_timer()
        
        if events:
            return self._invoke_streaming_events(task, response_start_time, **kwargs)
        else:
            return self._invoke_streaming(task, response_start_time, **kwargs)
    
    
    async def close(self) -> None:
        """Close the chat session and cleanup resources."""
        await self._session_manager.aclose_session()
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log("Chat session closed", "Chat")
    
    async def __aenter__(self) -> "Chat":
        """Async context manager entry."""
        return self
    
    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        await self.close()
    
    def __repr__(self) -> str:
        """String representation of the chat."""
        message_count = self._session_manager.get_message_count()
        return (
            f"Chat(session_id='{self.session_id}', user_id='{self.user_id}', "
            f"state={self.state.value}, messages={message_count}, "
            f"cost=${self.total_cost:.4f})"
        )
