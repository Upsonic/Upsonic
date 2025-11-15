import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Callable, Literal, TYPE_CHECKING, overload

from upsonic.tasks.tasks import Task
from upsonic.storage.memory.memory import Memory
from upsonic.storage.base import Storage
from upsonic.storage.providers.in_memory import InMemoryStorage
from upsonic.messages.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart, TextPart
from upsonic.agent.run_result import RunResult, StreamRunResult
from .cost_calculator import CostTracker, format_cost, format_tokens
from .session_manager import SessionManager, SessionState
from .message import ChatMessage

if TYPE_CHECKING:
    from upsonic.agent.agent import Agent
    from upsonic.models import Model
    from upsonic.schemas import UserTraits
    from upsonic.storage.session.sessions import InteractionSession, UserProfile
else:
    Agent = "Agent"
    Model = "Model"
    UserTraits = "UserTraits"
    InteractionSession = "InteractionSession"
    UserProfile = "UserProfile"


class Chat:
    """
    A comprehensive, high-level Chat interface for managing conversational sessions.
    
    The Chat class serves as a stateful session orchestrator that provides:
    - Session lifecycle management
    - Memory integration and persistence
    - Cost and token tracking
    - Both blocking and streaming interfaces
    - Middleware support for extensibility
    - Error handling and retry mechanisms
    
    Usage:
        Basic usage:
        ```python
        from upsonic import Chat, Agent
        
        agent = Agent("openai/gpt-4o")
        chat = Chat(session_id="user123_session1", user_id="user123", agent=agent)
        
        # Send a message
        response = await chat.invoke("Hello, how are you?")
        print(response)
        
        # Access chat history
        print(chat.all_messages)
        print(f"Total cost: ${chat.total_cost}")
        ```
        
        Advanced usage with streaming:
        ```python
        async for chunk in chat.invoke("Tell me a story", stream=True):
            print(chunk, end='', flush=True)
        ```
        
        With custom storage and memory settings:
        ```python
        from upsonic.storage.providers import SqliteStorage
        
        storage = SqliteStorage("chat.db", "sessions", "profiles")
        chat = Chat(
            session_id="session1",
            user_id="user1", 
            agent=agent,
            storage=storage,
            full_session_memory=True,
            summary_memory=True,
            user_analysis_memory=True
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
        # Memory configuration
        full_session_memory: bool = True,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        user_profile_schema: Optional[type] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update',
        # Chat configuration
        debug: bool = False,
        max_concurrent_invocations: int = 1,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
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
        
        # Initialize storage
        self._storage = storage or InMemoryStorage()
        
        # Initialize memory with all configuration
        self._memory = Memory(
            storage=self._storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            user_profile_schema=user_profile_schema,
            dynamic_user_profile=dynamic_user_profile,
            num_last_messages=num_last_messages,
            model=agent.model,
            debug=debug,
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        # Update agent's memory to use our configured memory
        self.agent.memory = self._memory
        
        # Initialize session manager
        self._session_manager = SessionManager(
            session_id=session_id,
            user_id=user_id,
            debug=debug,
            max_concurrent_invocations=max_concurrent_invocations
        )
        
        # Initialize retry configuration
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay
        
        # Performance optimization settings
        self._max_concurrent_invocations = max_concurrent_invocations
        
        
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log(f"Chat initialized: session_id={session_id}, user_id={user_id}", "Chat")
    
    @property
    def state(self) -> SessionState:
        """Current state of the chat session."""
        return self._session_manager.state
    
    @property
    def all_messages(self) -> List[ChatMessage]:
        """
        Get all messages in the current chat session.
        
        Returns:
            List of ChatMessage objects representing the conversation history
        """
        return self._session_manager.all_messages
    
    @property
    def input_tokens(self) -> int:
        """Total input tokens used in this chat session."""
        return self._session_manager.input_tokens
    
    @property
    def output_tokens(self) -> int:
        """Total output tokens used in this chat session."""
        return self._session_manager.output_tokens
    
    @property
    def total_cost(self) -> float:
        """Total cost of this chat session in USD."""
        return self._session_manager.total_cost
    
    @property
    def session_duration(self) -> float:
        """Duration of the chat session in seconds."""
        return self._session_manager.session_duration
    
    @property
    def last_activity(self) -> float:
        """Time since last activity in seconds."""
        return self._session_manager.last_activity
    
    def get_cost_history(self) -> List[Dict[str, Any]]:
        """Get detailed cost history for this session."""
        return self._session_manager.get_cost_history()
    
    def get_session_metrics(self):
        """Get comprehensive session metrics."""
        return self._session_manager.get_session_metrics()
    
    def get_session_summary(self) -> str:
        """Get a human-readable session summary."""
        return self._session_manager.get_session_summary()
    
    
    def _transition_state(self, new_state: SessionState) -> None:
        """Safely transition to a new state."""
        self._session_manager.transition_state(new_state)
    
    def get_recent_messages(self, count: int = 10) -> List[ChatMessage]:
        """Get the most recent messages."""
        return self._session_manager.get_recent_messages(count)
    
    def _normalize_input(self, input_data: Union[str, Task], context: Optional[List[str]] = None) -> Task:
        """Normalize various input types into a Task object."""
        if input_data is None:
            raise ValueError("Input data cannot be None")
        
        if isinstance(input_data, str):
            if not input_data.strip():
                raise ValueError("Input string cannot be empty or whitespace only")
            return Task(
                description=input_data.strip(),
                context=context
            )
        elif isinstance(input_data, Task):
            if not input_data.description or not input_data.description.strip():
                raise ValueError("Task description cannot be empty or whitespace only")
            return input_data
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}. Expected str or Task, got {type(input_data)}")
    
    async def _execute_with_retry(self, coro_func, *args, **kwargs) -> Any:
        """Execute a coroutine with retry logic."""
        last_exception = None
        
        for attempt in range(self._retry_attempts + 1):
            try:
                return await coro_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # Check if this is a retryable error
                if not self._is_retryable_error(e):
                    if self.debug:
                        from upsonic.utils.printing import debug_log
                        debug_log(f"Non-retryable error: {e}", "Chat")
                    self._transition_state(SessionState.ERROR)
                    raise e
                
                if attempt < self._retry_attempts:
                    if self.debug:
                        from upsonic.utils.printing import debug_log
                        debug_log(f"Attempt {attempt + 1} failed: {e}. Retrying in {self._retry_delay}s...", "Chat")
                    await asyncio.sleep(self._retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    if self.debug:
                        from upsonic.utils.printing import debug_log
                        debug_log(f"All {self._retry_attempts + 1} attempts failed. Last error: {e}", "Chat")
        
        self._transition_state(SessionState.ERROR)
        raise last_exception
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        # Network-related errors that might be temporary
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError,  # Network errors
        )
        
        # Check error type
        if isinstance(error, retryable_errors):
            return True
        
        # Check error message for common retryable patterns
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
        **kwargs
    ) -> str: ...

    @overload
    async def invoke(
        self,
        input_data: Union[str, Task],
        *,
        context: Optional[List[str]] = None,
        stream: Literal[True],
        **kwargs
    ) -> AsyncIterator[str]: ...

    async def invoke(
        self,
        input_data: Union[str, Task],
        *,
        context: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """
        Send a message to the chat and get a response.
        
        This is the primary method for interacting with the chat. It handles:
        - Input normalization and validation
        - State management and concurrency control
        - Memory preparation and context injection
        - Agent execution (blocking or streaming)
        - Response processing and cost tracking
        - Memory persistence and history updates
        
        Args:
            input_data: The message content (string) or Task object
            context: Optional list of file paths to attach
            stream: Whether to stream the response
            **kwargs: Additional arguments passed to the agent
            
        Returns:
            If stream=False: The response string
            If stream=True: AsyncIterator yielding response chunks
            
        Raises:
            RuntimeError: If chat is in an invalid state
            ValueError: If input validation fails
            Exception: If agent execution fails after retries
            
        Example:
            ```python
            # Blocking response
            response = await chat.invoke("Hello!")
            print(response)
            
            # Streaming response
            async for chunk in chat.invoke("Tell me a story", stream=True):
                print(chunk, end='', flush=True)
            
            # With context
            response = await chat.invoke("Analyze this", context=["data.csv"])
            ```
        """
        # State and concurrency checks
        if not self._session_manager.can_accept_invocation():
            if self._session_manager.state == SessionState.ERROR:
                raise RuntimeError("Chat is in error state. Reset or create a new chat session.")
            else:
                current = self._session_manager._concurrent_invocations
                max_allowed = self._session_manager._max_concurrent_invocations
                raise RuntimeError(f"Maximum concurrent invocations exceeded. Current: {current}, Max allowed: {max_allowed}. Wait for current operations to complete or increase max_concurrent_invocations.")
        
        # Normalize input
        task = self._normalize_input(input_data, context)
        
        # Add user message to history
        user_message = ChatMessage(
            content=task.description,
            role="user",
            timestamp=time.time(),
            attachments=task.attachments
        )
        self._session_manager.add_message(user_message)
        
        
        # Update state and activity
        self._session_manager.start_invocation()
        self._transition_state(SessionState.STREAMING if stream else SessionState.AWAITING_RESPONSE)
        
        # Start response timer
        response_start_time = self._session_manager.start_response_timer()
        
        if stream:
            # For streaming, return the AsyncIterator directly
            return self._invoke_streaming(task, response_start_time, **kwargs)
        else:
            # For blocking, execute and return the string result
            return await self._invoke_blocking_async(task, response_start_time, **kwargs)
    
    async def _invoke_blocking_async(self, task: Task, response_start_time: float, **kwargs) -> str:
        """Handle blocking invocation."""
        async def _execute():
            # Execute agent with retry logic
            result = await self.agent.do_async(task, debug=self.debug, **kwargs)
            
            # Extract response - result is the actual output, not a RunResult
            response_text = str(result)
            
            # Get the RunResult from the agent to access messages
            run_result = self.agent.get_run_result()
            
            # Update cost tracking
            for message in run_result.new_messages():
                if isinstance(message, ModelResponse) and message.usage:
                    self._session_manager.add_usage(message.usage, self.agent.model)
            
            # Update in-memory history (only add assistant messages to avoid duplicates)
            for message in run_result.new_messages():
                # Only process ModelResponse messages (assistant responses)
                if hasattr(message, 'kind') and message.kind == 'response':
                    chat_message = ChatMessage.from_model_message(message)
                    if chat_message.role == "assistant":
                        self._session_manager.add_message(chat_message)
            
            return response_text
        
        try:
            result = await self._execute_with_retry(_execute)
            # End response timer
            self._session_manager.end_response_timer(response_start_time)
            return result
        except Exception as e:
            # End response timer even on error
            self._session_manager.end_response_timer(response_start_time)
            raise
        finally:
            # Handle state transitions
            self._session_manager.end_invocation()
            self._transition_state(SessionState.IDLE)
    
    def _invoke_streaming(self, task: Task, response_start_time: float, **kwargs) -> AsyncIterator[str]:
        """Handle streaming invocation."""
        async def _execute_streaming():
            accumulated_text = ""
            
            # Get streaming result from agent
            stream_result = await self.agent.stream_async(task, debug=self.debug, **kwargs)
            
            async with stream_result:
                async for chunk in stream_result.stream_output():
                    accumulated_text += chunk
                    yield chunk
            
            # Get final result for cost tracking and history
            final_output = stream_result.get_final_output()
            
            # Update cost tracking
            if hasattr(stream_result, 'new_messages'):
                for message in stream_result.new_messages():
                    if isinstance(message, ModelResponse) and message.usage:
                        self._session_manager.add_usage(message.usage, self.agent.model)
            
            # Update in-memory history (only add assistant messages to avoid duplicates)
            if hasattr(stream_result, 'new_messages'):
                for message in stream_result.new_messages():
                    # Only process ModelResponse messages (assistant responses)
                    if hasattr(message, 'kind') and message.kind == 'response':
                        chat_message = ChatMessage.from_model_message(message)
                        if chat_message.role == "assistant":
                            self._session_manager.add_message(chat_message)
        
        # For streaming, we can't use the retry mechanism directly since it's an async generator
        # Instead, we'll handle retries within the generator
        async def _stream_with_retry():
            last_exception = None
            stream_generator = None
            
            try:
                for attempt in range(self._retry_attempts + 1):
                    try:
                        stream_generator = _execute_streaming()
                        async for chunk in stream_generator:
                            yield chunk
                        return  # Success, exit retry loop
                    except Exception as e:
                        last_exception = e
                        # Clean up the previous generator if it exists
                        if stream_generator:
                            try:
                                await stream_generator.aclose()
                            except:
                                pass
                            stream_generator = None
                        
                        # Check if it's a context manager error that we can handle
                        if "context manager is already active" in str(e):
                            if self.debug:
                                from upsonic.utils.printing import debug_log
                                debug_log(f"Streaming context manager conflict on attempt {attempt + 1}. Waiting longer...", "Chat")
                            await asyncio.sleep(self._retry_delay * (3 ** attempt))  # Longer wait for context conflicts
                        elif attempt < self._retry_attempts:
                            if self.debug:
                                from upsonic.utils.printing import debug_log
                                debug_log(f"Streaming attempt {attempt + 1} failed: {e}. Retrying in {self._retry_delay}s...", "Chat")
                            await asyncio.sleep(self._retry_delay * (2 ** attempt))
                        else:
                            if self.debug:
                                from upsonic.utils.printing import debug_log
                                debug_log(f"All {self._retry_attempts + 1} streaming attempts failed. Last error: {e}", "Chat")
                            raise last_exception
            finally:
                # Clean up the generator if it still exists
                if stream_generator:
                    try:
                        await stream_generator.aclose()
                    except:
                        pass
                
                # End response timer and clean up state when streaming is done
                self._session_manager.end_response_timer(response_start_time)
                self._session_manager.end_invocation()
                self._transition_state(SessionState.IDLE)
        
        return _stream_with_retry()
    
    def stream(
        self,
        input_data: Union[str, Task],
        *,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream a response from the chat.
        
        This is a dedicated streaming method that returns an async iterator.
        
        Args:
            input_data: The message content (string) or Task object
            context: Optional list of file paths to attach
            **kwargs: Additional arguments passed to the agent
            
        Returns:
            AsyncIterator yielding response chunks
            
        Example:
            ```python
            async for chunk in chat.stream("Tell me a story"):
                print(chunk, end='', flush=True)
            ```
        """
        # Normalize input
        task = self._normalize_input(input_data, context)
        
        # Add user message to history
        user_message = ChatMessage(
            content=task.description,
            role="user",
            timestamp=time.time(),
            attachments=task.attachments
        )
        self._session_manager.add_message(user_message)
        
        
        # Update state and activity
        self._session_manager.start_invocation()
        self._transition_state(SessionState.STREAMING)
        
        # Start response timer
        response_start_time = self._session_manager.start_response_timer()
        
        return self._invoke_streaming(task, response_start_time, **kwargs)
    
    def clear_history(self) -> None:
        """Clear the in-memory chat history."""
        self._session_manager.clear_history()
    
    def reset_session(self) -> None:
        """Reset the chat session to initial state."""
        self._session_manager.reset_session()
    
    async def close(self) -> None:
        """Close the chat session and cleanup resources."""
        if self._storage:
            try:
                if await self._storage.is_connected_async():
                    await self._storage.disconnect_async()
            except Exception as e:
                if self.debug:
                    from upsonic.utils.printing import debug_log
                    debug_log(f"Error closing storage connection: {e}", "Chat")
        
        self._session_manager.close_session()
        if self.debug:
            from upsonic.utils.printing import debug_log
            debug_log("Chat session closed", "Chat")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def __repr__(self) -> str:
        """String representation of the chat."""
        return (
            f"Chat(session_id='{self.session_id}', user_id='{self.user_id}', "
            f"state={self.state.value}, messages={len(self.all_messages)}, "
            f"cost=${self.total_cost:.4f})"
        )
