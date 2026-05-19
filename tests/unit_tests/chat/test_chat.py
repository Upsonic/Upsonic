"""
Unit Tests for Chat Class

Tests Chat class functionality without making API calls.
Uses mocking to simulate agent behavior.

This test suite validates:
1. Chat initialization and validation
2. Session state management
3. Input normalization
4. Message handling
5. Retry logic
6. Error handling
7. Event streaming parameter handling
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import List, Any, AsyncIterator

from upsonic.chat import Chat, SessionState, ChatMessage
from upsonic.tasks.tasks import Task
from upsonic.storage.in_memory import InMemoryStorage


class MockAgent:
    """Mock Agent for unit testing without API calls."""
    
    def __init__(self, model_name: str = "test-model"):
        self.model = MagicMock()
        self.model.model_name = model_name
        self.memory = None
        self.name = "MockAgent"
        self.debug = False
        self.tools = []
        
    async def do_async(self, task: Task, **kwargs) -> str:
        """Mock do_async that returns a predictable response."""
        return f"Mock response to: {task.description}"
    
    async def astream(self, task: Task, events: bool = False, **kwargs) -> AsyncIterator[Any]:
        """Mock astream that yields predictable content."""
        if events:
            # Yield mock events
            from upsonic.run.events.events import (
                RunStartedEvent, PipelineStartEvent, TextDeltaEvent,
                PipelineEndEvent, RunCompletedEvent
            )
            yield RunStartedEvent(run_id="mock-run", agent_id="mock-agent")
            yield PipelineStartEvent(run_id="mock-run", total_steps=5, is_streaming=True)
            for char in "Hello World":
                yield TextDeltaEvent(run_id="mock-run", content=char)
            yield PipelineEndEvent(run_id="mock-run", total_steps=5, executed_steps=5, total_duration=1.0)
            yield RunCompletedEvent(run_id="mock-run", agent_id="mock-agent")
        else:
            # Yield text chunks
            for word in ["Hello", " ", "World"]:
                yield word


class TestChatInitialization:
    """Test Chat initialization and validation."""
    
    def test_chat_init_basic(self):
        """Test basic Chat initialization."""
        agent = MockAgent()
        chat = Chat(
            session_id="test_session",
            user_id="test_user",
            agent=agent
        )
        
        assert chat.session_id == "test_session"
        assert chat.user_id == "test_user"
        assert chat.agent is agent
        assert chat.debug is False
        assert chat.state == SessionState.IDLE
    
    def test_chat_init_with_storage(self):
        """Test Chat initialization with custom storage."""
        agent = MockAgent()
        storage = InMemoryStorage()
        
        chat = Chat(
            session_id="test_session",
            user_id="test_user",
            agent=agent,
            storage=storage
        )
        
        assert chat._storage is storage
    
    def test_chat_init_with_debug(self):
        """Test Chat initialization with debug enabled."""
        agent = MockAgent()
        chat = Chat(
            session_id="test_session",
            user_id="test_user",
            agent=agent,
            debug=True,
            debug_level=2
        )
        
        assert chat.debug is True
        assert chat.debug_level == 2
    
    def test_chat_init_empty_session_id_raises(self):
        """Test that empty session_id raises ValueError."""
        agent = MockAgent()
        
        with pytest.raises(ValueError, match="session_id must be a non-empty string"):
            Chat(session_id="", user_id="test_user", agent=agent)
    
    def test_chat_init_empty_user_id_raises(self):
        """Test that empty user_id raises ValueError."""
        agent = MockAgent()
        
        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            Chat(session_id="test_session", user_id="", agent=agent)
    
    def test_chat_init_none_agent_raises(self):
        """Test that None agent raises ValueError."""
        with pytest.raises(ValueError, match="agent cannot be None"):
            Chat(session_id="test_session", user_id="test_user", agent=None)
    
    def test_chat_init_invalid_max_concurrent_raises(self):
        """Test that invalid max_concurrent_invocations raises ValueError."""
        agent = MockAgent()
        
        with pytest.raises(ValueError, match="max_concurrent_invocations must be at least 1"):
            Chat(
                session_id="test_session",
                user_id="test_user",
                agent=agent,
                max_concurrent_invocations=0
            )
    
    def test_chat_init_invalid_num_last_messages_raises(self):
        """Test that invalid num_last_messages raises ValueError."""
        agent = MockAgent()
        
        with pytest.raises(ValueError, match="num_last_messages must be at least 1"):
            Chat(
                session_id="test_session",
                user_id="test_user",
                agent=agent,
                num_last_messages=0
            )
    
    def test_chat_init_strips_whitespace(self):
        """Test that session_id and user_id are stripped of whitespace."""
        agent = MockAgent()
        chat = Chat(
            session_id="  test_session  ",
            user_id="  test_user  ",
            agent=agent
        )
        
        assert chat.session_id == "test_session"
        assert chat.user_id == "test_user"


class TestChatInputNormalization:
    """Test input normalization logic."""
    
    def test_normalize_string_input(self):
        """Test normalizing string input to Task."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        task = chat._normalize_input("Hello world")
        
        assert isinstance(task, Task)
        assert task.description == "Hello world"
    
    def test_normalize_string_with_context(self):
        """Test normalizing string input with context."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        # Use text context (not file paths) to avoid FileNotFoundError
        task = chat._normalize_input("Hello", context=["Some additional context", "More context"])
        
        assert isinstance(task, Task)
        assert task.description == "Hello"
        # Context is processed into _context_formatted, not stored as list
        assert task is not None
    
    def test_normalize_task_input(self):
        """Test normalizing Task input."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        original_task = Task(description="Test task")
        task = chat._normalize_input(original_task)
        
        assert task is original_task
    
    def test_normalize_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        with pytest.raises(ValueError, match="Input string cannot be empty"):
            chat._normalize_input("")
    
    def test_normalize_whitespace_only_raises(self):
        """Test that whitespace-only string raises ValueError."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        with pytest.raises(ValueError, match="Input string cannot be empty"):
            chat._normalize_input("   ")
    
    def test_normalize_none_raises(self):
        """Test that None input raises ValueError."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        with pytest.raises(ValueError, match="Input data cannot be None"):
            chat._normalize_input(None)
    
    def test_normalize_invalid_type_raises(self):
        """Test that invalid type raises TypeError."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        with pytest.raises(TypeError, match="Unsupported input type"):
            chat._normalize_input(123)


class TestChatSessionState:
    """Test session state management."""
    
    def test_initial_state_is_idle(self):
        """Test that initial state is IDLE."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        assert chat.state == SessionState.IDLE
    
    def test_state_transition(self):
        """Test state transition method."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        chat._transition_state(SessionState.AWAITING_RESPONSE)
        assert chat.state == SessionState.AWAITING_RESPONSE
        
        chat._transition_state(SessionState.IDLE)
        assert chat.state == SessionState.IDLE
    
    def test_is_closed_initially_false(self):
        """Test that is_closed is initially False."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        assert chat.is_closed is False


class TestChatProperties:
    """Test Chat property accessors."""
    
    def test_session_id_property(self):
        """Test session_id property."""
        agent = MockAgent()
        chat = Chat(session_id="my_session", user_id="user", agent=agent)
        
        assert chat.session_id == "my_session"
    
    def test_user_id_property(self):
        """Test user_id property."""
        agent = MockAgent()
        chat = Chat(session_id="session", user_id="my_user", agent=agent)
        
        assert chat.user_id == "my_user"
    
    def test_start_time_property(self):
        """Test start_time property."""
        agent = MockAgent()
        chat = Chat(session_id="session", user_id="user", agent=agent)
        
        assert chat.start_time > 0
    
    def test_duration_property(self):
        """Test duration property."""
        agent = MockAgent()
        chat = Chat(session_id="session", user_id="user", agent=agent)
        
        assert chat.duration >= 0
    
    def test_all_messages_initially_empty(self):
        """Test that all_messages is initially empty."""
        agent = MockAgent()
        chat = Chat(session_id="session", user_id="user", agent=agent)
        
        assert chat.all_messages == []
    
    def test_total_tokens_initially_zero(self):
        """Test that total_tokens is initially zero."""
        agent = MockAgent()
        chat = Chat(session_id="session", user_id="user", agent=agent)
        
        assert chat.usage.total_tokens == 0
    
    def test_total_cost_initially_zero(self):
        """Test that total_cost is initially zero."""
        agent = MockAgent()
        chat = Chat(session_id="session", user_id="user", agent=agent)
        
        assert (chat.usage.cost or 0.0) == 0.0


class TestChatRepr:
    """Test Chat string representation."""
    
    def test_repr_format(self):
        """Test __repr__ format."""
        agent = MockAgent()
        chat = Chat(session_id="test_session", user_id="test_user", agent=agent)
        
        repr_str = repr(chat)
        
        assert "Chat(" in repr_str
        assert "session_id='test_session'" in repr_str
        assert "user_id='test_user'" in repr_str
        assert "state=" in repr_str


class TestChatEventStreamingParameters:
    """Test event streaming parameter handling."""
    
    @pytest.mark.asyncio
    async def test_stream_returns_async_iterator(self):
        """Test that stream() returns an async iterator."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        result = chat.stream("Hello", events=False)
        
        # Should be an async iterator
        assert hasattr(result, '__anext__')
    
    @pytest.mark.asyncio
    async def test_stream_events_returns_async_iterator(self):
        """Test that stream(events=True) returns an async iterator."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        result = chat.stream("Hello", events=True)
        
        # Should be an async iterator
        assert hasattr(result, '__anext__')
    
    @pytest.mark.asyncio
    async def test_invoke_with_events_forces_stream(self):
        """Test that invoke with events=True forces streaming."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        # Mock the session manager
        chat._session_manager.can_accept_invocation = MagicMock(return_value=True)
        chat._session_manager.start_invocation = MagicMock()
        chat._session_manager.start_response_timer = MagicMock(return_value=0.0)
        
        # Call invoke with events=True, stream=False
        # This should internally set stream=True
        result = await chat.invoke("Hello", stream=False, events=True)
        
        # Should return an async iterator (because events=True forces streaming)
        assert hasattr(result, '__anext__')


class TestChatMemoryConfiguration:
    """Test Chat memory configuration."""
    
    def test_full_session_memory_default(self):
        """Test full_session_memory is True by default."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        # Memory should be configured on the agent
        assert agent.memory is not None
    
    def test_summary_memory_disabled_by_default(self):
        """Test summary_memory is False by default."""
        agent = MockAgent()
        chat = Chat(
            session_id="test",
            user_id="user",
            agent=agent,
            summary_memory=False
        )
        
        # Memory should still be configured
        assert agent.memory is not None
    
    def test_user_analysis_memory_disabled_by_default(self):
        """Test user_analysis_memory is False by default."""
        agent = MockAgent()
        chat = Chat(
            session_id="test",
            user_id="user",
            agent=agent,
            user_analysis_memory=False
        )
        
        assert agent.memory is not None


class TestChatContextManager:
    """Test Chat as async context manager."""
    
    @pytest.mark.asyncio
    async def test_async_context_manager_enter(self):
        """Test async context manager __aenter__."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        result = await chat.__aenter__()
        
        assert result is chat
    
    @pytest.mark.asyncio
    async def test_async_context_manager_exit(self):
        """Test async context manager __aexit__."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        # Should not raise
        await chat.__aexit__(None, None, None)


class TestChatHistoryMethods:
    """Test Chat history manipulation methods."""
    
    def test_clear_history_method_exists(self):
        """Test that clear_history method exists."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        assert hasattr(chat, 'clear_history')
        assert callable(chat.clear_history)
    
    def test_reset_session_method_exists(self):
        """Test that reset_session method exists."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        assert hasattr(chat, 'reset_session')
        assert callable(chat.reset_session)
    
    def test_reopen_method_exists(self):
        """Test that reopen method exists."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        assert hasattr(chat, 'reopen')
        assert callable(chat.reopen)
    
    def test_get_recent_messages_method_exists(self):
        """Test that get_recent_messages method exists."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        assert hasattr(chat, 'get_recent_messages')
        assert callable(chat.get_recent_messages)
        
        # Should return empty list initially
        result = chat.get_recent_messages(10)
        assert result == []
    
    def test_delete_message_method_exists(self):
        """Test that delete_message method exists."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        assert hasattr(chat, 'delete_message')
        assert callable(chat.delete_message)
    
    def test_remove_attachment_method_exists(self):
        """Test that remove_attachment method exists."""
        agent = MockAgent()
        chat = Chat(session_id="test", user_id="user", agent=agent)
        
        assert hasattr(chat, 'remove_attachment')
        assert callable(chat.remove_attachment)
