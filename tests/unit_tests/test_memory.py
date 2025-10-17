import asyncio
import pytest
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field

from upsonic.storage.memory.memory import Memory
from upsonic.storage.base import Storage
from upsonic.storage.providers.in_memory import InMemoryStorage
from upsonic.storage.session.sessions import InteractionSession, UserProfile
from upsonic.storage.types import SessionId, UserId
from upsonic.messages.messages import (
    ModelRequest, ModelResponse, TextPart, UserPromptPart, 
    SystemPromptPart, ModelMessagesTypeAdapter, ToolCallPart, ToolReturnPart
)
from upsonic.schemas import UserTraits


class MockStorage(Storage):
    """Mock storage implementation for testing."""
    
    def __init__(self):
        super().__init__()
        self._connected = False
        self._data = {}
    
    def is_connected(self) -> bool:
        return self._connected
    
    def connect(self) -> None:
        self._connected = True
    
    def disconnect(self) -> None:
        self._connected = False
    
    def create(self) -> None:
        pass
    
    def read(self, object_id: str, model_type: Type) -> Optional[Any]:
        return self._data.get((object_id, model_type))
    
    def upsert(self, data: Any) -> None:
        if isinstance(data, InteractionSession):
            self._data[(data.session_id, InteractionSession)] = data
        elif isinstance(data, UserProfile):
            self._data[(data.user_id, UserProfile)] = data
    
    def delete(self, object_id: str, model_type: Type) -> None:
        self._data.pop((object_id, model_type), None)
    
    def drop(self) -> None:
        self._data.clear()
    
    async def is_connected_async(self) -> bool:
        return self._connected
    
    async def connect_async(self) -> None:
        self._connected = True
    
    async def disconnect_async(self) -> None:
        self._connected = False
    
    async def create_async(self) -> None:
        pass
    
    async def read_async(self, object_id: str, model_type: Type) -> Optional[Any]:
        return self._data.get((object_id, model_type))
    
    async def upsert_async(self, data: Any) -> None:
        if isinstance(data, InteractionSession):
            self._data[(data.session_id, InteractionSession)] = data
        elif isinstance(data, UserProfile):
            self._data[(data.user_id, UserProfile)] = data
    
    async def delete_async(self, object_id: str, model_type: Type) -> None:
        self._data.pop((object_id, model_type), None)
    
    async def drop_async(self) -> None:
        self._data.clear()


class MockModel:
    """Mock model for testing memory functionality."""
    
    def __init__(self, model_name: str = "test-model"):
        self.model_name = model_name
    
    async def do_async(self, task):
        """Mock agent execution."""
        mock_result = Mock()
        mock_result.output = "Mocked response"
        return mock_result


class MockRunResult:
    """Mock run result for testing."""
    
    def __init__(self, messages: List[Any] = None):
        self._messages = messages or []
    
    def new_messages(self) -> List[Any]:
        return self._messages


class TestMemoryInitialization:
    """Test Memory class initialization and configuration."""
    
    def test_memory_init_basic(self):
        """Test basic memory initialization."""
        storage = MockStorage()
        memory = Memory(storage)
        
        assert memory.storage == storage
        assert memory.full_session_memory_enabled is False
        assert memory.summary_memory_enabled is False
        assert memory.user_analysis_memory_enabled is False
        assert memory.session_id is None
        assert memory.user_id is None
        assert memory.num_last_messages is None
        assert memory.model is None
        assert memory.debug is False
        assert memory.feed_tool_call_results is False
        assert memory.profile_schema_model == UserTraits
        assert memory.is_profile_dynamic is False
        assert memory.user_memory_mode == 'update'
    
    def test_memory_init_with_session_id(self):
        """Test memory initialization with session ID."""
        storage = MockStorage()
        memory = Memory(
            storage=storage,
            session_id="test-session",
            full_session_memory=True
        )
        
        assert memory.session_id == SessionId("test-session")
        assert memory.full_session_memory_enabled is True
    
    def test_memory_init_with_user_id(self):
        """Test memory initialization with user ID."""
        storage = MockStorage()
        memory = Memory(
            storage=storage,
            user_id="test-user",
            user_analysis_memory=True
        )
        
        assert memory.user_id == UserId("test-user")
        assert memory.user_analysis_memory_enabled is True
    
    def test_memory_init_session_id_required_error(self):
        """Test that session_id is required for session memory."""
        storage = MockStorage()
        
        with pytest.raises(ValueError, match="session_id.*required"):
            Memory(
                storage=storage,
                full_session_memory=True
            )
    
    def test_memory_init_user_id_required_error(self):
        """Test that user_id is required for user analysis memory."""
        storage = MockStorage()
        
        with pytest.raises(ValueError, match="user_id.*required"):
            Memory(
                storage=storage,
                user_analysis_memory=True
            )
    
    def test_memory_init_with_custom_schema(self):
        """Test memory initialization with custom user profile schema."""
        storage = MockStorage()
        
        class CustomSchema(BaseModel):
            name: str
            age: int
        
        memory = Memory(
            storage=storage,
            user_profile_schema=CustomSchema
        )
        
        assert memory.profile_schema_model == CustomSchema
    
    def test_memory_init_dynamic_profile(self):
        """Test memory initialization with dynamic user profile."""
        storage = MockStorage()
        
        class CustomSchema(BaseModel):
            name: str
        
        memory = Memory(
            storage=storage,
            user_profile_schema=CustomSchema,
            dynamic_user_profile=True
        )
        
        assert memory.is_profile_dynamic is True
        assert memory.profile_schema_model is None
    
    def test_memory_init_with_model(self):
        """Test memory initialization with model provider."""
        storage = MockStorage()
        model = MockModel()
        
        memory = Memory(
            storage=storage,
            model=model
        )
        
        assert memory.model == model
    
    def test_memory_init_with_all_options(self):
        """Test memory initialization with all options."""
        storage = MockStorage()
        model = MockModel()
        
        class CustomSchema(BaseModel):
            name: str
        
        memory = Memory(
            storage=storage,
            session_id="test-session",
            user_id="test-user",
            full_session_memory=True,
            summary_memory=True,
            user_analysis_memory=True,
            user_profile_schema=CustomSchema,
            dynamic_user_profile=False,
            num_last_messages=10,
            model=model,
            debug=True,
            feed_tool_call_results=True,
            user_memory_mode='replace'
        )
        
        assert memory.session_id == SessionId("test-session")
        assert memory.user_id == UserId("test-user")
        assert memory.full_session_memory_enabled is True
        assert memory.summary_memory_enabled is True
        assert memory.user_analysis_memory_enabled is True
        assert memory.profile_schema_model == CustomSchema
        assert memory.is_profile_dynamic is False
        assert memory.num_last_messages == 10
        assert memory.model == model
        assert memory.debug is True
        assert memory.feed_tool_call_results is True
        assert memory.user_memory_mode == 'replace'


class TestMemoryPrepareInputs:
    """Test prepare_inputs_for_task method."""
    
    @pytest.fixture
    def storage(self):
        """Create mock storage."""
        return MockStorage()
    
    @pytest.fixture
    def memory(self, storage):
        """Create memory instance."""
        return Memory(storage)
    
    @pytest.mark.asyncio
    async def test_prepare_inputs_basic(self, memory):
        """Test basic prepare_inputs_for_task."""
        inputs = await memory.prepare_inputs_for_task()
        
        assert isinstance(inputs, dict)
        assert "message_history" in inputs
        assert "context_injection" in inputs
        assert "system_prompt_injection" in inputs
        assert inputs["message_history"] == []
        assert inputs["context_injection"] == ""
        assert inputs["system_prompt_injection"] == ""
    
    @pytest.mark.asyncio
    async def test_prepare_inputs_with_user_profile(self, storage):
        """Test prepare_inputs_for_task with user profile."""
        # Create user profile
        profile = UserProfile(
            user_id=UserId("test-user"),
            profile_data={"name": "John", "age": 30}
        )
        storage._data[(UserId("test-user"), UserProfile)] = profile
        
        memory = Memory(
            storage=storage,
            user_id="test-user",
            user_analysis_memory=True
        )
        
        inputs = await memory.prepare_inputs_for_task()
        
        assert inputs["system_prompt_injection"] == "<UserProfile>\n- name: John\n- age: 30\n</UserProfile>"
    
    @pytest.mark.asyncio
    async def test_prepare_inputs_with_session_summary(self, storage):
        """Test prepare_inputs_for_task with session summary."""
        # Create session with summary
        session = InteractionSession(
            session_id=SessionId("test-session"),
            summary="Previous conversation summary"
        )
        storage._data[(SessionId("test-session"), InteractionSession)] = session
        
        memory = Memory(
            storage=storage,
            session_id="test-session",
            summary_memory=True
        )
        
        inputs = await memory.prepare_inputs_for_task()
        
        assert inputs["context_injection"] == "<SessionSummary>\nPrevious conversation summary\n</SessionSummary>"
    
    @pytest.mark.asyncio
    async def test_prepare_inputs_with_chat_history(self, storage):
        """Test prepare_inputs_for_task with chat history."""
        # Create session with chat history in the format expected by ModelMessagesTypeAdapter
        chat_history = [
            {
                "kind": "request",
                "parts": [
                    {"part_kind": "system-prompt", "content": "System prompt"},
                    {"part_kind": "user-prompt", "content": "User message"}
                ]
            },
            {
                "kind": "response", 
                "parts": [
                    {"part_kind": "text", "content": "Assistant response"}
                ]
            }
        ]

        session = InteractionSession(
            session_id=SessionId("test-session"),
            chat_history=chat_history
        )
        storage._data[(SessionId("test-session"), InteractionSession)] = session

        memory = Memory(
            storage=storage,
            session_id="test-session",
            full_session_memory=True
        )

        inputs = await memory.prepare_inputs_for_task()

        assert len(inputs["message_history"]) == 2
        assert isinstance(inputs["message_history"][0], ModelRequest)
        assert isinstance(inputs["message_history"][1], ModelResponse)
    
    @pytest.mark.asyncio
    async def test_prepare_inputs_with_tool_call_filtering(self, storage):
        """Test prepare_inputs_for_task with tool call filtering."""
        # Create session with tool calls
        chat_history = [
            {
                "parts": [
                    {"part_kind": "user-prompt", "content": "User message"}
                ]
            },
            {
                "parts": [
                    {"part_kind": "tool-call", "tool_name": "test_tool"},
                    {"part_kind": "tool-return", "content": "Tool result"}
                ]
            }
        ]
        
        session = InteractionSession(
            session_id=SessionId("test-session"),
            chat_history=chat_history
        )
        storage._data[(SessionId("test-session"), InteractionSession)] = session
        
        memory = Memory(
            storage=storage,
            session_id="test-session",
            full_session_memory=True,
            feed_tool_call_results=False
        )
        
        inputs = await memory.prepare_inputs_for_task()
        
        # Tool calls should be filtered out
        assert len(inputs["message_history"]) == 0
    
    @pytest.mark.asyncio
    async def test_prepare_inputs_with_tool_calls_included(self, storage):
        """Test prepare_inputs_for_task with tool calls included."""
        # Create session with chat history in the format expected by ModelMessagesTypeAdapter
        chat_history = [
            {
                "kind": "request",
                "parts": [
                    {"part_kind": "user-prompt", "content": "User message"}
                ]
            },
            {
                "kind": "response",
                "parts": [
                    {"part_kind": "tool-call", "tool_name": "test_tool", "tool_call_id": "call_1", "args": {}}
                ]
            },
            {
                "kind": "request",
                "parts": [
                    {"part_kind": "tool-return", "tool_name": "test_tool", "tool_call_id": "call_1", "content": "Tool result"}
                ]
            }
        ]

        session = InteractionSession(
            session_id=SessionId("test-session"),
            chat_history=chat_history
        )
        storage._data[(SessionId("test-session"), InteractionSession)] = session

        memory = Memory(
            storage=storage,
            session_id="test-session",
            full_session_memory=True,
            feed_tool_call_results=True
        )

        inputs = await memory.prepare_inputs_for_task()

        # Tool calls should be included
        assert len(inputs["message_history"]) == 3
    
    @pytest.mark.asyncio
    async def test_prepare_inputs_invalid_history_handling(self, storage):
        """Test prepare_inputs_for_task with invalid chat history."""
        # Create session with invalid chat history (wrong format)
        session = InteractionSession(
            session_id=SessionId("test-session"),
            chat_history=[{"invalid": "format"}]  # This will fail validation
        )
        storage._data[(SessionId("test-session"), InteractionSession)] = session
        
        memory = Memory(
            storage=storage,
            session_id="test-session",
            full_session_memory=True
        )
        
        inputs = await memory.prepare_inputs_for_task()
        
        # Should handle invalid history gracefully
        assert inputs["message_history"] == []


class TestMemoryUpdateMemories:
    """Test update_memories_after_task method."""
    
    @pytest.fixture
    def storage(self):
        """Create mock storage."""
        return MockStorage()
    
    @pytest.fixture
    def memory(self, storage):
        """Create memory instance."""
        return Memory(storage)
    
    @pytest.fixture
    def mock_run_result(self):
        """Create mock run result."""
        return MockRunResult([
            ModelRequest(parts=[UserPromptPart(content="Test message")]),
            ModelResponse(parts=[TextPart(content="Test response")])
        ])
    
    @pytest.mark.asyncio
    async def test_update_memories_basic(self, memory, mock_run_result):
        """Test basic update_memories_after_task."""
        # Should not raise any errors
        await memory.update_memories_after_task(mock_run_result)
    
    @pytest.mark.asyncio
    async def test_update_memories_with_session(self, storage, mock_run_result):
        """Test update_memories_after_task with session memory."""
        memory = Memory(
            storage=storage,
            session_id="test-session",
            full_session_memory=True
        )
        
        await memory.update_memories_after_task(mock_run_result)
        
        # Check that session was created/updated
        session = await storage.read_async(SessionId("test-session"), InteractionSession)
        assert session is not None
        assert len(session.chat_history) == 2
    
    @pytest.mark.asyncio
    async def test_update_memories_with_summary(self, storage, mock_run_result):
        """Test update_memories_after_task with summary memory."""
        memory = Memory(
            storage=storage,
            session_id="test-session",
            summary_memory=True,
            model=MockModel()
        )
        
        with patch('upsonic.agent.agent.Agent') as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.do_async.return_value = Mock(output="Generated summary")
            mock_agent_class.return_value = mock_agent
            
            await memory.update_memories_after_task(mock_run_result)
        
        # Check that session was created/updated with summary
        session = await storage.read_async(SessionId("test-session"), InteractionSession)
        assert session is not None
        assert session.summary is not None
    
    @pytest.mark.asyncio
    async def test_update_memories_with_user_profile(self, storage, mock_run_result):
        """Test update_memories_after_task with user profile analysis."""
        memory = Memory(
            storage=storage,
            user_id="test-user",
            user_analysis_memory=True,
            model=MockModel()
        )
        
        with patch('upsonic.agent.agent.Agent') as mock_agent_class:
            mock_agent = AsyncMock()
            # Create a proper mock response with model_dump method
            mock_output = Mock()
            mock_output.model_dump.return_value = {"name": "John", "age": 30}
            mock_agent.do_async.return_value = mock_output
            mock_agent_class.return_value = mock_agent

            await memory.update_memories_after_task(mock_run_result)

        # Check that user profile was created/updated
        profile = await storage.read_async(UserId("test-user"), UserProfile)
        assert profile is not None
        assert "name" in profile.profile_data
        assert "age" in profile.profile_data
    
    @pytest.mark.asyncio
    async def test_update_memories_user_memory_mode_replace(self, storage, mock_run_result):
        """Test update_memories_after_task with replace mode."""
        # Create existing profile
        existing_profile = UserProfile(
            user_id=UserId("test-user"),
            profile_data={"old_key": "old_value"}
        )
        storage._data[(UserId("test-user"), UserProfile)] = existing_profile
        
        memory = Memory(
            storage=storage,
            user_id="test-user",
            user_analysis_memory=True,
            model=MockModel(),
            user_memory_mode='replace'
        )
        
        with patch('upsonic.agent.agent.Agent') as mock_agent_class:
            mock_agent = AsyncMock()
            # Create a proper mock response with model_dump method
            mock_output = Mock()
            mock_output.model_dump.return_value = {"name": "John"}
            mock_agent.do_async.return_value = mock_output
            mock_agent_class.return_value = mock_agent

            await memory.update_memories_after_task(mock_run_result)

        # Check that profile was replaced
        profile = await storage.read_async(UserId("test-user"), UserProfile)
        assert profile is not None
        assert "old_key" not in profile.profile_data
        assert "name" in profile.profile_data
    
    @pytest.mark.asyncio
    async def test_update_memories_user_memory_mode_update(self, storage, mock_run_result):
        """Test update_memories_after_task with update mode."""
        # Create existing profile
        existing_profile = UserProfile(
            user_id=UserId("test-user"),
            profile_data={"old_key": "old_value"}
        )
        storage._data[(UserId("test-user"), UserProfile)] = existing_profile
        
        memory = Memory(
            storage=storage,
            user_id="test-user",
            user_analysis_memory=True,
            model=MockModel(),
            user_memory_mode='update'
        )
        
        with patch('upsonic.agent.agent.Agent') as mock_agent_class:
            mock_agent = AsyncMock()
            # Create a proper mock response with model_dump method
            mock_output = Mock()
            mock_output.model_dump.return_value = {"name": "John"}
            mock_agent.do_async.return_value = mock_output
            mock_agent_class.return_value = mock_agent

            await memory.update_memories_after_task(mock_run_result)

        # Check that profile was updated (merged)
        profile = await storage.read_async(UserId("test-user"), UserProfile)
        assert profile is not None
        assert "old_key" in profile.profile_data
        assert "name" in profile.profile_data


class TestMemoryMessageHistoryLimiting:
    """Test message history limiting functionality."""
    
    @pytest.fixture
    def storage(self):
        """Create mock storage."""
        return MockStorage()
    
    def test_limit_message_history_no_limit(self, storage):
        """Test message history limiting with no limit."""
        memory = Memory(storage)
        
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Message 1")]),
            ModelResponse(parts=[TextPart(content="Response 1")]),
            ModelRequest(parts=[UserPromptPart(content="Message 2")]),
            ModelResponse(parts=[TextPart(content="Response 2")])
        ]
        
        limited = memory._limit_message_history(messages)
        
        assert limited == messages
    
    def test_limit_message_history_with_limit(self, storage):
        """Test message history limiting with limit."""
        memory = Memory(storage, num_last_messages=2)
        
        messages = [
            ModelRequest(parts=[SystemPromptPart(content="System"), UserPromptPart(content="Message 1")]),
            ModelResponse(parts=[TextPart(content="Response 1")]),
            ModelRequest(parts=[UserPromptPart(content="Message 2")]),
            ModelResponse(parts=[TextPart(content="Response 2")]),
            ModelRequest(parts=[UserPromptPart(content="Message 3")]),
            ModelResponse(parts=[TextPart(content="Response 3")])
        ]
        
        limited = memory._limit_message_history(messages)
        
        # Should keep last 2 runs (4 messages total)
        assert len(limited) == 4
        assert isinstance(limited[0], ModelRequest)
        assert isinstance(limited[0].parts[0], SystemPromptPart)
        assert isinstance(limited[0].parts[1], UserPromptPart)
        assert limited[0].parts[1].content == "Message 2"
    
    def test_limit_message_history_empty(self, storage):
        """Test message history limiting with empty history."""
        memory = Memory(storage, num_last_messages=2)
        
        limited = memory._limit_message_history([])
        
        assert limited == []
    
    def test_limit_message_history_less_than_limit(self, storage):
        """Test message history limiting with fewer messages than limit."""
        memory = Memory(storage, num_last_messages=5)
        
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Message 1")]),
            ModelResponse(parts=[TextPart(content="Response 1")])
        ]
        
        limited = memory._limit_message_history(messages)
        
        assert limited == messages
    
    def test_limit_message_history_malformed(self, storage):
        """Test message history limiting with malformed history."""
        memory = Memory(storage, num_last_messages=2)
        
        # Malformed history without proper system prompt
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Message 1")]),
            ModelResponse(parts=[TextPart(content="Response 1")])
        ]
        
        limited = memory._limit_message_history(messages)
        
        # Should return original messages when malformed
        assert limited == messages


class TestMemoryUserProfileAnalysis:
    """Test user profile analysis functionality."""
    
    @pytest.fixture
    def storage(self):
        """Create mock storage."""
        return MockStorage()
    
    @pytest.fixture
    def memory(self, storage):
        """Create memory instance."""
        return Memory(
            storage=storage,
            user_id="test-user",
            user_analysis_memory=True,
            model=MockModel()
        )
    
    def test_extract_user_prompt_content(self, memory):
        """Test extracting user prompt content from messages."""
        messages = [
            ModelRequest(parts=[
                SystemPromptPart(content="System prompt"),
                UserPromptPart(content="User message 1")
            ]),
            ModelResponse(parts=[TextPart(content="Response")]),
            ModelRequest(parts=[UserPromptPart(content="User message 2")])
        ]
        
        prompts = memory._extract_user_prompt_content(messages)
        
        assert prompts == ["User message 1", "User message 2"]
    
    def test_extract_user_prompt_content_empty(self, memory):
        """Test extracting user prompt content from empty messages."""
        prompts = memory._extract_user_prompt_content([])
        
        assert prompts == []
    
    def test_extract_user_prompt_content_no_user_prompts(self, memory):
        """Test extracting user prompt content with no user prompts."""
        messages = [
            ModelRequest(parts=[SystemPromptPart(content="System prompt")]),
            ModelResponse(parts=[TextPart(content="Response")])
        ]
        
        prompts = memory._extract_user_prompt_content(messages)
        
        assert prompts == []
    
    @pytest.mark.asyncio
    async def test_analyze_interaction_for_traits_no_model(self, storage):
        """Test user trait analysis without model provider."""
        memory = Memory(
            storage=storage,
            user_id="test-user",
            user_analysis_memory=True
        )
        
        mock_result = MockRunResult()
        
        with pytest.raises(ValueError, match="model must be configured"):
            await memory._analyze_interaction_for_traits({}, mock_result)
    
    @pytest.mark.asyncio
    async def test_analyze_interaction_for_traits_no_prompts(self, memory):
        """Test user trait analysis with no user prompts."""
        mock_result = MockRunResult()
        
        traits = await memory._analyze_interaction_for_traits({}, mock_result)
        
        assert traits == {}
    
    @pytest.mark.asyncio
    async def test_analyze_interaction_for_traits_with_prompts(self, memory):
        """Test user trait analysis with user prompts."""
        # Create session with proper message objects
        session = InteractionSession(
            session_id=SessionId("test-session"),
            chat_history=[
                ModelRequest(parts=[
                    UserPromptPart(content="I love programming")
                ])
            ]
        )
        memory.storage._data[(SessionId("test-session"), InteractionSession)] = session
        memory.session_id = SessionId("test-session")
        
        mock_result = MockRunResult([
            ModelRequest(parts=[UserPromptPart(content="I use Python daily")])
        ])
        
        with patch('upsonic.agent.agent.Agent') as mock_agent_class:
            mock_agent = AsyncMock()
            # Create a proper mock response with model_dump method
            mock_output = Mock()
            mock_output.model_dump.return_value = {"programming_language": "Python"}
            mock_agent.do_async.return_value = mock_output
            mock_agent_class.return_value = mock_agent

            traits = await memory._analyze_interaction_for_traits({}, mock_result)

        assert "programming_language" in traits
        assert traits["programming_language"] == "Python"


class TestMemoryDynamicProfile:
    """Test dynamic user profile functionality."""
    
    @pytest.fixture
    def storage(self):
        """Create mock storage."""
        return MockStorage()
    
    @pytest.fixture
    def dynamic_memory(self, storage):
        """Create memory instance with dynamic profile."""
        return Memory(
            storage=storage,
            user_id="test-user",
            user_analysis_memory=True,
            model=MockModel(),
            dynamic_user_profile=True
        )
    
    @pytest.mark.asyncio
    async def test_dynamic_profile_schema_generation(self, dynamic_memory):
        """Test dynamic profile schema generation."""
        mock_result = MockRunResult([
            ModelRequest(parts=[UserPromptPart(content="I'm a software engineer")])
        ])
        
        with patch('upsonic.agent.agent.Agent') as mock_agent_class:
            mock_agent = AsyncMock()
            
            # Mock schema generation
            from upsonic.storage.memory.memory import Memory
            from pydantic import BaseModel, Field
            from typing import List
            
            class FieldDefinition(BaseModel):
                name: str = Field(..., description="Snake_case field name")
                description: str = Field(..., description="Description of what this field represents")
            
            class ProposedSchema(BaseModel):
                fields: List[FieldDefinition] = Field(..., min_length=2, description="List of 2-5 field definitions extracted from the conversation")
            
            schema_response = ProposedSchema(fields=[
                FieldDefinition(name="profession", description="User's profession"),
                FieldDefinition(name="experience_level", description="Years of experience")
            ])
            
            # Mock trait extraction
            trait_response = Mock()
            trait_response.model_dump.return_value = {
                "profession": "software_engineer",
                "experience_level": "senior"
            }
            
            mock_agent.do_async.side_effect = [schema_response, trait_response]
            mock_agent_class.return_value = mock_agent
            
            traits = await dynamic_memory._analyze_interaction_for_traits({}, mock_result)
        
        assert "profession" in traits
        assert "experience_level" in traits
        assert traits["profession"] == "software_engineer"
        assert traits["experience_level"] == "senior"


class TestMemoryErrorHandling:
    """Test error handling in Memory class."""
    
    @pytest.fixture
    def storage(self):
        """Create mock storage."""
        return MockStorage()
    
    @pytest.mark.asyncio
    async def test_prepare_inputs_storage_error(self, storage):
        """Test prepare_inputs_for_task with storage error."""
        # Mock storage to raise error
        storage.read_async = AsyncMock(side_effect=Exception("Storage error"))
        
        memory = Memory(
            storage=storage,
            session_id="test-session",
            full_session_memory=True
        )
        
        # Should raise the storage error since it's not caught
        with pytest.raises(Exception, match="Storage error"):
            await memory.prepare_inputs_for_task()
    
    @pytest.mark.asyncio
    async def test_update_memories_summary_error(self, storage):
        """Test update_memories_after_task with summary generation error."""
        memory = Memory(
            storage=storage,
            session_id="test-session",
            summary_memory=True,
            model=MockModel()
        )
        
        mock_result = MockRunResult()
        
        with patch('upsonic.agent.agent.Agent') as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.do_async.side_effect = Exception("Summary error")
            mock_agent_class.return_value = mock_agent
            
            # Should handle summary error gracefully
            await memory.update_memories_after_task(mock_result)
    
    @pytest.mark.asyncio
    async def test_update_memories_profile_error(self, storage):
        """Test update_memories_after_task with profile analysis error."""
        memory = Memory(
            storage=storage,
            user_id="test-user",
            user_analysis_memory=True,
            model=MockModel()
        )
        
        mock_result = MockRunResult()
        
        with patch('upsonic.agent.agent.Agent') as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.do_async.side_effect = Exception("Profile error")
            mock_agent_class.return_value = mock_agent
            
            # Should handle profile error gracefully
            await memory.update_memories_after_task(mock_result)


class TestMemoryIntegration:
    """Integration tests for Memory class."""
    
    @pytest.fixture
    def storage(self):
        """Create mock storage."""
        return MockStorage()
    
    @pytest.fixture
    def full_memory(self, storage):
        """Create memory instance with all features enabled."""
        return Memory(
            storage=storage,
            session_id="integration-session",
            user_id="integration-user",
            full_session_memory=True,
            summary_memory=True,
            user_analysis_memory=True,
            num_last_messages=3,
            model=MockModel(),
            debug=True,
            feed_tool_call_results=True
        )
    
    @pytest.mark.asyncio
    async def test_full_memory_workflow(self, full_memory):
        """Test complete memory workflow."""
        # First, prepare inputs
        inputs = await full_memory.prepare_inputs_for_task()
        
        assert "message_history" in inputs
        assert "context_injection" in inputs
        assert "system_prompt_injection" in inputs
        
        # Create mock run result
        mock_result = MockRunResult([
            ModelRequest(parts=[UserPromptPart(content="Hello, I'm John")]),
            ModelResponse(parts=[TextPart(content="Nice to meet you, John!")])
        ])
        
        # Update memories
        with patch('upsonic.agent.agent.Agent') as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent.do_async.return_value = Mock(output="Updated summary")
            mock_agent_class.return_value = mock_agent
            
            await full_memory.update_memories_after_task(mock_result)
        
        # Verify session was created
        session = await full_memory.storage.read_async(
            SessionId("integration-session"), 
            InteractionSession
        )
        assert session is not None
        assert len(session.chat_history) == 2
        
        # Verify user profile was created
        profile = await full_memory.storage.read_async(
            UserId("integration-user"), 
            UserProfile
        )
        assert profile is not None
    
    @pytest.mark.asyncio
    async def test_memory_with_multiple_sessions(self, storage):
        """Test memory with multiple sessions."""
        memory1 = Memory(
            storage=storage,
            session_id="session-1",
            full_session_memory=True
        )
        
        memory2 = Memory(
            storage=storage,
            session_id="session-2",
            full_session_memory=True
        )
        
        # Update memory1
        result1 = MockRunResult([
            ModelRequest(parts=[UserPromptPart(content="Session 1 message")])
        ])
        await memory1.update_memories_after_task(result1)
        
        # Update memory2
        result2 = MockRunResult([
            ModelRequest(parts=[UserPromptPart(content="Session 2 message")])
        ])
        await memory2.update_memories_after_task(result2)
        
        # Verify both sessions exist
        session1 = await storage.read_async(SessionId("session-1"), InteractionSession)
        session2 = await storage.read_async(SessionId("session-2"), InteractionSession)
        
        assert session1 is not None
        assert session2 is not None
        assert session1.session_id != session2.session_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
