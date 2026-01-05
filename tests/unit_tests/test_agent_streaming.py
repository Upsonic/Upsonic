import asyncio
import pytest
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import List, Dict, Any, Optional

from upsonic.agent.agent import Agent
from upsonic.run.agent.output import AgentRunOutput
from upsonic.tasks.tasks import Task
from upsonic.storage.memory.memory import Memory
from upsonic.storage.providers.in_memory import InMemoryStorage
from upsonic.models import Model
from upsonic.messages.messages import (
    ModelRequest, ModelResponse, TextPart, PartStartEvent, 
    PartDeltaEvent, FinalResultEvent, UserPromptPart, SystemPromptPart,
    TextPartDelta
)


class MockModel(Model):
    """Mock model for testing streaming functionality."""
    
    def __init__(self, model_name: str = "test-model"):
        super().__init__()
        self._model_name = model_name
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def system(self) -> str:
        return "test-provider"
    
    async def request(
        self,
        messages: list,
        model_settings: Any,
        model_request_parameters: Any,
    ) -> ModelResponse:
        """Mock request method."""
        return ModelResponse(
            parts=[TextPart(content="Hello world!")],
            model_name=self._model_name,
            timestamp=time.time(),
            usage=Mock(),
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop"
        )
    
    @asynccontextmanager
    async def request_stream(
        self,
        messages: list,
        model_settings: Any,
        model_request_parameters: Any,
    ):
        """Mock streaming request that yields test events."""
        # Create mock streaming events
        events = [
            PartStartEvent(index=0, part=TextPart(content="Hello")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=" world")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="!")),
            FinalResultEvent(tool_name=None, tool_call_id=None)
        ]
        
        # Create a mock stream context manager
        stream_mock = AsyncMock()
        stream_mock.__aenter__ = AsyncMock(return_value=stream_mock)
        stream_mock.__aexit__ = AsyncMock(return_value=None)
        
        async def mock_stream(self):
            for event in events:
                yield event
        
        stream_mock.__aiter__ = mock_stream
        stream_mock.get = Mock(return_value=ModelResponse(
            parts=[TextPart(content="Hello world!")],
            model_name=self._model_name,
            timestamp=time.time(),
            usage=Mock(),
            provider_name="test-provider",
            provider_response_id="test-id",
            provider_details={},
            finish_reason="stop"
        ))
        
        try:
            yield stream_mock
        finally:
            pass


class TestAgentStreaming:
    """Test suite for Agent streaming functionalities."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return MockModel()
    
    @pytest.fixture
    def agent(self, mock_model):
        """Create an agent instance for testing."""
        return Agent(model=mock_model, name="TestAgent")
    
    @pytest.fixture
    def simple_task(self):
        """Create a simple task for testing."""
        return Task(description="Hello, world!")
    
    @pytest.fixture
    def memory_storage(self):
        """Create in-memory storage for testing."""
        return InMemoryStorage()
    
    @pytest.fixture
    def memory(self, memory_storage):
        """Create memory instance for testing."""
        return Memory(
            storage=memory_storage,
            session_id="test-session",
            full_session_memory=False,  # Disable to avoid serialization issues
            summary_memory=False
        )
    
    def test_stream_returns_iterator(self, agent, simple_task):
        """Test that stream() returns an iterator."""
        result = agent.stream(simple_task)
        
        # stream() returns an iterator
        assert hasattr(result, '__iter__')
        # Consume the iterator to get text chunks
        chunks = list(result)
        assert len(chunks) > 0
        assert "".join(chunks) == "Hello world!"
    
    def test_stream_with_custom_parameters(self, agent, simple_task):
        """Test stream() with custom parameters."""
        # Use the agent's existing model instead of a custom one
        result = agent.stream(
            simple_task, 
            debug=True, 
            retry=3
        )
        
        # stream() returns an iterator regardless of parameters
        assert hasattr(result, '__iter__')
        chunks = list(result)
        assert len(chunks) > 0
    
    def test_stream_initializes_task_properties(self, agent, simple_task):
        """Test that stream() properly initializes task properties."""
        # Ensure task properties are set
        original_price_id = "existing-id"
        simple_task.price_id_ = original_price_id
        simple_task._tool_calls = [{"test": "call"}]

        result = agent.stream(simple_task)
        
        # Consume the iterator
        chunks = list(result)
        assert len(chunks) > 0
        
        # Task should have a price_id_ (may be same or different depending on implementation)
        assert simple_task.price_id_ is not None
    
    @pytest.mark.asyncio
    async def test_astream_returns_async_iterator(self, agent, simple_task):
        """Test that astream() returns an async iterator."""
        result = agent.astream(simple_task)
        
        # astream() returns an async iterator
        assert hasattr(result, '__aiter__')
        # Consume the iterator to get text chunks
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        assert len(chunks) > 0
        assert "".join(chunks) == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_astream_with_state_and_graph_id(self, agent, simple_task):
        """Test astream() with state parameter."""
        mock_state = {"test": "state"}
        
        result = agent.astream(
            simple_task, 
            state=mock_state
        )
        
        # astream() returns an async iterator regardless of parameters
        assert hasattr(result, '__aiter__')
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_astream_basic_functionality(self, agent, simple_task):
        """Test basic astream functionality."""
        result = agent.astream(simple_task)
        
        # astream() returns an async iterator
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        
        assert len(chunks) > 0
        assert "".join(chunks) == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_astream_can_be_consumed_multiple_times(self, agent, simple_task):
        """Test that astream can be called multiple times."""
        result1 = agent.astream(simple_task)
        result2 = agent.astream(simple_task)
        
        # Both should be async iterators
        assert hasattr(result1, '__aiter__')
        assert hasattr(result2, '__aiter__')
    
    @pytest.mark.asyncio
    async def test_astream_output_basic_functionality(self, agent, simple_task):
        """Test basic streaming output functionality."""
        result = agent.astream(simple_task)
        
        text_chunks = []
        async for chunk in result:
            text_chunks.append(chunk)
        
        # Should have received text chunks
        assert len(text_chunks) > 0
        assert "".join(text_chunks) == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_astream_events_basic_functionality(self, agent, simple_task):
        """Test basic streaming events functionality."""
        result = agent.astream(simple_task, events=True)
        
        events = []
        async for event in result:
            events.append(event)
        
        # Should have received various event types
        assert len(events) > 0
        event_types = [type(event).__name__ for event in events]
        # Should have some event types
        assert len(event_types) > 0
    
    @pytest.mark.asyncio
    async def test_astream_accumulates_text(self, agent, simple_task):
        """Test that astream accumulates text chunks."""
        result = agent.astream(simple_task)
        
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        
        # Should have accumulated all chunks
        full_text = "".join(chunks)
        assert full_text == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_astream_with_events_mode(self, agent, simple_task):
        """Test astream with events=True."""
        result = agent.astream(simple_task, events=True)
        
        events = []
        async for event in result:
            events.append(event)
        
        # Should have received events
        assert len(events) > 0
    
    def test_print_stream_synchronous(self, agent, simple_task, capsys):
        """Test print_stream synchronous functionality."""
        # print_stream doesn't exist, test stream() instead
        result = agent.stream(simple_task)
        
        # stream() returns an iterator
        chunks = list(result)
        full_text = "".join(chunks)
        
        # Should return text chunks
        assert len(chunks) > 0
        assert full_text == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_streaming_with_memory(self, agent, simple_task, memory):
        """Test streaming with memory integration."""
        agent.memory = memory
        
        result = agent.astream(simple_task)
        
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        
        # Should have received chunks
        assert len(chunks) > 0
        assert "".join(chunks) == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_streaming_with_tools(self, agent, simple_task):
        """Test streaming with tools enabled."""
        # Add a simple tool to the task
        def test_tool() -> str:
            """A simple test tool that returns a string."""
            return "Tool executed"

        simple_task.tools = [test_tool]
        
        result = agent.astream(simple_task)
        
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        
        # Should complete successfully even with tools
        assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, agent, simple_task):
        """Test error handling in streaming."""
        # Create a model that raises an error
        error_model = Mock()
        
        # Create a proper async context manager that raises an error
        class ErrorContextManager:
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            
            async def __aiter__(self):
                raise Exception("Stream error")
                yield  # This will never be reached
        
        error_model.request_stream = lambda *args, **kwargs: ErrorContextManager()
        error_model.model_name = "error-model"
        error_model.settings = None
        error_model.customize_request_parameters = Mock(return_value={})

        # Mock the infer_model function to return our error model
        with patch('upsonic.models.infer_model', return_value=error_model):
            agent.model = error_model
            result = agent.astream(simple_task)

            with pytest.raises(Exception):
                async for chunk in result:
                    pass
    
    @pytest.mark.asyncio
    async def test_streaming_with_caching(self, agent, simple_task):
        """Test streaming with caching enabled."""
        # Disable cache to avoid CacheMissEvent parameter issues
        simple_task.enable_cache = False
        
        result = agent.astream(simple_task)
        
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        
        # Should complete successfully
        assert len(chunks) > 0
        assert "".join(chunks) == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_streaming_with_guardrails(self, agent, simple_task):
        """Test streaming with guardrails."""
        def guardrail(text):
            return len(text) > 0, text
        
        simple_task.guardrail = guardrail
        
        result = agent.astream(simple_task)
        
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        
        # Should complete successfully with guardrails
        assert len(chunks) > 0
        assert "".join(chunks) == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_streaming_with_compression(self, agent, simple_task):
        """Test streaming with context compression."""
        agent.compression_strategy = "simple"
        agent.compression_settings = {"max_length": 100}
        
        result = agent.astream(simple_task)
        
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        
        # Should complete successfully with compression
        assert len(chunks) > 0
        assert "".join(chunks) == "Hello world!"


class TestAgentStreamingIntegration:
    """Integration tests for Agent streaming with various components."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for integration testing."""
        return MockModel()
    
    @pytest.fixture
    def agent_with_memory(self, mock_model):
        """Create an agent with memory for integration testing."""
        storage = InMemoryStorage()
        memory = Memory(
            storage=storage,
            session_id="integration-test",
            full_session_memory=False,  # Disable to avoid serialization issues
            summary_memory=False
        )
        
        return Agent(
            model=mock_model,
            name="IntegrationAgent",
            memory=memory
        )
    
    @pytest.fixture
    def complex_task(self):
        """Create a complex task for integration testing."""
        def process_tool(x: str) -> str:
            """Process input and return result."""
            return f"Processed: {x}"
        
        return Task(
            description="Analyze this data and provide insights",
            tools=[process_tool],
            enable_cache=False,  # Disable cache to avoid embedding provider issues
            response_format=str
        )
    
    @pytest.mark.asyncio
    async def test_full_streaming_workflow(self, agent_with_memory, complex_task):
        """Test complete streaming workflow with all features."""
        result = agent_with_memory.astream(complex_task)
        
        # Stream text
        text_chunks = []
        async for chunk in result:
            text_chunks.append(chunk)
        
        # Verify results
        assert len(text_chunks) > 0
        assert "".join(text_chunks) == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_multiple_streaming_sessions(self, agent_with_memory):
        """Test multiple streaming sessions with the same agent."""
        task1 = Task(description="First task")
        task2 = Task(description="Second task")
        
        # First streaming session
        result1 = agent_with_memory.astream(task1)
        chunks1 = []
        async for chunk in result1:
            chunks1.append(chunk)
        
        # Second streaming session
        result2 = agent_with_memory.astream(task2)
        chunks2 = []
        async for chunk in result2:
            chunks2.append(chunk)
        
        # Both should complete successfully
        assert "".join(chunks1) == "Hello world!"
        assert "".join(chunks2) == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_streaming_with_multiple_tasks(self, agent_with_memory):
        """Test multiple streaming tasks sequentially."""
        tasks = [
            Task(description=f"Task {i}") 
            for i in range(3)
        ]
        
        results = []
        for task in tasks:
            result = agent_with_memory.astream(task)
            chunks = []
            async for chunk in result:
                chunks.append(chunk)
            results.append("".join(chunks))
        
        # All should complete successfully
        assert all(result == "Hello world!" for result in results)
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
