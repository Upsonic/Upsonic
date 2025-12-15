import asyncio
import pytest
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import List, Dict, Any, Optional

from upsonic.agent.agent import Agent
from upsonic.agent.run_result import StreamRunResult
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
    
    def test_stream_returns_stream_run_result(self, agent, simple_task):
        """Test that stream() returns a StreamRunResult instance."""
        result = agent.stream(simple_task)
        
        assert isinstance(result, StreamRunResult)
        assert result._agent == agent
        assert result._task == simple_task
        assert result._debug is False
        assert result._retry == 1
    
    def test_stream_with_custom_parameters(self, agent, simple_task):
        """Test stream() with custom parameters."""
        result = agent.stream(
            simple_task, 
            model="custom-model", 
            debug=True, 
            retry=3
        )
        
        assert result._model == "custom-model"
        assert result._debug is True
        assert result._retry == 3
    
    def test_stream_initializes_task_properties(self, agent, simple_task):
        """Test that stream() properly initializes task properties."""
        # Ensure task properties are reset
        original_price_id = "existing-id"
        simple_task.price_id_ = original_price_id
        simple_task._tool_calls = [{"test": "call"}]

        result = agent.stream(simple_task)

        # Task should have a new price_id_ (not the original) and tool calls should be reset
        assert simple_task.price_id_ != original_price_id
        assert simple_task.price_id_ is not None  # Should have a new UUID
        assert simple_task._tool_calls == []
    
    @pytest.mark.asyncio
    async def test_stream_async_returns_stream_run_result(self, agent, simple_task):
        """Test that stream_async() returns a StreamRunResult instance."""
        result = await agent.stream_async(simple_task)
        
        assert isinstance(result, StreamRunResult)
        assert result._agent == agent
        assert result._task == simple_task
    
    @pytest.mark.asyncio
    async def test_stream_async_with_state_and_graph_id(self, agent, simple_task):
        """Test stream_async() with state and graph_execution_id."""
        mock_state = {"test": "state"}
        graph_id = "test-graph-id"
        
        result = await agent.stream_async(
            simple_task, 
            state=mock_state, 
            graph_execution_id=graph_id
        )
        
        assert result._state == mock_state
        assert result._graph_execution_id == graph_id
    
    @pytest.mark.asyncio
    async def test_stream_run_result_context_manager(self, agent, simple_task):
        """Test StreamRunResult as async context manager."""
        result = await agent.stream_async(simple_task)
        
        # Test context manager entry
        async with result as stream_result:
            assert stream_result._context_entered is True
            assert stream_result._start_time is not None
            assert stream_result is result
        
        # Test context manager exit
        assert result._context_entered is False
        assert result._end_time is not None
    
    @pytest.mark.asyncio
    async def test_stream_run_result_double_context_error(self, agent, simple_task):
        """Test that entering context manager twice raises error."""
        result = await agent.stream_async(simple_task)
        
        async with result:
            with pytest.raises(RuntimeError, match="context manager is already active"):
                async with result:
                    pass
    
    @pytest.mark.asyncio
    async def test_stream_output_basic_functionality(self, agent, simple_task):
        """Test basic streaming output functionality."""
        result = await agent.stream_async(simple_task)
        
        async with result:
            text_chunks = []
            async for chunk in result.stream_output():
                text_chunks.append(chunk)
            
            # Should have received text chunks
            assert len(text_chunks) > 0
            assert "".join(text_chunks) == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_stream_output_outside_context_error(self, agent, simple_task):
        """Test that stream_output() outside context raises error."""
        result = await agent.stream_async(simple_task)
        
        with pytest.raises(RuntimeError, match="must be called within async context manager"):
            async for chunk in result.stream_output():
                pass
    
    @pytest.mark.asyncio
    async def test_stream_output_without_agent_task_error(self, agent, simple_task):
        """Test stream_output() without agent/task raises error."""
        result = StreamRunResult()
        result._context_entered = True
        
        with pytest.raises(RuntimeError, match="No agent or task available"):
            async for chunk in result.stream_output():
                pass
    
    @pytest.mark.asyncio
    async def test_stream_events_basic_functionality(self, agent, simple_task):
        """Test basic streaming events functionality."""
        result = await agent.stream_async(simple_task)
        
        async with result:
            events = []
            async for event in result.stream_events():
                events.append(event)
            
            # Should have received various event types
            assert len(events) > 0
            event_types = [type(event).__name__ for event in events]
            # PartStartEvent is converted to TextDeltaEvent by the pipeline
            assert "TextDeltaEvent" in event_types
            # FinalResultEvent is converted to FinalOutputEvent by the pipeline
            assert "FinalOutputEvent" in event_types
    
    @pytest.mark.asyncio
    async def test_stream_events_outside_context_error(self, agent, simple_task):
        """Test that stream_events() outside context raises error."""
        result = await agent.stream_async(simple_task)
        
        with pytest.raises(RuntimeError, match="must be called within async context manager"):
            async for event in result.stream_events():
                pass
    
    @pytest.mark.asyncio
    async def test_get_final_output(self, agent, simple_task):
        """Test getting final output from stream result."""
        result = await agent.stream_async(simple_task)
        
        async with result:
            async for chunk in result.stream_output():
                pass  # Consume all chunks
        
        final_output = result.get_final_output()
        assert final_output == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_accumulated_text_property(self, agent, simple_task):
        """Test the accumulated text property."""
        result = await agent.stream_async(simple_task)
        
        async with result:
            async for chunk in result.stream_output():
                pass  # Consume all chunks
        
        assert result.output == "Hello world!"
        assert result.get_accumulated_text() == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_is_complete_status(self, agent, simple_task):
        """Test completion status tracking."""
        result = await agent.stream_async(simple_task)
        
        assert result.is_complete() is False
        
        async with result:
            async for chunk in result.stream_output():
                pass  # Consume all chunks
        
        assert result.is_complete() is True
    
    @pytest.mark.asyncio
    async def test_streaming_events_tracking(self, agent, simple_task):
        """Test that streaming events are properly tracked."""
        result = await agent.stream_async(simple_task)
        
        async with result:
            async for event in result.stream_events():
                pass  # Consume all events
        
        events = result.get_streaming_events()
        assert len(events) > 0
        
        # Test event filtering
        text_events = result.get_text_events()
        tool_events = result.get_tool_events()
        
        assert len(text_events) > 0
        assert len(tool_events) == 0  # No tool events in our mock
    
    @pytest.mark.asyncio
    async def test_streaming_stats(self, agent, simple_task):
        """Test streaming statistics collection."""
        result = await agent.stream_async(simple_task)
        
        async with result:
            async for chunk in result.stream_output():
                pass  # Consume all chunks
        
        stats = result.get_streaming_stats()
        
        assert "total_events" in stats
        assert "text_events" in stats
        assert "tool_events" in stats
        assert "accumulated_chars" in stats
        assert "is_complete" in stats
        assert "has_final_output" in stats
        assert "event_types" in stats
        
        assert stats["is_complete"] is True
        assert stats["has_final_output"] is True
        assert stats["accumulated_chars"] == len("Hello world!")
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, agent, simple_task):
        """Test performance metrics collection."""
        result = await agent.stream_async(simple_task)
        
        async with result:
            async for chunk in result.stream_output():
                pass  # Consume all chunks
        
        metrics = result.get_performance_metrics()
        
        assert "start_time" in metrics
        assert "end_time" in metrics
        assert "first_token_time" in metrics
        assert "total_duration" in metrics
        assert "time_to_first_token" in metrics
        assert "tokens_per_second" in metrics
        assert "characters_per_second" in metrics
        
        # All timing metrics should be present
        assert metrics["start_time"] is not None
        assert metrics["end_time"] is not None
        assert metrics["total_duration"] is not None
        assert metrics["total_duration"] > 0
    
    @pytest.mark.asyncio
    async def test_message_tracking(self, agent, simple_task):
        """Test message tracking in streaming."""
        result = await agent.stream_async(simple_task)
        
        async with result:
            async for chunk in result.stream_output():
                pass  # Consume all chunks
        
        # Test message methods
        all_messages = result.all_messages()
        new_messages = result.new_messages()
        
        # Should have messages after streaming
        assert len(all_messages) > 0
        assert len(new_messages) > 0
    
    @pytest.mark.asyncio
    async def test_print_stream_async(self, agent, simple_task, capsys):
        """Test print_stream_async functionality."""
        result = await agent.print_stream_async(simple_task)
        
        # Should return the final output
        assert result == "Hello world!"
        
        # Should have printed the output (captured by capsys)
        captured = capsys.readouterr()
        assert "Hello world!" in captured.out
    
    def test_print_stream_synchronous(self, agent, simple_task, capsys):
        """Test print_stream synchronous functionality."""
        result = agent.print_stream(simple_task)
        
        # Should return the final output
        assert result == "Hello world!"
        
        # Should have printed the output
        captured = capsys.readouterr()
        assert "Hello world!" in captured.out
    
    @pytest.mark.asyncio
    async def test_streaming_with_memory(self, agent, simple_task, memory):
        """Test streaming with memory integration."""
        agent.memory = memory
        
        result = await agent.stream_async(simple_task)
        
        async with result:
            async for chunk in result.stream_output():
                pass  # Consume all chunks
        
        # Memory should have been updated
        assert result.get_final_output() == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_streaming_with_tools(self, agent, simple_task):
        """Test streaming with tools enabled."""
        # Add a simple tool to the task
        def test_tool() -> str:
            """A simple test tool that returns a string."""
            return "Tool executed"

        simple_task.tools = [test_tool]
        
        result = await agent.stream_async(simple_task)
        
        async with result:
            async for chunk in result.stream_output():
                pass  # Consume all chunks
        
        # Should complete successfully even with tools
        assert result.get_final_output() is not None
    
    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, agent, simple_task):
        """Test error handling in streaming."""
        # Create a model that raises an error
        error_model = Mock()
        
        # Create a proper async context manager that raises an error
        class ErrorContextManager:
            async def __aenter__(self):
                raise Exception("Stream error")
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        
        error_model.request_stream = lambda *args, **kwargs: ErrorContextManager()
        error_model.model_name = "error-model"
        error_model.settings = None
        error_model.customize_request_parameters = Mock(return_value={})

        # Mock the infer_model function to return our error model
        with patch('upsonic.models.infer_model', return_value=error_model):
            agent.model = error_model
            result = await agent.stream_async(simple_task)

            with pytest.raises(Exception, match="Stream error"):
                async with result:
                    async for chunk in result.stream_output():
                        pass
    
    @pytest.mark.asyncio
    async def test_streaming_with_caching(self, agent, simple_task):
        """Test streaming with caching enabled."""
        simple_task.enable_cache = True
        
        result = await agent.stream_async(simple_task)
        
        async with result:
            async for chunk in result.stream_output():
                pass  # Consume all chunks
        
        # Should complete successfully with caching
        assert result.get_final_output() == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_streaming_with_guardrails(self, agent, simple_task):
        """Test streaming with guardrails."""
        def guardrail(text):
            return len(text) > 0, text
        
        simple_task.guardrail = guardrail
        
        result = await agent.stream_async(simple_task)
        
        async with result:
            async for chunk in result.stream_output():
                pass  # Consume all chunks
        
        # Should complete successfully with guardrails
        assert result.get_final_output() == "Hello world!"
    
    @pytest.mark.asyncio
    async def test_streaming_with_compression(self, agent, simple_task):
        """Test streaming with context compression."""
        agent.compression_strategy = "simple"
        agent.compression_settings = {"max_length": 100}
        
        result = await agent.stream_async(simple_task)
        
        async with result:
            async for chunk in result.stream_output():
                pass  # Consume all chunks
        
        # Should complete successfully with compression
        assert result.get_final_output() == "Hello world!"
    
    def test_stream_run_result_string_representation(self, agent, simple_task):
        """Test string representation of StreamRunResult."""
        result = agent.stream(simple_task)
        
        # Test __str__ method
        str_repr = str(result)
        assert str_repr == ""  # Empty before streaming
        
        # Test __repr__ method
        repr_str = repr(result)
        assert "StreamRunResult" in repr_str
        assert "streaming" in repr_str or "complete" in repr_str
    
    @pytest.mark.asyncio
    async def test_stream_run_result_repr_after_completion(self, agent, simple_task):
        """Test __repr__ after streaming completion."""
        result = await agent.stream_async(simple_task)
        
        async with result:
            async for chunk in result.stream_output():
                pass  # Consume all chunks
        
        repr_str = repr(result)
        assert "StreamRunResult" in repr_str
        assert "complete" in repr_str
        assert "accumulated_chars" in repr_str
        assert "events_count" in repr_str
        assert "messages_count" in repr_str


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
        result = await agent_with_memory.stream_async(complex_task)
        
        async with result:
            # Test both text and event streaming
            text_chunks = []
            events = []
            
            # Stream text
            async for chunk in result.stream_output():
                text_chunks.append(chunk)
            
            # Stream events
            async for event in result.stream_events():
                events.append(event)
        
        # Verify results
        assert result.get_final_output() == "Hello world!"
        assert len(text_chunks) > 0
        assert len(events) > 0
        assert result.is_complete() is True
        
        # Verify performance metrics
        metrics = result.get_performance_metrics()
        assert metrics["total_duration"] > 0
        
        # Verify streaming stats
        stats = result.get_streaming_stats()
        assert stats["is_complete"] is True
        assert stats["has_final_output"] is True
    
    @pytest.mark.asyncio
    async def test_multiple_streaming_sessions(self, agent_with_memory):
        """Test multiple streaming sessions with the same agent."""
        task1 = Task(description="First task")
        task2 = Task(description="Second task")
        
        # First streaming session
        result1 = await agent_with_memory.stream_async(task1)
        async with result1:
            async for chunk in result1.stream_output():
                pass
        
        # Second streaming session
        result2 = await agent_with_memory.stream_async(task2)
        async with result2:
            async for chunk in result2.stream_output():
                pass
        
        # Both should complete successfully
        assert result1.get_final_output() == "Hello world!"
        assert result2.get_final_output() == "Hello world!"
        assert result1.is_complete() is True
        assert result2.is_complete() is True
    
    @pytest.mark.asyncio
    async def test_streaming_with_multiple_tasks(self, agent_with_memory):
        """Test multiple streaming tasks sequentially."""
        tasks = [
            Task(description=f"Task {i}") 
            for i in range(3)
        ]
        
        results = []
        for task in tasks:
            result = await agent_with_memory.stream_async(task)
            async with result:
                async for chunk in result.stream_output():
                    pass  # Consume all chunks
            results.append(result.get_final_output())
        
        # All should complete successfully
        assert all(result == "Hello world!" for result in results)
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
