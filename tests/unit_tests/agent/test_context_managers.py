"""
Tests for Context Managers

This module contains tests for all context manager classes in the agent/context_managers/
directory, verifying their initialization, async context manager behavior, and methods.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from upsonic.agent.context_managers.call_manager import CallManager
from upsonic.agent.context_managers.task_manager import TaskManager
from upsonic.agent.context_managers.reliability_manager import ReliabilityManager
from upsonic.agent.context_managers.llm_manager import LLMManager
from upsonic.agent.context_managers.system_prompt_manager import SystemPromptManager
from upsonic.agent.context_managers.context_manager import ContextManager
from upsonic.agent.context_managers.memory_manager import MemoryManager


# ============================================================================
# Test Fixtures
# ============================================================================


class MockModel:
    """Mock model for testing."""

    def __init__(self, name="test-model"):
        self.model_name = name


class MockTask:
    """Mock task for testing."""

    def __init__(self, description="Test task"):
        self.description = description
        self.context = None
        self.response_format = None
        self.price_id = None
        self.context_formatted = None
        self.response = None
        self.tool_calls = []
        self.attachments = []
        self.response_lang = None
        self.not_main_task = False
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.total_cost = None
        self.total_input_token = None
        self.total_output_token = None
        self.enable_thinking_tool = None
        self.enable_reasoning_tool = None
        self.tools = []

    def get_task_id(self):
        return "test-task-id"

    def task_response(self, model_response):
        """Mock task_response method."""
        self.response = model_response


class MockAgent:
    """Mock agent for testing."""

    def __init__(self):
        self.agent_id = "test-agent-id"
        self.name = "Test Agent"
        self.debug = False
        self.retry = 1
        self.mode = "raise"
        self.show_tool_calls = True
        self.tool_call_limit = 10
        self.enable_thinking_tool = False
        self.enable_reasoning_tool = False
        self.memory = None
        self.knowledge = None
        self.canvas = None
        self.system_prompt = None
        self.role = None
        self.goal = None
        self.instructions = None
        self.education = None
        self.work_experience = None
        self.company_name = None
        self.company_url = None
        self.company_objective = None
        self.company_description = None

    def get_agent_id(self):
        return self.agent_id


class MockMemory:
    """Mock memory for testing."""

    def __init__(self):
        self.prepared_inputs = {
            "message_history": [],
            "context_injection": "test context",
            "system_prompt_injection": "test system prompt",
        }

    async def prepare_inputs_for_task(self):
        return self.prepared_inputs

    async def update_memories_after_task(self, model_response):
        return None


class MockModelResponse:
    """Mock model response for testing."""

    def __init__(self):
        self.output = "Test output"
        self.tool_calls = []
        # Mock usage object for llm_usage function
        self.usage = Mock()
        self.usage.input_tokens = 10
        self.usage.output_tokens = 5

    def all_messages(self):
        """Mock all_messages method that returns a list of messages with usage and parts."""
        mock_message = Mock()
        mock_message.usage = self.usage
        # Mock parts attribute for tool_usage function
        mock_message.parts = []
        return [mock_message]


# ============================================================================
# Test CallManager
# ============================================================================


class TestCallManager:
    """Test suite for CallManager."""

    def test_call_manager_initialization(self):
        """Test CallManager initialization."""
        model = MockModel()
        task = MockTask()

        manager = CallManager(model=model, task=task, debug=False, show_tool_calls=True)

        assert manager.model == model
        assert manager.task == task
        assert manager.debug is False
        assert manager.show_tool_calls is True
        assert manager.start_time is None
        assert manager.end_time is None
        assert manager.model_response is None

    def test_call_manager_process_response(self):
        """Test CallManager process_response method."""
        model = MockModel()
        task = MockTask()
        manager = CallManager(model, task)

        mock_response = MockModelResponse()
        result = manager.process_response(mock_response)

        assert manager.model_response == mock_response
        assert result == mock_response

    @pytest.mark.asyncio
    @patch("upsonic.utils.printing.call_end")
    @patch("upsonic.utils.tool_usage.tool_usage")
    @patch("upsonic.utils.llm_usage.llm_usage")
    async def test_call_manager_context_manager(
        self, mock_llm_usage, mock_tool_usage, mock_call_end
    ):
        """Test CallManager as async context manager."""
        model = MockModel()
        task = MockTask()
        manager = CallManager(model, task, show_tool_calls=True)

        mock_response = MockModelResponse()
        mock_llm_usage.return_value = {"input_tokens": 10, "output_tokens": 5}
        mock_tool_usage.return_value = [
            {"tool_name": "test_tool", "params": {}, "tool_result": None}
        ]

        async with manager.manage_call() as ctx:
            assert ctx is manager
            assert manager.start_time is not None
            manager.process_response(mock_response)

        assert manager.end_time is not None
        assert manager.end_time > manager.start_time
        mock_llm_usage.assert_called_once_with(mock_response)
        mock_tool_usage.assert_called_once_with(mock_response, task)
        mock_call_end.assert_called_once()

    @pytest.mark.asyncio
    @patch("upsonic.utils.printing.call_end")
    @patch("upsonic.utils.tool_usage.tool_usage")
    @patch("upsonic.utils.llm_usage.llm_usage")
    async def test_call_manager_no_response_no_call_end(
        self, mock_llm_usage, mock_tool_usage, mock_call_end
    ):
        """Test CallManager doesn't call call_end if no response."""
        model = MockModel()
        task = MockTask()
        manager = CallManager(model, task)

        async with manager.manage_call():
            pass

        mock_call_end.assert_not_called()

    @pytest.mark.asyncio
    @patch("upsonic.utils.printing.call_end")
    @patch("upsonic.utils.tool_usage.tool_usage")
    @patch("upsonic.utils.llm_usage.llm_usage")
    async def test_call_manager_show_tool_calls_false(
        self, mock_llm_usage, mock_tool_usage, mock_call_end
    ):
        """Test CallManager with show_tool_calls=False."""
        model = MockModel()
        task = MockTask()
        manager = CallManager(model, task, show_tool_calls=False)

        mock_response = MockModelResponse()
        mock_llm_usage.return_value = {"input_tokens": 10, "output_tokens": 5}

        async with manager.manage_call():
            manager.process_response(mock_response)

        mock_tool_usage.assert_not_called()
        mock_llm_usage.assert_called_once_with(mock_response)
        mock_call_end.assert_called_once()


# ============================================================================
# Test TaskManager
# ============================================================================


class TestTaskManager:
    """Test suite for TaskManager."""

    def test_task_manager_initialization(self):
        """Test TaskManager initialization."""
        task = MockTask()
        agent = MockAgent()

        manager = TaskManager(task, agent)

        assert manager.task == task
        assert manager.agent == agent
        assert manager.model_response is None

    def test_task_manager_process_response(self):
        """Test TaskManager process_response method."""
        task = MockTask()
        agent = MockAgent()
        manager = TaskManager(task, agent)

        mock_response = MockModelResponse()
        result = manager.process_response(mock_response)

        assert manager.model_response == mock_response
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_task_manager_context_manager(self):
        """Test TaskManager as async context manager."""
        task = MockTask()
        agent = MockAgent()
        manager = TaskManager(task, agent)

        mock_response = MockModelResponse()

        async with manager.manage_task() as ctx:
            assert ctx is manager
            manager.process_response(mock_response)

        assert task.response == mock_response

    @pytest.mark.asyncio
    async def test_task_manager_no_response_no_task_response(self):
        """Test TaskManager doesn't call task_response if no response."""
        task = MockTask()
        agent = MockAgent()
        manager = TaskManager(task, agent)

        async with manager.manage_task():
            pass

        assert task.response is None


# ============================================================================
# Test ReliabilityManager
# ============================================================================


class TestReliabilityManager:
    """Test suite for ReliabilityManager."""

    def test_reliability_manager_initialization(self):
        """Test ReliabilityManager initialization."""
        task = MockTask()
        reliability_layer = Mock()
        model = MockModel()

        manager = ReliabilityManager(task, reliability_layer, model)

        assert manager.task == task
        assert manager.reliability_layer == reliability_layer
        assert manager.model == model
        assert manager.processed_task is None

    @pytest.mark.asyncio
    @patch("upsonic.reliability_layer.reliability_layer.ReliabilityProcessor")
    async def test_reliability_manager_process_task(self, mock_processor_class):
        """Test ReliabilityManager process_task method."""
        task = MockTask()
        # Create a proper mock reliability_layer with prevent_hallucination attribute
        reliability_layer = Mock()
        reliability_layer.prevent_hallucination = 0  # Set to int, not Mock
        model = MockModel()
        manager = ReliabilityManager(task, reliability_layer, model)

        # When prevent_hallucination is 0, ReliabilityProcessor.process_task returns the task as-is
        # So we expect the task to be returned, not a mock
        mock_processor_class.process_task = AsyncMock(return_value=task)

        result = await manager.process_task(task)

        assert result == task
        assert manager.processed_task == task
        mock_processor_class.process_task.assert_called_once_with(
            task, reliability_layer, model
        )

    @pytest.mark.asyncio
    async def test_reliability_manager_context_manager(self):
        """Test ReliabilityManager as async context manager."""
        task = MockTask()
        reliability_layer = Mock()
        model = MockModel()
        manager = ReliabilityManager(task, reliability_layer, model)

        async with manager.manage_reliability() as ctx:
            assert ctx is manager


# ============================================================================
# Test LLMManager
# ============================================================================


class TestLLMManager:
    """Test suite for LLMManager."""

    def test_llm_manager_initialization(self):
        """Test LLMManager initialization."""
        default_model = "openai/gpt-4o"
        requested_model = None

        manager = LLMManager(default_model, requested_model)

        assert manager.default_model == default_model
        assert manager.requested_model == requested_model
        assert manager.selected_model is None

    def test_llm_manager_get_model(self):
        """Test LLMManager get_model method."""
        manager = LLMManager("openai/gpt-4o")
        manager.selected_model = "test-model"

        assert manager.get_model() == "test-model"

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    async def test_llm_manager_context_manager_default_model(self):
        """Test LLMManager context manager with default model."""
        default_model = "openai/gpt-4o"
        manager = LLMManager(default_model, None)

        async with manager.manage_llm() as ctx:
            assert ctx is manager
            assert manager.selected_model == default_model

    @pytest.mark.asyncio
    @patch.dict("os.environ", {}, clear=True)
    async def test_llm_manager_context_manager_requested_model(self):
        """Test LLMManager context manager with requested model."""
        default_model = "openai/gpt-4o"
        requested_model = "anthropic/claude-3"
        manager = LLMManager(default_model, requested_model)

        async with manager.manage_llm() as ctx:
            assert ctx is manager
            assert manager.selected_model == requested_model

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"LLM_MODEL_KEY": "openai/gpt-3.5-turbo"}, clear=False)
    async def test_llm_manager_context_manager_env_fallback(self):
        """Test LLMManager uses environment variable when model is None."""
        manager = LLMManager(None, None)

        async with manager.manage_llm():
            assert manager.selected_model == "openai/gpt-3.5-turbo"


# ============================================================================
# Test SystemPromptManager
# ============================================================================


class TestSystemPromptManager:
    """Test suite for SystemPromptManager."""

    def test_system_prompt_manager_initialization(self):
        """Test SystemPromptManager initialization."""
        agent = MockAgent()
        task = MockTask()

        manager = SystemPromptManager(agent, task)

        assert manager.agent == agent
        assert manager.task == task
        assert manager.system_prompt == ""

    def test_system_prompt_manager_get_system_prompt(self):
        """Test SystemPromptManager get_system_prompt method."""
        agent = MockAgent()
        task = MockTask()
        manager = SystemPromptManager(agent, task)
        manager.system_prompt = "test prompt"

        assert manager.get_system_prompt() == "test prompt"

    @pytest.mark.asyncio
    @patch("upsonic.agent.context_managers.system_prompt_manager.default_prompt")
    async def test_system_prompt_manager_context_manager_no_memory(
        self, mock_default_prompt
    ):
        """Test SystemPromptManager context manager without memory."""
        agent = MockAgent()
        task = MockTask()
        manager = SystemPromptManager(agent, task)

        mock_default_prompt.return_value.prompt = "Default prompt"

        async with manager.manage_system_prompt() as ctx:
            assert ctx is manager
            assert manager.system_prompt is not None
            assert isinstance(manager.system_prompt, str)

    @pytest.mark.asyncio
    async def test_system_prompt_manager_context_manager_with_memory(self):
        """Test SystemPromptManager context manager with memory."""
        agent = MockAgent()
        task = MockTask()
        manager = SystemPromptManager(agent, task)

        memory_manager = MemoryManager(MockMemory())
        memory_manager._prepared_inputs = {
            "system_prompt_injection": "Memory injection"
        }

        async with manager.manage_system_prompt(memory_manager) as ctx:
            assert ctx is manager
            assert manager.system_prompt is not None
            assert "Memory injection" in manager.system_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_manager_with_agent_attributes(self):
        """Test SystemPromptManager includes agent attributes in prompt."""
        agent = MockAgent()
        agent.role = "Test Role"
        agent.goal = "Test Goal"
        agent.instructions = "Test Instructions"
        task = MockTask()
        manager = SystemPromptManager(agent, task)

        async with manager.manage_system_prompt():
            prompt = manager.get_system_prompt()
            assert "Test Role" in prompt
            assert "Test Goal" in prompt
            assert "Test Instructions" in prompt


# ============================================================================
# Test ContextManager
# ============================================================================


class TestContextManager:
    """Test suite for ContextManager."""

    def test_context_manager_initialization(self):
        """Test ContextManager initialization."""
        agent = MockAgent()
        task = MockTask()

        manager = ContextManager(agent, task)

        assert manager.agent == agent
        assert manager.task == task
        assert manager.state is None
        assert manager.context_prompt == ""

    def test_context_manager_get_context_prompt(self):
        """Test ContextManager get_context_prompt method."""
        agent = MockAgent()
        task = MockTask()
        manager = ContextManager(agent, task)
        manager.context_prompt = "test context"

        assert manager.get_context_prompt() == "test context"

    @pytest.mark.asyncio
    async def test_context_manager_context_manager_no_context(self):
        """Test ContextManager with no context."""
        agent = MockAgent()
        task = MockTask()
        task.context = None
        manager = ContextManager(agent, task)

        async with manager.manage_context() as ctx:
            assert ctx is manager
            assert manager.context_prompt == ""
            assert task.context_formatted == ""

    @pytest.mark.asyncio
    async def test_context_manager_context_manager_with_string_context(self):
        """Test ContextManager with string context."""
        agent = MockAgent()
        task = MockTask()
        task.context = ["Additional context string"]
        manager = ContextManager(agent, task)

        async with manager.manage_context() as ctx:
            assert ctx is manager
            assert manager.context_prompt is not None
            assert "Additional context string" in manager.context_prompt
            assert task.context_formatted is not None

    @pytest.mark.asyncio
    async def test_context_manager_context_manager_with_memory(self):
        """Test ContextManager with memory handler."""
        agent = MockAgent()
        task = MockTask()
        manager = ContextManager(agent, task)

        memory_manager = MemoryManager(MockMemory())
        memory_manager._prepared_inputs = {
            "context_injection": "Memory context injection"
        }

        async with manager.manage_context(memory_manager) as ctx:
            assert ctx is manager
            assert "Memory context injection" in manager.context_prompt

    @pytest.mark.asyncio
    async def test_context_manager_get_context_summary(self):
        """Test ContextManager get_context_summary method."""
        agent = MockAgent()
        task = MockTask()
        manager = ContextManager(agent, task)

        summary = manager.get_context_summary()

        assert isinstance(summary, dict)
        assert "task" in summary
        assert "context" in summary
        assert "agent" in summary
        assert "state" in summary
        assert summary["agent"]["id"] == agent.agent_id
        assert summary["agent"]["name"] == agent.name


# ============================================================================
# Test MemoryManager
# ============================================================================


class TestMemoryManager:
    """Test suite for MemoryManager."""

    def test_memory_manager_initialization(self):
        """Test MemoryManager initialization."""
        memory = MockMemory()
        manager = MemoryManager(memory)

        assert manager.memory == memory
        assert manager._prepared_inputs == {
            "message_history": [],
            "context_injection": "",
            "system_prompt_injection": "",
        }
        assert manager._model_response is None

    def test_memory_manager_initialization_no_memory(self):
        """Test MemoryManager initialization without memory."""
        manager = MemoryManager(None)

        assert manager.memory is None
        assert manager._prepared_inputs == {
            "message_history": [],
            "context_injection": "",
            "system_prompt_injection": "",
        }

    def test_memory_manager_get_message_history(self):
        """Test MemoryManager get_message_history method."""
        memory = MockMemory()
        manager = MemoryManager(memory)
        manager._prepared_inputs["message_history"] = [
            {"role": "user", "content": "test"}
        ]

        history = manager.get_message_history()
        assert history == [{"role": "user", "content": "test"}]

    def test_memory_manager_get_context_injection(self):
        """Test MemoryManager get_context_injection method."""
        memory = MockMemory()
        manager = MemoryManager(memory)
        manager._prepared_inputs["context_injection"] = "test context"

        injection = manager.get_context_injection()
        assert injection == "test context"

    def test_memory_manager_get_system_prompt_injection(self):
        """Test MemoryManager get_system_prompt_injection method."""
        memory = MockMemory()
        manager = MemoryManager(memory)
        manager._prepared_inputs["system_prompt_injection"] = "test system prompt"

        injection = manager.get_system_prompt_injection()
        assert injection == "test system prompt"

    def test_memory_manager_process_response(self):
        """Test MemoryManager process_response method."""
        memory = MockMemory()
        manager = MemoryManager(memory)
        mock_response = MockModelResponse()

        result = manager.process_response(mock_response)

        assert manager._model_response == mock_response
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_memory_manager_context_manager_with_memory(self):
        """Test MemoryManager context manager with memory."""
        memory = MockMemory()
        manager = MemoryManager(memory)

        async with manager.manage_memory() as ctx:
            assert ctx is manager
            # After entering, prepared_inputs should be set
            assert manager._prepared_inputs is not None

    @pytest.mark.asyncio
    async def test_memory_manager_context_manager_no_memory(self):
        """Test MemoryManager context manager without memory."""
        manager = MemoryManager(None)

        async with manager.manage_memory() as ctx:
            assert ctx is manager

    @pytest.mark.asyncio
    async def test_memory_manager_context_manager_updates_memory(self):
        """Test MemoryManager updates memory after task."""
        memory = MockMemory()
        memory.update_memories_after_task = AsyncMock()
        manager = MemoryManager(memory)
        mock_response = MockModelResponse()

        async with manager.manage_memory():
            manager.process_response(mock_response)

        memory.update_memories_after_task.assert_called_once_with(mock_response)

    @pytest.mark.asyncio
    async def test_memory_manager_context_manager_no_response_no_update(self):
        """Test MemoryManager doesn't update if no response."""
        memory = MockMemory()
        memory.update_memories_after_task = AsyncMock()
        manager = MemoryManager(memory)

        async with manager.manage_memory():
            pass

        memory.update_memories_after_task.assert_not_called()
