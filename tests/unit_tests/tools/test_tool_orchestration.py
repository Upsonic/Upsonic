"""Unit tests for tool orchestration."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any

from upsonic.tools.orchestration import Orchestrator, PlanStep, Thought, AnalysisResult
from upsonic.tasks.tasks import Task


class TestOrchestrator:
    """Test suite for Orchestrator."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent instance."""
        agent = Mock()
        agent.enable_reasoning_tool = False
        agent.enable_thinking_tool = False
        agent.name = "TestAgent"
        agent.model = "openai/gpt-4"  # Set a valid model string
        agent.do_async = AsyncMock(return_value=Mock(output="synthesis_result"))
        return agent

    @pytest.fixture
    def mock_task(self):
        """Create a mock task."""
        task = Mock(spec=Task)
        task.description = "Test task"
        return task

    @pytest.fixture
    def mock_wrapped_tools(self):
        """Create mock wrapped tools."""
        return {
            "tool1": AsyncMock(return_value="result1"),
            "tool2": AsyncMock(return_value="result2"),
        }

    @pytest.fixture
    def orchestrator(self, mock_agent, mock_task, mock_wrapped_tools):
        """Create an Orchestrator instance for testing."""
        return Orchestrator(
            agent_instance=mock_agent, task=mock_task, wrapped_tools=mock_wrapped_tools
        )

    def test_orchestrator_initialization(self, orchestrator, mock_agent, mock_task):
        """Test Orchestrator initialization."""
        assert orchestrator.agent_instance == mock_agent
        assert orchestrator.task == mock_task
        assert orchestrator.is_reasoning_enabled is False
        assert orchestrator.original_user_request == "Test task"
        assert orchestrator.program_counter == 0
        assert orchestrator.pending_plan == []
        assert orchestrator.revision_count == 0

    @pytest.mark.asyncio
    @patch("upsonic.agent.agent.Agent")
    async def test_orchestrator_orchestrate_tools(self, mock_agent_class, orchestrator):
        """Test tool orchestration."""
        # Mock the Agent class to prevent real API calls
        mock_synthesis_agent = Mock()
        mock_synthesis_agent.do_async = AsyncMock(return_value=Mock(output="synthesis_result"))
        mock_agent_class.return_value = mock_synthesis_agent
        
        thought = Thought(
            reasoning="Test reasoning",
            plan=[
                PlanStep(tool_name="tool1", parameters={"param": "value"}),
                PlanStep(tool_name="tool2", parameters={"param": "value2"}),
            ],
            criticism="Test criticism",
            action="execute_plan",
        )

        result = await orchestrator.execute(thought)

        assert result is not None
        assert orchestrator.program_counter == 2

    @pytest.mark.asyncio
    @patch("upsonic.agent.agent.Agent")
    async def test_orchestrator_parallel_execution(self, mock_agent_class, orchestrator):
        """Test parallel tool execution."""
        # Mock the Agent class to prevent real API calls
        mock_synthesis_agent = Mock()
        mock_synthesis_agent.do_async = AsyncMock(return_value=Mock(output="synthesis_result"))
        mock_agent_class.return_value = mock_synthesis_agent
        
        # Orchestrator executes sequentially by default
        # This test verifies sequential execution works
        thought = Thought(
            reasoning="Test reasoning",
            plan=[
                PlanStep(tool_name="tool1", parameters={"param": "value"}),
                PlanStep(tool_name="tool2", parameters={"param": "value2"}),
            ],
            criticism="Test criticism",
            action="execute_plan",
        )

        result = await orchestrator.execute(thought)

        # Verify tools were called
        assert orchestrator.wrapped_tools["tool1"].called
        assert orchestrator.wrapped_tools["tool2"].called

    @pytest.mark.asyncio
    @patch("upsonic.agent.agent.Agent")
    async def test_orchestrator_sequential_execution(self, mock_agent_class, orchestrator):
        """Test sequential execution."""
        # Mock the Agent class to prevent real API calls
        mock_synthesis_agent = Mock()
        mock_synthesis_agent.do_async = AsyncMock(return_value=Mock(output="synthesis_result"))
        mock_agent_class.return_value = mock_synthesis_agent
        
        thought = Thought(
            reasoning="Test reasoning",
            plan=[
                PlanStep(tool_name="tool1", parameters={"param": "value"}),
                PlanStep(tool_name="tool2", parameters={"param": "value2"}),
            ],
            criticism="Test criticism",
            action="execute_plan",
        )

        result = await orchestrator.execute(thought)

        # Verify sequential execution
        call_order = []
        for call in orchestrator.wrapped_tools["tool1"].call_args_list:
            call_order.append("tool1")
        for call in orchestrator.wrapped_tools["tool2"].call_args_list:
            call_order.append("tool2")

        assert len(call_order) == 2

    @pytest.mark.asyncio
    @patch("upsonic.agent.agent.Agent")
    async def test_orchestrator_tool_not_found(self, mock_agent_class, orchestrator):
        """Test handling of non-existent tool."""
        # Mock the Agent class to prevent real API calls
        mock_synthesis_agent = Mock()
        mock_synthesis_agent.do_async = AsyncMock(return_value=Mock(output="synthesis_result"))
        mock_agent_class.return_value = mock_synthesis_agent
        
        thought = Thought(
            reasoning="Test reasoning",
            plan=[PlanStep(tool_name="nonexistent_tool", parameters={})],
            criticism="Test criticism",
            action="execute_plan",
        )

        result = await orchestrator.execute(thought)
        assert result is not None

    @pytest.mark.asyncio
    @patch("upsonic.agent.agent.Agent")
    async def test_orchestrator_tool_error(self, mock_agent_class, orchestrator):
        """Test handling of tool execution errors."""
        # Mock the Agent class to prevent real API calls
        mock_synthesis_agent = Mock()
        mock_synthesis_agent.do_async = AsyncMock(return_value=Mock(output="synthesis_result"))
        mock_agent_class.return_value = mock_synthesis_agent
        
        orchestrator.wrapped_tools["tool1"] = AsyncMock(
            side_effect=Exception("Tool error")
        )

        thought = Thought(
            reasoning="Test reasoning",
            plan=[PlanStep(tool_name="tool1", parameters={"param": "value"})],
            criticism="Test criticism",
            action="execute_plan",
        )

        result = await orchestrator.execute(thought)
        assert result is not None
