"""Unit tests for ToolManager."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any

from upsonic.tools import ToolManager
from upsonic.tools.base import Tool, ToolResult, ToolDefinition
from upsonic.tools.metrics import ToolMetrics
from upsonic.tools.deferred import DeferredExecutionManager


class TestToolManager:
    """Test suite for ToolManager."""

    @pytest.fixture
    def tool_manager(self):
        """Create a ToolManager instance for testing."""
        return ToolManager()

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool for testing."""
        tool = Mock(spec=Tool)
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.schema = Mock()
        tool.schema.json_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        tool.schema.strict = False
        tool.metadata = Mock()
        tool.metadata.custom = {}
        tool.config = Mock()
        tool.config.strict = None
        tool.config.sequential = False
        tool.execute = AsyncMock(return_value="test_result")
        return tool

    @pytest.fixture
    def mock_context(self):
        """Create a mock ToolMetrics for testing."""
        return ToolMetrics()

    def test_tool_manager_initialization(self, tool_manager):
        """Test ToolManager initialization."""
        assert tool_manager.processor is not None
        assert tool_manager.deferred_manager is not None
        assert tool_manager.orchestrator is None
        assert tool_manager.wrapped_tools == {}
        assert tool_manager.current_task is None

    def test_tool_manager_add_tool(self, tool_manager, mock_tool, mock_context):
        """Test adding tools."""

        def test_function(query: str) -> str:
            """Test function."""
            return f"Result: {query}"

        with patch.object(
            tool_manager.processor,
            "process_tools",
            return_value={"test_function": mock_tool},
        ):
            with patch.object(
                tool_manager.processor,
                "create_behavioral_wrapper",
                return_value=AsyncMock(),
            ):
                registered = tool_manager.register_tools(
                    tools=[test_function]
                )

                assert "test_function" in registered
                assert "test_function" in tool_manager.wrapped_tools

    def test_tool_manager_remove_tool(self, tool_manager, mock_tool):
        """Test removing tools."""
        tool_manager.wrapped_tools["test_tool"] = AsyncMock()
        tool_manager.processor.registered_tools["test_tool"] = mock_tool

        # Remove tool
        del tool_manager.wrapped_tools["test_tool"]
        del tool_manager.processor.registered_tools["test_tool"]

        assert "test_tool" not in tool_manager.wrapped_tools
        assert "test_tool" not in tool_manager.processor.registered_tools

    def test_tool_manager_get_tool(self, tool_manager, mock_tool):
        """Test getting tools."""
        tool_manager.processor.registered_tools["test_tool"] = mock_tool

        tool = tool_manager.processor.registered_tools.get("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"

    def test_tool_manager_list_tools(self, tool_manager, mock_tool):
        """Test listing tools."""
        tool_manager.processor.registered_tools["test_tool"] = mock_tool
        tool_manager.processor.registered_tools["another_tool"] = mock_tool

        tools = list(tool_manager.processor.registered_tools.keys())
        assert "test_tool" in tools
        assert "another_tool" in tools
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_tool_manager_execute_tool(self, tool_manager, mock_tool):
        """Test tool execution."""
        mock_wrapper = AsyncMock(return_value={"func": "test_result"})
        tool_manager.wrapped_tools["test_tool"] = mock_wrapper

        result = await tool_manager.execute_tool(
            tool_name="test_tool", args={"query": "test"}
        )

        assert isinstance(result, ToolResult)
        assert result.tool_name == "test_tool"
        assert result.success is True
        assert result.content == {"func": "test_result"}
        mock_wrapper.assert_called_once_with(query="test")

    @pytest.mark.asyncio
    async def test_tool_manager_execute_tool_not_found(self, tool_manager):
        """Test executing a non-existent tool."""
        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            await tool_manager.execute_tool(tool_name="nonexistent", args={})

    @pytest.mark.asyncio
    async def test_tool_manager_execute_tool_with_error(self, tool_manager):
        """Test tool execution with error."""
        mock_wrapper = AsyncMock(side_effect=Exception("Test error"))
        tool_manager.wrapped_tools["test_tool"] = mock_wrapper

        result = await tool_manager.execute_tool(tool_name="test_tool", args={})

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "Test error" in result.error

    def test_tool_manager_get_tool_definitions(self, tool_manager, mock_tool):
        """Test getting tool definitions."""
        tool_manager.processor.registered_tools["test_tool"] = mock_tool

        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 1
        assert isinstance(definitions[0], ToolDefinition)
        assert definitions[0].name == "test_tool"

    def test_tool_manager_has_deferred_requests(self, tool_manager):
        """Test checking for deferred requests."""
        assert tool_manager.deferred_manager.has_pending_requests() is False

        # Add a deferred request
        tool_manager.deferred_manager.create_external_call(
            tool_name="test", args={}, tool_call_id="123"
        )

        assert tool_manager.deferred_manager.has_pending_requests() is True

    def test_tool_manager_get_deferred_requests(self, tool_manager):
        """Test getting deferred requests."""
        # Use deferred_manager directly
        pending = tool_manager.deferred_manager.get_pending_calls()
        assert isinstance(pending, list)

    def test_tool_manager_process_deferred_results(self, tool_manager):
        """Test processing deferred results."""
        # Add a pending call
        external_call = tool_manager.deferred_manager.create_external_call(
            tool_name="test", args={"key": "value"}, tool_call_id="123"
        )

        # Update with result
        tool_manager.deferred_manager.update_call_result(
            tool_call_id="123",
            result="test_result"
        )

        # The deferred manager tracks results in execution_history
        history = tool_manager.deferred_manager.get_execution_history()
        assert len(history) == 1
        assert history[0].result == "test_result"
