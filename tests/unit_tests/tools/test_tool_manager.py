"""Unit tests for ToolManager."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any

from upsonic.tools import ToolManager
from upsonic.tools.base import Tool, ToolResult, ToolDefinition
from upsonic.tools.metrics import ToolMetrics


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
        """Test ToolManager initialization composes the five collaborators."""
        assert tool_manager.normalizer is not None
        assert tool_manager.registry is not None
        assert tool_manager.wrapper is not None
        assert tool_manager.pause_handler is not None
        assert tool_manager.orchestrator_lifecycle is not None
        assert tool_manager.registry.registered_tools == {}
        assert tool_manager.registry.wrapped_tools == {}
        assert tool_manager.orchestrator_lifecycle.get_orchestrator() is None

    def test_tool_manager_add_tool(self, tool_manager, mock_tool, mock_context):
        """Test adding tools."""

        def test_function(query: str) -> str:
            """Test function."""
            return f"Result: {query}"

        from upsonic.tools.normalizer import NormalizationResult

        with patch.object(
            tool_manager.normalizer,
            "normalize",
            return_value=NormalizationResult(tools={"test_function": mock_tool}),
        ):
            with patch.object(
                tool_manager.wrapper,
                "wrap",
                return_value=AsyncMock(),
            ):
                registered = tool_manager.register_tools(
                    tools=[test_function]
                )

                assert "test_function" in registered
                assert "test_function" in tool_manager.registry.wrapped_tools

    def test_tool_manager_remove_tool(self, tool_manager, mock_tool):
        """Test removing tools."""
        tool_manager.registry.wrapped_tools["test_tool"] = AsyncMock()
        tool_manager.registry.registered_tools["test_tool"] = mock_tool

        # Remove tool
        del tool_manager.registry.wrapped_tools["test_tool"]
        del tool_manager.registry.registered_tools["test_tool"]

        assert "test_tool" not in tool_manager.registry.wrapped_tools
        assert "test_tool" not in tool_manager.registry.registered_tools

    def test_tool_manager_get_tool(self, tool_manager, mock_tool):
        """Test getting tools."""
        tool_manager.registry.registered_tools["test_tool"] = mock_tool

        tool = tool_manager.registry.registered_tools.get("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"

    def test_tool_manager_list_tools(self, tool_manager, mock_tool):
        """Test listing tools."""
        tool_manager.registry.registered_tools["test_tool"] = mock_tool
        tool_manager.registry.registered_tools["another_tool"] = mock_tool

        tools = list(tool_manager.registry.registered_tools.keys())
        assert "test_tool" in tools
        assert "another_tool" in tools
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_tool_manager_execute_tool(self, tool_manager, mock_tool):
        """Test tool execution."""
        mock_wrapper = AsyncMock(return_value={"func": "test_result"})
        tool_manager.registry.wrapped_tools["test_tool"] = mock_wrapper

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
        tool_manager.registry.wrapped_tools["test_tool"] = mock_wrapper

        result = await tool_manager.execute_tool(tool_name="test_tool", args={})

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert "Test error" in result.error

    def test_tool_manager_get_tool_definitions(self, tool_manager, mock_tool):
        """Test getting tool definitions."""
        tool_manager.registry.registered_tools["test_tool"] = mock_tool

        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 1
        assert isinstance(definitions[0], ToolDefinition)
        assert definitions[0].name == "test_tool"
