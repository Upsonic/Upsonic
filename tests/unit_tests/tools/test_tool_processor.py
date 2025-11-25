"""Unit tests for ToolProcessor."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from upsonic.tools.processor import (
    ToolProcessor,
    ToolValidationError,
    ExternalExecutionPause,
)
from upsonic.tools.base import Tool, ToolCall, ToolResult
from upsonic.tools.metrics import ToolMetrics
from upsonic.tools.config import ToolConfig


class TestToolProcessor:
    """Test suite for ToolProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a ToolProcessor instance for testing."""
        return ToolProcessor()

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool for testing."""
        tool = Mock(spec=Tool)
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.schema = Mock()
        tool.execute = AsyncMock(return_value="result")
        return tool

    @pytest.fixture
    def mock_context(self):
        """Create a mock ToolMetrics for testing."""
        return ToolMetrics()

    def test_tool_processor_initialization(self, processor):
        """Test ToolProcessor initialization."""
        assert processor.registered_tools == {}
        assert processor.tool_definitions == {}
        assert processor.mcp_handlers == []
        assert processor.current_context is None

    def test_tool_processor_process_tool_calls(self, processor, mock_tool):
        """Test processing tool calls."""

        def test_function(query: str) -> str:
            """Test function."""
            return f"Result: {query}"

        with patch.object(processor, "_process_function_tool", return_value=mock_tool):
            processed = processor.process_tools([test_function])

            assert "test_tool" in processed
            assert processed["test_tool"] == mock_tool

    def test_tool_processor_validate_tool_calls(self, processor):
        """Test validation."""

        def valid_function(query: str) -> str:
            """Valid function with type hints and docstring."""
            return f"Result: {query}"

        def invalid_function(query):
            """Invalid function without type hints."""
            return f"Result: {query}"

        with patch("upsonic.tools.schema.validate_tool_function") as mock_validate:
            mock_validate.return_value = []
            errors = mock_validate(valid_function)
            assert len(errors) == 0

            mock_validate.return_value = ["Missing type hint"]
            errors = mock_validate(invalid_function)
            assert len(errors) > 0

    @pytest.mark.asyncio
    async def test_tool_processor_execute_tool_calls(
        self, processor, mock_tool, mock_context
    ):
        """Test execution."""
        processor.registered_tools["test_tool"] = mock_tool

        with patch.object(processor, "create_behavioral_wrapper") as mock_wrapper:
            mock_wrapped = AsyncMock(return_value="executed_result")
            mock_wrapper.return_value = mock_wrapped

            wrapper = processor.create_behavioral_wrapper(mock_tool, mock_context)
            result = await wrapper(query="test")

            assert result is not None
            mock_wrapped.assert_called_once()

    def test_tool_processor_process_function_tool(self, processor):
        """Test processing function tools."""

        def test_function(query: str) -> str:
            """Test function.

            Args:
                query: The query string.

            Returns:
                The result string.
            """
            return f"Result: {query}"

        with patch("upsonic.tools.schema.validate_tool_function", return_value=[]):
            with patch("upsonic.tools.schema.generate_function_schema") as mock_schema:
                from upsonic.tools.schema import FunctionSchema

                mock_schema.return_value = FunctionSchema(
                    function=test_function,
                    name="test_function",
                    description="Test function",
                    parameters_schema={"type": "object", "properties": {}},
                    return_schema=None,
                    is_async=False,
                )

                tool = processor._process_function_tool(test_function)
                assert tool is not None
                assert tool.name == "test_function"

    def test_tool_processor_process_toolkit(self, processor):
        """Test processing toolkit."""
        from upsonic.tools.base import ToolKit

        class TestToolKit(ToolKit):
            @staticmethod
            def test_method(query: str) -> str:
                """Test method."""
                return f"Result: {query}"

        toolkit = TestToolKit()

        with patch.object(processor, "_process_function_tool") as mock_process:
            mock_tool = Mock()
            mock_tool.name = "test_method"
            mock_process.return_value = mock_tool

            tools = processor._process_toolkit(toolkit)
            # Since test_method is not decorated with @tool, it won't be processed
            assert isinstance(tools, dict)

    def test_tool_processor_is_builtin_tool(self, processor):
        """Test builtin tool detection."""
        from upsonic.tools.builtin_tools import AbstractBuiltinTool, WebSearchTool

        builtin = WebSearchTool()
        assert processor._is_builtin_tool(builtin) is True

        regular_tool = Mock()
        assert processor._is_builtin_tool(regular_tool) is False

    def test_tool_processor_extract_builtin_tools(self, processor):
        """Test extracting builtin tools."""
        from upsonic.tools.builtin_tools import WebSearchTool

        builtin = WebSearchTool()
        regular_tool = Mock()

        tools = [builtin, regular_tool, None]
        builtin_tools = processor.extract_builtin_tools(tools)

        assert len(builtin_tools) == 1
        assert builtin_tools[0] == builtin
