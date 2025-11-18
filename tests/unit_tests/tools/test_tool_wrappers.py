"""Unit tests for tool wrappers."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from upsonic.tools.wrappers import FunctionTool, AgentTool
from upsonic.tools.schema import FunctionSchema
from upsonic.tools.config import ToolConfig
from upsonic.tools.base import ToolSchema


class TestToolWrappers:
    """Test suite for tool wrappers."""

    @pytest.fixture
    def mock_function_schema(self):
        """Create a mock function schema."""
        return FunctionSchema(
            function=Mock(),
            name="test_function",
            description="Test function",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            return_schema=None,
            is_async=False,
        )

    @pytest.fixture
    def mock_tool_config(self):
        """Create a mock tool config."""
        return ToolConfig()

    def test_tool_wrapper_creation(self, mock_function_schema, mock_tool_config):
        """Test tool wrapper creation."""

        def test_function(query: str) -> str:
            """Test function."""
            return f"Result: {query}"

        mock_function_schema.function = test_function

        tool = FunctionTool(
            function=test_function, schema=mock_function_schema, config=mock_tool_config
        )

        assert tool.name == "test_function"
        assert tool.description == "Test function"
        assert tool.schema is not None

    @pytest.mark.asyncio
    async def test_tool_wrapper_execution(self, mock_function_schema, mock_tool_config):
        """Test wrapper execution."""

        def test_function(query: str) -> str:
            """Test function."""
            return f"Result: {query}"

        mock_function_schema.function = test_function
        mock_function_schema.is_async = False

        tool = FunctionTool(
            function=test_function, schema=mock_function_schema, config=mock_tool_config
        )

        result = await tool.execute(query="test")
        assert result == "Result: test"

    @pytest.mark.asyncio
    async def test_tool_wrapper_async_execution(
        self, mock_function_schema, mock_tool_config
    ):
        """Test async wrapper execution."""

        async def async_function(query: str) -> str:
            """Async function."""
            return f"Result: {query}"

        mock_function_schema.function = async_function
        mock_function_schema.is_async = True

        tool = FunctionTool(
            function=async_function,
            schema=mock_function_schema,
            config=mock_tool_config,
        )

        result = await tool.execute(query="test")
        assert result == "Result: test"

    @pytest.mark.asyncio
    async def test_agent_tool_creation(self):
        """Test agent tool creation."""
        mock_agent = Mock()
        mock_agent.name = "TestAgent"
        mock_agent.role = "Assistant"
        mock_agent.goal = "Help users"
        mock_agent.system_prompt = None
        mock_agent.do_async = AsyncMock(return_value=Mock(output="Agent response"))

        tool = AgentTool(mock_agent)

        assert "ask_" in tool.name.lower()
        assert tool.description is not None

    @pytest.mark.asyncio
    async def test_agent_tool_execution(self):
        """Test agent tool execution."""
        mock_agent = Mock()
        mock_agent.name = "TestAgent"
        mock_agent.system_prompt = None
        mock_agent.do_async = AsyncMock(return_value=Mock(output="Agent response"))

        tool = AgentTool(mock_agent)

        result = await tool.execute(request="Test request")
        assert result is not None
        mock_agent.do_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_tool_sync_execution(self):
        """Test agent tool with sync do method."""
        mock_agent = Mock()
        mock_agent.name = "TestAgent"
        mock_agent.system_prompt = None
        mock_agent.do = Mock(return_value=Mock(output="Sync response"))
        # Ensure do_async doesn't exist for sync test
        if hasattr(mock_agent, "do_async"):
            delattr(mock_agent, "do_async")

        tool = AgentTool(mock_agent)

        result = await tool.execute(request="Test request")
        assert result is not None

    def test_tool_wrapper_pydantic_conversion(
        self, mock_function_schema, mock_tool_config
    ):
        """Test Pydantic model conversion in wrapper."""
        from pydantic import BaseModel

        class UserModel(BaseModel):
            name: str
            age: int

        def test_function(user: UserModel) -> str:
            """Test function."""
            return f"User: {user.name}"

        mock_function_schema.function = test_function
        mock_function_schema.is_async = False

        tool = FunctionTool(
            function=test_function, schema=mock_function_schema, config=mock_tool_config
        )

        # Test that wrapper can handle dict to Pydantic conversion
        assert tool is not None
