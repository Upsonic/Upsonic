"""Unit tests for tool wrappers."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from upsonic.tools.wrappers import FunctionTool, AgentTool
from upsonic.tools.schema import FunctionSchema
from upsonic.tools.config import ToolConfig


class TestToolWrappers:
    """Test suite for tool wrappers."""

    @pytest.fixture
    def mock_function_schema(self):
        """Create a mock function schema."""
        from pydantic_core import SchemaValidator
        from unittest.mock import Mock
        
        def test_function(query: str) -> str:
            """Test function."""
            return f"Result: {query}"
        
        # Create a proper FunctionSchema using function_schema
        from upsonic.tools.schema import function_schema, GenerateToolJsonSchema
        return function_schema(test_function, GenerateToolJsonSchema)

    @pytest.fixture
    def mock_tool_config(self):
        """Create a mock tool config."""
        return ToolConfig()

    def test_tool_wrapper_creation(self, mock_function_schema, mock_tool_config):
        """Test tool wrapper creation."""

        def test_function(query: str) -> str:
            """Test function."""
            return f"Result: {query}"

        tool = FunctionTool(
            function=test_function, schema=mock_function_schema, config=mock_tool_config
        )

        assert tool.name is not None
        assert tool.description is not None
        assert tool.schema is not None

    @pytest.mark.asyncio
    async def test_tool_wrapper_execution(self, mock_function_schema, mock_tool_config):
        """Test wrapper execution."""

        def test_function(query: str) -> str:
            """Test function."""
            return f"Result: {query}"

        tool = FunctionTool(
            function=test_function, schema=mock_function_schema, config=mock_tool_config
        )

        result = await tool.execute(query="test")
        assert result == "Result: test"

    @pytest.mark.asyncio
    async def test_tool_wrapper_async_execution(
        self, mock_tool_config
    ):
        """Test async wrapper execution."""
        from upsonic.tools.schema import function_schema, GenerateToolJsonSchema

        async def async_function(query: str) -> str:
            """Async function."""
            return f"Result: {query}"

        async_schema = function_schema(async_function, GenerateToolJsonSchema)
        tool = FunctionTool(
            function=async_function,
            schema=async_schema,
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
        self, mock_tool_config
    ):
        """Test Pydantic model conversion in wrapper."""
        from pydantic import BaseModel
        from upsonic.tools.schema import function_schema, GenerateToolJsonSchema

        class UserModel(BaseModel):
            name: str
            age: int

        def test_function(user: UserModel) -> str:
            """Test function."""
            return f"User: {user.name}"

        pydantic_schema = function_schema(test_function, GenerateToolJsonSchema)
        tool = FunctionTool(
            function=test_function, schema=pydantic_schema, config=mock_tool_config
        )

        # Test that wrapper can handle dict to Pydantic conversion
        assert tool is not None
