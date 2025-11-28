"""Unit tests for deferred tool execution."""

import pytest
from unittest.mock import Mock
from typing import Dict, Any

from upsonic.tools.deferred import (
    DeferredExecutionManager,
    ExternalToolCall,
)
from upsonic.tools.base import ToolResult


class TestDeferredTools:
    """Test suite for deferred tools."""

    @pytest.fixture
    def deferred_manager(self):
        """Create a DeferredExecutionManager instance for testing."""
        return DeferredExecutionManager()

    def test_deferred_tool_creation(self, deferred_manager):
        """Test deferred tool creation."""
        external_call = deferred_manager.create_external_call(
            tool_name="test_tool",
            args={"param": "value"},
            tool_call_id="123",
        )

        assert isinstance(external_call, ExternalToolCall)
        assert external_call.tool_name == "test_tool"
        assert external_call.tool_args == {"param": "value"}
        assert external_call.tool_call_id == "123"

    def test_deferred_tool_update_result(self, deferred_manager):
        """Test updating external call result."""
        external_call = deferred_manager.create_external_call(
            tool_name="test_tool", args={"param": "value"}, tool_call_id="123"
        )

        updated_call = deferred_manager.update_call_result(
            tool_call_id="123",
            result="execution_result"
        )

        assert updated_call is not None
        assert updated_call.result == "execution_result"
        assert updated_call.error is None

    def test_deferred_tool_update_error(self, deferred_manager):
        """Test updating external call with error."""
        external_call = deferred_manager.create_external_call(
            tool_name="test_tool", args={}, tool_call_id="123"
        )

        updated_call = deferred_manager.update_call_result(
            tool_call_id="123",
            error="Test error"
        )

        assert updated_call is not None
        assert updated_call.error == "Test error"
        assert updated_call.result is None

    def test_deferred_tool_execution_history(self, deferred_manager):
        """Test execution history tracking."""
        external_call = deferred_manager.create_external_call(
            tool_name="test_tool", args={}, tool_call_id="123"
        )

        history = deferred_manager.get_execution_history()
        assert len(history) == 1
        assert history[0] == external_call

    def test_deferred_tool_pending_requests(self, deferred_manager):
        """Test pending requests management."""
        assert deferred_manager.has_pending_requests() is False

        deferred_manager.create_external_call(
            tool_name="test", args={}, tool_call_id="123"
        )

        assert deferred_manager.has_pending_requests() is True

        pending_calls = deferred_manager.get_pending_calls()
        assert len(pending_calls) == 1
        assert pending_calls[0].tool_name == "test"
