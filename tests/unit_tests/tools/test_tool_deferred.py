"""Unit tests for deferred tool execution."""

import pytest
from unittest.mock import Mock
from typing import Dict, Any

from upsonic.tools.deferred import (
    DeferredExecutionManager,
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproval,
    ExternalToolCall,
)
from upsonic.tools.base import ToolCall, ToolResult


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
            requires_approval=False,
        )

        assert isinstance(external_call, ExternalToolCall)
        assert external_call.tool_name == "test_tool"
        assert external_call.tool_args == {"param": "value"}
        assert external_call.tool_call_id == "123"

    @pytest.mark.asyncio
    async def test_deferred_tool_execution(self, deferred_manager):
        """Test deferred execution."""
        # Create external call
        external_call = deferred_manager.create_external_call(
            tool_name="test_tool", args={"param": "value"}, tool_call_id="123"
        )

        # Create results
        results = DeferredToolResults()
        results.add_result("123", "execution_result")

        # Process results
        tool_results = deferred_manager.process_results(results)

        assert len(tool_results) == 1
        assert tool_results[0].tool_name == "test_tool"
        assert tool_results[0].content == "execution_result"
        assert tool_results[0].success is True

    def test_deferred_tool_requests(self):
        """Test deferred tool requests."""
        requests = DeferredToolRequests()

        call = ToolCall(tool_name="test", args={"param": "value"}, tool_call_id="123")
        requests.add_call(call)

        assert len(requests.calls) == 1
        assert requests.is_empty() is False

    def test_deferred_tool_approvals(self):
        """Test deferred tool approvals."""
        requests = DeferredToolRequests()

        call = ToolCall(tool_name="test", args={"param": "value"}, tool_call_id="123")
        requests.add_approval(call)

        assert len(requests.approvals) == 1
        assert requests.is_empty() is False

    def test_deferred_tool_results(self):
        """Test deferred tool results."""
        results = DeferredToolResults()

        results.add_result("123", "result_data")
        assert results.get_result("123") == "result_data"
        assert results.get_result("nonexistent") is None

    def test_deferred_tool_approval_decision(self):
        """Test approval decision."""
        results = DeferredToolResults()

        results.add_approval("123", approved=True, message="Approved")
        approval = results.get_approval("123")

        assert approval is not None
        assert approval.approved is True
        assert approval.message == "Approved"

    def test_deferred_tool_denial(self, deferred_manager):
        """Test tool denial."""
        call = ToolCall(tool_name="test", args={}, tool_call_id="123")
        deferred_manager.pending_requests.add_approval(call)

        results = DeferredToolResults()
        results.add_approval("123", approved=False, message="Denied")

        tool_results = deferred_manager.process_results(results)

        assert len(tool_results) == 1
        assert tool_results[0].success is False
        assert "Denied" in tool_results[0].content

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

        requests = deferred_manager.get_pending_requests()
        assert len(requests.calls) == 1
