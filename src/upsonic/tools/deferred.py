"""Deferred and external tool execution handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from upsonic.tools.base import ToolCall, ToolResult


@dataclass
class ExternalToolCall:
    """Represents a tool call that must be executed externally."""
    
    tool_name: str
    """Name of the tool to execute."""
    
    tool_args: Dict[str, Any]
    """Arguments for the tool."""
    
    tool_call_id: str
    """Unique identifier for this tool call."""
    
    result: Optional[Any] = None
    """Result after external execution."""
    
    error: Optional[str] = None
    """Error message if execution failed."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""
    
    @property
    def args(self) -> Dict[str, Any]:
        """Backward compatibility alias for tool_args."""
        return self.tool_args


@dataclass
class DeferredToolRequests:
    """
    Tool calls that require approval or external execution.
    
    This can be used as output when the model calls deferred tools.
    Results can be passed back using DeferredToolResults.
    """
    
    calls: List[ToolCall] = field(default_factory=list)
    """Tool calls that require external execution."""
    
    approvals: List[ToolCall] = field(default_factory=list)
    """Tool calls that require human-in-the-loop approval."""
    
    def add_call(self, call: ToolCall) -> None:
        """Add a tool call for external execution."""
        self.calls.append(call)
    
    def add_approval(self, call: ToolCall) -> None:
        """Add a tool call requiring approval."""
        self.approvals.append(call)
    
    def is_empty(self) -> bool:
        """Check if there are no deferred requests."""
        return len(self.calls) == 0 and len(self.approvals) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'calls': [{'tool_name': c.tool_name, 'args': c.args, 'tool_call_id': c.tool_call_id} for c in self.calls],
            'approvals': [{'tool_name': c.tool_name, 'args': c.args, 'tool_call_id': c.tool_call_id} for c in self.approvals]
        }


@dataclass
class ToolApproval:
    """Approval decision for a tool call."""
    
    approved: bool
    """Whether the tool call is approved."""
    
    override_args: Optional[Dict[str, Any]] = None
    """Optional override arguments."""
    
    message: Optional[str] = None
    """Optional message about the decision."""


@dataclass
class DeferredToolResults:
    """
    Results for deferred tool calls from a previous run.
    
    Tool call IDs must match those from the DeferredToolRequests.
    """
    
    calls: Dict[str, Any] = field(default_factory=dict)
    """Results for tool calls that required external execution."""
    
    approvals: Dict[str, ToolApproval] = field(default_factory=dict)
    """Approval decisions for tools requiring approval."""
    
    def add_result(self, tool_call_id: str, result: Any) -> None:
        """Add a result for an external tool call."""
        self.calls[tool_call_id] = result
    
    def add_approval(
        self,
        tool_call_id: str,
        approved: bool,
        override_args: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None
    ) -> None:
        """Add an approval decision."""
        self.approvals[tool_call_id] = ToolApproval(
            approved=approved,
            override_args=override_args,
            message=message
        )
    
    def get_result(self, tool_call_id: str) -> Optional[Any]:
        """Get result for a tool call."""
        return self.calls.get(tool_call_id)
    
    def get_approval(self, tool_call_id: str) -> Optional[ToolApproval]:
        """Get approval for a tool call."""
        return self.approvals.get(tool_call_id)


class DeferredExecutionManager:
    """Manager for deferred and external tool execution."""
    
    def __init__(self):
        self.pending_requests = DeferredToolRequests()
        self.execution_history: List[ExternalToolCall] = []
    
    def create_external_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        tool_call_id: str,
        requires_approval: bool = False
    ) -> ExternalToolCall:
        """Create an external tool call."""
        call = ToolCall(
            tool_name=tool_name,
            args=args,
            tool_call_id=tool_call_id
        )
        
        if requires_approval:
            self.pending_requests.add_approval(call)
        else:
            self.pending_requests.add_call(call)
        
        external_call = ExternalToolCall(
            tool_name=tool_name,
            tool_args=args,
            tool_call_id=tool_call_id
        )
        
        self.execution_history.append(external_call)
        return external_call
    
    def process_results(
        self,
        results: DeferredToolResults
    ) -> List[ToolResult]:
        """
        Process deferred tool results and create return results.
        
        Args:
            results: Results from external execution
            
        Returns:
            List of tool results
        """
        tool_results = []
        
        # Process external execution results
        for tool_call_id, result in results.calls.items():
            # Find the original call
            original_call = self._find_call_by_id(tool_call_id)
            if original_call:
                tool_result = ToolResult(
                    tool_name=original_call.tool_name,
                    content=result,
                    tool_call_id=tool_call_id,
                    success=True
                )
                tool_results.append(tool_result)
                
                # Update execution history
                self._update_execution_history(tool_call_id, result)
        
        # Process approvals
        for tool_call_id, approval in results.approvals.items():
            original_call = self._find_approval_by_id(tool_call_id)
            if original_call:
                if approval.approved:
                    # Create approval result
                    tool_result = ToolResult(
                        tool_name=original_call.tool_name,
                        content={
                            'approved': True,
                            'message': approval.message or 'Approved',
                            'override_args': approval.override_args
                        },
                        tool_call_id=tool_call_id,
                        success=True
                    )
                else:
                    # Create denial result
                    tool_result = ToolResult(
                        tool_name=original_call.tool_name,
                        content=approval.message or 'Tool execution denied',
                        tool_call_id=tool_call_id,
                        success=False,
                        error='User denied execution'
                    )
                tool_results.append(tool_result)
        
        # Clear processed requests
        self._clear_processed_requests(results)
        
        return tool_results
    
    def _find_call_by_id(self, tool_call_id: str) -> Optional[ToolCall]:
        """Find a call by ID in pending requests."""
        for call in self.pending_requests.calls:
            if call.tool_call_id == tool_call_id:
                return call
        return None
    
    def _find_approval_by_id(self, tool_call_id: str) -> Optional[ToolCall]:
        """Find an approval request by ID."""
        for call in self.pending_requests.approvals:
            if call.tool_call_id == tool_call_id:
                return call
        return None
    
    def _update_execution_history(
        self,
        tool_call_id: str,
        result: Any
    ) -> None:
        """Update execution history with result."""
        for call in self.execution_history:
            if call.tool_call_id == tool_call_id:
                call.result = result
                break
    
    def _clear_processed_requests(self, results: DeferredToolResults) -> None:
        """Clear processed requests from pending."""
        # Clear processed calls
        processed_call_ids = set(results.calls.keys())
        self.pending_requests.calls = [
            call for call in self.pending_requests.calls
            if call.tool_call_id not in processed_call_ids
        ]
        
        # Clear processed approvals
        processed_approval_ids = set(results.approvals.keys())
        self.pending_requests.approvals = [
            call for call in self.pending_requests.approvals
            if call.tool_call_id not in processed_approval_ids
        ]
    
    def get_pending_requests(self) -> DeferredToolRequests:
        """Get all pending deferred requests."""
        return self.pending_requests
    
    def has_pending_requests(self) -> bool:
        """Check if there are pending requests."""
        return not self.pending_requests.is_empty()
    
    def get_execution_history(self) -> List[ExternalToolCall]:
        """Get the full execution history."""
        return self.execution_history.copy()
