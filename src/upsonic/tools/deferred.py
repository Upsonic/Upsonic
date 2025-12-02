"""Deferred and external tool execution handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


class DeferredExecutionManager:
    """Manager for external tool execution."""
    
    def __init__(self):
        self.execution_history: List[ExternalToolCall] = []
    
    def create_external_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        tool_call_id: str
    ) -> ExternalToolCall:
        """Create an external tool call."""
        external_call = ExternalToolCall(
            tool_name=tool_name,
            tool_args=args,
            tool_call_id=tool_call_id
        )
        
        self.execution_history.append(external_call)
        return external_call
    
    def has_pending_requests(self) -> bool:
        """Check if there are pending external tool calls."""
        return any(
            call.result is None and call.error is None
            for call in self.execution_history
        )
    
    def get_pending_calls(self) -> List[ExternalToolCall]:
        """Get all pending external tool calls."""
        return [
            call for call in self.execution_history
            if call.result is None and call.error is None
        ]
    
    def update_call_result(
        self,
        tool_call_id: str,
        result: Any = None,
        error: Optional[str] = None
    ) -> Optional[ExternalToolCall]:
        """
        Update the result of an external tool call.
        
        Args:
            tool_call_id: ID of the tool call to update
            result: The result of the execution
            error: Error message if execution failed
            
        Returns:
            The updated ExternalToolCall or None if not found
        """
        for call in self.execution_history:
            if call.tool_call_id == tool_call_id:
                call.result = result
                call.error = error
                return call
        return None
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
    
    def get_execution_history(self) -> List[ExternalToolCall]:
        """Get the full execution history."""
        return self.execution_history.copy()
