"""Metrics system for tool execution tracking."""

from __future__ import annotations

import dataclasses
from typing import Optional

from upsonic._utils import dataclasses_no_defaults_repr


@dataclasses.dataclass(repr=False, kw_only=True)
class ToolMetrics:
    """Metrics for tracking tool execution."""
    
    tool_call_count: int = 0
    """Total number of tool calls made."""
    
    tool_call_limit: Optional[int] = None
    """Maximum number of tool calls allowed."""
    
    def can_call_tool(self) -> bool:
        """Check if another tool call is allowed."""
        if self.tool_call_limit is None:
            return True
        return self.tool_call_count < self.tool_call_limit
    
    def increment_tool_count(self) -> None:
        """Increment the tool call count."""
        self.tool_call_count += 1
    
    __repr__ = dataclasses_no_defaults_repr
