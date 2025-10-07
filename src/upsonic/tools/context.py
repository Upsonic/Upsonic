"""Context system for tool execution."""

from __future__ import annotations

import dataclasses
from dataclasses import field
from typing import TYPE_CHECKING, Any, Dict, Generic, Optional, TypeVar

from upsonic._utils import dataclasses_no_defaults_repr, now_utc

if TYPE_CHECKING:
    from upsonic.tools.base import ToolCall


AgentDepsT = TypeVar('AgentDepsT')
"""Type variable for agent dependencies."""


@dataclasses.dataclass(repr=False, kw_only=True)
class ToolContext(Generic[AgentDepsT]):
    """Context information for tool execution."""
    
    deps: AgentDepsT
    """Dependencies for the agent."""
    
    agent_id: Optional[str] = None
    """ID of the agent executing the tool."""
    
    task_id: Optional[str] = None
    """ID of the current task."""
    
    tool_call: Optional[ToolCall] = None
    """The current tool call being executed."""
    
    messages: list[Any] = field(default_factory=list)
    """Messages exchanged in the conversation so far."""
    
    retries: Dict[str, int] = field(default_factory=dict)
    """Number of retries for each tool so far."""
    
    retry: int = 0
    """Number of retries of current tool so far."""
    
    max_retries: int = 0
    """Maximum number of retries allowed."""
    
    tool_call_count: int = 0
    """Total number of tool calls made."""
    
    tool_call_limit: Optional[int] = None
    """Maximum number of tool calls allowed."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the context."""
    
    @property
    def last_attempt(self) -> bool:
        """Whether this is the last attempt at running this tool."""
        return self.retry == self.max_retries
    
    def can_call_tool(self) -> bool:
        """Check if another tool call is allowed."""
        if self.tool_call_limit is None:
            return True
        return self.tool_call_count < self.tool_call_limit
    
    def increment_tool_count(self) -> None:
        """Increment the tool call count."""
        self.tool_call_count += 1
    
    def get_retry_count(self, tool_name: str) -> int:
        """Get the retry count for a specific tool."""
        return self.retries.get(tool_name, 0)
    
    def increment_retry(self, tool_name: str) -> None:
        """Increment the retry count for a tool."""
        self.retries[tool_name] = self.retries.get(tool_name, 0) + 1
        if self.tool_call and self.tool_call.tool_name == tool_name:
            self.retry = self.retries[tool_name]
    
    __repr__ = dataclasses_no_defaults_repr
