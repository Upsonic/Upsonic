"""
Run Events Module

This module provides event classes for agent run streaming.
Events are emitted during agent execution to provide real-time visibility
into the agent's activity, including tool calls, content generation, and errors.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import time
import uuid

if TYPE_CHECKING:
    from upsonic.usage import RequestUsage
    from upsonic.messages.messages import ToolCallPart, ToolReturnPart


class RunStatus(str, Enum):
    """Status of an agent run."""
    
    pending = "pending"
    """Run is pending start."""
    
    running = "running"
    """Run is currently executing."""
    
    completed = "completed"
    """Run completed successfully."""
    
    paused = "paused"
    """Run is paused (e.g., waiting for external tool execution)."""
    
    cancelled = "cancelled"
    """Run was cancelled."""
    
    error = "error"
    """Run encountered an error."""


class RunEvent(str, Enum):
    """Events that can be sent by the run() functions.
    
    These events map to the agent execution pipeline steps and provide
    visibility into the agent's activity during streaming.
    """

    # Run lifecycle events
    run_started = "RunStarted"
    """Emitted when the run begins."""
    
    run_content = "RunContent"
    """Emitted for each delta of content (text chunk)."""
    
    run_content_completed = "RunContentCompleted"
    """Emitted when the current content generation is complete."""
    
    run_completed = "RunCompleted"
    """Emitted when the run completes successfully."""
    
    run_error = "RunError"
    """Emitted when an error occurs."""
    
    run_cancelled = "RunCancelled"
    """Emitted when the run is cancelled."""
    
    run_paused = "RunPaused"
    """Emitted when the run is paused for external execution."""
    
    run_continued = "RunContinued"
    """Emitted when a paused run is continued."""

    # Tool events
    tool_call_started = "ToolCallStarted"
    """Emitted when a tool call begins."""
    
    tool_call_completed = "ToolCallCompleted"
    """Emitted when a tool call completes."""

    # Thinking/Reasoning events (for enable_thinking_tool)
    thinking_started = "ThinkingStarted"
    """Emitted when thinking/planning begins."""
    
    thinking_step = "ThinkingStep"
    """Emitted for each thinking step."""
    
    thinking_completed = "ThinkingCompleted"
    """Emitted when thinking/planning completes."""

    # Cache events
    cache_hit = "CacheHit"
    """Emitted when a cache hit occurs."""
    
    cache_miss = "CacheMiss"
    """Emitted when a cache miss occurs."""

    # Policy events
    policy_check_started = "PolicyCheckStarted"
    """Emitted when policy validation starts."""
    
    policy_check_completed = "PolicyCheckCompleted"
    """Emitted when policy validation completes."""


@dataclass
class BaseAgentRunEvent:
    """Base class for all agent run events.
    
    All events share common fields for identification and tracking.
    
    Attributes:
        event: The event type identifier
        created_at: Unix timestamp when the event was created
        agent_id: The unique identifier of the agent
        agent_name: The display name of the agent
        run_id: The unique identifier of the run
        session_id: Optional session identifier for conversation tracking
        content: Optional content associated with the event
    """
    
    event: str = ""
    """The event type identifier (from RunEvent enum)."""
    
    created_at: int = field(default_factory=lambda: int(time.time()))
    """Unix timestamp when the event was created."""
    
    agent_id: str = ""
    """The unique identifier of the agent."""
    
    agent_name: str = ""
    """The display name of the agent."""
    
    run_id: Optional[str] = None
    """The unique identifier of the run."""
    
    session_id: Optional[str] = None
    """Optional session identifier for conversation tracking."""
    
    content: Optional[Any] = None
    """Optional content associated with the event."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the event
        """
        result = {
            "event": self.event,
            "created_at": self.created_at,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
        }
        
        if self.run_id is not None:
            result["run_id"] = self.run_id
        if self.session_id is not None:
            result["session_id"] = self.session_id
        if self.content is not None:
            if hasattr(self.content, 'to_dict'):
                result["content"] = self.content.to_dict()
            elif hasattr(self.content, 'model_dump'):
                result["content"] = self.content.model_dump(exclude_none=True)
            else:
                result["content"] = self.content
                
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseAgentRunEvent":
        """Create event from dictionary.
        
        Args:
            data: Dictionary representation of the event
            
        Returns:
            BaseAgentRunEvent: Reconstructed event instance
        """
        return cls(
            event=data.get("event", ""),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content")
        )


@dataclass
class RunStartedEvent(BaseAgentRunEvent):
    """Event emitted when a run starts.
    
    Contains information about the model and agent configuration.
    """
    
    event: str = field(default=RunEvent.run_started.value)
    """Event type identifier."""
    
    model: str = ""
    """The model being used for this run."""
    
    model_provider: str = ""
    """The provider of the model."""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["model"] = self.model
        result["model_provider"] = self.model_provider
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunStartedEvent":
        return cls(
            event=data.get("event", RunEvent.run_started.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            model=data.get("model", ""),
            model_provider=data.get("model_provider", ""),
        )


@dataclass
class RunContentEvent(BaseAgentRunEvent):
    """Event emitted for each content delta during streaming.
    
    This is the primary event for text content streaming.
    """
    
    event: str = field(default=RunEvent.run_content.value)
    """Event type identifier."""
    
    content: Optional[str] = None
    """The text content delta."""
    
    content_type: str = "str"
    """The type of content (str, json, etc)."""
    
    thinking_content: Optional[str] = None
    """Optional thinking/reasoning content if available."""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["content_type"] = self.content_type
        if self.thinking_content is not None:
            result["thinking_content"] = self.thinking_content
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunContentEvent":
        return cls(
            event=data.get("event", RunEvent.run_content.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            content_type=data.get("content_type", "str"),
            thinking_content=data.get("thinking_content"),
        )


@dataclass
class RunContentCompletedEvent(BaseAgentRunEvent):
    """Event emitted when content generation is complete."""
    
    event: str = field(default=RunEvent.run_content_completed.value)
    """Event type identifier."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunContentCompletedEvent":
        return cls(
            event=data.get("event", RunEvent.run_content_completed.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
        )


@dataclass
class RunCompletedEvent(BaseAgentRunEvent):
    """Event emitted when the run completes successfully.
    
    Contains the final output and usage metrics.
    """
    
    event: str = field(default=RunEvent.run_completed.value)
    """Event type identifier."""
    
    content: Optional[Any] = None
    """The final output content."""
    
    content_type: str = "str"
    """The type of content."""
    
    usage: Optional[Dict[str, Any]] = None
    """Token usage information."""
    
    duration_ms: Optional[int] = None
    """Duration of the run in milliseconds."""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["content_type"] = self.content_type
        if self.usage is not None:
            result["usage"] = self.usage
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunCompletedEvent":
        return cls(
            event=data.get("event", RunEvent.run_completed.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            content_type=data.get("content_type", "str"),
            usage=data.get("usage"),
            duration_ms=data.get("duration_ms"),
        )


@dataclass
class RunErrorEvent(BaseAgentRunEvent):
    """Event emitted when an error occurs during the run."""
    
    event: str = field(default=RunEvent.run_error.value)
    """Event type identifier."""
    
    error_type: Optional[str] = None
    """The type/class of the error."""
    
    error_message: Optional[str] = None
    """The error message."""
    
    error_id: Optional[str] = None
    """Optional unique identifier for the error."""
    
    additional_data: Optional[Dict[str, Any]] = None
    """Additional error context."""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.error_type is not None:
            result["error_type"] = self.error_type
        if self.error_message is not None:
            result["error_message"] = self.error_message
        if self.error_id is not None:
            result["error_id"] = self.error_id
        if self.additional_data is not None:
            result["additional_data"] = self.additional_data
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunErrorEvent":
        return cls(
            event=data.get("event", RunEvent.run_error.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            error_id=data.get("error_id"),
            additional_data=data.get("additional_data"),
        )


@dataclass
class RunCancelledEvent(BaseAgentRunEvent):
    """Event emitted when the run is cancelled."""
    
    event: str = field(default=RunEvent.run_cancelled.value)
    """Event type identifier."""
    
    reason: Optional[str] = None
    """Reason for cancellation."""

    @property
    def is_cancelled(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.reason is not None:
            result["reason"] = self.reason
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunCancelledEvent":
        return cls(
            event=data.get("event", RunEvent.run_cancelled.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            reason=data.get("reason"),
        )


@dataclass
class RunPausedEvent(BaseAgentRunEvent):
    """Event emitted when the run is paused for external execution."""
    
    event: str = field(default=RunEvent.run_paused.value)
    """Event type identifier."""
    
    tools_awaiting_execution: Optional[List[Dict[str, Any]]] = None
    """List of tools waiting for external execution."""
    
    pause_reason: Optional[str] = None
    """Reason for the pause."""

    @property
    def is_paused(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.tools_awaiting_execution is not None:
            result["tools_awaiting_execution"] = self.tools_awaiting_execution
        if self.pause_reason is not None:
            result["pause_reason"] = self.pause_reason
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunPausedEvent":
        return cls(
            event=data.get("event", RunEvent.run_paused.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            tools_awaiting_execution=data.get("tools_awaiting_execution"),
            pause_reason=data.get("pause_reason"),
        )


@dataclass
class RunContinuedEvent(BaseAgentRunEvent):
    """Event emitted when a paused run is continued."""
    
    event: str = field(default=RunEvent.run_continued.value)
    """Event type identifier."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunContinuedEvent":
        return cls(
            event=data.get("event", RunEvent.run_continued.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
        )


@dataclass
class ToolCallStartedEvent(BaseAgentRunEvent):
    """Event emitted when a tool call starts.
    
    Contains tool name and arguments.
    """
    
    event: str = field(default=RunEvent.tool_call_started.value)
    """Event type identifier."""
    
    tool_name: str = ""
    """The name of the tool being called."""
    
    tool_call_id: Optional[str] = None
    """The unique identifier for this tool call."""
    
    tool_args: Optional[Dict[str, Any]] = None
    """The arguments passed to the tool."""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["tool_name"] = self.tool_name
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_args is not None:
            result["tool_args"] = self.tool_args
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallStartedEvent":
        return cls(
            event=data.get("event", RunEvent.tool_call_started.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            tool_name=data.get("tool_name", ""),
            tool_call_id=data.get("tool_call_id"),
            tool_args=data.get("tool_args"),
        )


@dataclass
class ToolCallCompletedEvent(BaseAgentRunEvent):
    """Event emitted when a tool call completes.
    
    Contains the tool result.
    """
    
    event: str = field(default=RunEvent.tool_call_completed.value)
    """Event type identifier."""
    
    tool_name: str = ""
    """The name of the tool that was called."""
    
    tool_call_id: Optional[str] = None
    """The unique identifier for this tool call."""
    
    tool_result: Optional[Any] = None
    """The result returned by the tool."""
    
    duration_ms: Optional[int] = None
    """Duration of the tool call in milliseconds."""
    
    error: Optional[str] = None
    """Error message if the tool call failed."""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["tool_name"] = self.tool_name
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_result is not None:
            if hasattr(self.tool_result, 'to_dict'):
                result["tool_result"] = self.tool_result.to_dict()
            elif hasattr(self.tool_result, 'model_dump'):
                result["tool_result"] = self.tool_result.model_dump(exclude_none=True)
            else:
                result["tool_result"] = self.tool_result
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.error is not None:
            result["error"] = self.error
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallCompletedEvent":
        return cls(
            event=data.get("event", RunEvent.tool_call_completed.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            tool_name=data.get("tool_name", ""),
            tool_call_id=data.get("tool_call_id"),
            tool_result=data.get("tool_result"),
            duration_ms=data.get("duration_ms"),
            error=data.get("error"),
        )


@dataclass
class ThinkingStartedEvent(BaseAgentRunEvent):
    """Event emitted when thinking/planning begins."""
    
    event: str = field(default=RunEvent.thinking_started.value)
    """Event type identifier."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThinkingStartedEvent":
        return cls(
            event=data.get("event", RunEvent.thinking_started.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
        )


@dataclass
class ThinkingStepEvent(BaseAgentRunEvent):
    """Event emitted for each thinking step."""
    
    event: str = field(default=RunEvent.thinking_step.value)
    """Event type identifier."""
    
    thinking_content: str = ""
    """The thinking/reasoning content."""
    
    step_number: Optional[int] = None
    """The step number in the thinking process."""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["thinking_content"] = self.thinking_content
        if self.step_number is not None:
            result["step_number"] = self.step_number
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThinkingStepEvent":
        return cls(
            event=data.get("event", RunEvent.thinking_step.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            thinking_content=data.get("thinking_content", ""),
            step_number=data.get("step_number"),
        )


@dataclass
class ThinkingCompletedEvent(BaseAgentRunEvent):
    """Event emitted when thinking/planning completes."""
    
    event: str = field(default=RunEvent.thinking_completed.value)
    """Event type identifier."""
    
    thinking_summary: Optional[str] = None
    """Summary of the thinking process."""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.thinking_summary is not None:
            result["thinking_summary"] = self.thinking_summary
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThinkingCompletedEvent":
        return cls(
            event=data.get("event", RunEvent.thinking_completed.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            thinking_summary=data.get("thinking_summary"),
        )


@dataclass 
class CacheHitEvent(BaseAgentRunEvent):
    """Event emitted when a cache hit occurs."""
    
    event: str = field(default=RunEvent.cache_hit.value)
    """Event type identifier."""
    
    cache_method: Optional[str] = None
    """The cache method used (vector_search, llm_call)."""
    
    similarity: Optional[float] = None
    """Similarity score for vector search cache."""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.cache_method is not None:
            result["cache_method"] = self.cache_method
        if self.similarity is not None:
            result["similarity"] = self.similarity
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheHitEvent":
        return cls(
            event=data.get("event", RunEvent.cache_hit.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            cache_method=data.get("cache_method"),
            similarity=data.get("similarity"),
        )


@dataclass
class CacheMissEvent(BaseAgentRunEvent):
    """Event emitted when a cache miss occurs."""
    
    event: str = field(default=RunEvent.cache_miss.value)
    """Event type identifier."""
    
    cache_method: Optional[str] = None
    """The cache method used."""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.cache_method is not None:
            result["cache_method"] = self.cache_method
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheMissEvent":
        return cls(
            event=data.get("event", RunEvent.cache_miss.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            cache_method=data.get("cache_method"),
        )


@dataclass
class PolicyCheckStartedEvent(BaseAgentRunEvent):
    """Event emitted when policy validation starts."""
    
    event: str = field(default=RunEvent.policy_check_started.value)
    """Event type identifier."""
    
    policy_type: Optional[str] = None
    """The type of policy being checked (user, agent, tool_pre, tool_post)."""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.policy_type is not None:
            result["policy_type"] = self.policy_type
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyCheckStartedEvent":
        return cls(
            event=data.get("event", RunEvent.policy_check_started.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            policy_type=data.get("policy_type"),
        )


@dataclass
class PolicyCheckCompletedEvent(BaseAgentRunEvent):
    """Event emitted when policy validation completes."""
    
    event: str = field(default=RunEvent.policy_check_completed.value)
    """Event type identifier."""
    
    policy_type: Optional[str] = None
    """The type of policy that was checked."""
    
    action_taken: Optional[str] = None
    """The action taken by the policy (ALLOW, BLOCK, REPLACE, etc)."""
    
    blocked: bool = False
    """Whether the policy blocked the input/output."""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.policy_type is not None:
            result["policy_type"] = self.policy_type
        if self.action_taken is not None:
            result["action_taken"] = self.action_taken
        result["blocked"] = self.blocked
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyCheckCompletedEvent":
        return cls(
            event=data.get("event", RunEvent.policy_check_completed.value),
            created_at=data.get("created_at", int(time.time())),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            run_id=data.get("run_id"),
            session_id=data.get("session_id"),
            content=data.get("content"),
            policy_type=data.get("policy_type"),
            action_taken=data.get("action_taken"),
            blocked=data.get("blocked", False),
        )


# Type alias for all agent run event types
AgentRunEvent = Union[
    RunStartedEvent,
    RunContentEvent,
    RunContentCompletedEvent,
    RunCompletedEvent,
    RunErrorEvent,
    RunCancelledEvent,
    RunPausedEvent,
    RunContinuedEvent,
    ToolCallStartedEvent,
    ToolCallCompletedEvent,
    ThinkingStartedEvent,
    ThinkingStepEvent,
    ThinkingCompletedEvent,
    CacheHitEvent,
    CacheMissEvent,
    PolicyCheckStartedEvent,
    PolicyCheckCompletedEvent,
]
"""Union type of all possible agent run events."""


# Event type registry for deserialization
RUN_EVENT_TYPE_REGISTRY: Dict[str, type] = {
    RunEvent.run_started.value: RunStartedEvent,
    RunEvent.run_content.value: RunContentEvent,
    RunEvent.run_content_completed.value: RunContentCompletedEvent,
    RunEvent.run_completed.value: RunCompletedEvent,
    RunEvent.run_error.value: RunErrorEvent,
    RunEvent.run_cancelled.value: RunCancelledEvent,
    RunEvent.run_paused.value: RunPausedEvent,
    RunEvent.run_continued.value: RunContinuedEvent,
    RunEvent.tool_call_started.value: ToolCallStartedEvent,
    RunEvent.tool_call_completed.value: ToolCallCompletedEvent,
    RunEvent.thinking_started.value: ThinkingStartedEvent,
    RunEvent.thinking_step.value: ThinkingStepEvent,
    RunEvent.thinking_completed.value: ThinkingCompletedEvent,
    RunEvent.cache_hit.value: CacheHitEvent,
    RunEvent.cache_miss.value: CacheMissEvent,
    RunEvent.policy_check_started.value: PolicyCheckStartedEvent,
    RunEvent.policy_check_completed.value: PolicyCheckCompletedEvent,
}


def agent_run_event_from_dict(data: Dict[str, Any]) -> BaseAgentRunEvent:
    """Create an AgentRunEvent from a dictionary.
    
    Args:
        data: Dictionary representation of the event
        
    Returns:
        BaseAgentRunEvent: The appropriate event type instance
        
    Raises:
        ValueError: If the event type is unknown
    """
    event_type = data.get("event", "")
    cls = RUN_EVENT_TYPE_REGISTRY.get(event_type)
    if not cls:
        raise ValueError(f"Unknown event type: {event_type}")
    return cls.from_dict(data)
