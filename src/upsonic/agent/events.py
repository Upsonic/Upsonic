"""
Agent Pipeline Events Module

This module provides comprehensive event data classes for the agent pipeline execution.
These events allow users to have full visibility and control over every step of agent
execution when streaming.

Event Hierarchy:
- AgentEvent (base class)
  ├── Pipeline Events
  │   ├── PipelineStartEvent
  │   └── PipelineEndEvent
  ├── Step Events
  │   ├── StepStartEvent
  │   └── StepEndEvent
  ├── Step-Specific Events
  │   ├── AgentInitializedEvent
  │   ├── CacheCheckEvent (CacheHitEvent, CacheMissEvent)
  │   ├── PolicyCheckEvent
  │   ├── ModelSelectedEvent
  │   ├── ToolsConfiguredEvent
  │   ├── MessagesBuiltEvent
  │   ├── ModelRequestStartEvent
  │   ├── ModelResponseEvent
  │   ├── ToolCallEvent
  │   ├── ToolResultEvent
  │   ├── ReflectionEvent
  │   ├── MemoryUpdateEvent
  │   └── ExecutionCompleteEvent
  └── LLM Stream Events (wrapped from messages module)
  ├── PolicyFeedbackEvent

Usage:
    ```python
    async with agent.stream(task) as result:
        async for event in result.stream_events():
            if isinstance(event, StepStartEvent):
                print(f"Starting step: {event.step_name}")
            elif isinstance(event, ToolCallEvent):
                print(f"Calling tool: {event.tool_name}({event.tool_args})")
            elif isinstance(event, TextDeltaEvent):
                print(event.content, end='', flush=True)
    ```
"""

from __future__ import annotations

import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union, TYPE_CHECKING, Annotated

import pydantic
from pydantic import BaseModel, Field

from upsonic._utils import now_utc as _now_utc

if TYPE_CHECKING:
    from upsonic.messages import (
        ModelResponseStreamEvent,
        PartStartEvent,
        PartDeltaEvent,
        PartEndEvent,
        FinalResultEvent,
        TextPart,
        ToolCallPart,
    )


# =============================================================================
# Base Event Classes
# =============================================================================

@dataclass(repr=False, kw_only=True)
class AgentEvent(ABC):
    """
    Base class for all agent pipeline events.
    
    All events share common attributes for identification and timing.
    
    Attributes:
        event_id: Unique identifier for this event
        timestamp: When the event occurred
        event_type: The type name of this event (class name)
    """
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    """Unique identifier for this event."""
    
    timestamp: datetime = field(default_factory=_now_utc)
    """When this event occurred."""
    
    @property
    def event_type(self) -> str:
        """Return the event type name (class name without 'Event' suffix if present)."""
        return self.__class__.__name__
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(event_id={self.event_id!r})"


# =============================================================================
# Pipeline-Level Events
# =============================================================================

@dataclass(repr=False, kw_only=True)
class PipelineStartEvent(AgentEvent):
    """
    Event emitted when the pipeline execution starts.
    
    Attributes:
        total_steps: Total number of steps in the pipeline
        is_streaming: Whether this is a streaming execution
        task_description: Brief description of the task being executed
    """
    
    total_steps: int
    """Total number of steps in the pipeline."""
    
    is_streaming: bool = False
    """Whether this is a streaming execution."""
    
    task_description: Optional[str] = None
    """Brief description of the task being executed."""
    
    event_kind: Literal['pipeline_start'] = 'pipeline_start'
    """Event type identifier."""
    
    def __repr__(self) -> str:
        return f"PipelineStartEvent(steps={self.total_steps}, streaming={self.is_streaming})"


@dataclass(repr=False, kw_only=True)
class PipelineEndEvent(AgentEvent):
    """
    Event emitted when the pipeline execution ends.
    
    Attributes:
        total_steps: Total number of steps in the pipeline
        executed_steps: Number of steps that were actually executed
        total_duration: Total execution time in seconds
        status: Final status ('success', 'error', 'paused')
        error_message: Error message if status is 'error'
    """
    
    total_steps: int
    """Total number of steps in the pipeline."""
    
    executed_steps: int
    """Number of steps that were actually executed."""
    
    total_duration: float
    """Total execution time in seconds."""
    
    status: Literal['success', 'error', 'paused'] = 'success'
    """Final status of the pipeline execution."""
    
    error_message: Optional[str] = None
    """Error message if status is 'error'."""
    
    event_kind: Literal['pipeline_end'] = 'pipeline_end'
    """Event type identifier."""
    
    def __repr__(self) -> str:
        return f"PipelineEndEvent(executed={self.executed_steps}/{self.total_steps}, status={self.status})"


# =============================================================================
# Step-Level Events
# =============================================================================

@dataclass(repr=False, kw_only=True)
class StepStartEvent(AgentEvent):
    """
    Event emitted when a pipeline step starts execution.
    
    Attributes:
        step_name: Name of the step (e.g., 'initialization', 'cache_check')
        step_description: Human-readable description of the step
        step_index: Index of this step in the pipeline (0-based)
        total_steps: Total number of steps in the pipeline
    """
    
    step_name: str
    """Name of the step."""
    
    step_description: str
    """Human-readable description of the step."""
    
    step_index: int
    """Index of this step in the pipeline (0-based)."""
    
    total_steps: int
    """Total number of steps in the pipeline."""
    
    event_kind: Literal['step_start'] = 'step_start'
    """Event type identifier."""
    
    def __repr__(self) -> str:
        return f"StepStartEvent(step={self.step_name}, index={self.step_index}/{self.total_steps})"


@dataclass(repr=False, kw_only=True)
class StepEndEvent(AgentEvent):
    """
    Event emitted when a pipeline step completes.
    
    Attributes:
        step_name: Name of the step
        step_index: Index of this step in the pipeline
        status: Status of the step ('success', 'error', 'pending', 'skipped')
        message: Status message from the step
        execution_time: Execution time in seconds
    """
    
    step_name: str
    """Name of the step."""
    
    step_index: int
    """Index of this step in the pipeline."""
    
    status: Literal['success', 'error', 'pending', 'skipped']
    """Status of the step execution."""
    
    message: str
    """Status message from the step."""
    
    execution_time: float
    """Execution time in seconds."""
    
    event_kind: Literal['step_end'] = 'step_end'
    """Event type identifier."""
    
    def __repr__(self) -> str:
        return f"StepEndEvent(step={self.step_name}, status={self.status}, time={self.execution_time:.3f}s)"


# =============================================================================
# Step-Specific Events
# =============================================================================

@dataclass(repr=False, kw_only=True)
class AgentInitializedEvent(AgentEvent):
    """
    Event emitted when the agent is initialized for execution.
    
    Attributes:
        agent_id: Unique identifier of the agent
        is_streaming: Whether streaming mode is enabled
    """
    
    agent_id: str
    """Unique identifier of the agent."""
    
    is_streaming: bool = False
    """Whether streaming mode is enabled."""
    
    event_kind: Literal['agent_initialized'] = 'agent_initialized'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class CacheCheckEvent(AgentEvent):
    """
    Event emitted when cache is checked for an existing response.
    
    Attributes:
        cache_enabled: Whether caching is enabled for this task
        cache_method: Cache method used ('exact_match', 'vector_search')
        cache_hit: Whether a cache hit occurred
        similarity: Similarity score for vector search (if applicable)
        input_preview: Preview of the input being cached
    """
    
    cache_enabled: bool
    """Whether caching is enabled for this task."""
    
    cache_method: Optional[str] = None
    """Cache method used ('exact_match', 'vector_search')."""
    
    cache_hit: bool = False
    """Whether a cache hit occurred."""
    
    similarity: Optional[float] = None
    """Similarity score for vector search (if applicable)."""
    
    input_preview: Optional[str] = None
    """Preview of the input being cached (truncated)."""
    
    event_kind: Literal['cache_check'] = 'cache_check'
    """Event type identifier."""
    
    def __repr__(self) -> str:
        return f"CacheCheckEvent(hit={self.cache_hit}, method={self.cache_method})"


@dataclass(repr=False, kw_only=True)
class CacheHitEvent(AgentEvent):
    """
    Event emitted when a cache hit occurs.
    
    Attributes:
        cache_method: Cache method used
        similarity: Similarity score (for vector search)
        cached_response_preview: Preview of the cached response
    """
    
    cache_method: str
    """Cache method used."""
    
    similarity: Optional[float] = None
    """Similarity score for vector search."""
    
    cached_response_preview: Optional[str] = None
    """Preview of the cached response (truncated)."""
    
    event_kind: Literal['cache_hit'] = 'cache_hit'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class CacheMissEvent(AgentEvent):
    """
    Event emitted when a cache miss occurs.
    
    Attributes:
        cache_method: Cache method used
        reason: Reason for cache miss (if known)
    """
    
    cache_method: str
    """Cache method used."""
    
    reason: Optional[str] = None
    """Reason for cache miss (if known)."""
    
    event_kind: Literal['cache_miss'] = 'cache_miss'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class PolicyCheckEvent(AgentEvent):
    """
    Event emitted when a policy check is performed.
    
    Attributes:
        policy_type: Type of policy ('user_policy', 'agent_policy')
        action: Action taken ('pass', 'block', 'modify', 'anonymize')
        policies_checked: Number of policies checked
        content_modified: Whether content was modified
        blocked_reason: Reason for blocking (if blocked)
    """
    
    policy_type: Literal['user_policy', 'agent_policy']
    """Type of policy being applied."""
    
    action: Literal['pass', 'block', 'modify', 'anonymize']
    """Action taken by the policy."""
    
    policies_checked: int = 0
    """Number of policies checked."""
    
    content_modified: bool = False
    """Whether content was modified by the policy."""
    
    blocked_reason: Optional[str] = None
    """Reason for blocking (if action is 'block')."""
    
    event_kind: Literal['policy_check'] = 'policy_check'
    """Event type identifier."""
    
    def __repr__(self) -> str:
        return f"PolicyCheckEvent(type={self.policy_type}, action={self.action})"


@dataclass(repr=False, kw_only=True)
class PolicyFeedbackEvent(AgentEvent):
    """
    Event emitted when policy feedback is generated for a retry attempt.
    
    This event is emitted when the feedback loop mechanism is active and
    a policy violation triggers the generation of constructive feedback
    for the agent to retry with.
    
    Attributes:
        policy_type: Type of policy ('user_policy', 'agent_policy')
        feedback_message: The generated feedback message
        retry_count: Current retry attempt number
        max_retries: Maximum number of retries allowed
        violated_policy: Name of the violated policy (if available)
    """
    
    policy_type: Literal['user_policy', 'agent_policy']
    """Type of policy that generated feedback."""
    
    feedback_message: str
    """The constructive feedback message generated."""
    
    retry_count: int = 0
    """Current retry attempt number."""
    
    max_retries: int = 1
    """Maximum number of retries allowed."""
    
    violated_policy: Optional[str] = None
    """Name of the violated policy."""
    
    event_kind: Literal['policy_feedback'] = 'policy_feedback'
    """Event type identifier."""
    
    def __repr__(self) -> str:
        preview = self.feedback_message[:50] + "..." if len(self.feedback_message) > 50 else self.feedback_message
        return f"PolicyFeedbackEvent(type={self.policy_type}, retry={self.retry_count}/{self.max_retries}, feedback={preview!r})"


@dataclass(repr=False, kw_only=True)
class ModelSelectedEvent(AgentEvent):
    """
    Event emitted when a model is selected for execution.
    
    Attributes:
        model_name: Name of the selected model
        provider: Model provider name
        is_override: Whether this is an override model (not the agent default)
    """
    
    model_name: str
    """Name of the selected model."""
    
    provider: Optional[str] = None
    """Model provider name."""
    
    is_override: bool = False
    """Whether this is an override model."""
    
    event_kind: Literal['model_selected'] = 'model_selected'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class ToolsConfiguredEvent(AgentEvent):
    """
    Event emitted when tools are configured for the task.
    
    Attributes:
        tool_count: Number of tools configured
        tool_names: List of configured tool names
        has_mcp_handlers: Whether MCP handlers are included
    """
    
    tool_count: int
    """Number of tools configured."""
    
    tool_names: List[str] = field(default_factory=list)
    """List of configured tool names."""
    
    has_mcp_handlers: bool = False
    """Whether MCP handlers are included."""
    
    event_kind: Literal['tools_configured'] = 'tools_configured'
    """Event type identifier."""
    
    def __repr__(self) -> str:
        return f"ToolsConfiguredEvent(count={self.tool_count}, tools={self.tool_names[:3]}...)" if len(self.tool_names) > 3 else f"ToolsConfiguredEvent(count={self.tool_count}, tools={self.tool_names})"


@dataclass(repr=False, kw_only=True)
class MessagesBuiltEvent(AgentEvent):
    """
    Event emitted when model request messages are built.
    
    Attributes:
        message_count: Number of messages built
        has_system_prompt: Whether a system prompt is included
        has_memory_messages: Whether memory messages are included
        is_continuation: Whether this is a continuation from paused state
    """
    
    message_count: int
    """Number of messages built."""
    
    has_system_prompt: bool = False
    """Whether a system prompt is included."""
    
    has_memory_messages: bool = False
    """Whether memory messages are included."""
    
    is_continuation: bool = False
    """Whether this is a continuation from paused state."""
    
    event_kind: Literal['messages_built'] = 'messages_built'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class ModelRequestStartEvent(AgentEvent):
    """
    Event emitted when a model request is starting.
    
    Attributes:
        model_name: Name of the model being called
        is_streaming: Whether streaming mode is enabled
        has_tools: Whether tools are available
        tool_call_count: Current tool call count
        tool_call_limit: Maximum tool calls allowed
    """
    
    model_name: str
    """Name of the model being called."""
    
    is_streaming: bool = False
    """Whether streaming mode is enabled."""
    
    has_tools: bool = False
    """Whether tools are available."""
    
    tool_call_count: int = 0
    """Current tool call count."""
    
    tool_call_limit: Optional[int] = None
    """Maximum tool calls allowed."""
    
    event_kind: Literal['model_request_start'] = 'model_request_start'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class ModelResponseEvent(AgentEvent):
    """
    Event emitted when a model response is received (non-streaming).
    
    Attributes:
        model_name: Name of the model
        has_text: Whether response contains text content
        has_tool_calls: Whether response contains tool calls
        tool_call_count: Number of tool calls in response
        finish_reason: Reason the model stopped generating
    """
    
    model_name: str
    """Name of the model."""
    
    has_text: bool = False
    """Whether response contains text content."""
    
    has_tool_calls: bool = False
    """Whether response contains tool calls."""
    
    tool_call_count: int = 0
    """Number of tool calls in response."""
    
    finish_reason: Optional[str] = None
    """Reason the model stopped generating."""
    
    event_kind: Literal['model_response'] = 'model_response'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class ToolCallEvent(AgentEvent):
    """
    Event emitted when a tool call is initiated.
    
    Attributes:
        tool_name: Name of the tool being called
        tool_call_id: Unique identifier for this tool call
        tool_args: Arguments passed to the tool
        tool_index: Index of this tool call (when multiple tools called)
        is_parallel: Whether this is part of parallel tool calls
    """
    
    tool_name: str
    """Name of the tool being called."""
    
    tool_call_id: str
    """Unique identifier for this tool call."""
    
    tool_args: Dict[str, Any] = field(default_factory=dict)
    """Arguments passed to the tool."""
    
    tool_index: int = 0
    """Index of this tool call (when multiple tools called)."""
    
    is_parallel: bool = False
    """Whether this is part of parallel tool calls."""
    
    event_kind: Literal['tool_call'] = 'tool_call'
    """Event type identifier."""
    
    def __repr__(self) -> str:
        return f"ToolCallEvent(tool={self.tool_name}, id={self.tool_call_id})"


@dataclass(repr=False, kw_only=True)
class ToolResultEvent(AgentEvent):
    """
    Event emitted when a tool call returns a result.
    
    Attributes:
        tool_name: Name of the tool that was called
        tool_call_id: Unique identifier for this tool call
        result: Result returned by the tool
        result_preview: Preview of the result (truncated for large results)
        execution_time: Time taken to execute the tool (seconds)
        is_error: Whether the result is an error
        error_message: Error message if is_error is True
    """
    
    tool_name: str
    """Name of the tool that was called."""
    
    tool_call_id: str
    """Unique identifier for this tool call."""
    
    result: Any = None
    """Result returned by the tool."""
    
    result_preview: Optional[str] = None
    """Preview of the result (truncated for large results)."""
    
    execution_time: Optional[float] = None
    """Time taken to execute the tool (seconds)."""
    
    is_error: bool = False
    """Whether the result is an error."""
    
    error_message: Optional[str] = None
    """Error message if is_error is True."""
    
    event_kind: Literal['tool_result'] = 'tool_result'
    """Event type identifier."""
    
    def __repr__(self) -> str:
        status = "error" if self.is_error else "success"
        return f"ToolResultEvent(tool={self.tool_name}, status={status})"


@dataclass(repr=False, kw_only=True)
class ExternalToolPauseEvent(AgentEvent):
    """
    Event emitted when execution pauses for external tool execution.
    
    Attributes:
        tool_name: Name of the external tool
        tool_call_id: Unique identifier for this tool call
        tool_args: Arguments passed to the tool
    """
    
    tool_name: str
    """Name of the external tool."""
    
    tool_call_id: str
    """Unique identifier for this tool call."""
    
    tool_args: Dict[str, Any] = field(default_factory=dict)
    """Arguments passed to the tool."""
    
    event_kind: Literal['external_tool_pause'] = 'external_tool_pause'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class ReflectionEvent(AgentEvent):
    """
    Event emitted when reflection processing is applied.
    
    Attributes:
        reflection_applied: Whether reflection was actually applied
        improvement_made: Whether the output was improved
        original_preview: Preview of original output
        improved_preview: Preview of improved output
    """
    
    reflection_applied: bool = False
    """Whether reflection was actually applied."""
    
    improvement_made: bool = False
    """Whether the output was improved."""
    
    original_preview: Optional[str] = None
    """Preview of original output."""
    
    improved_preview: Optional[str] = None
    """Preview of improved output."""
    
    event_kind: Literal['reflection'] = 'reflection'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class MemoryUpdateEvent(AgentEvent):
    """
    Event emitted when memory is updated.
    
    Attributes:
        messages_added: Number of messages added to memory
        memory_type: Type of memory ('session', 'full_session', 'custom')
    """
    
    messages_added: int = 0
    """Number of messages added to memory."""
    
    memory_type: Optional[str] = None
    """Type of memory being updated."""
    
    event_kind: Literal['memory_update'] = 'memory_update'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class ReliabilityEvent(AgentEvent):
    """
    Event emitted when reliability layer processing occurs.
    
    Attributes:
        reliability_applied: Whether reliability layer was applied
        modifications_made: Whether output was modified
    """
    
    reliability_applied: bool = False
    """Whether reliability layer was applied."""
    
    modifications_made: bool = False
    """Whether output was modified."""
    
    event_kind: Literal['reliability'] = 'reliability'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class CacheStoredEvent(AgentEvent):
    """
    Event emitted when a response is stored in cache.
    
    Attributes:
        cache_method: Cache method used
        duration_minutes: Cache duration in minutes
    """
    
    cache_method: str
    """Cache method used."""
    
    duration_minutes: Optional[int] = None
    """Cache duration in minutes."""
    
    event_kind: Literal['cache_stored'] = 'cache_stored'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class ExecutionCompleteEvent(AgentEvent):
    """
    Event emitted when execution is complete.
    
    Attributes:
        output_type: Type of output ('text', 'structured', 'cached')
        has_output: Whether output was produced
        output_preview: Preview of the output
        total_tool_calls: Total number of tool calls made
        total_duration: Total execution time
    """
    
    output_type: Literal['text', 'structured', 'cached', 'blocked'] = 'text'
    """Type of output produced."""
    
    has_output: bool = False
    """Whether output was produced."""
    
    output_preview: Optional[str] = None
    """Preview of the output (truncated)."""
    
    total_tool_calls: int = 0
    """Total number of tool calls made during execution."""
    
    total_duration: Optional[float] = None
    """Total execution time in seconds."""
    
    event_kind: Literal['execution_complete'] = 'execution_complete'
    """Event type identifier."""



@dataclass(repr=False, kw_only=True)
class TextDeltaEvent(AgentEvent):
    """
    Event emitted for text streaming deltas from the model.
    
    This wraps the LLM's PartDeltaEvent for text content,
    providing a simpler interface for text streaming.
    
    Attributes:
        content: The text content delta
        accumulated_content: Total accumulated text so far
        part_index: Index of the part being updated
    """
    
    content: str
    """The text content delta."""
    
    accumulated_content: Optional[str] = None
    """Total accumulated text so far."""
    
    part_index: int = 0
    """Index of the part being updated."""
    
    event_kind: Literal['text_delta'] = 'text_delta'
    """Event type identifier."""
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"TextDeltaEvent(content={preview!r})"


@dataclass(repr=False, kw_only=True)
class TextCompleteEvent(AgentEvent):
    """
    Event emitted when text streaming is complete for a part.
    
    Attributes:
        content: The complete text content
        part_index: Index of the completed part
    """
    
    content: str
    """The complete text content."""
    
    part_index: int = 0
    """Index of the completed part."""
    
    event_kind: Literal['text_complete'] = 'text_complete'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class ThinkingDeltaEvent(AgentEvent):
    """
    Event emitted for thinking/reasoning content deltas from the model.
    
    Attributes:
        content: The thinking content delta
        part_index: Index of the part being updated
    """
    
    content: str
    """The thinking content delta."""
    
    part_index: int = 0
    """Index of the part being updated."""
    
    event_kind: Literal['thinking_delta'] = 'thinking_delta'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class ToolCallDeltaEvent(AgentEvent):
    """
    Event emitted for tool call streaming deltas from the model.
    
    Attributes:
        tool_name: Name of the tool (may be partial during streaming)
        tool_call_id: Tool call identifier
        args_delta: Partial arguments delta
        part_index: Index of the part being updated
    """
    
    tool_name: Optional[str] = None
    """Name of the tool (may be partial during streaming)."""
    
    tool_call_id: Optional[str] = None
    """Tool call identifier."""
    
    args_delta: Optional[str] = None
    """Partial arguments delta."""
    
    part_index: int = 0
    """Index of the part being updated."""
    
    event_kind: Literal['tool_call_delta'] = 'tool_call_delta'
    """Event type identifier."""


@dataclass(repr=False, kw_only=True)
class FinalOutputEvent(AgentEvent):
    """
    Event emitted when the final output is ready.
    
    This is the last event in a streaming session and indicates
    that the complete response is available.
    
    Attributes:
        output: The final output (may be text or structured)
        output_type: Type of the output
    """
    
    output: Any = None
    """The final output."""
    
    output_type: Literal['text', 'structured'] = 'text'
    """Type of the output."""
    
    event_kind: Literal['final_output'] = 'final_output'
    """Event type identifier."""



# Pipeline events
PipelineEvent = Annotated[
    PipelineStartEvent | PipelineEndEvent,
    pydantic.Discriminator('event_kind')
]

# Step events
StepEvent = Annotated[
    StepStartEvent | StepEndEvent,
    pydantic.Discriminator('event_kind')
]

# Step-specific events
StepSpecificEvent = Annotated[
    AgentInitializedEvent |
    CacheCheckEvent |
    CacheHitEvent |
    CacheMissEvent |
    PolicyCheckEvent |
    ModelSelectedEvent |
    ToolsConfiguredEvent |
    MessagesBuiltEvent |
    ModelRequestStartEvent |
    ModelResponseEvent |
    ToolCallEvent |
    ToolResultEvent |
    ExternalToolPauseEvent |
    ReflectionEvent |
    MemoryUpdateEvent |
    ReliabilityEvent |
    CacheStoredEvent |
    ExecutionCompleteEvent,
    pydantic.Discriminator('event_kind')
]

# LLM streaming events (our wrappers)
LLMStreamEvent = Annotated[
    TextDeltaEvent |
    TextCompleteEvent |
    ThinkingDeltaEvent |
    ToolCallDeltaEvent |
    FinalOutputEvent,
    pydantic.Discriminator('event_kind')
]

# All agent events combined
AgentStreamEvent = Annotated[
    PipelineStartEvent |
    PipelineEndEvent |
    StepStartEvent |
    StepEndEvent |
    AgentInitializedEvent |
    CacheCheckEvent |
    CacheHitEvent |
    CacheMissEvent |
    PolicyCheckEvent |
    PolicyFeedbackEvent |
    ModelSelectedEvent |
    ToolsConfiguredEvent |
    MessagesBuiltEvent |
    ModelRequestStartEvent |
    ModelResponseEvent |
    ToolCallEvent |
    ToolResultEvent |
    ExternalToolPauseEvent |
    ReflectionEvent |
    MemoryUpdateEvent |
    ReliabilityEvent |
    CacheStoredEvent |
    ExecutionCompleteEvent |
    TextDeltaEvent |
    TextCompleteEvent |
    ThinkingDeltaEvent |
    ToolCallDeltaEvent |
    FinalOutputEvent,
    pydantic.Discriminator('event_kind')
]



def convert_llm_event_to_agent_event(
    llm_event: "ModelResponseStreamEvent",
    accumulated_text: str = ""
) -> Optional[AgentEvent]:
    """
    Convert an LLM streaming event to an AgentEvent.
    
    This function wraps the low-level LLM events in our higher-level
    agent event classes for a consistent streaming interface.
    
    Args:
        llm_event: The LLM streaming event to convert
        accumulated_text: Accumulated text so far (for context)
        
    Returns:
        The corresponding AgentEvent, or None if no conversion is needed
    """
    from upsonic.messages import (
        PartStartEvent,
        PartDeltaEvent,
        PartEndEvent,
        FinalResultEvent,
        TextPart,
        TextPartDelta,
        ThinkingPart,
        ThinkingPartDelta,
        ToolCallPart,
        ToolCallPartDelta,
    )
    
    if isinstance(llm_event, PartStartEvent):
        part = llm_event.part
        if isinstance(part, TextPart):
            return TextDeltaEvent(
                content=part.content,
                accumulated_content=accumulated_text + part.content,
                part_index=llm_event.index
            )
        elif isinstance(part, ThinkingPart):
            return ThinkingDeltaEvent(
                content=part.content,
                part_index=llm_event.index
            )
        elif isinstance(part, ToolCallPart):
            return ToolCallDeltaEvent(
                tool_name=part.tool_name,
                tool_call_id=part.tool_call_id,
                args_delta=part.args_as_json_str() if part.args else None,
                part_index=llm_event.index
            )
    
    elif isinstance(llm_event, PartDeltaEvent):
        delta = llm_event.delta
        if isinstance(delta, TextPartDelta):
            return TextDeltaEvent(
                content=delta.content_delta,
                accumulated_content=accumulated_text + delta.content_delta,
                part_index=llm_event.index
            )
        elif isinstance(delta, ThinkingPartDelta):
            return ThinkingDeltaEvent(
                content=delta.content_delta or "",
                part_index=llm_event.index
            )
        elif isinstance(delta, ToolCallPartDelta):
            return ToolCallDeltaEvent(
                tool_name=delta.tool_name_delta,
                tool_call_id=delta.tool_call_id,
                args_delta=delta.args_delta if isinstance(delta.args_delta, str) else None,
                part_index=llm_event.index
            )
    
    elif isinstance(llm_event, PartEndEvent):
        part = llm_event.part
        if isinstance(part, TextPart):
            return TextCompleteEvent(
                content=part.content,
                part_index=llm_event.index
            )
    
    elif isinstance(llm_event, FinalResultEvent):
        return FinalOutputEvent(
            output=None,  # Will be set by the caller
            output_type='text'
        )
    
    return None


def is_text_event(event: AgentEvent) -> bool:
    """Check if an event is a text-related streaming event."""
    return isinstance(event, (TextDeltaEvent, TextCompleteEvent))


def is_tool_event(event: AgentEvent) -> bool:
    """Check if an event is a tool-related event."""
    return isinstance(event, (ToolCallEvent, ToolResultEvent, ToolCallDeltaEvent))


def is_pipeline_event(event: AgentEvent) -> bool:
    """Check if an event is a pipeline-level event."""
    return isinstance(event, (PipelineStartEvent, PipelineEndEvent, StepStartEvent, StepEndEvent))


def extract_text_from_event(event: AgentEvent) -> Optional[str]:
    """Extract text content from an event if available."""
    if isinstance(event, TextDeltaEvent):
        return event.content
    elif isinstance(event, TextCompleteEvent):
        return event.content
    return None
