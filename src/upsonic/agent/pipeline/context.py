"""
Step Context - Shared state across pipeline steps.

The StepContext is a Pydantic model that holds all the state information
needed during the agent's execution pipeline. It gets passed through each
step and can be modified by steps to communicate state changes.
"""

from typing import Any, Optional, List, Dict, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task
    from upsonic.models import Model, ModelRequest, ModelResponse
    from upsonic.agent.context_managers import MemoryManager
    from upsonic.graph.graph import State
    from upsonic.agent.events import AgentEvent


class StepContext(BaseModel):
    """
    Context object that flows through the pipeline steps.
    
    This context holds all the state needed for agent execution,
    and can be modified by each step to pass information to
    subsequent steps.
    
    Attributes:
        task: The task being executed
        agent: Reference to the agent instance
        model: Model to use for execution
        state: Graph execution state (optional)
        
        # Execution mode
        is_streaming: Whether this is a streaming execution
        stream_result: StreamRunResult reference for streaming mode
        
        # Event tracking
        emitted_events: Events emitted during execution
        emit_step_events: Whether to emit step-level events (for streaming)
        
        # Continuation mode (for resuming from external execution pause)
        is_continuation: Whether this is continuing from a paused state
        continuation_messages: Messages to restore when continuing
        continuation_tool_results: Tool results to inject when continuing
        
        # Step outputs
        messages: Built messages for model request
        response: Model response
        final_output: Final processed output
        
        # Streaming-specific
        streaming_events: List of events collected during streaming
    """
    
    task: Any = Field(description="Task being executed")
    agent: Any = Field(description="Agent instance")
    
    model: Any = Field(default=None, description="Model override for execution")
    state: Any = Field(default=None, description="Graph execution state")
    
    is_streaming: bool = Field(default=False, description="Whether this is streaming execution")
    stream_result: Any = Field(default=None, description="StreamRunResult reference for streaming mode")
    
    # Event tracking for comprehensive streaming
    emitted_events: List[Any] = Field(default_factory=list, description="Events emitted during execution")
    emit_step_events: bool = Field(default=True, description="Whether to emit step-level events")
    accumulated_text: str = Field(default="", description="Accumulated text during streaming")
    
    # Continuation-specific attributes
    is_continuation: bool = Field(default=False, description="Whether this is continuing from a paused state")
    continuation_messages: List[Any] = Field(default_factory=list, description="Messages to restore when continuing")
    continuation_tool_results: List[Any] = Field(default_factory=list, description="Tool results to inject when continuing")
    continuation_response_with_tool_calls: Any = Field(default=None, description="Response with tool_calls that triggered the pause")
    
    # Step outputs and intermediate state
    messages: List[Any] = Field(default_factory=list, description="Model request messages")
    response: Any = Field(default=None, description="Model response")
    final_output: Any = Field(default=None, description="Final processed output")
    
    # Streaming-specific attributes
    streaming_events: List[Any] = Field(default_factory=list, description="Events collected during streaming")

    # DeepAgent-specific attributes
    plan_context: Optional[str] = Field(default=None, description="Formatted plan for system prompt injection")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def emit_event(self, event: "AgentEvent") -> None:
        """
        Emit an event during pipeline execution.
        
        Events are collected and can be yielded during streaming.
        
        Args:
            event: The event to emit
        """
        self.emitted_events.append(event)
        
        # Also add to streaming events if streaming
        if self.is_streaming:
            self.streaming_events.append(event)
    
    def get_emitted_events(self) -> List["AgentEvent"]:
        """
        Get all events emitted during execution.
        
        Returns:
            List of emitted events
        """
        return list(self.emitted_events)
    
    def clear_emitted_events(self) -> None:
        """Clear the emitted events list."""
        self.emitted_events.clear()
    
    def pop_pending_events(self) -> List["AgentEvent"]:
        """
        Get and clear all pending events.
        
        This is useful for streaming to get events that haven't been yielded yet.
        
        Returns:
            List of pending events (list is cleared after this call)
        """
        events = list(self.emitted_events)
        self.emitted_events.clear()
        return events
