"""
Step Context - Shared state across pipeline steps.

The StepContext is a Pydantic model that holds all the state information
needed during the agent's execution pipeline. It gets passed through each
step and can be modified by steps to communicate state changes.
"""

from typing import Any, Optional, List, Dict, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task
    from upsonic.models import Model, ModelRequest, ModelResponse
    from upsonic.agent.context_managers import MemoryManager
    from upsonic.graph.graph import State


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
    
    # Step outputs and intermediate state
    messages: List[Any] = Field(default_factory=list, description="Model request messages")
    response: Any = Field(default=None, description="Model response")
    final_output: Any = Field(default=None, description="Final processed output")
    
    # Streaming-specific attributes
    streaming_events: List[Any] = Field(default_factory=list, description="Events collected during streaming")
    
    class Config:
        arbitrary_types_allowed = True

