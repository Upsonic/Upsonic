"""
Base Step class for pipeline execution.

Steps are the building blocks of the agent execution pipeline.
Each step performs a specific operation and can modify the context.
Steps can also emit events during execution for streaming visibility.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional, List, Any, AsyncIterator, TYPE_CHECKING
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

from .context import StepContext

if TYPE_CHECKING:
    from upsonic.agent.events import AgentEvent


class StepStatus(str, Enum):
    """Status of step execution."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    SKIPPED = "skipped"


class StepResult(BaseModel):
    """Result of a step execution."""
    status: StepStatus = Field(description="Step execution status")
    message: Optional[str] = Field(default=None, description="Optional message")
    execution_time: float = Field(description="Execution time in seconds")
    
    # Events emitted during step execution
    events: List[Any] = Field(default_factory=list, description="Events emitted during execution")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Step(ABC):
    """
    Base class for pipeline steps.
    
    Each step performs a specific operation in the agent execution pipeline.
    All steps must execute - there is no skipping. If an error occurs,
    the pipeline stops and the error is raised to the user.
    
    Steps can emit events during execution by:
    1. Implementing `get_step_events()` to emit step-specific events after execution
    2. Using `_emit_event()` during execution for real-time event streaming
    
    Usage:
        ```python
        class MyStep(Step):
            @property
            def name(self) -> str:
                return "my_step"
            
            async def execute(self, context: StepContext) -> StepResult:
                # Do something with context
                context.messages.append(some_message)
                
                # Emit an event during execution
                self._emit_event(MyCustomEvent(data="value"))
                
                return StepResult(
                    status=StepStatus.SUCCESS,
                    execution_time=0.0  # Set by run()
                )
            
            def get_step_events(self, context: StepContext, result: StepResult) -> List[AgentEvent]:
                # Return step-specific events after execution
                return [MyStepCompletedEvent(info="details")]
        ```
    """
    
    def __init__(self):
        """Initialize the step with an empty events list."""
        self._emitted_events: List["AgentEvent"] = []
        self._last_result: Optional[StepResult] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of this step.
        
        The name is used for logging, debugging, and event emission.
        """
        pass
    
    @property
    def description(self) -> str:
        """
        Return a description of what this step does.
        
        Override this to provide detailed information about the step's purpose.
        """
        return f"Executes {self.name}"
    
    @property
    def supports_streaming(self) -> bool:
        """
        Return True if this step supports streaming execution.
        
        Override this in steps that implement custom execute_stream() logic
        and need to yield events during execution.
        """
        return False
    
    @abstractmethod
    async def execute(self, context: StepContext) -> StepResult:
        """
        Execute the step's main logic.
        
        This is where the step performs its work. It can read from
        and modify the context as needed.
        
        Args:
            context: The current step context
            
        Returns:
            StepResult: The result of the execution (execution_time will be set by run())
            
        Raises:
            Exception: If an error occurs, it will be raised to stop the pipeline
        """
        pass
    
    def _emit_event(self, event: "AgentEvent", context: Optional["StepContext"] = None) -> None:
        """
        Emit an event during step execution.
        
        Events emitted this way will be collected and yielded during streaming.
        Only emits events if streaming is enabled.
        
        Args:
            event: The event to emit
            context: Optional step context to check if streaming is enabled
        """
        # Only emit events if streaming is enabled
        if context is not None and not context.is_streaming:
            return

        self._emitted_events.append(event)
    
    def get_emitted_events(self) -> List["AgentEvent"]:
        """
        Get all events emitted during the last execution.
        
        Returns:
            List of emitted events
        """
        return list(self._emitted_events)
    
    def clear_emitted_events(self) -> None:
        """Clear the emitted events list."""
        self._emitted_events.clear()
    
    def get_step_events(
        self, 
        context: StepContext, 
        result: StepResult
    ) -> List["AgentEvent"]:
        """
        Get step-specific events to emit after execution.
        
        Override this method to provide custom events based on
        the execution result and context state.
        
        Args:
            context: The step context after execution
            result: The result of the step execution
            
        Returns:
            List of events to emit for this step
        """
        return []
    
    async def run(self, context: StepContext) -> StepResult:
        """
        Run the step with time tracking and error handling.
        
        This method orchestrates the execution flow:
        1. Clear any previous events
        2. Record start time
        3. Execute the step
        4. Record end time and set execution_time
        5. Collect emitted events and step-specific events
        6. If error occurs, create ERROR result and raise it
        
        Args:
            context: The current step context
            
        Returns:
            StepResult: The result of the execution with execution_time set
            
        Raises:
            Exception: Any exception from execute() is raised after creating ERROR result
        """
        # Clear events from previous execution
        self.clear_emitted_events()
        
        start_time = time.time()
        
        try:
            result = await self.execute(context)
            execution_time = time.time() - start_time
            
            # Set execution time in result
            result.execution_time = execution_time
            
            # Collect all events (emitted during execution + step-specific)
            all_events = self.get_emitted_events() + self.get_step_events(context, result)
            result.events = all_events
            
            # Store last result for streaming steps
            self._last_result = result
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create an ERROR result to track in stats
            error_result = StepResult(
                status=StepStatus.ERROR,
                message=f"Error in {self.name}: {str(e)}",
                execution_time=execution_time,
                events=self.get_emitted_events()  # Include any events emitted before error
            )
            
            # Store error result in context for tracking
            if not hasattr(context, '_error_result'):
                context._error_result = error_result
            
            self._last_result = error_result
            
            # Re-raise the exception to stop pipeline
            raise
    
    async def execute_stream(self, context: StepContext) -> AsyncIterator[Any]:
        """
        Execute the step in streaming mode.
        
        Override this method for steps that need to yield events during execution
        (like model streaming). The default implementation just executes normally
        without yielding any events.
        
        Args:
            context: The current step context
            
        Yields:
            Events during streaming execution
        """
        # Default implementation: just execute normally
        await self.run(context)
        # Yield nothing - non-streaming step
        return
        yield  # Make this a generator
