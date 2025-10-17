"""
Base Step class for pipeline execution.

Steps are the building blocks of the agent execution pipeline.
Each step performs a specific operation and can modify the context.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

from .context import StepContext


class StepStatus(str, Enum):
    """Status of step execution."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


class StepResult(BaseModel):
    """Result of a step execution."""
    status: StepStatus = Field(description="Step execution status")
    message: Optional[str] = Field(default=None, description="Optional message")
    execution_time: float = Field(description="Execution time in seconds")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Step(ABC):
    """
    Base class for pipeline steps.
    
    Each step performs a specific operation in the agent execution pipeline.
    All steps must execute - there is no skipping. If an error occurs,
    the pipeline stops and the error is raised to the user.
    
    Usage:
        ```python
        class MyStep(Step):
            @property
            def name(self) -> str:
                return "my_step"
            
            async def execute(self, context: StepContext) -> StepResult:
                # Do something with context
                context.messages.append(some_message)
                return StepResult(
                    status=StepStatus.SUCCESS,
                    execution_time=0.0  # Set by run()
                )
        ```
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of this step.
        
        The name is used for logging and debugging purposes.
        """
        pass
    
    @property
    def description(self) -> str:
        """
        Return a description of what this step does.
        
        Override this to provide detailed information about the step's purpose.
        """
        return f"Executes {self.name}"
    
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
    
    async def run(self, context: StepContext) -> StepResult:
        """
        Run the step with time tracking and error handling.
        
        This method orchestrates the execution flow:
        1. Record start time
        2. Execute the step
        3. Record end time and set execution_time
        4. If error occurs, create ERROR result and raise it
        
        Args:
            context: The current step context
            
        Returns:
            StepResult: The result of the execution with execution_time set
            
        Raises:
            Exception: Any exception from execute() is raised after creating ERROR result
        """
        start_time = time.time()
        
        try:
            result = await self.execute(context)
            execution_time = time.time() - start_time
            
            # Set execution time in result
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create an ERROR result to track in stats
            error_result = StepResult(
                status=StepStatus.ERROR,
                message=f"Error in {self.name}: {str(e)}",
                execution_time=execution_time
            )
            
            # Store error result in context for tracking
            if not hasattr(context, '_error_result'):
                context._error_result = error_result
            
            # Re-raise the exception to stop pipeline
            raise

