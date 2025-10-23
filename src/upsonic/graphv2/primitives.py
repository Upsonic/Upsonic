"""
Core primitives for graph control flow.

This module provides Command, interrupt, and Send primitives that enable
dynamic routing, human-in-the-loop workflows, and dynamic parallelization.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union


# Type variable for Command goto targets
GotoT = TypeVar('GotoT', bound=str)

# Context variable for interrupt resume values
_interrupt_resume_values: ContextVar[Optional[List[Any]]] = ContextVar('interrupt_resume_values', default=None)
_interrupt_counter: ContextVar[int] = ContextVar('interrupt_counter', default=0)


class InterruptException(Exception):
    """Exception raised by interrupt() to pause graph execution.
    
    This exception is caught by the graph execution engine to save state
    and return control to the caller. It should never be caught by user code.
    """
    
    def __init__(self, value: Any):
        """Initialize the interrupt exception.
        
        Args:
            value: The value to return to the caller (will be in __interrupt__)
        """
        self.value = value
        super().__init__("Graph execution interrupted")


def interrupt(value: Any) -> Any:
    """Pause graph execution and wait for external input.
    
    This function immediately pauses the graph, saves the current state,
    and returns control to the caller. The value parameter is returned
    in the result's __interrupt__ field.
    
    To resume, call graph.invoke(Command(resume=<value>), config=config).
    The value passed to resume will be returned from this interrupt() call.
    
    Important rules:
    - Never wrap interrupt() in try/except
    - Keep interrupt order consistent (no conditional interrupts)
    - Only pass JSON-serializable values
    - Make operations before interrupt idempotent
    
    Args:
        value: Data to show to the caller (must be JSON-serializable)
        
    Returns:
        The value passed to Command(resume=...) when resuming
        
    Raises:
        InterruptException: When not resuming (caught internally by graph)
        
    Example:
        ```python
        def approval_node(state: State) -> dict:
            approved = interrupt({
                "question": "Approve this action?",
                "details": state["action"]
            })
            
            if approved:
                return {"status": "approved"}
            else:
                return {"status": "rejected"}
        ```
    """
    # Check if we're resuming from an interrupt
    resume_values = _interrupt_resume_values.get()
    
    if resume_values is not None:
        # We're in resume mode - return the resume value for this interrupt
        counter = _interrupt_counter.get()
        
        if counter < len(resume_values):
            # Get the resume value for this specific interrupt call
            resume_value = resume_values[counter]
            
            # Increment counter for next interrupt call
            _interrupt_counter.set(counter + 1)
            
            # Return the resume value instead of throwing
            return resume_value
        else:
            # No more resume values - this interrupt wasn't in the original execution
            raise RuntimeError(
                f"Interrupt called but no resume value available (counter={counter}, "
                f"resume_values={len(resume_values)}). This may indicate a conditional "
                "interrupt or inconsistent interrupt order."
            )
    
    # Not in resume mode - throw the exception to pause execution
    # Also increment counter to track which interrupt this is
    counter = _interrupt_counter.get()
    _interrupt_counter.set(counter + 1)
    
    raise InterruptException(value)


@dataclass
class Command(Generic[GotoT]):
    """Control flow command for nodes to specify routing and state updates.
    
    The Command primitive allows nodes to both update state AND specify
    where execution should go next, making control flow more explicit.
    
    Attributes:
        update: Dictionary of state updates to apply
        goto: Node name to execute next, or END to finish, or Send object(s)
        resume: Value to provide when resuming from an interrupt
        graph: Graph to navigate in (None for current, PARENT for parent graph)
        
    Example:
        ```python
        def classify_and_route(state: State) -> Command[Literal["urgent", "normal", END]]:
            intent = classify(state["message"])
            
            if intent == "urgent":
                return Command(
                    update={"classification": "urgent"},
                    goto="urgent"
                )
            else:
                return Command(
                    update={"classification": "normal"},
                    goto="normal"
                )
        ```
    """
    
    update: Optional[Dict[str, Any]] = None
    """State updates to apply (merged using reducers)."""
    
    goto: Optional[Union[GotoT, str, 'Send', List['Send']]] = None
    """Node name to execute next, or END to finish execution, or Send object(s) for dynamic routing."""
    
    resume: Any = None
    """Value to provide when resuming from an interrupt."""
    
    graph: Optional[str] = None
    """Graph to navigate in. None = current graph, PARENT = parent graph."""
    
    # Class constant for parent graph navigation
    PARENT: str = "__parent__"


@dataclass
class Send:
    """Instruction to dynamically create and invoke a subgraph or worker.
    
    The Send API enables dynamic parallelization where a node can create
    multiple worker instances that execute in parallel.
    
    Attributes:
        node: Name of the node to invoke
        state: State to pass to that node
        
    Example:
        ```python
        def fan_out(state: State) -> List[Send]:
            # Create a worker for each item
            return [
                Send("worker", {"item": item})
                for item in state["items"]
            ]
        
        def worker(state: WorkerState) -> dict:
            result = process_item(state["item"])
            return {"results": [result]}
        
        # Results from all workers are merged using reducers
        ```
    """
    
    node: str
    """Name of the node to invoke."""
    
    state: Dict[str, Any]
    """State to pass to that node."""


# Special END marker for graph termination
END = "__end__"
"""Special constant representing graph termination.

Use this with edges or Command.goto to indicate the graph should finish.

Example:
    ```python
    builder.add_edge("final_node", END)
    
    # Or with Command
    return Command(update={...}, goto=END)
    ```
"""

