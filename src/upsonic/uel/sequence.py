from __future__ import annotations

from typing import Any, Optional

from upsonic.uel.runnable import Runnable


class RunnableSequence(Runnable[Any, Any]):
    """A sequence of Runnables that execute in order.
    
    This is the core primitive for chaining components together. When invoked,
    it passes the input through each step in sequence, with each step receiving
    the output of the previous step.
    
    Example:
        ```python
        sequence = prompt | model | parser
        result = sequence.invoke({"topic": "bears"})
        ```
    """
    
    def __init__(self, steps: list[Runnable]):
        """Initialize the sequence.
        
        Args:
            steps: List of Runnables to execute in order
        """
        if not steps:
            raise ValueError("RunnableSequence must have at least one step")
        
        self.steps = steps
    
    def invoke(self, input: Any, config: Optional[dict[str, Any]] = None) -> Any:
        """Execute all steps in sequence.
        
        Args:
            input: The initial input
            config: Optional runtime configuration passed to all steps
            
        Returns:
            The output of the final step
        """
        result = input
        
        # Pass the result through each step
        for step in self.steps:
            result = step.invoke(result, config)
        
        return result
    
    async def ainvoke(self, input: Any, config: Optional[dict[str, Any]] = None) -> Any:
        """Execute all steps in sequence asynchronously.
        
        Args:
            input: The initial input
            config: Optional runtime configuration passed to all steps
            
        Returns:
            The output of the final step
        """
        result = input
        
        # Pass the result through each step asynchronously
        for step in self.steps:
            result = await step.ainvoke(result, config)
        
        return result
    
    def __or__(self, other: Runnable) -> RunnableSequence:
        """Extend this sequence with another runnable.
        
        Args:
            other: The runnable, callable, or dict to append
            
        Returns:
            A new RunnableSequence with the additional step
        """
        from upsonic.uel.lambda_runnable import coerce_to_runnable
        
        # Coerce to runnable (handles functions, dicts, etc.)
        other_runnable = coerce_to_runnable(other)
        
        return RunnableSequence(steps=self.steps + [other_runnable])
    
    def get_graph(self):
        """Get a graph representation of this sequence.
        
        Returns:
            A RunnableGraph object that can be visualized
        """
        from upsonic.uel.graph import RunnableGraph
        return RunnableGraph(self)
    
    def get_prompts(self):
        """Extract all ChatPromptTemplate instances from this sequence.
        
        Returns:
            List of ChatPromptTemplate instances found in the sequence
        """
        from upsonic.uel.prompt import ChatPromptTemplate
        
        prompts = []
        
        def extract_prompts(runnable):
            """Recursively extract prompts from runnables."""
            if isinstance(runnable, ChatPromptTemplate):
                prompts.append(runnable)
            elif hasattr(runnable, 'steps'):
                # Handle sequences and parallels
                if isinstance(runnable.steps, list):
                    for step in runnable.steps:
                        extract_prompts(step)
                elif isinstance(runnable.steps, dict):
                    for step in runnable.steps.values():
                        extract_prompts(step)
            elif hasattr(runnable, 'runnable'):
                # Handle wrapped runnables
                extract_prompts(runnable.runnable)
        
        # Extract prompts from all steps
        for step in self.steps:
            extract_prompts(step)
        
        return prompts
    
    def __repr__(self) -> str:
        """Return a string representation of the sequence."""
        step_names = [step.__class__.__name__ for step in self.steps]
        return f"RunnableSequence({' | '.join(step_names)})"

