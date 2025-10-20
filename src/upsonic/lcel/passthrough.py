from __future__ import annotations

from typing import Any, Optional, Dict, Union, Callable

from upsonic.lcel.runnable import Runnable


class RunnablePassthrough(Runnable[Any, Any]):
    """A Runnable that passes its input through, optionally assigning new values.
    
    This is useful for:
    - Passing input unchanged in parallel branches
    - Adding or modifying keys in the input dictionary
    - Dynamic chain construction based on input
    
    Example:
        ```python
        from operator import itemgetter
        from upsonic.lcel import RunnablePassthrough
        
        # Pass through unchanged
        chain = RunnablePassthrough() | model
        
        # Assign new values
        chain = (
            RunnablePassthrough.assign(
                question=itemgetter("question"),
                context=itemgetter("question") | retriever
            )
            | prompt
            | model
        )
        ```
    """
    
    def __init__(self, assignments: Dict[str, Union[Runnable, Callable]] = None):
        """
        Initialize RunnablePassthrough.
        
        Args:
            assignments: Optional dictionary of key-runnable pairs to assign
        """
        self.assignments = assignments or {}
    
    def invoke(self, input: Any, config: Optional[dict[str, Any]] = None) -> Any:
        """Pass through the input, optionally with assignments.
        
        Args:
            input: The input to pass through
            config: Optional runtime configuration
            
        Returns:
            The input, potentially modified with assignments
        """
        if not self.assignments:
            return input
        
        # For assignments, input must be a dict
        if not isinstance(input, dict):
            raise TypeError(
                f"RunnablePassthrough with assignments expects dict input, got {type(input)}"
            )
        
        # Start with a copy of the input
        result = input.copy()
        
        # Process each assignment
        for key, runnable in self.assignments.items():
            # Convert to runnable if needed
            from upsonic.lcel.lambda_runnable import coerce_to_runnable
            runnable = coerce_to_runnable(runnable)
            
            # Invoke the runnable with the original input
            value = runnable.invoke(input, config)
            result[key] = value
        
        return result
    
    async def ainvoke(self, input: Any, config: Optional[dict[str, Any]] = None) -> Any:
        """Asynchronously pass through the input, optionally with assignments.
        
        Args:
            input: The input to pass through
            config: Optional runtime configuration
            
        Returns:
            The input, potentially modified with assignments
        """
        if not self.assignments:
            return input
        
        # For assignments, input must be a dict
        if not isinstance(input, dict):
            raise TypeError(
                f"RunnablePassthrough with assignments expects dict input, got {type(input)}"
            )
        
        # Start with a copy of the input
        result = input.copy()
        
        # Process each assignment
        import asyncio
        tasks = []
        keys = []
        
        for key, runnable in self.assignments.items():
            # Convert to runnable if needed
            from upsonic.lcel.lambda_runnable import coerce_to_runnable
            runnable = coerce_to_runnable(runnable)
            
            # Create task for each assignment
            task = asyncio.create_task(runnable.ainvoke(input, config))
            tasks.append(task)
            keys.append(key)
        
        # Wait for all assignments to complete
        if tasks:
            values = await asyncio.gather(*tasks)
            
            # Update result with assigned values
            for key, value in zip(keys, values):
                result[key] = value
        
        return result
    
    @classmethod
    def assign(cls, **kwargs: Union[Runnable, Callable]) -> "RunnablePassthrough":
        """Create a RunnablePassthrough that assigns new values.
        
        Args:
            **kwargs: Key-value pairs where values are Runnables or callables
            
        Returns:
            A RunnablePassthrough configured with assignments
            
        Example:
            ```python
            chain = (
                RunnablePassthrough.assign(
                    formatted_question=lambda x: f"Question: {x['question']}",
                    context=lambda x: retrieve_context(x['question'])
                )
                | prompt
                | model
            )
            ```
        """
        return cls(assignments=kwargs)
    
    def __repr__(self) -> str:
        """Return a string representation of the passthrough."""
        if self.assignments:
            keys = ", ".join(self.assignments.keys())
            return f"RunnablePassthrough.assign({keys})"
        return "RunnablePassthrough()"

