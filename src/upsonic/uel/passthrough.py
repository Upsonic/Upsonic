from __future__ import annotations

from typing import Any, Optional, Dict, Union, Callable

from upsonic.uel.runnable import Runnable


class AssignDescriptor:
    """Descriptor that makes assign() work as both classmethod and instance method."""
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            # Called on class: RunnablePassthrough.assign(...)
            def class_assign(**kwargs):
                return objtype(assignments=kwargs)
            return class_assign
        else:
            # Called on instance: instance.assign(...)
            def instance_assign(**kwargs):
                merged_assignments = {**obj.assignments, **kwargs}
                return RunnablePassthrough(assignments=merged_assignments)
            return instance_assign


class RunnablePassthrough(Runnable[Any, Any]):
    """A Runnable that passes its input through, optionally assigning new values.
    
    This is useful for:
    - Passing input unchanged in parallel branches
    - Adding or modifying keys in the input dictionary
    - Dynamic chain construction based on input
    
    Example:
        ```python
        from operator import itemgetter
        from upsonic.uel import RunnablePassthrough
        
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
        
        # Process each assignment sequentially, so later assignments can use earlier ones
        for key, runnable in self.assignments.items():
            # Convert to runnable if needed
            from upsonic.uel.lambda_runnable import coerce_to_runnable
            runnable = coerce_to_runnable(runnable)
            
            # Invoke the runnable with the current result (so it can see previous assignments)
            value = runnable.invoke(result, config)
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
        
        # Process each assignment sequentially, so later assignments can use earlier ones
        for key, runnable in self.assignments.items():
            # Convert to runnable if needed
            from upsonic.uel.lambda_runnable import coerce_to_runnable
            runnable = coerce_to_runnable(runnable)
            
            # Invoke the runnable with the current result (so it can see previous assignments)
            value = await runnable.ainvoke(result, config)
            result[key] = value
        
        return result
    
    assign = AssignDescriptor()
    """Create or extend a RunnablePassthrough with assignments.
    
    Works as both a classmethod and instance method:
    - RunnablePassthrough.assign(a=...) creates a new instance
    - instance.assign(b=...) merges assignments and returns a new instance
    - RunnablePassthrough.assign(a=...).assign(b=...) chains assignments
    
    Args:
        **kwargs: Key-value pairs where values are Runnables or callables
        
    Returns:
        A RunnablePassthrough configured with assignments
        
    Example:
        ```python
        chain = (
            RunnablePassthrough.assign(
                formatted_question=lambda x: f"Question: {x['question']}",
            ).assign(
                context=lambda x: retrieve_context(x['question'])
            )
            | prompt
            | model
        )
        ```
    """
    
    def __repr__(self) -> str:
        """Return a string representation of the passthrough."""
        if self.assignments:
            keys = ", ".join(self.assignments.keys())
            return f"RunnablePassthrough.assign({keys})"
        return "RunnablePassthrough()"

