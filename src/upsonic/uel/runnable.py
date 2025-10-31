from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Generic, Union, Callable

Input = TypeVar('Input')
Output = TypeVar('Output')


class Runnable(ABC, Generic[Input, Output]):
    """Base class for all UEL components.
    
    The Runnable protocol is the foundation of UEL. It provides a unified interface
    for all components (prompts, models, parsers, chains) allowing them to be composed
    and executed in a consistent way.
    
    Every Runnable must implement:
    - invoke(): Synchronous execution
    - ainvoke(): Asynchronous execution (optional, defaults to sync version)
    
    The pipe operator (|) is supported to chain Runnables together.
    """
    
    @abstractmethod
    def invoke(self, input: Input, config: Optional[dict[str, Any]] = None) -> Output:
        """Execute this runnable synchronously.
        
        Args:
            input: The input to process
            config: Optional runtime configuration (for tags, callbacks, etc.)
            
        Returns:
            The output of this runnable
        """
        raise NotImplementedError()
    
    async def ainvoke(self, input: Input, config: Optional[dict[str, Any]] = None) -> Output:
        """Execute this runnable asynchronously.
        
        Default implementation calls the synchronous invoke method.
        Override this for true async execution.
        
        Args:
            input: The input to process
            config: Optional runtime configuration
            
        Returns:
            The output of this runnable
        """
        # Default: run in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, input, config)
    
    def __or__(self, other: Union["Runnable", Callable[[Any], Any], dict]) -> "Runnable":
        """Implement the pipe operator (|) for chaining runnables.
        
        This allows intuitive composition like:
            chain = prompt | model | parser
        
        Args:
            other: The runnable to chain after this one
            
        Returns:
            A new RunnableSequence that chains this runnable with other
        """
        from upsonic.uel.sequence import RunnableSequence
        from upsonic.uel.lambda_runnable import coerce_to_runnable
        
        # Coerce to runnable
        other_runnable = coerce_to_runnable(other)
        
        # If self is already a sequence, extend it
        if isinstance(self, RunnableSequence):
            return RunnableSequence(steps=self.steps + [other_runnable])
        
        # Otherwise create a new sequence
        return RunnableSequence(steps=[self, other_runnable])
    
    def __ror__(self, other: Any) -> "Runnable":
        """Support reverse pipe operator for non-Runnable objects.
        
        This allows syntax like: `dict | runnable`
        """
        from upsonic.uel.sequence import RunnableSequence
        from upsonic.uel.lambda_runnable import coerce_to_runnable
        
        # Coerce to runnable
        other_runnable = coerce_to_runnable(other)
        
        return RunnableSequence(steps=[other_runnable, self])
    
    def __repr__(self) -> str:
        """Return a string representation of this runnable."""
        return f"{self.__class__.__name__}()"
