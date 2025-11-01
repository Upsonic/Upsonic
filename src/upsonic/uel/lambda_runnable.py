import asyncio
import inspect
from typing import Any, Callable, Union, Coroutine, TypeVar, Awaitable

from upsonic.uel.runnable import Runnable, Input, Output


class RunnableLambda(Runnable[Input, Output]):
    """
    A Runnable that wraps a Python function or coroutine.
    
    RunnableLambda allows you to use any Python function in an UEL chain
    by wrapping it in the Runnable interface.
    
    Example:
        ```python
        from upsonic.uel import RunnableLambda
        
        def length_function(text):
            return len(text)
        
        chain = RunnableLambda(length_function) | other_runnable
        result = chain.invoke("hello")  # Passes 5 to other_runnable
        ```
    """
    
    def __init__(self, func: Callable[[Input], Output]):
        """Initialize the RunnableLambda.
        
        Args:
            func: A function or coroutine to wrap
        """
        self.func = func
        self.is_coroutine = inspect.iscoroutinefunction(func)
    
    def invoke(self, input: Input, config: dict[str, Any] | None = None) -> Output:
        """Execute the wrapped function.
        
        Args:
            input: The input to pass to the function
            config: Optional runtime configuration (unused)
            
        Returns:
            The output of the function
        """
        if self.is_coroutine:
            # Handle coroutine in sync context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, create a new loop in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.func(input))
                    return future.result()
            except RuntimeError:
                # No running loop, safe to create one
                return asyncio.run(self.func(input))
        else:
            return self.func(input)
    
    async def ainvoke(self, input: Input, config: dict[str, Any] | None = None) -> Output:
        """Execute the wrapped function asynchronously.
        
        Args:
            input: The input to pass to the function
            config: Optional runtime configuration (unused)
            
        Returns:
            The output of the function
        """
        if self.is_coroutine:
            return await self.func(input)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.func, input)
    
    def __repr__(self) -> str:
        """Return a string representation."""
        func_name = getattr(self.func, '__name__', repr(self.func))
        return f"RunnableLambda({func_name})"


def coerce_to_runnable(thing: Union[Runnable, Callable, dict]) -> Runnable:
    """Convert various types to Runnables.
    
    This is a utility function to handle automatic conversion of:
    - Runnables: returned as-is
    - Callables: wrapped in RunnableLambda
    - Dicts: converted to RunnableParallel
    
    Args:
        thing: The object to convert
        
    Returns:
        A Runnable
        
    Raises:
        TypeError: If the type cannot be converted
    """
    if isinstance(thing, Runnable):
        return thing
    elif isinstance(thing, dict):
        from upsonic.uel.parallel import RunnableParallel
        return RunnableParallel.from_dict(thing)
    elif callable(thing):
        return RunnableLambda(thing)
    else:
        raise TypeError(f"Cannot coerce {type(thing)} to Runnable")
