import functools
import inspect
from typing import Any, Callable, TypeVar, Union

from upsonic.uel.runnable import Runnable, Input, Output
from upsonic.uel.lambda_runnable import RunnableLambda, coerce_to_runnable


F = TypeVar('F', bound=Callable[..., Any])


def chain(func: F) -> Runnable:
    """
    Decorator that converts a function into a Runnable.
    
    The decorated function can:
    - Take regular inputs and return outputs
    - Return other Runnables (for dynamic chain construction)
    - Use async/await for asynchronous operations
    
    Example:
        ```python
        from upsonic.uel import chain, ChatPromptTemplate
        from upsonic.models import infer_model
        
        prompt1 = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
        prompt2 = ChatPromptTemplate.from_template("What is the subject of this joke: {joke}")
        
        @chain
        def custom_chain(text):
            # Use runnables inside the function
            prompt_val1 = prompt1.invoke({"topic": text})
            output1 = infer_model("gpt-4o").invoke(prompt_val1)
            
            # Create and return a chain
            chain2 = prompt2 | infer_model("gpt-4o")
            return chain2.invoke({"joke": output1})
        
        # Use the chain
        result = custom_chain.invoke("bears")
        ```
    
    The decorator also supports returning Runnables for dynamic chain construction:
        ```python
        @chain
        def dynamic_chain(input_: dict) -> Runnable:
            if input_.get("use_complex"):
                return complex_prompt | model
            else:
                return simple_prompt | model
        
        # The returned Runnable is automatically invoked
        result = dynamic_chain.invoke({"use_complex": True, "text": "hello"})
        ```
    """
    
    class ChainRunnable(Runnable[Input, Output]):
        """Runnable wrapper for decorated functions."""
        
        def __init__(self, wrapped_func: Callable):
            self.func = wrapped_func
            self.is_async = inspect.iscoroutinefunction(wrapped_func)
            functools.update_wrapper(self, wrapped_func)
        
        def invoke(self, input: Input, config: dict[str, Any] | None = None) -> Output:
            """Execute the wrapped function."""
            # Call the function
            if self.is_async:
                # Handle async function in sync context
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    raise RuntimeError(
                        f"Cannot call async @chain function '{self.func.__name__}' "
                        "synchronously from within an async context. "
                        "Use await runnable.ainvoke() instead."
                    )
                except RuntimeError:
                    # No running loop, safe to create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(self.func(input))
                    finally:
                        loop.close()
            else:
                result = self.func(input)
            
            # If the result is a Runnable, invoke it with the same input
            if isinstance(result, Runnable):
                return result.invoke(input, config)
            
            return result
        
        async def ainvoke(self, input: Input, config: dict[str, Any] | None = None) -> Output:
            """Asynchronously execute the wrapped function."""
            # Call the function
            if self.is_async:
                result = await self.func(input)
            else:
                # Run sync function in thread pool
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.func, input)
            
            # If the result is a Runnable, invoke it with the same input
            if isinstance(result, Runnable):
                return await result.ainvoke(input, config)
            
            return result
        
        def __repr__(self) -> str:
            return f"@chain({self.func.__name__})"
    
    return ChainRunnable(func)
