import asyncio
from typing import Any, Dict, List, Tuple, Union, Callable, Awaitable

from upsonic.uel.runnable import Runnable, Input, Output


class RunnableParallel(Runnable[Input, Dict[str, Any]]):
    """
    A Runnable that executes multiple runnables in parallel.
    
    RunnableParallel allows you to run multiple chains concurrently and 
    collect their results into a dictionary. It can be created by passing
    either a dictionary of runnables or by using dict syntax in a chain.
    
    Example:
        ```python
        from upsonic.uel import ChatPromptTemplate, RunnableParallel
        from upsonic.models import infer_model
        
        model = infer_model("gpt-4o")
        
        joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
        poem_chain = ChatPromptTemplate.from_template("write a 2-line poem about {topic}") | model
        
        # Method 1: Using RunnableParallel directly
        parallel_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)
        
        # Method 2: Using dict syntax (automatically creates RunnableParallel)
        parallel_chain = {
            "joke": joke_chain,
            "poem": poem_chain
        }
        
        result = parallel_chain.invoke({"topic": "bears"})
        # result: {"joke": ModelResponse(...), "poem": ModelResponse(...)}
        ```
    """
    
    def __init__(self, **kwargs: Union[Runnable, Callable[[Any], Any]]):
        """
        Initialize RunnableParallel with named runnables.
        
        Args:
            **kwargs: Named runnables or callables to execute in parallel
        """
        self.steps: Dict[str, Runnable] = {}
        
        for name, runnable in kwargs.items():
            if isinstance(runnable, Runnable):
                self.steps[name] = runnable
            elif isinstance(runnable, dict):
                # Recursively handle nested dicts by creating nested RunnableParallel
                self.steps[name] = RunnableParallel.from_dict(runnable)
            elif callable(runnable):
                # Wrap callable in RunnableLambda
                from upsonic.uel.lambda_runnable import RunnableLambda
                self.steps[name] = RunnableLambda(runnable)
            else:
                raise TypeError(
                    f"RunnableParallel expects Runnable, callable, or dict values, "
                    f"got {type(runnable)} for key '{name}'"
                )
    
    @classmethod
    def from_dict(cls, steps: Dict[str, Union[Runnable, Callable[[Any], Any]]]) -> "RunnableParallel":
        """
        Create a RunnableParallel from a dictionary of steps.
        
        Args:
            steps: Dictionary mapping names to runnables or callables
            
        Returns:
            A new RunnableParallel instance
        """
        return cls(**steps)
    
    def invoke(self, input: Input, config: dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Execute all runnables in parallel using async approach.
        
        Uses asyncio.run() to execute the async version for true parallelism
        without threading issues.
        
        Args:
            input: The input to pass to all runnables
            config: Optional configuration
            
        Returns:
            Dictionary mapping step names to their results
        """
        import asyncio
        
        # Use async approach for true parallelism
        return asyncio.run(self.ainvoke(input, config))
    
    async def ainvoke(self, input: Input, config: dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Execute all runnables in parallel asynchronously.
        
        Args:
            input: The input to pass to all runnables
            config: Optional configuration
            
        Returns:
            Dictionary mapping step names to their results
        """
        # Create tasks for all runnables
        tasks = []
        names = []
        
        for name, runnable in self.steps.items():
            task = asyncio.create_task(runnable.ainvoke(input, config))
            tasks.append(task)
            names.append(name)
        
        # Wait for all tasks to complete
        try:
            results_list = await asyncio.gather(*tasks)
        except Exception as e:
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise
        
        # Build results dictionary
        results = {}
        for name, result in zip(names, results_list):
            results[name] = result
        
        return results
    
    def __or__(self, other: Union[Runnable, Callable[[Any], Any]]) -> Runnable:
        """
        Override pipe operator to handle dictionary outputs properly.
        
        When chaining after RunnableParallel, the next runnable receives
        the dictionary output.
        """
        from upsonic.uel.sequence import RunnableSequence
        if isinstance(other, Runnable):
            return RunnableSequence(steps=[self, other])
        elif callable(other):
            from upsonic.uel.lambda_runnable import RunnableLambda
            return RunnableSequence(steps=[self, RunnableLambda(other)])
        else:
            raise TypeError(f"Unsupported operand type for |: {type(other)}")
    
    def __repr__(self) -> str:
        step_names = ", ".join(self.steps.keys())
        return f"RunnableParallel({step_names})"
