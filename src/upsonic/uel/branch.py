from typing import Any, Callable, List, Tuple, Union

from upsonic.uel.runnable import Runnable, Input, Output
from upsonic.uel.lambda_runnable import coerce_to_runnable


class RunnableBranch(Runnable[Input, Output]):
    """
    A Runnable that routes to different runnables based on conditions.
    
    RunnableBranch allows you to define conditional logic in your chains.
    It evaluates conditions in order and executes the first matching branch.
    
    Example:
        ```python
        from upsonic.uel import RunnableBranch, ChatPromptTemplate
        from upsonic.models import infer_model
        
        model = infer_model("gpt-4o")
        
        # Define branch chains
        upsonic_chain = (
            ChatPromptTemplate.from_template("Expert response about Upsonic: {question}")
            | model
        )
        
        anthropic_chain = (
            ChatPromptTemplate.from_template("Expert response about Anthropic: {question}")
            | model
        )
        
        general_chain = (
            ChatPromptTemplate.from_template("General response: {question}")
            | model
        )
        
        # Create branch with conditions
        branch = RunnableBranch(
            (lambda x: "upsonic" in x["topic"].lower(), upsonic_chain),
            (lambda x: "anthropic" in x["topic"].lower(), anthropic_chain),
            general_chain  # default branch
        )
        
        result = branch.invoke({"topic": "upsonic", "question": "How does it work?"})
        ```
    """
    
    def __init__(
        self,
        *branches: Union[
            Tuple[Callable[[Input], bool], Union[Runnable, Callable]],
            Union[Runnable, Callable]
        ]
    ):
        """
        Initialize RunnableBranch with condition-runnable pairs and a default.
        
        Args:
            *branches: Variable number of (condition, runnable) tuples, 
                      with the last argument being the default runnable
        """
        if len(branches) < 1:
            raise ValueError("RunnableBranch requires at least a default branch")
        
        self.conditions_and_runnables: List[Tuple[Callable[[Input], bool], Runnable]] = []
        self.default_runnable: Runnable
        
        # Process all branches
        for i, branch in enumerate(branches):
            if i == len(branches) - 1:
                # Last item is the default
                if isinstance(branch, tuple):
                    raise ValueError(
                        "The last argument to RunnableBranch must be a default runnable, "
                        "not a (condition, runnable) tuple"
                    )
                self.default_runnable = coerce_to_runnable(branch)
            else:
                # Previous items must be (condition, runnable) tuples
                if not isinstance(branch, tuple) or len(branch) != 2:
                    raise ValueError(
                        f"Branch {i} must be a (condition, runnable) tuple, "
                        f"got {type(branch)}"
                    )
                condition, runnable = branch
                if not callable(condition):
                    raise ValueError(
                        f"Condition in branch {i} must be callable, "
                        f"got {type(condition)}"
                    )
                self.conditions_and_runnables.append(
                    (condition, coerce_to_runnable(runnable))
                )
    
    def invoke(self, input: Input, config: dict[str, Any] | None = None) -> Output:
        """
        Evaluate conditions and execute the first matching branch.
        
        Args:
            input: The input to evaluate conditions against and pass to the selected runnable
            config: Optional configuration
            
        Returns:
            The output from the selected branch's runnable
        """
        # Check each condition in order
        for condition, runnable in self.conditions_and_runnables:
            try:
                if condition(input):
                    return runnable.invoke(input, config)
            except Exception as e:
                # If condition evaluation fails, skip to next
                continue
        
        # No conditions matched, use default
        return self.default_runnable.invoke(input, config)
    
    async def ainvoke(self, input: Input, config: dict[str, Any] | None = None) -> Output:
        """
        Asynchronously evaluate conditions and execute the first matching branch.
        
        Args:
            input: The input to evaluate conditions against and pass to the selected runnable
            config: Optional configuration
            
        Returns:
            The output from the selected branch's runnable
        """
        # Check each condition in order
        for condition, runnable in self.conditions_and_runnables:
            try:
                # Handle async conditions
                import inspect
                if inspect.iscoroutinefunction(condition):
                    condition_result = await condition(input)
                else:
                    condition_result = condition(input)
                    
                if condition_result:
                    return await runnable.ainvoke(input, config)
            except Exception as e:
                # If condition evaluation fails, skip to next
                continue
        
        # No conditions matched, use default
        return await self.default_runnable.ainvoke(input, config)
    
    def __repr__(self) -> str:
        num_conditions = len(self.conditions_and_runnables)
        return f"RunnableBranch(conditions={num_conditions}, default={self.default_runnable})"
