from typing import Any, Callable, List, Optional
import inspect


class FunctionToolkit:
    """Toolkit wrapper for standalone decorated functions"""

    def __init__(self, func: Callable, name: Optional[str] = None):
        """
        Initialize the toolkit with a single function.

        Args:
            func: The decorated function to wrap
            name: Optional name for the toolkit
        """
        # Store original function without making it accessible as an attribute
        self._original_func = func
        self._name = name or f"{func.__name__}Toolkit"

        # Create a bound method that accepts only keyword arguments but preserves signature
        def bound_method(**kwargs):
            return func(**kwargs)

        # Copy function attributes to the bound method
        bound_method.__name__ = func.__name__
        bound_method.__doc__ = func.__doc__
        bound_method._is_tool = True
        bound_method._tool_description = getattr(func, "_tool_description", None)

        # Preserve the original function's signature for proper schema generation
        bound_method.__signature__ = inspect.signature(func)
        bound_method.__annotations__ = func.__annotations__

        # Set the bound method as an attribute and store reference for functions()
        setattr(self, func.__name__, bound_method)
        self._bound_method = bound_method

    def __control__(self) -> bool:
        """Validate that the toolkit is ready to use."""
        return True

    @property
    def name(self) -> str:
        """Get the toolkit name."""
        return self._name

    def functions(self) -> List[Any]:
        """Return the list of tool functions - only the bound method."""
        return [self._bound_method]

    def get_description(self) -> str:
        """Return a description of what this toolkit does."""
        description = (
            getattr(self._original_func, "_tool_description", None)
            or self._original_func.__doc__
        )
        return (
            description or f"Toolkit containing {self._original_func.__name__} function"
        )
