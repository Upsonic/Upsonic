from __future__ import annotations

from typing import Optional, List, Callable, Any
from pydantic import BaseModel, Field

class ToolKit:
    """
    A base class for creating organized collections of tools.

    When a class inherits from ToolKit, the ToolProcessor will only
    register methods that are explicitly decorated with @tool as available tools.
    Methods without the decorator will be treated as internal helper methods
    and will not be exposed to the AI agent.

    Example:
        class MyToolSet(ToolKit):
            @tool
            def public_tool_for_ai(self, query: str) -> str:
                '''This tool is exposed to the AI.'''
                return self._internal_helper(query)

            def _internal_helper(self, text: str) -> str:
                '''This method is NOT exposed to the AI.'''
                return f"Processed: {text}"
    """
    pass

class ToolHooks(BaseModel):
    """
    A model to hold optional 'before' and 'after' callables for tool execution.
    This allows for custom logic injection at specific points in the tool's lifecycle,
    enabling advanced logging, state modification, or validation.
    """
    before: Optional[Callable] = None
    after: Optional[Callable] = None

    class Config:
        arbitrary_types_allowed = True


class ToolConfig(BaseModel):
    """
    A Pydantic model to holistically define the configuration and behavior
    of a tool within the Upsonic framework. This configuration is typically
    provided via the @tool decorator and is used by the ToolProcessor to
    generate a behavioral wrapper around the user's original tool function.
    """

    requires_confirmation: bool = Field(
        default=False,
        description="If True, the agent will pause and require user confirmation in the console before executing the tool."
    )

    requires_user_input: bool = Field(
        default=False,
        description="If True, the agent will pause and prompt the user for input for the fields specified in `user_input_fields`."
    )
    
    user_input_fields: List[str] = Field(
        default_factory=list,
        description="A list of argument names that the user should be prompted to provide when `requires_user_input` is True."
    )

    external_execution: bool = Field(
        default=False,
        description="If True, signals that the tool's execution is handled by an external process outside the agent's direct control. (For advanced use-cases)."
    )

    show_result: bool = Field(
        default=False,
        description="If True, the output of the tool is shown directly to the user and is NOT sent back to the LLM for further processing."
    )

    stop_after_tool_call: bool = Field(
        default=False,
        description="If True, the agent's run will terminate immediately after this tool call is completed."
    )

    tool_hooks: Optional[ToolHooks] = Field(
        default=None,
        description="An object containing custom functions to run before and/or after the tool's main logic is executed."
    )

    cache_results: bool = Field(
        default=False,
        description="If True, the result of the tool call will be cached based on its arguments."
    )

    cache_dir: Optional[str] = Field(
        default=None,
        description="The directory to store cache files. Defaults to a system-appropriate temporary directory if not set."
    )

    cache_ttl: Optional[int] = Field(
        default=None,
        description="Time-to-live for cache entries, in seconds. If None, cache entries do not expire."
    )


class _ToolDecorator:
    """An internal helper class used by the `tool` factory."""
    def __init__(self, config: ToolConfig):
        self.config = config

    def __call__(self, func: Callable) -> Callable:
        """
        This method is called when the decorator instance is applied to a function.
        It attaches the configuration object to the function itself.
        """
        setattr(func, '_upsonic_tool_config', self.config)
        return func


def tool(*args: Any, **kwargs: Any) -> Callable:
    """
    A decorator to configure the behavior of a tool function within the Upsonic framework.

    This decorator can be used in two ways:

    1.  Without parentheses (`@tool`):
        Applies a default configuration to the tool. This is the simplest way to mark
        a function as a valid tool.

        Example:
        ```
        @tool
        def my_simple_tool(arg: str) -> str:
            ...
        ```

    2.  With parentheses (`@tool(...)`):
        Allows for detailed configuration of the tool's behavior by passing
        arguments that correspond to the `ToolConfig` model.

        Example:
        ```
        @tool(requires_confirmation=True, cache_results=True, cache_ttl=3600)
        def my_complex_tool(arg: str) -> str:
            ...
        ```

    Args:
        *args: Used to detect the decoration syntax.
        **kwargs: Configuration parameters corresponding to `ToolConfig` fields.

    Returns:
        A callable that is either the decorated function (now tagged with a config)
        or a decorator instance ready to be applied to a function.
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        func = args[0]
        default_config = ToolConfig()
        return _ToolDecorator(default_config)(func)

    else:
        config = ToolConfig(**kwargs)
        return _ToolDecorator(config)