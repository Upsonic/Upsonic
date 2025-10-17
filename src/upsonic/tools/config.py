"""Tool configuration and decorator system."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class ToolHooks(BaseModel):
    """Optional before/after callables for tool execution lifecycle hooks."""
    before: Optional[Callable] = None
    after: Optional[Callable] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ToolConfig(BaseModel):
    """Tool behavior configuration for the Upsonic framework."""
    
    requires_confirmation: bool = Field(
        default=False,
        description="If True, the agent will pause and require user confirmation before executing the tool."
    )
    
    requires_user_input: bool = Field(
        default=False,
        description="If True, the agent will pause and prompt the user for input for specified fields."
    )
    
    user_input_fields: List[str] = Field(
        default_factory=list,
        description="Field names that the user should provide when requires_user_input is True."
    )
    
    external_execution: bool = Field(
        default=False,
        description="If True, the tool's execution is handled by an external process."
    )
    
    show_result: bool = Field(
        default=False,
        description="If True, the output is shown to the user and NOT sent back to the LLM."
    )
    
    stop_after_tool_call: bool = Field(
        default=False,
        description="If True, the agent's run will terminate after this tool call."
    )
    
    sequential: bool = Field(
        default=False,
        description="If True, this tool requires sequential execution (no parallelization)."
    )
    
    cache_results: bool = Field(
        default=False,
        description="If True, the result will be cached based on arguments."
    )
    
    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory to store cache files."
    )
    
    cache_ttl: Optional[int] = Field(
        default=None,
        description="Time-to-live for cache entries in seconds."
    )
    
    tool_hooks: Optional[ToolHooks] = Field(
        default=None,
        description="Custom functions to run before/after tool execution."
    )
    
    max_retries: Optional[int] = Field(
        default=5,
        description="Maximum number of retries allowed for this tool."
    )
    
    timeout: Optional[float] = Field(
        default=30.0,
        description="Timeout for tool execution in seconds."
    )
    
    strict: Optional[bool] = Field(
        default=None,
        description="Whether to enforce strict JSON schema validation on tool parameters."
    )
    
    docstring_format: str = Field(
        default='auto',
        description="Format of the docstring: 'google', 'numpy', 'sphinx', or 'auto'."
    )
    
    require_parameter_descriptions: bool = Field(
        default=False,
        description="If True, raise error if required parameter descriptions are missing."
    )


class _ToolDecorator:
    """Internal helper class for the tool decorator."""
    
    def __init__(self, config: ToolConfig):
        self.config = config
    
    def __call__(self, func: Callable) -> Callable:
        """Apply the decorator to a function."""
        # Store the config on the function
        setattr(func, '_upsonic_tool_config', self.config)
        # Mark it as a tool
        setattr(func, '_upsonic_is_tool', True)
        return func


def tool(*args: Any, **kwargs: Any) -> Union[Callable, _ToolDecorator]:
    """Decorator to configure tool behavior. Use @tool or @tool(...) for configuration."""
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        func = args[0]
        default_config = ToolConfig()
        return _ToolDecorator(default_config)(func)
    
    else:
        config = ToolConfig(**kwargs)
        return _ToolDecorator(config)
