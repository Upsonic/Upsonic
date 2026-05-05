"""Base interfaces and types for the Upsonic tool system."""

from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional,
    Literal, TypeAlias, TYPE_CHECKING
)

if TYPE_CHECKING:
    from upsonic.tools.metrics import ToolMetrics
    from upsonic.tools.schema import FunctionSchema

# Type aliases for compatibility
DocstringFormat: TypeAlias = Literal['google', 'numpy', 'sphinx', 'auto']
"""Supported docstring formats."""

ObjectJsonSchema: TypeAlias = Dict[str, Any]
"""Type representing JSON schema of an object."""

# Tool kinds
ToolKind: TypeAlias = Literal['function', 'output', 'external', 'unapproved', 'mcp']

class ToolValidationError(Exception):
    """Error raised when tool validation fails."""
    pass


@dataclass
class ToolMetadata:
    """Universal metadata for all tools."""
    name: str
    description: Optional[str] = None
    
    # Universal attributes
    kind: ToolKind = 'function'
    """Type of tool ('function', 'output', 'external', 'unapproved', 'mcp')"""
    
    is_async: bool = False
    """Whether the tool is async"""
    
    strict: bool = False
    """Whether to use strict schema validation"""
    
    # Tool-specific metadata
    custom: Dict[str, Any] = field(default_factory=dict)
    """Tool-specific metadata that doesn't fit in universal fields"""


class Tool:
    """
    Central base class for all tools in the Upsonic framework.
    
    This is the main class that all tools (except builtin tools) inherit from.
    It provides:
    - Standard tool interface (name, description, metadata)
    - Metrics tracking for each tool instance  
    - Abstract execute method for tool logic
    
    Usage:
        Create custom tools by inheriting from Tool and implementing execute():
        
        ```python
        class MyTool(Tool):
            def __init__(self):
                super().__init__(
                    name="my_tool",
                    description="Does something useful",
                    metadata=ToolMetadata(name="my_tool", kind='function')
                )
            
            async def execute(self, **kwargs):
                # Tool logic here
                return result
        ```
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        schema: Optional['FunctionSchema'] = None,
        metadata: Optional[ToolMetadata] = None,
        tool_id: Optional[str] = None,
    ):
        """
        Initialize a tool.
        
        Args:
            name: Tool name
            description: Tool description
            schema: Tool's input schema
            metadata: Tool metadata
            tool_id: Optional unique identifier. If not provided, auto-generated from class name + name
        """
        self.name = name
        self.description = description or ""
        self.schema = schema
        self.metadata = metadata or ToolMetadata(name=name)
        
        # Auto-generate stable tool_id if not provided
        if tool_id is None:
            tool_id = f"{self.__class__.__name__}_{name}"
        self.tool_id = tool_id
        
        # Tool-specific metrics tracking
        from upsonic.tools.metrics import ToolMetrics
        self._metrics = ToolMetrics()
    
    @property
    def metrics(self) -> "ToolMetrics":
        """The metrics for this tool instance."""
        return self._metrics

    
    
    def record_execution(
        self, 
        execution_time: float, 
        args: Dict[str, Any] = None,
        result: Any = None,
        success: bool = True
    ) -> None:
        """
        Record a tool execution in metrics and history.
        
        Args:
            execution_time: Time taken to execute in seconds
            args: Arguments passed to the tool
            result: Result returned by the tool
            success: Whether the execution was successful
        """
        self._metrics.increment_tool_count()
        
        # Store execution history in metadata custom dict
        if 'execution_history' not in self.metadata.custom:
            self.metadata.custom['execution_history'] = []
        
        self.metadata.custom['execution_history'].append({
            'timestamp': time.time(),
            'execution_time': execution_time,
            'args': args,
            'result': str(result) if result is not None else None,
            'success': success
        })
        if len(self.metadata.custom['execution_history']) > 100:
            self.metadata.custom['execution_history'] = self.metadata.custom['execution_history'][-100:]
    
    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the tool.
        
        This method must be implemented by all tool subclasses.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Tool execution result
        """
        raise NotImplementedError


class ToolKit:
    """
    Base class for organized tool collections (config-only).

    Stores user-provided configuration: filtering (``include_tools`` /
    ``exclude_tools``), ``use_async`` mode, and toolkit-wide ``ToolConfig``
    defaults.  **No tool discovery or registration happens here** -- that
    is the ``ToolProcessor``'s responsibility.

    Tool discovery rules:

    * ``@tool``-decorated methods form the base candidate set.
    * ``use_async=True`` replaces candidates with **all** async methods
      and drops every sync method (even if decorated).
    * ``include_tools`` is **additive** -- listed names are added to the
      candidate set even without ``@tool``.
    * ``exclude_tools`` is **supreme** -- listed names are always removed.

    Config merge priority (highest first):

    1. Toolkit ``__init__`` defaults -- toolkit-wide, set at instantiation.
    2. ``@tool`` decorator -- per-method, set at class definition time.

    After the processor processes this toolkit it writes the final list
    of registered callables back into ``self.tools``, which is also
    exposed via the ``functions`` property for introspection.

    Usage::

        from upsonic.tools import tool, ToolKit

        class MyToolKit(ToolKit):
            def __init__(self, api_key: str, **kwargs):
                super().__init__(**kwargs)
                self.api_key = api_key

            @tool(requires_confirmation=True)
            def dangerous_action(self, x: int) -> int:
                return x * 2

            @tool
            def safe_action(self, y: str) -> str:
                return y.upper()

        tk = MyToolKit(
            api_key="...",
            include_tools=["extra_helper"],
            timeout=60.0,
        )
        agent = Agent(tools=[tk])  # processor registers & populates tk.functions
    """

    def __init__(
        self,
        # --- Filtering & async mode ---
        include_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None,
        use_async: bool = False,
        # --- Toolkit-level instructions ---
        instructions: Optional[str] = None,
        add_instructions: bool = False,
        # --- Toolkit-wide defaults (mirror every ToolConfig field) ---
        requires_confirmation: Optional[bool] = None,
        requires_confirmation_tools: Optional[List[str]] = None,
        requires_user_input: Optional[bool] = None,
        requires_user_input_tools: Optional[List[str]] = None,
        user_input_fields: Optional[List[str]] = None,
        external_execution: Optional[bool] = None,
        requires_external_execution_tools: Optional[List[str]] = None,
        show_result: Optional[bool] = None,
        stop_after_tool_call: Optional[bool] = None,
        sequential: Optional[bool] = None,
        cache_results: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        tool_hooks: Optional[Any] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
        strict: Optional[bool] = None,
        docstring_format: Optional[str] = None,
        require_parameter_descriptions: Optional[bool] = None,
    ) -> None:
        self._toolkit_include_tools: Optional[List[str]] = include_tools
        self._toolkit_exclude_tools: Optional[List[str]] = exclude_tools
        self._toolkit_use_async: bool = use_async
        self.instructions: Optional[str] = instructions
        self.add_instructions: bool = add_instructions

        self._requires_confirmation_tools: Optional[List[str]] = requires_confirmation_tools
        self._requires_user_input_tools: Optional[List[str]] = requires_user_input_tools
        self._requires_external_execution_tools: Optional[List[str]] = requires_external_execution_tools
        
        self._toolkit_defaults: Dict[str, Any] = {
            "requires_confirmation": requires_confirmation,
            "requires_user_input": requires_user_input,
            "user_input_fields": user_input_fields,
            "external_execution": external_execution,
            "show_result": show_result,
            "stop_after_tool_call": stop_after_tool_call,
            "sequential": sequential,
            "cache_results": cache_results,
            "cache_dir": cache_dir,
            "cache_ttl": cache_ttl,
            "tool_hooks": tool_hooks,
            "max_retries": max_retries,
            "timeout": timeout,
            "strict": strict,
            "docstring_format": docstring_format,
            "require_parameter_descriptions": require_parameter_descriptions,
        }

        # Populated by ToolProcessor._process_toolkit() after registration
        self.tools: List[Callable] = []

    @property
    def functions(self) -> List[Callable]:
        """Return the list of registered tool callables (populated by the processor)."""
        return self.tools


@dataclass
class ToolDefinition:
    """Tool definition passed to a model."""
    
    name: str
    """The name of the tool."""
    
    parameters_json_schema: Dict[str, Any] = field(default_factory=lambda: {'type': 'object', 'properties': {}})
    """The JSON schema for the tool's parameters."""
    
    description: Optional[str] = None
    """The description of the tool."""
    
    kind: ToolKind = 'function'
    """The kind of tool."""
    
    strict: Optional[bool] = None
    """Whether to enforce strict JSON schema validation."""
    
    sequential: bool = False
    """Whether this tool requires a sequential/serial execution environment."""
    
    metadata: Optional[Dict[str, Any]] = None
    """Tool metadata that is not sent to the model."""
    
    @property
    def defer(self) -> bool:
        """Whether calls to this tool will be deferred."""
        return self.kind in ('external', 'unapproved')



@dataclass
class ToolResult:
    """Internal representation of a tool execution result."""
    
    tool_name: str
    """The name of the tool that was called."""
    
    content: Any
    """The return value."""
    
    tool_call_id: Optional[str] = None
    """The tool call identifier."""
    
    success: bool = True
    """Whether the tool execution was successful."""
    
    error: Optional[str] = None
    """Error message if the tool execution failed."""
    
    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata for the result."""
    
    execution_time: Optional[float] = None
    """Time taken to execute the tool in seconds."""
