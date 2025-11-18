"""Base interfaces and types for the Upsonic tool system."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, 
    Literal, TypeAlias, TYPE_CHECKING
)

if TYPE_CHECKING:
    from upsonic.tools.context import ToolMetrics

# Type aliases for compatibility
DocstringFormat: TypeAlias = Literal['google', 'numpy', 'sphinx', 'auto']
"""Supported docstring formats."""

ObjectJsonSchema: TypeAlias = Dict[str, Any]
"""Type representing JSON schema of an object."""

# Tool kinds
ToolKind: TypeAlias = Literal['function', 'output', 'external', 'unapproved', 'mcp']


@dataclass
class ToolMetadata:
    """Metadata for a tool."""
    name: str
    description: Optional[str] = None
    version: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSchema:
    """Schema information for a tool."""
    parameters: Dict[str, Any]  # JSON Schema for parameters
    return_type: Optional[Dict[str, Any]] = None  # JSON Schema for return type
    strict: bool = False
    
    @property
    def json_schema(self) -> Dict[str, Any]:
        """Get the full JSON schema for the tool."""
        return {
            "type": "object",
            "properties": self.parameters.get("properties", {}),
            "required": self.parameters.get("required", []),
            "additionalProperties": not self.strict
        }


class Tool:
    """
    Central base class for all tools in the Upsonic framework.
    
    This is the main class that all tools (except builtin tools) inherit from.
    It provides:
    - Standard tool interface (name, description, schema, metadata)
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
                    schema=ToolSchema(parameters={...})
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
        schema: Optional[ToolSchema] = None,
        metadata: Optional[ToolMetadata] = None,
    ):
        """
        Initialize a tool.
        
        Args:
            name: Tool name
            description: Tool description
            schema: Tool parameter schema
            metadata: Tool metadata
        """
        self._name = name
        self._description = description
        self._schema = schema or ToolSchema(parameters={})
        self._metadata = metadata or ToolMetadata(name=name)
        
        # Tool-specific metrics tracking
        from upsonic.tools.context import ToolMetrics
        self._metrics = ToolMetrics()
    
    @property
    def name(self) -> str:
        """The name of the tool."""
        return self._name
    
    @property
    def description(self) -> Optional[str]:
        """The description of the tool."""
        return self._description
    
    @property
    def schema(self) -> ToolSchema:
        """The schema for the tool."""
        return self._schema
    
    @property
    def metadata(self) -> ToolMetadata:
        """The metadata for the tool."""
        return self._metadata
    
    @property
    def metrics(self) -> "ToolMetrics":
        """The metrics for this tool instance."""
        return self._metrics
    
    def record_execution(self, execution_time: float, success: bool = True) -> None:
        """
        Record a tool execution in metrics.
        
        Args:
            execution_time: Time taken to execute in seconds
            success: Whether the execution was successful
        """
        self._metrics.increment_tool_count()
        
        # Store execution history in metadata custom dict
        if 'execution_history' not in self._metadata.custom:
            self._metadata.custom['execution_history'] = []
        
        self._metadata.custom['execution_history'].append({
            'execution_time': execution_time,
            'success': success,
            'total_calls': self._metrics.tool_call_count
        })
        
        # Keep only last 100 executions to avoid memory bloat
        if len(self._metadata.custom['execution_history']) > 100:
            self._metadata.custom['execution_history'] = self._metadata.custom['execution_history'][-100:]
    
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
    Base class for organized tool collections.
    
    Only @tool decorated methods are exposed as tools.
    
    Usage:
        ```python
        from upsonic.tools import tool, ToolKit
        
        class MyToolKit(ToolKit):
            @tool
            def tool1(self, x: int) -> int:
                '''Tool 1 description'''
                return x * 2
            
            @tool
            def tool2(self, y: str) -> str:
                '''Tool 2 description'''
                return y.upper()
        ```
    """
    pass


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
class ToolCall:
    """Internal representation of a tool call request."""
    
    tool_name: str
    """The name of the tool to call."""
    
    args: Optional[Dict[str, Any]] = None
    """The arguments to pass to the tool."""
    
    tool_call_id: Optional[str] = None
    """The tool call identifier."""
    
    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata for the tool call."""


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
