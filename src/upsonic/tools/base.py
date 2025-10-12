"""Base interfaces and types for the Upsonic tool system."""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Type, Union, 
    Literal, TypeAlias, Protocol, runtime_checkable
)

# Type aliases for compatibility
DocstringFormat: TypeAlias = Literal['google', 'numpy', 'sphinx', 'auto']
"""Supported docstring formats."""

ObjectJsonSchema: TypeAlias = Dict[str, Any]
"""Type representing JSON schema of an object."""

from pydantic import BaseModel


# Type aliases for tool functions
ToolCallResult: TypeAlias = Union[str, Dict[str, Any], BaseModel, Any]
ToolFunction: TypeAlias = Callable[..., Union[ToolCallResult, asyncio.Future[ToolCallResult]]]

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


@runtime_checkable
class Tool(Protocol):
    """Protocol for all tools in the Upsonic framework."""
    
    @property
    def name(self) -> str:
        """The name of the tool."""
        ...
    
    @property
    def description(self) -> Optional[str]:
        """The description of the tool."""
        ...
    
    @property
    def schema(self) -> ToolSchema:
        """The schema for the tool."""
        ...
    
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool."""
        ...


class ToolBase(ABC):
    """Abstract base class for tools."""
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        schema: Optional[ToolSchema] = None,
        metadata: Optional[ToolMetadata] = None,
    ):
        self._name = name
        self._description = description
        self._schema = schema or ToolSchema(parameters={})
        self._metadata = metadata or ToolMetadata(name=name)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> Optional[str]:
        return self._description
    
    @property
    def schema(self) -> ToolSchema:
        return self._schema
    
    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata
    
    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool."""
        raise NotImplementedError


class ToolKit:
    """Base class for organized tool collections. Only @tool decorated methods are exposed."""
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
