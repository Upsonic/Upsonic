from typing import Any, Dict, List

from upsonic.tools.base.tool import Tool


class ToolRegistry:
    """Registry for all available tools"""

    _instance = None

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the tools dictionary"""
        if not hasattr(self, "_tools"):
            self._tools = {}

    def register(self, tool: Tool) -> None:
        """Register a tool in the registry"""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool:
        """Get a tool by name"""
        if name not in self._tools:
            raise ValueError(f"Tool {name} not found in registry")
        return self._tools[name]

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools as dicts for Claude API"""
        return [tool.as_dict() for tool in self._tools.values()]

    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with given parameters"""
        tool = self.get_tool(name)
        return tool.execute(**kwargs)

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists in the registry"""
        return name in self._tools
