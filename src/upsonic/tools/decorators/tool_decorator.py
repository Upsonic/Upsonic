import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from ..base.tool import Tool


class FunctionTool(Tool):
    """Tool implementation for a decorated function"""

    def __init__(self, func: Callable, custom_description: Optional[str] = None):
        self.func = func
        self.func_name = func.__name__
        self.__name__ = func.__name__
        self.__qualname__ = getattr(func, "__qualname__", func.__name__)
        self.__doc__ = func.__doc__
        self._description = (
            custom_description or func.__doc__ or f"Execute {self.func_name} function"
        )
        self.type_hints = get_type_hints(func)
        self.signature = inspect.signature(func)

    @property
    def name(self) -> str:
        return self.func_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> Dict[str, Any]:
        properties = {}
        required = []

        for param_name, param in self.signature.parameters.items():
            if param_name == "self":
                continue

            param_type = self.type_hints.get(param_name, Any)
            param_desc = self._extract_param_description(param_name)

            properties[param_name] = {
                "type": self._get_json_type(param_type),
                "description": param_desc,
            }

            if hasattr(param_type, "__args__") and all(
                isinstance(arg, str) for arg in param_type.__args__
            ):
                properties[param_name]["enum"] = list(param_type.__args__)

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {"type": "object", "properties": properties, "required": required}

    def execute(self, **kwargs) -> Any:
        """Execute the function with the given parameters"""
        return self.func(**kwargs)

    def __call__(self, **kwargs) -> Any:
        """Make the FunctionTool callable like a function"""
        if hasattr(self.func, "__self__"):
            return self.func(**kwargs)
        else:
            return self.func(**kwargs)

    def _get_json_type(self, py_type: Any) -> str:
        """Convert Python type to JSON schema type"""
        if py_type in (str, type(None)):
            return "string"
        elif py_type in (int, float):
            return "number"
        elif py_type == bool:
            return "boolean"
        elif py_type == list or getattr(py_type, "__origin__", None) == list:
            return "array"
        elif py_type == dict or getattr(py_type, "__origin__", None) == dict:
            return "object"
        else:
            return "string"

    def _extract_param_description(self, param_name: str) -> str:
        """Extract parameter description from docstring"""
        if not self.func.__doc__:
            return ""

        docstring = self.func.__doc__
        lines = docstring.split("\n")
        param_section = False

        for line in lines:
            line = line.strip()

            if line.lower().startswith("args:") or line.lower().startswith(
                "parameters:"
            ):
                param_section = True
                continue

            if (
                param_section
                and line
                and line.endswith(":")
                and not line.startswith(" ")
            ):
                param_section = False

            if param_section and ":" in line:
                parts = line.split(":", 1)
                if parts[0].strip() == param_name:
                    return parts[1].strip()

        return ""


def tool(func: Optional[Callable] = None, *, description: Optional[str] = None):
    """
    Decorator to mark a function as a tool.

    Args:
        func: The function to decorate
        description: Optional description to override function docstring

    Usage:
        @tool
        def my_function(...):
            ...

        @tool(description="Custom description")
        def my_function(...):
            ...
    """

    def decorator(f):
        f._is_tool = True
        f._tool_description = description

        return f

    if func is None:
        return decorator
    return decorator(func)


def get_tools_from_instance(instance: Any) -> List[Any]:
    """
    Get all tools from a class instance that have been marked with @tool decorator.

    Args:
        instance: The class instance to scan for decorated methods

    Returns:
        List of bound methods for each decorated method
    """
    tools = []

    for method_name in dir(instance):
        method = getattr(instance, method_name)

        if callable(method) and hasattr(method, "_is_tool") and method._is_tool:
            tools.append(method)

    return tools
