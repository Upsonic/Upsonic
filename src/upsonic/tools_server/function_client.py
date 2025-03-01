import httpx
from typing import Dict, List, Any, Callable, Optional
from functools import wraps
import inspect


class FunctionToolManager:
    """Client for interacting with the Upsonic Functions API."""

    def __init__(self):
        """Initialize the Upsonic Function client."""
        self.base_url = "http://localhost:8086"

    def get_tools_by_name(self, name: list[str]):
        """
        Get tools by name, supporting wildcard patterns.
        
        Args:
            name: List of tool names or patterns (e.g. ["FileSystem.*", "MyTools.*"])
            
        Returns:
            List of matching tools
        """
        matching_tools = []
        for tool in self.tools():
            tool_name = tool.__name__
            for pattern in name:
                # Handle wildcard pattern
                if pattern.endswith(".*"):
                    prefix = pattern[:-2]  # Remove .* from the end
                    if tool_name.startswith(prefix):
                        matching_tools.append(tool)
                        break
                # Exact match
                elif tool_name == pattern:
                    matching_tools.append(tool)
                    break
        return matching_tools

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def close(self):
        """Close the client session."""
        pass

    def list_tools(self) -> Dict[str, Any]:
        """List all available tools."""
        with httpx.Client(timeout=600.0) as session:
            response = session.post(f"{self.base_url}/functions/tools")
            response.raise_for_status()
            return response.json()

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a specific tool with the given arguments.

        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments to pass to the tool

        Returns:
            Tool execution results
        """
        print("Tool Calling")
        print(tool_name)
        print(arguments)
        with httpx.Client(timeout=600.0) as session:
            response = session.post(
                f"{self.base_url}/functions/call_tool",
                json={"tool_name": tool_name, "arguments": arguments},
            )
            response.raise_for_status()
            print("return")
            print(response.json())
            return response.json()

    def tools(self) -> List[Callable[..., Dict[str, Any]]]:
        """Initialize tool-specific methods based on available tools."""
        tools_response = self.list_tools()



        tools = tools_response.get("available_tools", {}).get("tools", [])

        functions: List[Callable[..., Dict[str, Any]]] = []

        def get_python_type(schema_type: str, format: Optional[str] = None) -> type:
            """Convert JSON schema type to Python type."""
            type_mapping = {
                "string": str,
                "integer": int,
                "boolean": bool,
                "number": float,
                "array": list,
                "object": dict,
            }
            return type_mapping.get(schema_type, Any)

        for tool in tools:
            tool_name: str = tool["name"]
            tool_desc: str = tool.get("description", "")
            input_schema: Dict[str, Any] = tool.get("inputSchema", {})
            properties: Dict[str, Dict[str, Any]] = input_schema.get("properties", {})
            required: List[str] = input_schema.get("required", [])

            def create_tool_function(
                tool_name: str,
                properties: Dict[str, Dict[str, Any]],
                required: List[str],
            ) -> Callable[..., Dict[str, Any]]:
                annotations = {}
                defaults = {}
                parameters = []

                # Build parameters for both required and optional arguments
                for param_name in required:
                    param_info = properties[param_name]
                    param_type = get_python_type(param_info.get("type", "any"))
                    annotations[param_name] = param_type
                    parameters.append(
                        inspect.Parameter(
                            param_name,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=param_type
                        )
                    )

                for param_name, param_info in properties.items():
                    if param_name not in required:
                        param_type = get_python_type(param_info.get("type", "any"))
                        annotations[param_name] = param_type
                        default_value = param_info.get("default", None)
                        defaults[param_name] = default_value
                        parameters.append(
                            inspect.Parameter(
                                param_name,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                default=default_value,
                                annotation=param_type
                            )
                        )

                def tool_function(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                    all_kwargs = kwargs.copy()
                    for i, arg in enumerate(args):
                        if i < len(required):
                            all_kwargs[required[i]] = arg

                    for param, default in defaults.items():
                        if param not in all_kwargs:
                            all_kwargs[param] = default

                    return self.call_tool(tool_name, all_kwargs)

                # Create a signature object and apply it to the function
                sig = inspect.Signature(parameters=parameters, return_annotation=Dict[str, Any])
                tool_function.__signature__ = sig
                tool_function.__name__ = tool_name
                tool_function.__annotations__ = {
                    **annotations,
                    "return": Dict[str, Any],
                }
                tool_function.__doc__ = f"{tool_desc}\n\nReturns:\n    Tool execution results"

                return tool_function


            func = create_tool_function(tool_name, properties, required)
            functions.append(func)

        return functions
