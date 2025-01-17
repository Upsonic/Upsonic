import inspect
import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
import dill
import base64
import httpx
from typing import Any, List, Dict, Optional, Type, Union, Callable
from pydantic import BaseModel
from functools import wraps

from ..tasks.tasks import Task

class NoAPIKeyException(Exception):
    pass

class UnsupportedLLMModelException(Exception):
    pass

class ComputerUse:
    pass

class Search:
    pass


class Tools:
    def tool(self, library: Optional[Union[str, List[str]]] = None):
        """
        Decorator to register a function or class as a tool.
        Can be used as @tool(), @tool("pandas"), or @tool(["pandas", "numpy"])

        Args:
            library: Optional library name or list of library names to install before registering the tool
        """
        def decorator(obj: Union[Callable, Type]):
            # Install libraries first if specified
            if library:
                if isinstance(library, str):
                    self.install_library(library)
                else:
                    for lib in library:
                        self.install_library(lib)
            
            # If it's a class, register each method as a tool
            if isinstance(obj, type):
                class_name = obj.__name__
                # Get all methods that don't start with underscore
                methods = [(name, getattr(obj, name)) for name in dir(obj) 
                          if not name.startswith('_') and callable(getattr(obj, name))]
                
                # Register each method as a tool
                for name, method in methods:
                    # Convert the method to a standalone function
                    def create_standalone(method, full_name):
                        @wraps(method)
                        def standalone(*args, **kwargs):
                            return method(*args, **kwargs)
                        standalone.__name__ = full_name
                        return standalone
                    
                    full_name = f"{class_name}__{name}"
                    standalone = create_standalone(method, full_name)
                    self.add_tool(standalone)
                
                return obj
            else:
                # Register the function as a tool
                @wraps(obj)
                def wrapper(*args, **kwargs):
                    return obj(*args, **kwargs)
                self.add_tool(wrapper)
                return wrapper
                
        return decorator

    def add_tool(
        self,
        function,
    ) -> Any:



        # Get the function then make a cloudpickle of it
        the_module = dill.detect.getmodule(function)
        if the_module is not None:
            cloudpickle.register_pickle_by_value(the_module)

        the_dumped_function = cloudpickle.dumps(function)

        data = {
            "function": base64.b64encode(the_dumped_function).decode("utf-8"),
        }
        
        result = self.send_request("/tools/add_tool", data)
        return result
    


    def add_mcp_tool(self, name: str, command: str, args: List[str], env: Dict[str, str] = {}) -> Dict[str, Any]:
        result = self.send_request("/tools/add_mcp_tool", {"name": name, "command": command, "args": args, "env": env})
        return result

    def install_library(self, library: str) -> Dict[str, Any]:
        result = self.send_request("/tools/install_library", {"library": library})
        return result

    def uninstall_library(self, library: str) -> Dict[str, Any]:
        result = self.send_request("/tools/uninstall_library", {"library": library})
        return result

    def mcp(self):
        """
        Decorator to register a class as an MCP tool.
        Usage:
        @client.mcp()
        class ToolName:
            command = "command-name"
            args = ["arg1", "arg2"]
            env = {"key": "value"}
        """
        def decorator(cls):
            command = getattr(cls, "command", None)
            args = getattr(cls, "args", [])
            env = getattr(cls, "env", {})

            name = cls.__name__
            
            if not command:
                raise ValueError("MCP tool class must have a 'command' attribute")
                
            self.add_mcp_tool(name, command, args, env)
            return cls
        return decorator
