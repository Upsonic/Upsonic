import inspect
import cloudpickle

from ..level_utilized.utility import error_handler
cloudpickle.DEFAULT_PROTOCOL = 2
import dill
import base64
import httpx
from typing import Any, List, Dict, Optional, Type, Union, Callable
from pydantic import BaseModel
from functools import wraps

import inspect
import functools

from ..tasks.tasks import Task
from ...exception import NoAPIKeyException, UnsupportedLLMModelException
from ..printing import mcp_tool_operation

class ComputerUse:
    pass

class BrowserUse:
    pass

class Search:
    pass


def generate_static_method_class(instance):
    # Store instance attributes
    instance_attrs = {name: value for name, value in inspect.getmembers(instance)
                     if not name.startswith('__') and not callable(value)}
    
    # Create new class with the same name as the original class
    original_class_name = type(instance).__name__
    TransformedClass = type(original_class_name, (), {})
    
    # Set instance attributes as class attributes
    for attr_name, attr_value in instance_attrs.items():
        setattr(TransformedClass, attr_name, attr_value)

    # Dynamically add each method as static method to the new class
    for method_name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
        if not method_name.startswith('__'):
            # Create a closure that captures the instance attributes
            def create_static_method(method, instance_attrs):
                if inspect.iscoroutinefunction(method):
                    @functools.wraps(method)
                    async def static_wrapper(*args, **kwargs):
                        # Create a new instance with the stored attributes
                        temp_instance = type(instance)(**{k: v for k, v in instance_attrs.items()})
                        return await method.__get__(temp_instance, type(instance))(*args, **kwargs)
                    return static_wrapper
                else:
                    @functools.wraps(method)
                    def static_wrapper(*args, **kwargs):
                        # Create a new instance with the stored attributes
                        temp_instance = type(instance)(**{k: v for k, v in instance_attrs.items()})
                        return method.__get__(temp_instance, type(instance))(*args, **kwargs)
                    return static_wrapper

            static_method = staticmethod(create_static_method(method, instance_attrs))
            setattr(TransformedClass, method_name, static_method)

    return TransformedClass

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
                        if inspect.iscoroutinefunction(method):
                            @wraps(method)
                            async def standalone(*args, **kwargs):
                                return await method(*args, **kwargs)
                        else:
                            @wraps(method)
                            def standalone(*args, **kwargs):
                                return method(*args, **kwargs)
                        standalone.__name__ = full_name
                        return standalone
                    
                    full_name = f"{class_name}__{name}"
                    standalone = create_standalone(method, full_name)
                    self.add_tool(standalone)
                
                return obj
            
            if isinstance(obj, object):
                obj = generate_static_method_class(obj)

                # Get all methods that don't start with underscore
                methods = [(name, getattr(obj, name)) for name in dir(obj) 
                          if not name.startswith('_') and callable(getattr(obj, name))]
                
                # Register each method as a tool
                for name, method in methods:
                    # Convert the method to a standalone function
                    def create_standalone(method, full_name):
                        if inspect.iscoroutinefunction(method):
                            @wraps(method)
                            async def standalone(*args, **kwargs):
                                return await method(*args, **kwargs)
                        else:
                            @wraps(method)
                            def standalone(*args, **kwargs):
                                return method(*args, **kwargs)
                        standalone.__name__ = full_name
                        return standalone
                    
                    full_name = f"{obj.__name__}__{name}"
                    standalone = create_standalone(method, full_name)
                    self.add_tool(standalone)
                
            else:
                # Register the function as a tool
                if inspect.iscoroutinefunction(obj):
                    @wraps(obj)
                    async def wrapper(*args, **kwargs):
                        return await obj(*args, **kwargs)
                else:
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
        error_handler(result)
        mcp_tool_operation(f"MCP Tool: {name}", "Successfully Added")
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

    def sse_mcp(self):
        """
        Decorator to register a class as an MCP tool that uses Server-Sent Events (SSE).
        Usage:
        @client.sse_mcp()
        class ToolName:
            url = "https://example.com/sse"
        
        """
        def decorator(cls):
            url = getattr(cls, "url", None)

            name = cls.__name__

            if not url:
                raise ValueError("SSE MCP tool class must have a 'url' attribute")
            
            self.add_sse_mcp(name, url)
            return cls
        return decorator


    def add_sse_mcp(self, name: str, url: str) -> Dict[str, Any]:
        result = self.send_request("/tools/add_sse_mcp", {"name": name, "url": url})
        return result
