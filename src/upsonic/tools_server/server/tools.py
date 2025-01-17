import base64
import inspect
import subprocess
import traceback
import asyncio
from typing import List, Dict, Any, Optional, Union, Callable
from pydantic import BaseModel



from fastapi import HTTPException
from pydantic import BaseModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.stdio import get_default_environment
import asyncio
from contextlib import asynccontextmanager
@asynccontextmanager
async def managed_session(command: str, args: list, env: dict | None = None):

    if env is None:
        env = get_default_environment()
    else:
        default_env = get_default_environment()
        default_env.update(env)
        env = default_env


    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=env,
    )

    
    client = None
    session = None
    
    try:
        client = stdio_client(server_params)

        read, write = await client.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()
        yield session
    finally:
        if session:
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
        if client:
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass


def install_library_(library):
    try:
        result = subprocess.run(
            ["uv", "pip", "install", library],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:

        return False


def uninstall_library_(library):
    try:
        result = subprocess.run(
            ["uv", "pip", "uninstall", "-y", library],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:

        return False
    

def add_tool_(function, description: str = "", properties: Dict[str, Any] = None, required: List[str] = None):
    """
    Add a tool to the registered functions.
    
    Args:
        function: The function to be registered as a tool
    """
    from ..server.function_tools import tool
    # Apply the tool decorator with empty description


    
    decorated_function = tool(description=description, custom_properties=properties, custom_required=required)(function)
    return decorated_function

    






import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
from fastapi import HTTPException
from pydantic import BaseModel
from mcp import ClientSession, StdioServerParameters

import asyncio
from contextlib import asynccontextmanager
# Create server parameters for stdio connection

from .api import app, timeout


prefix = "/tools"


class InstallLibraryRequest(BaseModel):
    library: str



@app.post(f"{prefix}/install_library")
@timeout(30.0)
async def install_library(request: InstallLibraryRequest):
    """
    Endpoint to install a library.

    Args:
        library: The library to install

    Returns:
        A success message
    """


    install_library_(request.library)

    return {"message": "Library installed successfully"}



@app.post(f"{prefix}/uninstall_library")
@timeout(30.0)
async def uninstall_library(request: InstallLibraryRequest):
    """
    Endpoint to uninstall a library.
    """
    uninstall_library_(request.library)
    return {"message": "Library uninstalled successfully"}





class AddToolRequest(BaseModel):
    function: str

@app.post(f"{prefix}/add_tool")
@timeout(30.0)
async def add_tool(request: AddToolRequest):
    """
    Endpoint to add a tool.
    """
    # Cloudpickle the function
    decoded_function = base64.b64decode(request.function)
    deserialized_function = cloudpickle.loads(decoded_function)



    add_tool_(deserialized_function)
    return {"message": "Tool added successfully"}



class AddMCPToolRequest(BaseModel):
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]


async def add_mcp_tool_(name: str, command: str, args: List[str], env: Dict[str, str]):
    """
    Add a tool.
    """
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

    async with managed_session(command=command, args=args, env=env) as session:
        tools = await session.list_tools()


        tools = tools.tools
        for tool in tools:

            tool_name: str = tool.name
            tool_desc: str = tool.description
            input_schema: Dict[str, Any] = tool.inputSchema
            properties: Dict[str, Dict[str, Any]] = input_schema.get("properties", {})
            required: List[str] = input_schema.get("required", [])

            def create_tool_function(
                tool_name: str,
                properties: Dict[str, Dict[str, Any]],
                required: List[str],
            ) -> Callable[..., Dict[str, Any]]:
                # Create function parameters type annotations
                annotations = {}
                defaults = {}

                # First add required parameters
                for param_name in required:
                    param_info = properties[param_name]
                    param_type = get_python_type(param_info.get("type", "any"))
                    annotations[param_name] = param_type

                # Then add optional parameters
                for param_name, param_info in properties.items():
                    if param_name not in required:
                        param_type = get_python_type(param_info.get("type", "any"))
                        annotations[param_name] = param_type
                        defaults[param_name] = param_info.get("default", None)

                # Create the signature parameters
                from inspect import Parameter, Signature
                
                parameters = []
                # Add required parameters first
                for param_name in required:
                    param_type = annotations[param_name]
                    parameters.append(
                        Parameter(
                            name=param_name,
                            kind=Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=param_type
                        )
                    )
                
                # Add optional parameters
                for param_name, param_type in annotations.items():
                    if param_name not in required:
                        parameters.append(
                            Parameter(
                                name=param_name,
                                kind=Parameter.POSITIONAL_OR_KEYWORD,
                                annotation=param_type,
                                default=defaults[param_name]
                            )
                        )

                async def tool_function(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                    # Convert positional args to kwargs
                    if len(args) > len(required):
                        raise TypeError(
                            f"{tool_name}() takes {len(required)} positional arguments but {len(args)} were given"
                        )

                    # Combine positional args with kwargs
                    all_kwargs = kwargs.copy()
                    for i, arg in enumerate(args):
                        if i < len(required):
                            all_kwargs[required[i]] = arg

                    # Validate required parameters
                    for req in required:
                        if req not in all_kwargs:
                            raise ValueError(f"Missing required parameter: {req}")

                    # Add defaults for optional parameters
                    for param, default in defaults.items():
                        if param not in all_kwargs:
                            all_kwargs[param] = default

                    async with managed_session(command=tool_function.command, args=tool_function.args, env=tool_function.env) as new_session:
                        # Remove None kwargs
                        all_kwargs = {k: v for k, v in all_kwargs.items() if v is not None}
                        result = await new_session.call_tool(name=tool_name, arguments=all_kwargs)
                        return {"result": result}

                # Set function name and annotations
                tool_function.__name__ = tool_name
                tool_function.__annotations__ = {
                    **annotations,
                    "return": Dict[str, Any],
                }
                tool_function.__doc__ = f"{tool_desc}\n\nReturns:\n    Tool execution results"

                # Create and set the signature
                tool_function.__signature__ = Signature(
                    parameters=parameters,
                    return_annotation=Dict[str, Any]
                )

                # Store session parameters as attributes of the function
                tool_function.command = command
                tool_function.args = args
                tool_function.env = env

                return tool_function

            # Create function with proper annotations
            func = create_tool_function(tool_name, properties, required)
            #name should be name__function_name
            full_name = f"{name}__{tool_name}"
            func.__name__ = full_name



            add_tool_(func, description=tool_desc, properties=properties, required=required)


@app.post(f"{prefix}/add_mcp_tool")
@timeout(60.0)
async def add_mcp_tool(request: AddMCPToolRequest):
    """
    Endpoint to add a tool.
    """
    await add_mcp_tool_(request.name, request.command, request.args, request.env)
    return {"message": "Tool added successfully"}