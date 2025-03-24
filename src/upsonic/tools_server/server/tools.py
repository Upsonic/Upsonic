import base64
import inspect
import subprocess
import traceback
import asyncio
import logging
import os
import shutil
from typing import List, Dict, Any, Optional, Union, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pydantic import BaseModel

from fastapi import HTTPException
from pydantic import BaseModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.stdio import get_default_environment
from mcp.client.sse import sse_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import the shared server instances dictionary
from .server_utils import _server_instances

class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, command: str, args: list, env: dict | None = None, name: str = "default") -> None:
        """Initialize a server with connection parameters.
        
        Args:
            command: The command to execute.
            args: Arguments for the command.
            env: Environment variables for the command.
            name: A name for this server instance.
        """
        self.name: str = name
        self.command: str = command
        self.args: list = args
        
        if env is None:
            self.env = get_default_environment()
        else:
            default_env = get_default_environment()
            default_env.update(env)
            self.env = default_env
            
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        if self.command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )
        
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
            logging.info(f"Server {self.name} initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
    
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries + 1:  # +1 because first attempt is not a retry
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                attempt += 1
                if attempt <= retries:
                    logging.warning(
                        f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                    )
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def list_tools(self) -> Any:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        return await self.session.list_tools()

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                
                # Remove this server from the global instances dictionary
                for key, srv in list(_server_instances.items()):
                    if srv is self:
                        del _server_instances[key]
                        logging.info(f"Removed server {self.name} from instances registry")
                        break
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")

class SSEServer:
    """Manages SSE-based MCP server connections and tool execution."""

    def __init__(self, url: str, name: str = "default") -> None:
        """Initialize an SSE server with connection parameters.
        
        Args:
            url: The SSE server URL.
            name: A name for this server instance.
        """
        self.name: str = name
        self.url: str = url
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the SSE server connection."""
        try:
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(self.url)
            )
            read, write = sse_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
            logging.info(f"SSE Server {self.name} initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing SSE server {self.name}: {e}")
            await self.cleanup()
            raise

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"SSE Server {self.name} not initialized")

        attempt = 0
        while attempt < retries + 1:  # +1 because first attempt is not a retry
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                attempt += 1
                if attempt <= retries:
                    logging.warning(
                        f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                    )
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def list_tools(self) -> Any:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"SSE Server {self.name} not initialized")

        return await self.session.list_tools()

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                
                # Remove this server from the global instances dictionary
                for key, srv in list(_server_instances.items()):
                    if srv is self:
                        del _server_instances[key]
                        logging.info(f"Removed SSE server {self.name} from instances registry")
                        break
            except Exception as e:
                logging.error(f"Error during cleanup of SSE server {self.name}: {e}")

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


class AddSSEMCPToolRequest(BaseModel):
    name: str
    url: str


async def add_mcp_tool_(name: str, command: str, args: List[str], env: Dict[str, str]):
    """
    Add a tool from an MCP server.
    
    Args:
        name: Name prefix for the tools.
        command: Command to execute.
        args: Arguments for the command.
        env: Environment variables for the command.
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

    # Create a hashable key for the server instance
    env_items = frozenset(env.items()) if env else frozenset()
    server_key = (name, command, tuple(args), env_items)
    
    # Check if we already have a server instance with this configuration
    if server_key in _server_instances:
        server = _server_instances[server_key]
        logging.info(f"Reusing existing server instance for {name}")
    else:
        # Create a new server instance
        server = Server(command=command, args=args, env=env, name=name)
        _server_instances[server_key] = server
        if server.session is None:
                    await server.initialize()
        logging.info(f"Created new server instance for {name}")
    
    try:
        # Only initialize if the session is not already initialized
        
        tools_response = await server.list_tools()
        
        tools = tools_response.tools
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

                    # Get the server that was created at the higher level
                    env_items = frozenset(tool_function.env.items()) if tool_function.env else frozenset()

                    
                    try:
                            
                        # Remove None kwargs
                        all_kwargs = {k: v for k, v in all_kwargs.items() if v is not None}
                        result = await server.execute_tool(tool_name=tool_name, arguments=all_kwargs)
                        return {"result": result}
                    except Exception as e:
                        # Log the error but don't clean up the server as it's managed at the higher level
                        logging.error(f"Error executing tool {tool_name}: {str(e)}")
                        raise

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
            # name should be name__function_name
            full_name = f"{name}__{tool_name}"
            func.__name__ = full_name

            add_tool_(func, description=tool_desc, properties=properties, required=required)
    except Exception as e:
        # Only clean up the server if there was an error
        logging.error(f"Error in add_mcp_tool_: {e}")
        await server.cleanup()
        raise
    # We don't clean up the server here to keep it alive for future use


@app.post(f"{prefix}/add_mcp_tool")
@timeout(60.0)
async def add_mcp_tool(request: AddMCPToolRequest):
    """
    Endpoint to add a tool.
    """
    await add_mcp_tool_(request.name, request.command, request.args, request.env)
    return {"message": "Tool added successfully"}



@app.post(f"{prefix}/add_sse_mcp")
@timeout(60.0)
async def add_sse_mcp(request: AddSSEMCPToolRequest):
    """
    Endpoint to add a tool.
    """
    await add_sse_mcp_(request.name, request.url)
    return {"message": "Tool added successfully"}

async def add_sse_mcp_(name: str, url: str):
    """
    Add a tool from an SSE MCP server.
    
    Args:
        name: Name prefix for the tools.
        url: The SSE server URL.
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

    # Create a hashable key for the server instance
    server_key = (name, url)
    
    # Check if we already have a server instance with this configuration
    if server_key in _server_instances:
        server = _server_instances[server_key]
        logging.info(f"Reusing existing SSE server instance for {name}")
    else:
        # Create a new server instance
        server = SSEServer(url=url, name=name)
        _server_instances[server_key] = server
        if server.session is None:
            await server.initialize()
        logging.info(f"Created new SSE server instance for {name}")
    
    try:
        tools_response = await server.list_tools()
        
        tools = tools_response.tools
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

                    try:
                        # Remove None kwargs
                        all_kwargs = {k: v for k, v in all_kwargs.items() if v is not None}
                        result = await server.execute_tool(tool_name=tool_name, arguments=all_kwargs)
                        return {"result": result}
                    except Exception as e:
                        # Log the error but don't clean up the server as it's managed at the higher level
                        logging.error(f"Error executing tool {tool_name}: {str(e)}")
                        raise

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

                # Store server URL as an attribute of the function
                tool_function.url = url

                return tool_function

            # Create function with proper annotations
            func = create_tool_function(tool_name, properties, required)
            # name should be name__function_name
            full_name = f"{name}__{tool_name}"
            func.__name__ = full_name

            add_tool_(func, description=tool_desc, properties=properties, required=required)
    except Exception as e:
        # Only clean up the server if there was an error
        logging.error(f"Error in add_sse_mcp_: {e}")
        await server.cleanup()
        raise
    # We don't clean up the server here to keep it alive for future use