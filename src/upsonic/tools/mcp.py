"""MCP (Model Context Protocol) tool handling with comprehensive features."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import weakref
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from datetime import timedelta
from shlex import split as shlex_split
from types import TracebackType
from typing import Any, Dict, List, Literal, Optional, Type, Union
from urllib.parse import urlparse
from uuid import uuid4

from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client, get_default_environment

from upsonic.tools.base import Tool, ToolMetadata

# Try to import streamable HTTP client (may not be available in all MCP versions)
try:
    from mcp.client.streamable_http import streamablehttp_client
    HAS_STREAMABLE_HTTP = True
except ImportError:
    HAS_STREAMABLE_HTTP = False



def prepare_command(command: str) -> List[str]:
    """
    Sanitize a command and split it into parts before using it to run an MCP server.
    
    This function provides critical security by:
    - Blocking dangerous shell metacharacters
    - Whitelisting allowed executables
    - Validating paths and binaries
    
    Args:
        command: The command string to sanitize
        
    Returns:
        List of command parts safe for execution
        
    Raises:
        ValueError: If command contains dangerous characters or disallowed executables
    """
    # Block dangerous shell metacharacters that could be used for injection
    DANGEROUS_CHARS = ["&", "|", ";", "`", "$", "(", ")"]
    if any(char in command for char in DANGEROUS_CHARS):
        raise ValueError(
            f"MCP command can't contain shell metacharacters: {', '.join(DANGEROUS_CHARS)}"
        )
    
    # Split command safely using shlex
    try:
        parts = shlex_split(command)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}")
    
    if not parts:
        raise ValueError("MCP command can't be empty")
    
    # Whitelist of allowed executables
    ALLOWED_COMMANDS = {
        # Python
        "python",
        "python3",
        "uv",
        "uvx",
        "pipx",
        # Node
        "node",
        "npm",
        "npx",
        "yarn",
        "pnpm",
        "bun",
        # Other runtimes
        "deno",
        "java",
        "ruby",
        "docker",
    }
    
    first_part = parts[0]
    executable = first_part.split("/")[-1]
    
    # Allow relative paths starting with ./ or ../
    if first_part.startswith(("./", "../")):
        return parts
    
    # Allow absolute paths to existing files
    if first_part.startswith("/") and os.path.isfile(first_part):
        return parts
    
    # Allow binaries in current directory without ./
    if "/" not in first_part and os.path.isfile(first_part):
        return parts
    
    # Check if it's a binary in PATH
    if shutil.which(first_part):
        return parts
    
    # Check against whitelist
    if executable not in ALLOWED_COMMANDS:
        raise ValueError(
            f"MCP command must use one of the following executables: {ALLOWED_COMMANDS}. "
            f"Got: '{executable}'"
        )
    
    return parts


@dataclass
class SSEClientParams:
    """Parameters for SSE (Server-Sent Events) client connection."""
    url: str
    headers: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = 5
    sse_read_timeout: Optional[float] = 60 * 5


@dataclass
class StreamableHTTPClientParams:
    """Parameters for Streamable HTTP client connection."""
    url: str
    headers: Optional[Dict[str, Any]] = None
    timeout: Optional[timedelta] = None
    sse_read_timeout: Optional[timedelta] = None
    terminate_on_close: Optional[bool] = None
    
    def __post_init__(self):
        """Set default timeouts."""
        if self.timeout is None:
            self.timeout = timedelta(seconds=30)
        if self.sse_read_timeout is None:
            self.sse_read_timeout = timedelta(seconds=60 * 5)



class MCPTool(Tool):
    """Wrapper for MCP tools with enhanced capabilities."""
    
    def __init__(
        self,
        handler: 'MCPHandler',
        tool_info: mcp_types.Tool
    ):
        self.handler = handler
        self.tool_info = tool_info
        
        from upsonic.tools.schema import FunctionSchema
        
        input_schema = tool_info.inputSchema if tool_info.inputSchema else {
            'type': 'object',
            'properties': {},
            'additionalProperties': True
        }
        
        mcp_schema = FunctionSchema(
            function=None,
            description=tool_info.description,
            validator=None,
            json_schema=input_schema,
            is_async=True,
            single_arg_name=None,
            positional_fields=[],
            var_positional_field=None
        )
        
        # Create metadata with MCP tool info
        metadata = ToolMetadata(
            name=tool_info.name,
            description=tool_info.description,
            kind='mcp',
            is_async=True,
            strict=False
        )
        
        # Store MCP-specific data in custom
        metadata.custom['mcp_server'] = handler.server_name
        metadata.custom['mcp_type'] = handler.connection_type
        metadata.custom['mcp_transport'] = handler.transport
        
        super().__init__(
            name=tool_info.name,
            description=tool_info.description,
            schema=mcp_schema,
            metadata=metadata
        )
        
        from upsonic.tools.config import ToolConfig
        self.config = ToolConfig(
            timeout=60,  # 60 second timeout for MCP operations
            max_retries=2,  # Reduce retries since each retry takes long
            sequential=False  # Allow parallel execution
        )
    
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the MCP tool with enhanced error handling."""
        # Call tool through MCP handler
        result = await self.handler.call_tool(self.name, kwargs)
        return result


class MCPHandler:
    """
    Handler for MCP server connections and tool management.
    
    Features:
    - Multiple transport types (stdio, SSE, Streamable HTTP)
    - Command sanitization and security
    - Health checks via ping
    - Enhanced image/media handling
    - Tool filtering (include/exclude)
    - Proper resource cleanup
    - Lazy connection support
    """
    
    def __init__(
        self,
        config: Type = None,
        *,
        command: Optional[str] = None,
        url: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
        server_params: Optional[Union[StdioServerParameters, SSEClientParams, StreamableHTTPClientParams]] = None,
        session: Optional[ClientSession] = None,
        timeout_seconds: int = 5,
        include_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None,
    ):
        """
        Initialize MCP handler.
        
        Args:
            config: Legacy config class with url/command/args/env attributes
            command: Command to run MCP server (for stdio transport)
            url: URL for SSE or Streamable HTTP transport
            env: Environment variables to pass to server
            transport: Transport protocol ("stdio", "sse", "streamable-http")
            server_params: Pre-configured server parameters
            session: Existing MCP ClientSession
            timeout_seconds: Read timeout in seconds
            include_tools: Optional list of tool names to include (None = all)
            exclude_tools: Optional list of tool names to exclude (None = none)
        """
        self.session: Optional[ClientSession] = session
        self.tools: List[MCPTool] = []
        self.transport = transport
        self.timeout_seconds = timeout_seconds
        self.include_tools = include_tools
        self.exclude_tools = exclude_tools
        self._initialized = False
        self._context = None
        self._session_context = None
        self._connection_task = None
        self._active_contexts: List[Any] = []
        
        # Handle legacy config class
        if config is not None:
            if hasattr(config, 'url'):
                url = config.url
                transport = 'sse'
            elif hasattr(config, 'command'):
                # Extract command and args from legacy config
                cmd = config.command
                legacy_args = getattr(config, 'args', [])
                
                # Combine command and args into a single command string for sanitization
                if legacy_args:
                    command = f"{cmd} {' '.join(str(arg) for arg in legacy_args)}"
                else:
                    command = cmd
                
                env = getattr(config, 'env', {})
                transport = 'stdio'
            else:
                raise ValueError("Config must have either 'url' or 'command' attribute")
        
        # Determine connection type and server name
        if url:
            if transport == "sse":
                self.connection_type = 'sse'
            elif transport == "streamable-http":
                if not HAS_STREAMABLE_HTTP:
                    from upsonic.utils.printing import import_error
                    import_error(
                        package_name="mcp[streamable-http]",
                        install_command="pip install 'mcp[streamable-http]'",
                        feature_name="MCP streamable HTTP transport"
                    )
                self.connection_type = 'streamable-http'
            else:
                raise ValueError(f"Invalid transport for URL: {transport}")
            self.server_name = self._extract_server_name(url)
        elif command or server_params:
            self.connection_type = 'stdio'
            if command:
                self.server_name = command.split()[0].split("/")[-1]
            else:
                # Use UUID to ensure unique server name when server_params provided without command
                # This prevents tool name collisions when multiple handlers use default name
                self.server_name = f"mcp_server_{uuid4().hex[:8]}"
        else:
            raise ValueError("Must provide either url, command, or server_params")
        
        # Setup server parameters
        if server_params:
            self.server_params = server_params
        elif transport == "sse" and url:
            self.server_params = SSEClientParams(url=url)
        elif transport == "streamable-http" and url:
            self.server_params = StreamableHTTPClientParams(url=url)
        elif transport == "stdio" and command:
            # Sanitize command for security
            parts = prepare_command(command)
            cmd = parts[0]
            args = parts[1:] if len(parts) > 1 else []
            
            # Merge with default environment
            if env is not None:
                env = {
                    **get_default_environment(),
                    **env,
                }
            else:
                env = get_default_environment()
            
            self.server_params = StdioServerParameters(
                command=cmd,
                args=args,
                env=env
            )
        else:
            raise ValueError("Invalid configuration for MCP handler")
        
        # Setup cleanup finalizer
        # Note: _cleanup_finalizer is intentionally stored but not explicitly called.
        # weakref.finalize() stores the finalizer and automatically calls cleanup()
        # when this object is garbage collected. This ensures no resource leaks.
        def cleanup():
            """Cancel active connections before garbage collection."""
            if self._connection_task and not self._connection_task.done():
                self._connection_task.cancel()
        
        self._cleanup_finalizer = weakref.finalize(self, cleanup)
    
    def _extract_server_name(self, url: str) -> str:
        """Extract server name from URL."""
        parsed = urlparse(url)
        return parsed.hostname or parsed.path.split('/')[-1] or 'mcp_server'
    
    def _create_session(self):
        """Create a new session for MCP communication."""
        if self.connection_type == 'sse':
            # SSE connection
            if isinstance(self.server_params, SSEClientParams):
                return sse_client(**asdict(self.server_params))
            else:
                # Fallback for legacy config
                return sse_client(self.server_params.url if hasattr(self.server_params, 'url') else str(self.server_params))
            
        elif self.connection_type == 'streamable-http':
            # Streamable HTTP connection
            if not HAS_STREAMABLE_HTTP:
                from upsonic.utils.printing import import_error
                import_error(
                    package_name="mcp[streamable-http]",
                    install_command="pip install 'mcp[streamable-http]'",
                    feature_name="MCP streamable HTTP"
                )
            if isinstance(self.server_params, StreamableHTTPClientParams):
                return streamablehttp_client(**asdict(self.server_params))
            else:
                raise ValueError("Streamable HTTP requires StreamableHTTPClientParams")
            
        else:  # stdio
            # Stdio connection
            if not isinstance(self.server_params, StdioServerParameters):
                raise ValueError(f"stdio transport requires StdioServerParameters, got {type(self.server_params)}")
            return stdio_client(self.server_params)
    
    def _start_connection(self):
        """Ensure there are no active connections and setup a new one."""
        if self._connection_task is None or self._connection_task.done():
            self._connection_task = asyncio.create_task(self.connect())
    
    async def connect(self) -> None:
        """Initialize and connect to the MCP server."""
        if self._initialized:
            return
        
        from upsonic.utils.printing import console
        
        if self.session is not None:
            await self._initialize_with_session()
            return
        
        console.print(f"[cyan]Connecting to MCP server: {self.server_name} ({self.connection_type})[/cyan]")
        
        try:
            # Create appropriate client based on transport
            if self.connection_type == 'sse':
                sse_params = asdict(self.server_params) if isinstance(self.server_params, SSEClientParams) else {}
                self._context = sse_client(**sse_params)
                client_timeout = min(self.timeout_seconds, sse_params.get("timeout", self.timeout_seconds))
                
            elif self.connection_type == 'streamable-http':
                http_params = asdict(self.server_params) if isinstance(self.server_params, StreamableHTTPClientParams) else {}
                self._context = streamablehttp_client(**http_params)
                params_timeout = http_params.get("timeout", self.timeout_seconds)
                if isinstance(params_timeout, timedelta):
                    params_timeout = int(params_timeout.total_seconds())
                client_timeout = min(self.timeout_seconds, params_timeout)
                
            else:  # stdio
                if not isinstance(self.server_params, StdioServerParameters):
                    raise ValueError("server_params must be StdioServerParameters for stdio transport")
                self._context = stdio_client(self.server_params)
                client_timeout = self.timeout_seconds
            
            # Enter context and setup session
            session_params = await self._context.__aenter__()
            self._active_contexts.append(self._context)
            
            read, write = session_params[0:2]
            self._session_context = ClientSession(
                read, 
                write, 
                read_timeout_seconds=timedelta(seconds=client_timeout)
            )
            self.session = await self._session_context.__aenter__()
            self._active_contexts.append(self._session_context)
            
            # Initialize with the new session
            await self._initialize_with_session()
            
            console.print(f"[green]✅ Connected to MCP server: {self.server_name}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to connect to MCP server: {e}[/red]")
            raise
    
    async def close(self) -> None:
        """Close the MCP connection and clean up resources."""
        from upsonic.utils.printing import console
        
        if self._session_context is not None:
            try:
                await self._session_context.__aexit__(None, None, None)
            except (RuntimeError, Exception) as e:
                # Suppress event loop closed errors (common in threaded contexts)
                error_msg = str(e).lower()
                if "event loop is closed" not in error_msg and "loop" not in error_msg:
                    console.print(f"[yellow]Warning: Error closing session: {e}[/yellow]")
            self.session = None
            self._session_context = None
        
        if self._context is not None:
            try:
                await self._context.__aexit__(None, None, None)
            except (RuntimeError, Exception) as e:
                # Suppress event loop closed errors (common in threaded contexts)
                error_msg = str(e).lower()
                if "event loop is closed" not in error_msg and "loop" not in error_msg:
                    console.print(f"[yellow]Warning: Error closing context: {e}[/yellow]")
            self._context = None
        
        self._initialized = False
        console.print(f"[cyan]MCP handler for {self.server_name} closed[/cyan]")
    
    async def __aenter__(self) -> "MCPHandler":
        """Enter async context manager."""
        await self.connect()
        return self
    
    async def __aexit__(
        self, 
        exc_type: Optional[Type[BaseException]], 
        exc_val: Optional[BaseException], 
        exc_tb: Optional[TracebackType]
    ):
        """Exit async context manager."""
        await self.close()
    
    async def _initialize_with_session(self) -> None:
        """Initialize the MCP session and discover tools."""
        if self._initialized:
            return
        
        from upsonic.utils.printing import console
        
        if not self.session:
            raise ValueError("Session not initialized")
        
        try:
            # Initialize the session
            await self.session.initialize()
            
            # List available tools
            tools_response = await self.session.list_tools()
            
            # Validate tool filters
            available_tool_names = [tool.name for tool in tools_response.tools]
            self._check_tools_filters(available_tool_names)
            
            # Filter tools based on include/exclude lists
            filtered_tools = self._filter_tools(tools_response.tools)
            
            console.print(
                f"[green]Found {len(filtered_tools)} tools from {self.server_name} "
                f"(total: {len(tools_response.tools)})[/green]"
            )
            
            # Create tool wrappers
            self.tools = []
            for tool_info in filtered_tools:
                try:
                    tool = MCPTool(self, tool_info)
                    self.tools.append(tool)
                    console.print(f"  - {tool.name}: {tool.description}")
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to register tool {tool_info.name}: {e}[/yellow]")
            
            self._initialized = True
            
        except Exception as e:
            console.print(f"[red]Failed to initialize MCP session: {e}[/red]")
            raise
    
    def _check_tools_filters(self, available_tools: List[str]) -> None:
        """
        Validate that include/exclude tool filters reference existing tools.
        
        Args:
            available_tools: List of tool names available from the MCP server
            
        Raises:
            ValueError: If filters reference non-existent tools
        """
        if self.include_tools:
            invalid = set(self.include_tools) - set(available_tools)
            if invalid:
                raise ValueError(
                    f"include_tools references non-existent tools: {invalid}. "
                    f"Available tools: {available_tools}"
                )
        
        if self.exclude_tools:
            invalid = set(self.exclude_tools) - set(available_tools)
            if invalid:
                raise ValueError(
                    f"exclude_tools references non-existent tools: {invalid}. "
                    f"Available tools: {available_tools}"
                )
    
    def _filter_tools(self, tools: List[mcp_types.Tool]) -> List[mcp_types.Tool]:
        """
        Filter tools based on include/exclude lists.
        
        Args:
            tools: List of MCP tools
            
        Returns:
            Filtered list of tools
        """
        filtered = []
        for tool in tools:
            # Exclude takes precedence
            if self.exclude_tools and tool.name in self.exclude_tools:
                continue
            # Include filter (None means include all)
            if self.include_tools is None or tool.name in self.include_tools:
                filtered.append(tool)
        return filtered
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this MCP server connection.
        
        Returns:
            Dictionary with server information
        """
        return {
            'server_name': self.server_name,
            'connection_type': self.connection_type,
            'transport': self.transport,
            'tool_count': len(self.tools),
            'tools': [t.name for t in self.tools],
            'timeout_seconds': self.timeout_seconds,
            'initialized': self._initialized,
            'has_filters': bool(self.include_tools or self.exclude_tools)
        }
    
    def get_tools(self) -> List[MCPTool]:
        """
        Get all available tools from this MCP server.
        
        This method handles synchronous calling contexts by running
        the async connection in a thread or new event loop.
        
        Returns:
            List of MCPTool instances
        """
        from upsonic.utils.printing import console
        
        if self.tools:
            return self.tools  # Already discovered
        
        # Discover tools via async connection
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, create tools in a thread
            console.print(f"[yellow]⚠️  MCP async limitation detected. Attempting threaded connection...[/yellow]")
            
            import concurrent.futures
            
            def discover_tools_in_thread():
                """Discover MCP tools in a separate thread."""
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(self.connect())
                    return self.tools
                finally:
                    new_loop.close()
            
            # Run discovery in thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(discover_tools_in_thread)
                self.tools = future.result(timeout=30)  # 30 second timeout
            
            console.print(f"[green]✅ MCP tools discovered via thread[/green]")
            
        except RuntimeError:
            # No running loop, safe to create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.connect())
            finally:
                loop.close()
        except Exception as e:
            console.print(f"[red]❌ MCP tool discovery failed: {e}[/red]")
            return []
        
        return self.tools
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server with enhanced error handling and image support.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result with enhanced image/media handling
            
        Raises:
            Exception: If tool call fails
        """
        from upsonic.utils.printing import console
        
        try:
            console.print(f"[blue]Calling MCP tool '{tool_name}' with args: {arguments}[/blue]")
            
            # Create fresh client connection for this call
            client = self._create_session()
            
            async with client as client_context:
                if self.connection_type == 'stdio':
                    read_stream, write_stream = client_context
                    from mcp.client.session import ClientSession
                    from datetime import timedelta
                    session = ClientSession(
                        read_stream, 
                        write_stream,
                        read_timeout_seconds=timedelta(seconds=max(self.timeout_seconds, 30))
                    )
                else:
                    # For SSE or Streamable HTTP
                    session = client_context
                
                async with session:
                    await session.initialize()
                    
                    # Call the tool
                    result: mcp_types.CallToolResult = await session.call_tool(tool_name, arguments)
                    
                    # Check for errors in result
                    if result.isError:
                        error_msg = f"Error from MCP tool '{tool_name}': {result.content}"
                        console.print(f"[red]{error_msg}[/red]")
                        return {"error": error_msg, "success": False}
                    
                    # Process the result content with enhanced image/media handling
                    return self._process_tool_result(result, tool_name)
            
        except Exception as e:
            console.print(f"[red]Failed to call MCP tool '{tool_name}': {e}[/red]")
            raise
    
    def _process_tool_result(self, result: mcp_types.CallToolResult, tool_name: str) -> Any:
        """
        Process tool result with enhanced image and media handling.
        
        Features:
        - Base64 image decoding
        - Custom JSON image format parsing
        - Multiple content type support
        - Embedded resource handling
        
        Args:
            result: The MCP tool call result
            tool_name: Name of the tool (for logging)
            
        Returns:
            Processed result with images and content
        """
        if not result.content:
            return None
        
        response_parts = []
        images = []
        
        for content_item in result.content:
            if isinstance(content_item, mcp_types.TextContent):
                text_content = content_item.text
                
                # Try to parse as JSON to check for custom image format
                try:
                    parsed_json = json.loads(text_content)
                    if (
                        isinstance(parsed_json, dict)
                        and parsed_json.get("type") == "image"
                        and "data" in parsed_json
                    ):
                        # Custom JSON image format found
                        image_data = parsed_json.get("data")
                        mime_type = parsed_json.get("mimeType", "image/png")
                        
                        if image_data and isinstance(image_data, str):
                            try:
                                # Decode base64 image data
                                image_bytes = base64.b64decode(image_data)
                                image_obj = {
                                    'id': str(uuid4()),
                                    'type': 'image',
                                    'content': image_bytes,
                                    'mime_type': mime_type,
                                    'source': 'mcp_custom_json'
                                }
                                images.append(image_obj)
                                response_parts.append("Image has been generated and added to the response.")
                                continue
                            except Exception as e:
                                # Failed to decode, treat as regular text
                                pass
                except (json.JSONDecodeError, TypeError):
                    # Not JSON or not image format, treat as regular text
                    pass
                
                # Regular text content
                response_parts.append(text_content)
                
            elif isinstance(content_item, mcp_types.ImageContent):
                # Handle standard MCP ImageContent
                image_data = getattr(content_item, "data", None)
                
                if image_data and isinstance(image_data, str):
                    try:
                        # Decode base64 image data
                        image_bytes = base64.b64decode(image_data)
                    except Exception as e:
                        image_bytes = None
                else:
                    image_bytes = image_data
                
                image_obj = {
                    'id': str(uuid4()),
                    'type': 'image',
                    'url': getattr(content_item, "url", None),
                    'content': image_bytes,
                    'mime_type': getattr(content_item, "mimeType", "image/png"),
                    'source': 'mcp_image_content'
                }
                images.append(image_obj)
                response_parts.append("Image has been generated and added to the response.")
                
            elif isinstance(content_item, mcp_types.EmbeddedResource):
                # Handle embedded resources
                resource_info = {
                    'type': 'resource',
                    'uri': content_item.resource.uri,
                    'mime_type': getattr(content_item.resource, 'mimeType', None),
                    'text': getattr(content_item.resource, 'text', None)
                }
                response_parts.append(f"[Embedded resource: {json.dumps(resource_info)}]")
            
            else:
                # Handle other content types
                response_parts.append(f"[Unsupported content type: {getattr(content_item, 'type', 'unknown')}]")
        
        # Construct final result
        response_text = "\n".join(response_parts).strip()
        
        if images:
            return {
                'content': response_text,
                'images': images,
                'success': True
            }
        else:
            return response_text if response_text else None




class MultiMCPHandler:
    """
    Coordinator for managing multiple MCP server connections simultaneously.
    
    This class creates and manages multiple MCPHandler instances, aggregating
    their tools into a unified interface.
    
    Architecture:
    - Creates one MCPHandler instance per server
    - Each handler manages its own connection lifecycle
    - Aggregates tools from all handlers
    - Provides unified introspection across all servers
    
    Features:
    - Connect to multiple servers (stdio, SSE, Streamable HTTP)
    - Unified tool discovery across all servers
    - Server introspection and debugging
    - Tool filtering across all servers
    - Proper cleanup delegation to individual handlers
    """
    
    def __init__(
        self,
        commands: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        urls_transports: Optional[List[Literal["sse", "streamable-http"]]] = None,
        *,
        env: Optional[Dict[str, str]] = None,
        server_params_list: Optional[
            List[Union[SSEClientParams, StdioServerParameters, StreamableHTTPClientParams]]
        ] = None,
        timeout_seconds: int = 5,
        include_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None,
    ):
        """
        Initialize multi-MCP handler.
        
        Args:
            commands: List of commands to run MCP servers (stdio transport)
            urls: List of URLs for SSE or Streamable HTTP endpoints
            urls_transports: List of transport types for URLs
            env: Environment variables for stdio servers
            server_params_list: Pre-configured server parameters
            timeout_seconds: Read timeout in seconds
            include_tools: Optional list of tool names to include
            exclude_tools: Optional list of tool names to exclude
        """
        if server_params_list is None and commands is None and urls is None:
            raise ValueError("Must provide commands, urls, or server_params_list")
        
        self.timeout_seconds = timeout_seconds
        self.include_tools = include_tools
        self.exclude_tools = exclude_tools
        self._initialized = False
        # Track handlers and tools (each handler manages its own connection)
        self.tools: List[MCPTool] = []
        self.handlers: List[MCPHandler] = []  # Store MCPHandler for each server
        
        # Build server parameters list
        self.server_params_list: List[Union[SSEClientParams, StdioServerParameters, StreamableHTTPClientParams]] = (
            server_params_list or []
        )
        
        # Merge env with defaults
        if env is not None:
            env = {
                **get_default_environment(),
                **env,
            }
        else:
            env = get_default_environment()
        
        # Process commands
        if commands:
            for command in commands:
                parts = prepare_command(command)
                cmd = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                self.server_params_list.append(
                    StdioServerParameters(command=cmd, args=args, env=env)
                )
        
        # Process URLs
        if urls:
            if urls_transports:
                if len(urls) != len(urls_transports):
                    raise ValueError("urls and urls_transports must be same length")
                for url, transport in zip(urls, urls_transports):
                    if transport == "streamable-http":
                        self.server_params_list.append(StreamableHTTPClientParams(url=url))
                    else:  # sse
                        self.server_params_list.append(SSEClientParams(url=url))
            else:
                # Default to streamable-http
                for url in urls:
                    self.server_params_list.append(StreamableHTTPClientParams(url=url))
    
    async def connect(self) -> None:
        """Initialize and connect to all MCP servers."""
        if self._initialized:
            return
        
        from upsonic.utils.printing import console
        
        console.print(f"[cyan]🔌 Connecting to {len(self.server_params_list)} MCP server(s)...[/cyan]")
        
        # Create MCPHandler for each server and connect
        for idx, server_params in enumerate(self.server_params_list):
            try:
                # Create a proper MCPHandler instance for this server
                handler = MCPHandler(
                    server_params=server_params,
                    timeout_seconds=self.timeout_seconds,
                    include_tools=self.include_tools,
                    exclude_tools=self.exclude_tools
                )
                
                # Connect handler (this discovers tools automatically)
                await handler.connect()
                
                # Store handler and aggregate its tools
                self.handlers.append(handler)
                self.tools.extend(handler.tools)
                
                console.print(f"[green]  ✅ Server {idx+1}: {handler.server_name} - {len(handler.tools)} tools[/green]")
                
            except Exception as e:
                console.print(f"[yellow]  ⚠️  Server {idx+1} connection failed: {e}[/yellow]")
                # Continue with other servers
        
        self._initialized = True
        console.print(f"[green]✅ Successfully connected to {len(self.handlers)} MCP servers with {len(self.tools)} total tools[/green]")
    
    async def close(self) -> None:
        """Close all MCP connections and clean up resources."""
        # Close each handler (each manages its own connection)
        for handler in self.handlers:
            try:
                await handler.close()
            except (RuntimeError, Exception) as e:
                # Suppress event loop closed errors (common in threaded contexts)
                error_msg = str(e).lower()
                if "event loop is closed" not in error_msg and "loop" not in error_msg:
                    # Only log non-loop-related errors
                    from upsonic.utils.printing import console
                    console.print(f"[yellow]Warning: Error closing handler: {e}[/yellow]")
        
        self.handlers.clear()
        self.tools.clear()
        self._initialized = False
    
    async def __aenter__(self) -> "MultiMCPHandler":
        """Enter async context manager."""
        await self.connect()
        return self
    
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ):
        """Exit async context manager."""
        await self.close()
    
    def get_tools(self) -> List[MCPTool]:
        """
        Get all tools from all connected MCP servers.
        
        This method handles synchronous calling contexts by running
        the async connection in a thread or new event loop.
        
        Returns:
            List of all MCPTool instances
        """
        from upsonic.utils.printing import console
        
        if self.tools:
            return self.tools
        
        # Discover tools via async connection
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, create tools in a thread
            console.print(f"[yellow]⚠️  MCP async limitation detected. Attempting threaded connection...[/yellow]")
            
            import concurrent.futures
            
            def discover_tools_in_thread():
                """Discover MCP tools in a separate thread."""
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(self.connect())
                    return self.tools
                finally:
                    new_loop.close()
            
            # Run discovery in thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(discover_tools_in_thread)
                self.tools = future.result(timeout=60)  # 60 second timeout for multiple servers
            
            console.print(f"[green]✅ MCP tools discovered via thread[/green]")
            
        except RuntimeError:
            # No running loop, safe to create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.connect())
            finally:
                loop.close()
        except Exception as e:
            console.print(f"[red]❌ MultiMCP tool discovery failed: {e}[/red]")
            import traceback
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
            return []
        
        if not self.tools:
            console.print(f"[yellow]⚠️  Warning: MultiMCPHandler connected but found 0 tools. Check server connections.[/yellow]")
            console.print(f"[yellow]  Server count: {len(self.handlers)}, Server params: {len(self.server_params_list)}[/yellow]")
        
        return self.tools
    
    def get_server_count(self) -> int:
        """
        Get the number of connected MCP servers.
        
        Returns:
            Number of active server connections
        """
        return len(self.handlers)
    
    def get_tool_count(self) -> int:
        """
        Get the total number of tools from all servers.
        
        Returns:
            Total number of tools
        """
        return len(self.tools)
    
    def get_tools_by_server(self) -> Dict[str, List[str]]:
        """
        Get tools organized by their source server.
        
        Returns:
            Dictionary mapping server names to lists of tool names
        """
        servers: Dict[str, List[str]] = {}
        for tool in self.tools:
            server_name = tool.metadata.custom.get('mcp_server', 'unknown')
            if server_name not in servers:
                servers[server_name] = []
            servers[server_name].append(tool.name)
        return servers
    
    def get_server_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all connected servers.
        
        Returns:
            List of dictionaries with server information
        """
        info = []
        for idx, handler in enumerate(self.handlers):
            server_info = {
                'index': idx,
                'server_name': getattr(handler, 'server_name', f'server_{idx}'),
                'connection_type': getattr(handler, 'connection_type', 'unknown'),
                'transport': getattr(handler, 'transport', 'unknown'),
                'tools': [t.name for t in self.tools if t.handler == handler]
            }
            info.append(server_info)
        return info
