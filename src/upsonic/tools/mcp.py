"""MCP (Model Context Protocol) tool handling."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Type, Union
from urllib.parse import urlparse

from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from upsonic.tools.base import ToolBase, ToolSchema, ToolMetadata


class MCPTool(ToolBase):
    """Wrapper for MCP tools."""
    
    def __init__(
        self,
        handler: 'MCPHandler',
        tool_info: mcp_types.Tool
    ):
        self.handler = handler
        self.tool_info = tool_info
        
        # Extract schema from tool info
        input_schema = tool_info.inputSchema if tool_info.inputSchema else {}
        
        # Convert MCP schema to our schema format
        tool_schema = ToolSchema(
            parameters=input_schema,
            strict=True  # MCP tools are typically strict
        )
        
        # Create metadata
        metadata = ToolMetadata(
            name=tool_info.name,
            description=tool_info.description,
            custom={
                'mcp_server': handler.server_name,
                'mcp_type': handler.connection_type
            }
        )
        
        super().__init__(
            name=tool_info.name,
            description=tool_info.description,
            schema=tool_schema,
            metadata=metadata
        )
    
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the MCP tool."""
        # Convert arguments to MCP format
        arguments = kwargs
        
        # Call tool through MCP handler
        result = await self.handler.call_tool(self.name, arguments)
        
        return result


class MCPHandler:
    """Handler for MCP server connections and tool management."""
    
    def __init__(self, config: Type):
        """
        Initialize MCP handler from configuration class.
        
        Args:
            config: Class with MCP configuration (url, command, args, env)
        """
        self.config = config
        self.session: Optional[ClientSession] = None
        self.tools: List[MCPTool] = []
        self._connection_context = None
        self._client_context = None
        
        # Determine connection type
        if hasattr(config, 'url'):
            self.connection_type = 'sse'
            self.server_name = self._extract_server_name(config.url)
        elif hasattr(config, 'command'):
            self.connection_type = 'stdio'
            self.server_name = getattr(config, '__name__', config.command)
        else:
            raise ValueError("MCP config must have either 'url' or 'command' attribute")
    
    def _extract_server_name(self, url: str) -> str:
        """Extract server name from URL."""
        parsed = urlparse(url)
        return parsed.hostname or 'mcp_server'
    
    def _create_session(self):
        """Create a new session for MCP communication."""
        if self.connection_type == 'sse':
            # SSE connection
            url = self.config.url
            return sse_client(url)
            
        elif self.connection_type == 'stdio':
            # Stdio connection
            command = self.config.command
            args = getattr(self.config, 'args', [])
            env = getattr(self.config, 'env', {})
            
            # Merge with current environment
            full_env = os.environ.copy()
            full_env.update(env)
            
            params = StdioServerParameters(
                command=command,
                args=args,
                env=full_env
            )
            
            return stdio_client(params)
        else:
            raise ValueError("Unknown connection type")
    
    async def _initialize_session(self) -> None:
        """Initialize the MCP session and discover tools."""
        from upsonic.utils.printing import console
        
        if not self.session:
            return
        
        # Initialize the session
        await self.session.initialize()
        
        # List available tools
        tools_response = await self.session.list_tools()
        
        console.print(f"[green]Found {len(tools_response.tools)} tools from {self.server_name}[/green]")
        
        # Create tool wrappers
        self.tools = []
        for tool_info in tools_response.tools:
            tool = MCPTool(self, tool_info)
            self.tools.append(tool)
            console.print(f"  - {tool.name}: {tool.description}")
    
    def get_tools(self) -> List[MCPTool]:
        """Get all available tools from this MCP server."""
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
                    return new_loop.run_until_complete(self._discover_tools_async())
                finally:
                    new_loop.close()
            
            # Run discovery in thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(discover_tools_in_thread)
                self.tools = future.result(timeout=10)  # 10 second timeout
                
            console.print(f"[green]✅ MCP tools discovered via thread[/green]")
            
        except RuntimeError:
            # No running loop, safe to create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                self.tools = loop.run_until_complete(self._discover_tools_async())
            finally:
                loop.close()
        except Exception as e:
            console.print(f"[red]❌ MCP tool discovery failed: {e}[/red]")
            return []
        
        return self.tools
    
    async def _discover_tools_async(self) -> List[MCPTool]:
        """Discover tools asynchronously."""
        from upsonic.utils.printing import console
        console.print(f"[cyan]Connecting to MCP server: {self.server_name}[/cyan]")
        
        try:
            client = self._create_session()
            
            async with client as client_context:
                if self.connection_type == 'stdio':
                    read_stream, write_stream = client_context
                    from mcp.client.session import ClientSession
                    session = ClientSession(read_stream, write_stream)
                else:
                    # For SSE, handle differently if needed
                    session = client_context
                
                async with session:
                    await session.initialize()
                    tools_response = await session.list_tools()
                
                console.print(f"[green]Found {len(tools_response.tools)} tools from {self.server_name}[/green]")
                
                # Create tool wrappers
                tools = []
                for tool_info in tools_response.tools:
                    tool = MCPTool(self, tool_info)
                    tools.append(tool)
                    console.print(f"  - {tool.name}: {tool.description}")
                
                return tools
        except Exception as e:
            console.print(f"[red]Failed to discover MCP tools: {e}[/red]")
            raise
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        from upsonic.utils.printing import console
        
        try:
            client = self._create_session()
            
            async with client as client_context:
                if self.connection_type == 'stdio':
                    read_stream, write_stream = client_context
                    from mcp.client.session import ClientSession
                    session = ClientSession(read_stream, write_stream)
                else:
                    # For SSE, handle differently if needed
                    session = client_context
                
                async with session:
                    await session.initialize()
                    
                    # Call the tool
                    result = await session.call_tool(tool_name, arguments)
                    
                    # Extract result content
                    if result.content:
                        # Handle different content types
                        if len(result.content) == 1:
                            content = result.content[0]
                            if isinstance(content, mcp_types.TextContent):
                                return content.text
                            elif isinstance(content, mcp_types.ImageContent):
                                return {
                                    'type': 'image',
                                    'data': content.data,
                                    'mime_type': content.mimeType
                                }
                            elif isinstance(content, mcp_types.EmbeddedResource):
                                return {
                                    'type': 'resource',
                                    'uri': content.resource.uri,
                                    'mime_type': content.resource.mimeType,
                                    'text': content.resource.text if hasattr(content.resource, 'text') else None
                                }
                        else:
                            # Multiple content items
                            return [self._convert_content(c) for c in result.content]
                    
                    return None
        except Exception as e:
            console.print(f"[red]Failed to call MCP tool '{tool_name}': {e}[/red]")
            raise
    
    def _convert_content(self, content: Any) -> Any:
        """Convert MCP content to standard format."""
        if isinstance(content, mcp_types.TextContent):
            return content.text
        elif isinstance(content, mcp_types.ImageContent):
            return {
                'type': 'image',
                'data': content.data,
                'mime_type': content.mimeType
            }
        elif isinstance(content, mcp_types.EmbeddedResource):
            return {
                'type': 'resource',
                'uri': content.resource.uri,
                'mime_type': content.resource.mimeType,
                'text': content.resource.text if hasattr(content.resource, 'text') else None
            }
        else:
            return str(content)
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server (no-op since we use on-demand connections)."""
        from upsonic.utils.printing import console
        # Since we use on-demand connections, there's nothing to disconnect
        console.print(f"[cyan]MCP handler for {self.server_name} uses on-demand connections[/cyan]")
