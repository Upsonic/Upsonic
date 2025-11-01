"""Tool processor for handling, validating, and wrapping tools."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
import re
import time
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union, TYPE_CHECKING
)

from upsonic.tools.base import (
    Tool, ToolBase, ToolCall, ToolDefinition, ToolKit, ToolResult
)
from upsonic.tools.config import ToolConfig
from upsonic.tools.context import ToolContext
from upsonic.tools.schema import (
    FunctionSchema, generate_function_schema, validate_tool_function
)
from upsonic.tools.wrappers import FunctionTool

if TYPE_CHECKING:
    from upsonic.tools.mcp import MCPTool
    from upsonic.tasks.tasks import Task


class ToolValidationError(Exception):
    """Exception raised for invalid tool definitions."""
    pass


class ExternalExecutionPause(Exception):
    """Exception for pausing agent execution for external tool execution."""
    def __init__(self):
        super().__init__(f"Agent paused for external execution of a tool.")


class ToolProcessor:
    """Main engine for processing, validating, and managing tools."""
    
    def __init__(self):
        self.registered_tools: Dict[str, Tool] = {}
        self.tool_definitions: Dict[str, ToolDefinition] = {}
        self.mcp_handlers: List[Any] = []
        self.current_context: Optional[ToolContext] = None
    
    def process_tools(
        self,
        tools: List[Any],
        context: Optional[ToolContext] = None
    ) -> Dict[str, Tool]:
        """Process a list of raw tools and return registered Tool instances."""
        processed_tools = {}
        
        for tool_item in tools:
            if tool_item is None:
                continue
                
            if self._is_builtin_tool(tool_item):
                continue
            # Process based on tool type
            if self._is_mcp_tool(tool_item):
                # Process MCP tool
                mcp_tools = self._process_mcp_tool(tool_item)
                for name, tool in mcp_tools.items():
                    processed_tools[name] = tool
                    
            elif inspect.isfunction(tool_item):
                # Process function tool
                tool = self._process_function_tool(tool_item)
                processed_tools[tool.name] = tool
                
            elif inspect.isclass(tool_item):
                # Check if it's a ToolKit
                if issubclass(tool_item, ToolKit):
                    # Process ToolKit instance
                    toolkit_tools = self._process_toolkit(tool_item())
                    processed_tools.update(toolkit_tools)
                else:
                    # Process regular class with methods
                    class_tools = self._process_class_tools(tool_item())
                    processed_tools.update(class_tools)
                    
            elif hasattr(tool_item, '__class__'):
                # Process instance
                if isinstance(tool_item, ToolKit):
                    # Process ToolKit instance
                    toolkit_tools = self._process_toolkit(tool_item)
                    processed_tools.update(toolkit_tools)
                elif self._is_agent_instance(tool_item):
                    # Process agent as tool
                    agent_tool = self._process_agent_tool(tool_item)
                    processed_tools[agent_tool.name] = agent_tool
                else:
                    # Process regular instance with methods
                    instance_tools = self._process_class_tools(tool_item)
                    processed_tools.update(instance_tools)
        
        # Register all processed tools
        self.registered_tools.update(processed_tools)
        
        return processed_tools
    
    def _is_mcp_tool(self, tool_item: Any) -> bool:
        """Check if an item is an MCP tool configuration."""
        if not inspect.isclass(tool_item):
            return False
        return hasattr(tool_item, 'url') or hasattr(tool_item, 'command')
    
    def _is_builtin_tool(self, tool_item: Any) -> bool:
        """Check if an item is a built-in tool."""
        from upsonic.tools.builtin_tools import AbstractBuiltinTool
        return isinstance(tool_item, AbstractBuiltinTool)
    
    def extract_builtin_tools(self, tools: List[Any]) -> List[Any]:
        """Extract built-in tools from a list of tools."""
        builtin_tools = []
        for tool_item in tools:
            if tool_item is not None and self._is_builtin_tool(tool_item):
                builtin_tools.append(tool_item)
        return builtin_tools
    
    def _process_mcp_tool(self, mcp_config: Type) -> Dict[str, Tool]:
        """Process MCP tool configuration."""
        from upsonic.tools.mcp import MCPHandler
        
        handler = MCPHandler(mcp_config)
        self.mcp_handlers.append(handler)
        
        # Get tools from MCP server
        mcp_tools = handler.get_tools()
        return {tool.name: tool for tool in mcp_tools}
    
    def _process_function_tool(self, func: Callable) -> Tool:
        """Process a function into a Tool."""
        # Validate function
        errors = validate_tool_function(func)
        if errors:
            raise ToolValidationError(
                f"Invalid tool function '{func.__name__}': " + "; ".join(errors)
            )
        
        # Get tool config
        config = getattr(func, '_upsonic_tool_config', ToolConfig())
        
        # Generate schema
        schema = generate_function_schema(
            func,
            docstring_format=config.docstring_format,
            require_parameter_descriptions=config.require_parameter_descriptions
        )
        
        # Create wrapped tool
        return FunctionTool(
            function=func,
            schema=schema,
            config=config
        )
    
    def _process_toolkit(self, toolkit: ToolKit) -> Dict[str, Tool]:
        """Process a ToolKit instance."""
        tools = {}
        
        for name, method in inspect.getmembers(toolkit, inspect.ismethod):
            # Only process methods marked with @tool
            if hasattr(method, '_upsonic_is_tool'):
                tool = self._process_function_tool(method)
                tools[tool.name] = tool
        
        return tools
    
    def _process_class_tools(self, instance: Any) -> Dict[str, Tool]:
        """Process all public methods of a class instance as tools."""
        tools = {}
        
        for name, method in inspect.getmembers(instance, inspect.ismethod):
            # Skip private methods
            if name.startswith('_'):
                continue
                
            # Process as tool
            try:
                tool = self._process_function_tool(method)
                tools[tool.name] = tool
            except ToolValidationError:
                # Skip invalid methods
                continue
        
        return tools
    
    def _is_agent_instance(self, obj: Any) -> bool:
        """Check if an object is an agent instance."""
        # Check for agent-like attributes
        return hasattr(obj, 'name') and (
            hasattr(obj, 'do_async') or 
            hasattr(obj, 'do') or
            hasattr(obj, 'agent_id')
        )
    
    def _process_agent_tool(self, agent: Any) -> Tool:
        """Process an agent instance as a tool."""
        from upsonic.tools.wrappers import AgentTool
        
        return AgentTool(agent)
    
    def create_behavioral_wrapper(
        self,
        tool: Tool,
        context: ToolContext
    ) -> Callable:
        """Create a wrapper function with behavioral logic for a tool."""
        # Track if this tool requires sequential execution
        config = getattr(tool, 'config', ToolConfig())
        is_sequential = config.sequential
        
        @functools.wraps(tool.execute)
        async def wrapper(**kwargs: Any) -> Any:
            from upsonic.utils.printing import console, spacing
            
            # Get tool config (re-fetch to ensure latest)
            config = getattr(tool, 'config', ToolConfig())

            func_dict: Dict[str, Any] = {}
            # Before hook
            if config.tool_hooks and config.tool_hooks.before:
                try:
                    result = config.tool_hooks.before(**kwargs)
                    if result is not None:
                        func_dict["func_before"] = result
                except Exception as e:
                    console.print(f"[red]Before hook error: {e}[/red]")
                    raise
            
            # User confirmation
            if config.requires_confirmation:
                if not self._get_user_confirmation(tool.name, kwargs):
                    return "Tool execution cancelled by user"
            
            # User input
            if config.requires_user_input and config.user_input_fields:
                kwargs = self._get_user_input(
                    tool.name, 
                    kwargs, 
                    config.user_input_fields
                )
            
            # External execution
            if config.external_execution:
                # Don't create ToolCall here - ToolManager will create ExternalToolCall with ID
                raise ExternalExecutionPause()
            
            # Caching
            cache_key = None
            if config.cache_results:
                cache_key = self._get_cache_key(tool.name, kwargs)
                cached = self._get_cached_result(cache_key, config)
                if cached is not None:
                    console.print(f"[green]✓ Cache hit for {tool.name}[/green]")
                    func_dict["func_cache"] = cached
                    return func_dict
            
            # Execute tool with retry logic
            start_time = time.time()
            
            max_retries = config.max_retries
            last_error = None
            result = None
            
            for attempt in range(max_retries + 1):
                try:
                    # Apply timeout if configured
                    if config.timeout:
                        result = await asyncio.wait_for(
                            tool.execute(**kwargs),
                            timeout=config.timeout
                        )
                    else:
                        result = await tool.execute(**kwargs)
                    
                    # Success - break out of retry loop
                    break
                    
                except asyncio.TimeoutError as e:
                    last_error = e
                    if attempt < max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        console.print(f"[yellow]Tool '{tool.name}' timed out, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries + 1})[/yellow]")
                        await asyncio.sleep(wait_time)
                    else:
                        raise TimeoutError(f"Tool '{tool.name}' timed out after {config.timeout}s and {max_retries} retries")
                        
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        console.print(f"[yellow]Tool '{tool.name}' failed, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries + 1})[/yellow]")
                        await asyncio.sleep(wait_time)
                    else:
                        console.print(f"[bold red]Tool error after {max_retries} retries: {e}[/bold red]")
                        raise
            
            execution_time = time.time() - start_time
            
            # Cache result
            if config.cache_results and cache_key:
                self._cache_result(cache_key, result, config)
            
            # Show result if configured
            if config.show_result:
                console.print(f"[bold green]Tool Result:[/bold green] {result}")
                spacing()
            
            # After hook
            if config.tool_hooks and config.tool_hooks.after:
                try:
                    hook_result = config.tool_hooks.after(result)
                    if hook_result is not None:
                        func_dict["func_after"] = hook_result
                except Exception as e:
                    console.print(f"[bold red]After hook error: {e}[/bold red]")
            
            func_dict["func"] = result
            
            # Stop after call if configured
            if config.stop_after_tool_call:
                console.print("[bold yellow]Stopping after tool call[/bold yellow]")
                func_dict["_stop_execution"] = True
            
            return func_dict
        
        return wrapper
    
    def _get_user_confirmation(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Get user confirmation for tool execution."""
        from upsonic.utils.printing import console
        console.print(f"[bold yellow]⚠️ Confirmation Required[/bold yellow]")
        console.print(f"Tool: [cyan]{tool_name}[/cyan]")
        console.print(f"Arguments: {args}")
        
        try:
            response = input("Proceed? (y/n): ").lower().strip()
            return response in ('y', 'yes')
        except KeyboardInterrupt:
            return False
    
    def _get_user_input(
        self,
        tool_name: str,
        args: Dict[str, Any],
        fields: List[str]
    ) -> Dict[str, Any]:
        """Get user input for specified fields."""
        from upsonic.utils.printing import console
        console.print(f"[bold blue]📝 Input Required for {tool_name}[/bold blue]")
        
        for field in fields:
            try:
                value = input(f"Enter value for '{field}': ")
                args[field] = value
            except KeyboardInterrupt:
                console.print("[bold red]Input cancelled[/bold red]")
                break
        
        return args
    
    def _get_cache_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Generate cache key for tool call."""
        key_data = json.dumps(
            {"tool": tool_name, "args": args},
            sort_keys=True,
            default=str
        )
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str, config: ToolConfig) -> Any:
        """Get cached result if available and valid."""
        cache_dir = Path(config.cache_dir or Path.home() / '.upsonic' / 'cache')
        cache_file = cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Check TTL
            if config.cache_ttl:
                age = time.time() - data.get('timestamp', 0)
                if age > config.cache_ttl:
                    cache_file.unlink()
                    return None
            
            return data.get('result')
            
        except Exception:
            return None
    
    def _cache_result(self, cache_key: str, result: Any, config: ToolConfig) -> None:
        """Cache tool result."""
        cache_dir = Path(config.cache_dir or Path.home() / '.upsonic' / 'cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / f"{cache_key}.json"
        
        try:
            data = {
                'timestamp': time.time(),
                'result': result
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            from upsonic.utils.printing import warning_log
            warning_log(f"Could not cache result: {e}", "ToolProcessor")