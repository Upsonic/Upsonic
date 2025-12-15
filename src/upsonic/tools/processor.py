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
    Tool, ToolDefinition, ToolKit, ToolResult
)
from upsonic.tools.config import ToolConfig
from upsonic.tools.metrics import ToolMetrics
from upsonic.tools.schema import (
    FunctionSchema,
    function_schema,
    SchemaGenerationError,
    GenerateToolJsonSchema,
)
from upsonic.tools.wrappers import FunctionTool, AgentTool
from upsonic.tools.deferred import DeferredExecutionManager, ExternalToolCall

if TYPE_CHECKING:
    from upsonic.tools.base import Tool, ToolConfig, ToolMetadata
    from upsonic.tools.orchestration import PlanStep


class ToolValidationError(Exception):
    """Error raised when tool validation fails."""
    pass


class ExternalExecutionPause(Exception):
    """Exception to pause execution when external tool execution is required."""
    
    def __init__(self, external_calls: List[ExternalToolCall] = None):
        self.external_calls = external_calls or []
        super().__init__(f"Paused for {len(self.external_calls)} external tool calls")


class ToolProcessor:
    """Processes and validates tools before registration."""
    
    def __init__(
        self,
    ):
        self.registered_tools: Dict[str, Tool] = {}
        self.mcp_handlers: List[Any] = []
        # Track which tools belong to which MCP handler
        self.mcp_handler_to_tools: Dict[int, List[str]] = {}  # handler id -> tool names
        # Track which tools belong to which class instance (ToolKit or regular class)
        self.class_instance_to_tools: Dict[int, List[str]] = {}  # class instance id -> tool names
        # Track KnowledgeBase instances that need setup_async() called
        self.knowledge_base_instances: Dict[int, Any] = {}  # instance id -> KnowledgeBase instance
    
    def process_tools(
        self,
        tools: List[Any]
    ) -> Dict[str, Tool]:
        """Process a list of raw tools and return registered Tool instances."""
        processed_tools = {}
        
        for tool_item in tools:
            if tool_item is None:
                continue
            
            # Optimization: If tool already inherits from Tool base class, skip processing
            if isinstance(tool_item, Tool):
                # Tool is already properly formed, register directly
                processed_tools[tool_item.name] = tool_item
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

            elif inspect.ismethod(tool_item):
                # Process bound method (e.g., from YFinanceTools.functions())
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
        # Check for MCPHandler or MultiMCPHandler instances
        from upsonic.tools.mcp import MCPHandler, MultiMCPHandler
        if isinstance(tool_item, (MCPHandler, MultiMCPHandler)):
            return True
        
        # Check for legacy config class
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
    
    def _process_mcp_tool(self, mcp_config: Any) -> Dict[str, Tool]:
        """
        Process MCP tool configuration.
        
        Supports:
        - Legacy config classes (with url/command attributes)
        - MCPHandler instances
        - MultiMCPHandler instances
        """
        from upsonic.tools.mcp import MCPHandler, MultiMCPHandler
        
        # If already a handler instance, use it directly
        if isinstance(mcp_config, (MCPHandler, MultiMCPHandler)):
            handler = mcp_config
        else:
            # Legacy config class - create handler
            handler = MCPHandler(config=mcp_config)
        
        self.mcp_handlers.append(handler)
        
        # Get tools from MCP server(s)
        mcp_tools = handler.get_tools()
        tools_dict = {tool.name: tool for tool in mcp_tools}
        
        # Track which tools belong to this handler
        handler_id = id(handler)
        if handler_id not in self.mcp_handler_to_tools:
            self.mcp_handler_to_tools[handler_id] = []
        self.mcp_handler_to_tools[handler_id].extend(tools_dict.keys())
        
        return tools_dict
    
    def _process_function_tool(self, func: Callable) -> Tool:
        """Process a function into a Tool."""
        # Get tool config
        config = getattr(func, '_upsonic_tool_config', ToolConfig())
        
        # Generate schema using new function
        try:
            schema = function_schema(
                func,
                schema_generator=GenerateToolJsonSchema,
                docstring_format=config.docstring_format,
                require_parameter_descriptions=config.require_parameter_descriptions
            )
        except SchemaGenerationError as e:
            raise ToolValidationError(
                f"Invalid tool function '{func.__name__}': {e}"
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
        
        # Check if this is a KnowledgeBase instance
        try:
            from upsonic.knowledge_base.knowledge_base import KnowledgeBase
            if isinstance(toolkit, KnowledgeBase):
                toolkit_id = id(toolkit)
                self.knowledge_base_instances[toolkit_id] = toolkit
        except ImportError:
            # KnowledgeBase might not be available, skip
            pass
        
        for name, method in inspect.getmembers(toolkit, inspect.ismethod):
            # Only process methods marked with @tool
            if hasattr(method, '_upsonic_is_tool'):
                tool = self._process_function_tool(method)
                tools[tool.name] = tool
        
        # Track which tools belong to this toolkit instance
        if tools:
            toolkit_id = id(toolkit)
            if toolkit_id not in self.class_instance_to_tools:
                self.class_instance_to_tools[toolkit_id] = []
            self.class_instance_to_tools[toolkit_id].extend(tools.keys())
        
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
        
        # Track which tools belong to this class instance
        if tools:
            instance_id = id(instance)
            if instance_id not in self.class_instance_to_tools:
                self.class_instance_to_tools[instance_id] = []
            self.class_instance_to_tools[instance_id].extend(tools.keys())
        
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
        tool: Tool
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

            # Ensure KnowledgeBase setup_async() is called if this tool belongs to a KnowledgeBase
            if isinstance(tool, FunctionTool) and hasattr(tool, 'function'):
                try:
                    from upsonic.knowledge_base.knowledge_base import KnowledgeBase
                    # Check if the function is a bound method of a KnowledgeBase instance
                    func = tool.function
                    if inspect.ismethod(func) and hasattr(func, '__self__'):
                        instance = func.__self__
                        if isinstance(instance, KnowledgeBase):
                            # Ensure setup_async() is called
                            await instance.setup_async()
                except ImportError:
                    # KnowledgeBase might not be available, skip
                    pass
                except Exception as e:
                    # Log but don't fail - setup_async() might already be called or fail for other reasons
                    from upsonic.utils.printing import warning_log
                    warning_log(
                        f"Could not ensure KnowledgeBase setup for tool '{tool.name}': {e}",
                        "ToolProcessor"
                    )

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
            execution_success = False
            
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
                    execution_success = True
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
            
            # Record execution in tool metrics
            tool.record_execution(
                execution_time=execution_time,
                args=kwargs,
                result=result,
                success=execution_success
            )
            
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

    def register_tools(
        self,
        tools: List[Any]
    ) -> Dict[str, Tool]:
        """
        Register new tools (similar to process_tools but only processes new tools).
        
        This method:
        1. Filters out tools that are already registered (object-level comparison)
        2. Processes only new tools
        3. Registers them
        4. Returns the newly registered tools
        
        Args:
            tools: List of raw tools to register
            
        Returns:
            Dict mapping tool names to Tool instances (only newly registered tools)
        """
        if not tools:
            return {}
        
        # Track registered tool objects by their identity (object-level comparison)
        registered_tool_objects = set(id(t) for t in self.registered_tools.values())
        
        # Filter out already registered tools (same object instance)
        tools_to_register = []
        for tool in tools:
            if tool is None:
                continue
            # Check if this exact object is already registered
            if id(tool) not in registered_tool_objects:
                tools_to_register.append(tool)
        
        # Process only new tools
        if not tools_to_register:
            return {}
        
        # Use process_tools for the actual processing
        newly_registered = self.process_tools(tools_to_register)
        
        return newly_registered
    
    def unregister_tools(
        self,
        tool_names: List[str]
    ) -> None:
        """
        Unregister tools by name.
        
        This method:
        1. Removes tools from registered_tools
        2. Removes from MCP handler tracking
        
        Args:
            tool_names: List of tool names to unregister
        """
        if not tool_names:
            return
        
        for tool_name in tool_names:
            if tool_name in self.registered_tools:
                tool = self.registered_tools[tool_name]
                
                # If this is an MCP tool, remove from handler tracking
                if hasattr(tool, 'handler'):
                    # This is an MCPTool - remove from tracking
                    handler_id = id(tool.handler)
                    if handler_id in self.mcp_handler_to_tools:
                        if tool_name in self.mcp_handler_to_tools[handler_id]:
                            self.mcp_handler_to_tools[handler_id].remove(tool_name)
                        # If no more tools from this handler, cleanup tracking
                        if not self.mcp_handler_to_tools[handler_id]:
                            del self.mcp_handler_to_tools[handler_id]
                
                # Remove from registered tools
                del self.registered_tools[tool_name]
    
    def unregister_mcp_handlers(
        self,
        handlers: List[Any]
    ) -> List[str]:
        """
        Unregister MCP handlers and ALL their tools.
        
        This method:
        1. Gets all tools from each handler
        2. Removes all those tools from registered_tools
        3. Removes handlers from mcp_handlers list
        4. Cleans up tracking
        
        Args:
            handlers: List of MCPHandler or MultiMCPHandler instances
            
        Returns:
            List of tool names that were removed
        """
        if not handlers:
            return []
        
        from upsonic.tools.mcp import MCPHandler, MultiMCPHandler
        
        removed_tool_names = []
        
        for handler in handlers:
            if not isinstance(handler, (MCPHandler, MultiMCPHandler)):
                continue
            
            handler_id = id(handler)
            
            # Get all tool names from this handler
            tool_names = self.mcp_handler_to_tools.get(handler_id, [])
            
            # Remove all tools from registered_tools
            for tool_name in tool_names:
                if tool_name in self.registered_tools:
                    del self.registered_tools[tool_name]
                    removed_tool_names.append(tool_name)
            
            # Remove from handler tracking
            if handler_id in self.mcp_handler_to_tools:
                del self.mcp_handler_to_tools[handler_id]
            
            # Remove handler from mcp_handlers list
            if handler in self.mcp_handlers:
                self.mcp_handlers.remove(handler)
        
        return removed_tool_names
    
    def unregister_class_instances(
        self,
        class_instances: List[Any]
    ) -> List[str]:
        """
        Unregister class instances (ToolKit or regular classes) and ALL their tools.
        
        This method:
        1. Gets all tools from each class instance
        2. Removes all those tools from registered_tools
        3. Cleans up tracking
        
        Args:
            class_instances: List of ToolKit or regular class instances
            
        Returns:
            List of tool names that were removed
        """
        if not class_instances:
            return []
        
        removed_tool_names = []
        
        for instance in class_instances:
            instance_id = id(instance)
            
            # Get all tool names from this class instance
            tool_names = self.class_instance_to_tools.get(instance_id, [])
            
            # Remove all tools from registered_tools
            for tool_name in tool_names:
                if tool_name in self.registered_tools:
                    del self.registered_tools[tool_name]
                    removed_tool_names.append(tool_name)
            
            # Remove from class instance tracking
            if instance_id in self.class_instance_to_tools:
                del self.class_instance_to_tools[instance_id]
        
        return removed_tool_names