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
    Any, Callable, Dict, List, Optional, TYPE_CHECKING
)

from upsonic.tools.base import (
    Tool, ToolKit
)
from upsonic.tools.config import ToolConfig
from upsonic.tools.schema import (
    function_schema,
    SchemaGenerationError,
    GenerateToolJsonSchema,
)
from upsonic.tools.wrappers import FunctionTool
from upsonic.tools.deferred import PausedToolCall

if TYPE_CHECKING:
    from upsonic.tools.base import Tool, ToolConfig
    from upsonic.tools.user_input import UserInputField

class ToolValidationError(Exception):
    """Error raised when tool validation fails."""
    pass


class ExternalExecutionPause(Exception):
    """Exception to pause execution when external tool execution is required."""
    
    def __init__(self, paused_calls: List[PausedToolCall] = None):
        self.paused_calls: List[PausedToolCall] = paused_calls or []
        super().__init__(f"Paused for {len(self.paused_calls)} external tool calls")


class ConfirmationPause(Exception):
    """Exception to pause execution when user confirmation is required."""

    def __init__(self, paused_calls: List[PausedToolCall] = None):
        self.paused_calls: List[PausedToolCall] = paused_calls or []
        super().__init__(f"Paused for {len(self.paused_calls)} tool(s) requiring confirmation")


class UserInputPause(Exception):
    """Exception to pause execution when user input is required."""

    def __init__(
        self,
        paused_calls: List[PausedToolCall] = None,
        user_input_schema: Optional[List["UserInputField"]] = None,
    ):
        self.paused_calls: List[PausedToolCall] = paused_calls or []
        self.user_input_schema = user_input_schema or []
        super().__init__(f"Paused for {len(self.paused_calls)} tool(s) requiring user input")


class ToolProcessor:
    """Processes and validates tools before registration."""
    
    def __init__(
        self,
    ):
        self.registered_tools: Dict[str, Tool] = {}
        self.mcp_handlers: List[Any] = []
        # Track which tools belong to which MCP handler
        self.mcp_handler_to_tools: Dict[int, List[str]] = {}  # handler id -> tool names
        # Track which tools belong to which class instance (ToolKit, tool provider, or regular class)
        self.class_instance_to_tools: Dict[int, List[str]] = {}  # class instance id -> tool names
        # Track KnowledgeBase instances that need setup_async() called
        self.knowledge_base_instances: Dict[int, Any] = {}  # instance id -> KnowledgeBase instance
        self.toolkit_instances: Dict[int, Any] = {}  # instance id -> ToolKit instance
        # Track tool provider instances (objects implementing the get_tools() protocol)
        self.tool_provider_instances: Dict[int, Any] = {}  # instance id -> provider instance
        # Track raw tool object IDs for deduplication (prevents re-processing same objects)
        self._raw_tool_ids: set = set()
    
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
                    toolkit_tools = self._process_toolkit(tool_item)
                    processed_tools.update(toolkit_tools)
                elif self._is_tool_provider(tool_item):
                    provider_tools = self._process_tool_provider(tool_item)
                    processed_tools.update(provider_tools)
                elif self._is_agent_instance(tool_item):
                    agent_tool = self._process_agent_tool(tool_item)
                    processed_tools[agent_tool.name] = agent_tool
                else:
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
        
        # Track which tools belong to this handler (avoid duplicates)
        handler_id = id(handler)
        if handler_id not in self.mcp_handler_to_tools:
            self.mcp_handler_to_tools[handler_id] = []
        existing_tools = set(self.mcp_handler_to_tools[handler_id])
        for tool_name in tools_dict.keys():
            if tool_name not in existing_tools:
                self.mcp_handler_to_tools[handler_id].append(tool_name)
        
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
        
        tool_obj = FunctionTool(
            function=func,
            schema=schema,
            config=config
        )

        # Allow tools to provide a pre-built JSON schema (e.g. Apify actors
        # whose input schema is richer than what Python type hints express).
        json_override = getattr(func, '_json_schema_override', None)
        if json_override is not None:
            tool_obj.schema.json_schema = json_override

        if config.requires_confirmation:
            confirm_suffix: str = (
                "\n\nIMPORTANT: This tool requires confirmation before execution. "
                "You MUST call this tool directly with the required parameters — "
                "do NOT ask the user for confirmation yourself. The system will "
                "automatically pause and request confirmation from the user after "
                "you make the call."
            )
            if tool_obj.schema.description:
                tool_obj.schema.description += confirm_suffix
            else:
                tool_obj.schema.description = confirm_suffix.lstrip("\n")
            tool_obj.description = tool_obj.schema.description

            if not config.instructions:
                config.instructions = (
                    f"Tool '{func.__name__}' requires user confirmation. You MUST call "
                    f"this tool directly — never ask the user for confirmation in your "
                    f"response text. The framework will automatically pause execution "
                    f"and collect confirmation from the user."
                )
                config.add_instructions = True

        if config.requires_user_input and config.user_input_fields:
            required: list = tool_obj.schema.json_schema.get("required", [])
            tool_obj.schema.json_schema["required"] = [
                f for f in required if f not in config.user_input_fields
            ]
            field_list: str = ", ".join(config.user_input_fields)
            suffix: str = (
                f"\n\nIMPORTANT: The following field(s) will be provided by the user "
                f"after you call this tool — do NOT ask the user for them and do NOT "
                f"include them in the call. Just call this tool with the fields you "
                f"already have. User-provided fields: {field_list}"
            )
            if tool_obj.schema.description:
                tool_obj.schema.description += suffix
            else:
                tool_obj.schema.description = suffix.lstrip("\n")
            tool_obj.description = tool_obj.schema.description

            if not config.instructions:
                config.instructions = (
                    f"Tool '{func.__name__}' requires user input for the following "
                    f"field(s): {field_list}. You MUST call this tool without providing "
                    f"those fields — the framework will pause and collect them from the "
                    f"user. Never ask the user for these values in your response text."
                )
                config.add_instructions = True

        return tool_obj
    
    def _process_toolkit(self, toolkit: ToolKit) -> Dict[str, Tool]:
        """Process a ToolKit instance using a two-phase algorithm.

        Phase 1 -- Discover which methods become tools:
            * ``@tool``-decorated methods form the base candidate set.
            * ``use_async=True`` replaces candidates with **all** non-private
              async methods and drops every sync method.
            * ``include_tools`` is **additive** -- names are added on top.
            * ``exclude_tools`` is **supreme** -- names are always removed.

        Phase 2 -- Assign configs and register:
            * Methods with ``_upsonic_tool_config`` (from ``@tool``): merge
              with toolkit defaults (toolkit init overrides decorator).
            * Methods without config (added via ``include_tools`` or
              ``use_async`` discovery): use toolkit defaults merged with
              fresh ``ToolConfig()``.
        """
        tools: Dict[str, Tool] = {}

        self.toolkit_instances[id(toolkit)] = toolkit

        try:
            from upsonic.knowledge_base.knowledge_base import KnowledgeBase
            if isinstance(toolkit, KnowledgeBase):
                self.knowledge_base_instances[id(toolkit)] = toolkit
        except ImportError:
            pass

        use_async: bool = getattr(toolkit, '_toolkit_use_async', False)
        include_tools: List[str] | None = getattr(toolkit, '_toolkit_include_tools', None)
        exclude_tools: List[str] | None = getattr(toolkit, '_toolkit_exclude_tools', None)

        # ── Phase 1: discover candidates ──────────────────────────
        candidates: Dict[str, Any] = {}

        if use_async:
            for name, method in inspect.getmembers(toolkit, inspect.ismethod):
                if name.startswith('_'):
                    continue
                if inspect.iscoroutinefunction(method):
                    candidates[name] = method
        else:
            for name, method in inspect.getmembers(toolkit, inspect.ismethod):
                if getattr(method, '_upsonic_is_tool', False):
                    candidates[name] = method

        if include_tools is not None:
            for name in include_tools:
                if name not in candidates:
                    method: Any = getattr(toolkit, name, None)
                    if method is not None and inspect.ismethod(method):
                        candidates[name] = method

        if exclude_tools is not None:
            for name in exclude_tools:
                candidates.pop(name, None)

        # ── Phase 2: build configs & register ─────────────────────
        registered_callables: List[Any] = []
        confirmation_tools: Optional[List[str]] = getattr(toolkit, '_requires_confirmation_tools', None)
        user_input_tools: Optional[List[str]] = getattr(toolkit, '_requires_user_input_tools', None)
        external_execution_tools: Optional[List[str]] = getattr(toolkit, '_requires_external_execution_tools', None)

        for name, method in candidates.items():
            decorator_config: ToolConfig = getattr(
                method, '_upsonic_tool_config', ToolConfig()
            )
            config: ToolConfig = self._apply_toolkit_config_overrides(
                toolkit, decorator_config
            )

            if confirmation_tools and name in confirmation_tools:
                config.requires_confirmation = True
            if user_input_tools and name in user_input_tools:
                config.requires_user_input = True
            if external_execution_tools and name in external_execution_tools:
                config.external_execution = True

            wrapper = self._make_tool_wrapper(method, name, config)
            tool = self._process_function_tool(wrapper)
            tools[tool.name] = tool
            registered_callables.append(wrapper)

        toolkit.tools = registered_callables

        if tools:
            toolkit_id: int = id(toolkit)
            if toolkit_id not in self.class_instance_to_tools:
                self.class_instance_to_tools[toolkit_id] = []
            existing_tools: set = set(self.class_instance_to_tools[toolkit_id])
            for tool_name in tools:
                if tool_name not in existing_tools:
                    self.class_instance_to_tools[toolkit_id].append(tool_name)

        return tools

    def _is_tool_provider(self, obj: Any) -> bool:
        """Check if an object implements the tool provider protocol.

        A tool provider is any object that exposes a ``get_tools()`` method
        returning a list of callables, *without* being a ``ToolKit`` subclass.
        This is the preferred integration path for components like
        ``KnowledgeBase`` that produce tools on demand.

        Args:
            obj: The object to check.

        Returns:
            True if the object implements the tool provider protocol.
        """
        return (
            hasattr(obj, 'get_tools')
            and callable(obj.get_tools)
            and not isinstance(obj, ToolKit)
        )

    def _process_tool_provider(self, provider: Any) -> Dict[str, Tool]:
        """Process an object implementing the tool provider protocol.

        Calls ``provider.get_tools()`` to obtain ready-to-use ``Tool``
        objects (or raw callables as fallback) and registers them.  The
        provider is tracked for lifecycle management (unregistration,
        context instructions, KB setup, etc.).

        Args:
            provider: An object with a ``get_tools()`` method returning a
                list of ``Tool`` instances or callables.

        Returns:
            Dict mapping tool names to ``Tool`` instances.
        """
        tools: Dict[str, Tool] = {}

        provider_id: int = id(provider)
        self.tool_provider_instances[provider_id] = provider

        try:
            from upsonic.knowledge_base.knowledge_base import KnowledgeBase
            if isinstance(provider, KnowledgeBase):
                self.knowledge_base_instances[provider_id] = provider
        except ImportError:
            pass

        provided_tools: list = provider.get_tools()

        for item in provided_tools:
            if isinstance(item, Tool):
                tools[item.name] = item
            else:
                processed: Tool = self._process_function_tool(item)
                tools[processed.name] = processed

        if tools:
            if provider_id not in self.class_instance_to_tools:
                self.class_instance_to_tools[provider_id] = []
            existing: set = set(self.class_instance_to_tools[provider_id])
            for tool_name in tools:
                if tool_name not in existing:
                    self.class_instance_to_tools[provider_id].append(tool_name)

        return tools

    @staticmethod
    def _make_tool_wrapper(
        method: Any,
        tool_name: str,
        config: ToolConfig,
    ) -> Any:
        """Wrap a bound method in a plain function with its own ``__dict__``.

        Bound methods do not support arbitrary ``setattr``, so we create a
        per-instance wrapper that carries ``_upsonic_tool_config`` and
        ``_upsonic_is_tool``.
        """
        # Preserve __self__ from bound methods so unregister_tools
        # can look up the owning instance for tracking cleanup.
        bound_self = getattr(method, "__self__", None)

        if inspect.iscoroutinefunction(method):
            @functools.wraps(method)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await method(*args, **kwargs)

            _async_wrapper.__name__ = tool_name
            _async_wrapper._upsonic_tool_config = config  # type: ignore[attr-defined]
            _async_wrapper._upsonic_is_tool = True  # type: ignore[attr-defined]
            if bound_self is not None:
                _async_wrapper.__self__ = bound_self  # type: ignore[attr-defined]
            return _async_wrapper
        else:
            @functools.wraps(method)
            def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return method(*args, **kwargs)

            _sync_wrapper.__name__ = tool_name
            _sync_wrapper._upsonic_tool_config = config  # type: ignore[attr-defined]
            _sync_wrapper._upsonic_is_tool = True  # type: ignore[attr-defined]
            if bound_self is not None:
                _sync_wrapper.__self__ = bound_self  # type: ignore[attr-defined]
            return _sync_wrapper

    def _apply_toolkit_config_overrides(
        self,
        toolkit: ToolKit,
        decorator_config: ToolConfig,
    ) -> ToolConfig:
        """Build a merged ``ToolConfig``.

        Priority (highest first):
            1. Toolkit ``__init__`` defaults (toolkit-wide, set at instantiation)
            2. ``@tool`` decorator values (per-method, set at class definition)

        The decorator config is used as the base layer; then any
        toolkit ``__init__`` value that was explicitly provided
        (non-``None``) overwrites the decorator value.
        """
        from copy import deepcopy

        tk_defaults: Dict[str, Any] = getattr(toolkit, '_toolkit_defaults', {})
        if not tk_defaults:
            return deepcopy(decorator_config)

        merged: ToolConfig = deepcopy(decorator_config)

        for field_name in ToolConfig.model_fields.keys():
            tk_val: Any = tk_defaults.get(field_name)
            if tk_val is not None:
                setattr(merged, field_name, tk_val)

        return merged
    
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
        
        # Track which tools belong to this class instance (avoid duplicates)
        if tools:
            instance_id = id(instance)
            if instance_id not in self.class_instance_to_tools:
                self.class_instance_to_tools[instance_id] = []
            existing_tools = set(self.class_instance_to_tools[instance_id])
            for tool_name in tools.keys():
                if tool_name not in existing_tools:
                    self.class_instance_to_tools[instance_id].append(tool_name)
        
        return tools
    
    def _is_agent_instance(self, obj: Any) -> bool:
        """Check if an object is an agent instance."""
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
            if self.knowledge_base_instances:
                for kb_id, kb in self.knowledge_base_instances.items():
                    if tool.name in (self.class_instance_to_tools.get(kb_id) or []):
                        try:
                            await kb.setup_async()
                        except Exception as e:
                            from upsonic.utils.printing import warning_log
                            warning_log(
                                f"Could not ensure KnowledgeBase setup for tool '{tool.name}': {e}",
                                "ToolProcessor"
                            )
                        break

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
            
            # User confirmation — pause for user approval
            if config.requires_confirmation:
                raise ConfirmationPause()

            # User input — pause for user-provided field values
            if config.requires_user_input:
                raise UserInputPause()

            # External execution
            if config.external_execution:
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
                        
                except (ExternalExecutionPause, ConfirmationPause, UserInputPause):
                    raise
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
        1. Filters out tools that are already registered (object-level comparison using raw tool IDs)
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
        
        # Filter out already registered tools using raw tool object IDs
        # This correctly tracks the original objects (functions, class instances, etc.)
        # not the processed Tool wrappers
        tools_to_register = []
        for tool in tools:
            if tool is None:
                continue
            tool_id = id(tool)
            # Check if this exact raw tool object was already registered
            if tool_id not in self._raw_tool_ids:
                tools_to_register.append(tool)
                # Track this raw tool ID immediately to prevent duplicates
                self._raw_tool_ids.add(tool_id)
        
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
        2. Removes from MCP handler tracking (mcp_handler_to_tools)
        3. Removes from class instance tracking (class_instance_to_tools)
        
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
                    handler = tool.handler
                    handler_id = id(handler)
                    if handler_id in self.mcp_handler_to_tools:
                        if tool_name in self.mcp_handler_to_tools[handler_id]:
                            self.mcp_handler_to_tools[handler_id].remove(tool_name)
                        # If no more tools from this handler, cleanup tracking and remove handler
                        if not self.mcp_handler_to_tools[handler_id]:
                            del self.mcp_handler_to_tools[handler_id]
                            # Also remove from mcp_handlers list
                            if handler in self.mcp_handlers:
                                self.mcp_handlers.remove(handler)
                            # Remove from raw tool IDs tracking
                            if handler_id in self._raw_tool_ids:
                                self._raw_tool_ids.discard(handler_id)
                
                # If this is a class instance tool (method), remove from class instance tracking
                if hasattr(tool, 'function') and hasattr(tool.function, '__self__'):
                    # This is a bound method - get the instance
                    instance = tool.function.__self__
                    instance_id = id(instance)
                    if instance_id in self.class_instance_to_tools:
                        if tool_name in self.class_instance_to_tools[instance_id]:
                            self.class_instance_to_tools[instance_id].remove(tool_name)
                        # If no more tools from this instance, cleanup tracking
                        if not self.class_instance_to_tools[instance_id]:
                            del self.class_instance_to_tools[instance_id]
                            if instance_id in self.knowledge_base_instances:
                                del self.knowledge_base_instances[instance_id]
                            if instance_id in self.toolkit_instances:
                                del self.toolkit_instances[instance_id]
                            if instance_id in self._raw_tool_ids:
                                self._raw_tool_ids.discard(instance_id)
                
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
            
            # Remove from raw tool IDs tracking
            if handler_id in self._raw_tool_ids:
                self._raw_tool_ids.discard(handler_id)
        
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
            
            if instance_id in self.knowledge_base_instances:
                del self.knowledge_base_instances[instance_id]
            
            if instance_id in self.toolkit_instances:
                del self.toolkit_instances[instance_id]
            
            if instance_id in self.tool_provider_instances:
                del self.tool_provider_instances[instance_id]
            
            # Remove from raw tool IDs tracking
            if instance_id in self._raw_tool_ids:
                self._raw_tool_ids.discard(instance_id)
        
        return removed_tool_names

    def collect_instructions(self) -> List[str]:
        """Collect all active instructions from toolkits and individual tools.

        Returns a deduplicated, ordered list of instruction strings. Each string
        is prefixed with the source (toolkit name or tool name) so the model
        knows which tool/toolkit the instructions apply to.

        Sources:
        1. ``ToolKit`` instances whose ``add_instructions`` is True.
        2. Individual registered tools whose ``ToolConfig.add_instructions``
           is True.
        """
        from upsonic.tools.base import ToolKit

        seen: set = set()
        instructions: List[str] = []

        for toolkit in self.toolkit_instances.values():
            if not isinstance(toolkit, ToolKit):
                continue
            if not getattr(toolkit, "add_instructions", False):
                continue
            text: Optional[str] = getattr(toolkit, "instructions", None)
            if not text:
                continue
            toolkit_name: str = getattr(toolkit, "name", None) or type(toolkit).__name__
            key = ("toolkit", toolkit_name, text)
            if key in seen:
                continue
            seen.add(key)
            instructions.append(f"Instructions for toolkit «{toolkit_name}»:\n{text.strip()}")

        # Tool provider context (e.g. KnowledgeBase.build_context())
        for provider in self.tool_provider_instances.values():
            if not (hasattr(provider, 'build_context') and callable(provider.build_context)):
                continue
            try:
                context: str = provider.build_context()
            except Exception:
                continue
            if not context:
                continue
            provider_name: str = getattr(provider, 'name', None) or type(provider).__name__
            key = ("provider", provider_name, context)
            if key in seen:
                continue
            seen.add(key)
            instructions.append(context)

        for tool_name, tool in self.registered_tools.items():
            config: Optional[Any] = getattr(tool, "config", None)
            if config is None:
                continue
            if not getattr(config, "add_instructions", False):
                continue
            text = getattr(config, "instructions", None)
            if not text:
                continue
            key = ("tool", tool_name, text)
            if key in seen:
                continue
            seen.add(key)
            instructions.append(f"Instructions for tool «{tool_name}»:\n{text.strip()}")

        return instructions