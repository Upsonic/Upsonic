"""Tool normalization layer.

Stateless type-dispatch over the 8 input kinds the framework accepts:
raw function, bound method, ``ToolKit`` class, ``ToolKit`` instance,
tool-provider class, agent instance, ``MCPHandler``, plain class with
public methods. Produces a ``NormalizationResult`` consumed by
``ToolRegistry.add()``.
"""

from __future__ import annotations

import functools
import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from upsonic.tools.base import (
    Tool,
    ToolKit,
    ToolValidationError,
)
from upsonic.tools.config import ToolConfig
from upsonic.tools.schema import (
    function_schema,
    SchemaGenerationError,
    GenerateToolJsonSchema,
)
from upsonic.tools.wrappers import FunctionTool


@dataclass
class NormalizationResult:
    """Result of normalizing a list of tools.

    Holds the new tools plus the side-data the registry needs to update
    its ownership maps in a single atomic merge.
    """

    tools: Dict[str, Tool] = field(default_factory=dict)
    raw_object_ids: Set[int] = field(default_factory=set)
    mcp_handlers: List[Any] = field(default_factory=list)
    mcp_handler_owners: Dict[int, List[str]] = field(default_factory=dict)
    class_instance_owners: Dict[int, List[str]] = field(default_factory=dict)
    knowledge_base_instances: List[Any] = field(default_factory=list)
    toolkit_instances: List[Any] = field(default_factory=list)
    tool_provider_instances: List[Any] = field(default_factory=list)


class ToolNormalizer:
    """Stateless type dispatch.

    Same ``_process_*`` / ``_is_*`` semantics as the legacy
    ``ToolProcessor``, lifted out into its own collaborator. The
    normalizer never mutates a registry — it produces a
    ``NormalizationResult`` describing what to add.
    """

    def normalize(
        self,
        items: List[Any],
        already_registered: Set[int],
    ) -> NormalizationResult:
        """Normalize raw tool inputs into a ``NormalizationResult``.

        ``already_registered`` is read-only on the caller's set of raw
        object IDs (typically ``ToolRegistry.raw_object_ids``). The
        normalizer adds new IDs to the returned result, leaving the
        caller's set untouched.
        """
        result = NormalizationResult()

        if not items:
            return result

        # Filter out objects we've already registered (object-level
        # dedup). Keeps order; tracks dedup IDs in the result.
        filtered: List[Any] = []
        for item in items:
            if item is None:
                continue
            item_id = id(item)
            if item_id in already_registered or item_id in result.raw_object_ids:
                continue
            filtered.append(item)
            result.raw_object_ids.add(item_id)

        for tool_item in filtered:
            # Optimization: already a Tool subclass — register directly
            if isinstance(tool_item, Tool):
                result.tools[tool_item.name] = tool_item
                continue

            if self._is_builtin_tool(tool_item):
                continue

            if self._is_mcp_tool(tool_item):
                mcp_tools = self._process_mcp_tool(tool_item, result)
                for name, tool in mcp_tools.items():
                    result.tools[name] = tool

            elif inspect.isfunction(tool_item):
                tool = self._process_function_tool(tool_item)
                result.tools[tool.name] = tool

            elif inspect.ismethod(tool_item):
                tool = self._process_function_tool(tool_item)
                result.tools[tool.name] = tool

            elif inspect.isclass(tool_item):
                if issubclass(tool_item, ToolKit):
                    toolkit_tools = self._process_toolkit(tool_item(), result)
                    result.tools.update(toolkit_tools)
                else:
                    class_tools = self._process_class_tools(tool_item(), result)
                    result.tools.update(class_tools)

            elif hasattr(tool_item, '__class__'):
                if isinstance(tool_item, ToolKit):
                    toolkit_tools = self._process_toolkit(tool_item, result)
                    result.tools.update(toolkit_tools)
                elif self._is_tool_provider(tool_item):
                    provider_tools = self._process_tool_provider(tool_item, result)
                    result.tools.update(provider_tools)
                elif self._is_agent_instance(tool_item):
                    agent_tool = self._process_agent_tool(tool_item)
                    result.tools[agent_tool.name] = agent_tool
                else:
                    instance_tools = self._process_class_tools(tool_item, result)
                    result.tools.update(instance_tools)

        return result

    def extract_builtin_tools(self, tools: List[Any]) -> List[Any]:
        """Extract built-in tools from a list of tools."""
        return [t for t in tools if t is not None and self._is_builtin_tool(t)]

    # ------------------------------------------------------------------
    # Type predicates
    # ------------------------------------------------------------------

    def _is_mcp_tool(self, tool_item: Any) -> bool:
        from upsonic.tools.mcp import MCPHandler, MultiMCPHandler
        if isinstance(tool_item, (MCPHandler, MultiMCPHandler)):
            return True
        if not inspect.isclass(tool_item):
            return False
        return hasattr(tool_item, 'url') or hasattr(tool_item, 'command')

    def _is_builtin_tool(self, tool_item: Any) -> bool:
        from upsonic.tools.builtin_tools import AbstractBuiltinTool
        return isinstance(tool_item, AbstractBuiltinTool)

    def _is_tool_provider(self, obj: Any) -> bool:
        """A tool provider exposes ``get_tools()`` without being a ``ToolKit``."""
        return (
            hasattr(obj, 'get_tools')
            and callable(obj.get_tools)
            and not isinstance(obj, ToolKit)
        )

    def _is_agent_instance(self, obj: Any) -> bool:
        return hasattr(obj, 'name') and (
            hasattr(obj, 'do_async')
            or hasattr(obj, 'do')
            or hasattr(obj, 'agent_id')
        )

    # ------------------------------------------------------------------
    # Per-kind processors
    # ------------------------------------------------------------------

    def _process_mcp_tool(self, mcp_config: Any, result: NormalizationResult) -> Dict[str, Tool]:
        """Process an MCP tool configuration / handler."""
        from upsonic.tools.mcp import MCPHandler, MultiMCPHandler

        if isinstance(mcp_config, (MCPHandler, MultiMCPHandler)):
            handler = mcp_config
        else:
            handler = MCPHandler(config=mcp_config)

        result.mcp_handlers.append(handler)

        mcp_tools = handler.get_tools()
        tools_dict = {tool.name: tool for tool in mcp_tools}

        handler_id = id(handler)
        if handler_id not in result.mcp_handler_owners:
            result.mcp_handler_owners[handler_id] = []
        existing_tools = set(result.mcp_handler_owners[handler_id])
        for tool_name in tools_dict.keys():
            if tool_name not in existing_tools:
                result.mcp_handler_owners[handler_id].append(tool_name)

        return tools_dict

    def _process_function_tool(self, func: Callable) -> Tool:
        """Process a function/bound method into a ``Tool``."""
        config = getattr(func, '_upsonic_tool_config', ToolConfig())

        try:
            schema = function_schema(
                func,
                schema_generator=GenerateToolJsonSchema,
                docstring_format=config.docstring_format,
                require_parameter_descriptions=config.require_parameter_descriptions,
            )
        except SchemaGenerationError as e:
            raise ToolValidationError(
                f"Invalid tool function '{func.__name__}': {e}"
            )

        tool_obj = FunctionTool(
            function=func,
            schema=schema,
            config=config,
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

    def _process_toolkit(self, toolkit: ToolKit, result: NormalizationResult) -> Dict[str, Tool]:
        """Process a ``ToolKit`` instance (two-phase discovery + register)."""
        tools: Dict[str, Tool] = {}

        result.toolkit_instances.append(toolkit)

        try:
            from upsonic.knowledge_base.knowledge_base import KnowledgeBase
            if isinstance(toolkit, KnowledgeBase):
                result.knowledge_base_instances.append(toolkit)
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
            owners = result.class_instance_owners.setdefault(toolkit_id, [])
            existing: set = set(owners)
            for tool_name in tools:
                if tool_name not in existing:
                    owners.append(tool_name)

        return tools

    def _process_tool_provider(self, provider: Any, result: NormalizationResult) -> Dict[str, Tool]:
        """Process an object implementing the ``get_tools()`` protocol."""
        tools: Dict[str, Tool] = {}

        result.tool_provider_instances.append(provider)

        try:
            from upsonic.knowledge_base.knowledge_base import KnowledgeBase
            if isinstance(provider, KnowledgeBase):
                result.knowledge_base_instances.append(provider)
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
            provider_id: int = id(provider)
            owners = result.class_instance_owners.setdefault(provider_id, [])
            existing: set = set(owners)
            for tool_name in tools:
                if tool_name not in existing:
                    owners.append(tool_name)

        return tools

    def _process_class_tools(self, instance: Any, result: NormalizationResult) -> Dict[str, Tool]:
        """Process all public methods of a class instance as tools."""
        tools: Dict[str, Tool] = {}

        for name, method in inspect.getmembers(instance, inspect.ismethod):
            if name.startswith('_'):
                continue

            try:
                tool = self._process_function_tool(method)
                tools[tool.name] = tool
            except ToolValidationError:
                continue

        if tools:
            instance_id: int = id(instance)
            owners = result.class_instance_owners.setdefault(instance_id, [])
            existing: set = set(owners)
            for tool_name in tools.keys():
                if tool_name not in existing:
                    owners.append(tool_name)

        return tools

    def _process_agent_tool(self, agent: Any) -> Tool:
        """Process an agent instance as a tool."""
        from upsonic.tools.wrappers import AgentTool
        return AgentTool(agent)

    # ------------------------------------------------------------------
    # Toolkit / wrapper helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_tool_wrapper(
        method: Any,
        tool_name: str,
        config: ToolConfig,
    ) -> Any:
        """Wrap a bound method in a plain function with its own ``__dict__``."""
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
        """Build a merged ``ToolConfig`` (toolkit init overrides decorator)."""
        tk_defaults: Dict[str, Any] = getattr(toolkit, '_toolkit_defaults', {})
        if not tk_defaults:
            return deepcopy(decorator_config)

        merged: ToolConfig = deepcopy(decorator_config)

        for field_name in ToolConfig.model_fields.keys():
            tk_val: Any = tk_defaults.get(field_name)
            if tk_val is not None:
                setattr(merged, field_name, tk_val)

        return merged
