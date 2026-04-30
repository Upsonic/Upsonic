"""Tool registry — owns all tool state.

Holds every dict that the legacy ``ToolProcessor`` owned plus
``wrapped_tools``. Provides cascade-delete semantics for the 8 input
kinds and ``collect_instructions`` / ``all_definitions`` aggregation.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

from upsonic.tools.base import (
    Tool,
    ToolDefinition,
    ToolKit,
)
from upsonic.utils.printing import warning_log

if TYPE_CHECKING:
    from upsonic.tools.normalizer import NormalizationResult


class ToolRegistry:
    """Owns every dict that ``processor.py`` owns today plus cascade-delete
    and instruction aggregation.
    """

    def __init__(self) -> None:
        self.registered_tools: Dict[str, Tool] = {}
        self.wrapped_tools: Dict[str, Callable] = {}
        self.raw_object_ids: Set[int] = set()
        self.mcp_handlers: List[Any] = []
        self.mcp_handler_to_tools: Dict[int, List[str]] = {}
        self.class_instance_to_tools: Dict[int, List[str]] = {}
        self.knowledge_base_instances: Dict[int, Any] = {}
        self.toolkit_instances: Dict[int, Any] = {}
        self.tool_provider_instances: Dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Tool]:
        return self.registered_tools.get(name)

    def all_tools(self) -> Dict[str, Tool]:
        return self.registered_tools

    def store_wrapped(self, name: str, fn: Callable) -> None:
        self.wrapped_tools[name] = fn

    def get_wrapped(self, name: str) -> Optional[Callable]:
        return self.wrapped_tools.get(name)

    def is_raw_object_registered(self, raw_id: int) -> bool:
        return raw_id in self.raw_object_ids

    def all_definitions(self) -> List[ToolDefinition]:
        """Build ``ToolDefinition`` instances for every registered tool."""
        definitions: List[ToolDefinition] = []
        for tool in self.registered_tools.values():
            config = getattr(tool, 'config', None)

            if tool.schema:
                json_schema = tool.schema.json_schema
            else:
                json_schema = {'type': 'object', 'properties': {}}

            sequential = config.sequential if config else False

            definitions.append(
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=json_schema,
                    kind=tool.metadata.kind if hasattr(tool, 'metadata') else 'function',
                    strict=tool.metadata.strict if hasattr(tool, 'metadata') else False,
                    sequential=sequential,
                    metadata=tool.metadata if tool.metadata else None,
                )
            )
        return definitions

    # ------------------------------------------------------------------
    # add / remove
    # ------------------------------------------------------------------

    def add(self, result: "NormalizationResult") -> Dict[str, Tool]:
        """Atomic merge of a ``NormalizationResult`` into this registry.

        Returns the newly added tools (``result.tools``) so callers can
        wrap or otherwise post-process them.
        """
        # New tools
        self.registered_tools.update(result.tools)

        # Raw object IDs (already-registered set)
        self.raw_object_ids.update(result.raw_object_ids)

        # MCP handlers
        for handler in result.mcp_handlers:
            if handler not in self.mcp_handlers:
                self.mcp_handlers.append(handler)
        for handler_id, tool_names in result.mcp_handler_owners.items():
            if handler_id not in self.mcp_handler_to_tools:
                self.mcp_handler_to_tools[handler_id] = []
            existing = set(self.mcp_handler_to_tools[handler_id])
            for tool_name in tool_names:
                if tool_name not in existing:
                    self.mcp_handler_to_tools[handler_id].append(tool_name)

        # Class instance owners
        for instance_id, tool_names in result.class_instance_owners.items():
            if instance_id not in self.class_instance_to_tools:
                self.class_instance_to_tools[instance_id] = []
            existing = set(self.class_instance_to_tools[instance_id])
            for tool_name in tool_names:
                if tool_name not in existing:
                    self.class_instance_to_tools[instance_id].append(tool_name)

        # Knowledge base / toolkit / tool provider instances
        for kb in result.knowledge_base_instances:
            self.knowledge_base_instances[id(kb)] = kb
        for tk in result.toolkit_instances:
            self.toolkit_instances[id(tk)] = tk
        for tp in result.tool_provider_instances:
            self.tool_provider_instances[id(tp)] = tp

        return result.tools

    def remove(
        self,
        target: Union[Any, List[Any]],
    ) -> Tuple[List[str], List[Any]]:
        """Remove tools from the registry.

        Handles all 8 input kinds (tool name string, function, agent,
        MCP handler, ``ToolKit``, tool provider, class, instance).
        Returns ``(removed_tool_names, removed_original_objects)``.
        """
        from upsonic.tools.mcp import MCPHandler, MultiMCPHandler

        if not isinstance(target, list):
            tools = [target]
        else:
            tools = target

        if not tools:
            return ([], [])

        registered_tools = self.registered_tools

        mcp_handlers: List[Any] = []
        class_instances: List[Any] = []
        tool_names_to_remove: List[str] = []
        original_objects_to_remove: set = set()

        for tool_identifier in tools:
            # String — tool name
            if isinstance(tool_identifier, str):
                tool_names_to_remove.append(tool_identifier)

                # Find the original object for this tool name. ONLY add to
                # original_objects_to_remove for 1:1 relationships
                # (functions, agents) — NOT for 1:many relationships
                # (MCP handlers, ToolKits, class instances).
                if tool_identifier in registered_tools:
                    registered_tool = registered_tools[tool_identifier]

                    is_one_to_many = False

                    # MCP handler (1:many) — partial removal check
                    if hasattr(registered_tool, 'handler'):
                        handler_id = id(registered_tool.handler)
                        if handler_id in self.mcp_handler_to_tools:
                            handler_tools = self.mcp_handler_to_tools[handler_id]
                            if len(handler_tools) > 1:
                                is_one_to_many = True

                    # ToolKit / class instance (1:many) — partial removal check
                    elif hasattr(registered_tool, 'function') and hasattr(registered_tool.function, '__self__'):
                        instance = registered_tool.function.__self__
                        instance_id = id(instance)
                        if instance_id in self.class_instance_to_tools:
                            instance_tools = self.class_instance_to_tools[instance_id]
                            if len(instance_tools) > 1:
                                is_one_to_many = True

                    if not is_one_to_many:
                        if hasattr(registered_tool, 'agent'):
                            original_objects_to_remove.add(registered_tool.agent)
                        elif hasattr(registered_tool, 'function'):
                            original_objects_to_remove.add(registered_tool.function)

            # MCP handler
            elif isinstance(tool_identifier, (MCPHandler, MultiMCPHandler)):
                mcp_handlers.append(tool_identifier)
                original_objects_to_remove.add(tool_identifier)

            # ToolKit instance
            elif isinstance(tool_identifier, ToolKit):
                class_instances.append(tool_identifier)
                original_objects_to_remove.add(tool_identifier)

            # Tool provider (e.g. KnowledgeBase implementing get_tools())
            elif hasattr(tool_identifier, 'get_tools') and callable(tool_identifier.get_tools):
                class_instances.append(tool_identifier)
                original_objects_to_remove.add(tool_identifier)

            # Class (not instance) — find all instances of this class
            elif inspect.isclass(tool_identifier):
                found_instances = set()
                for name, registered_tool in registered_tools.items():
                    if hasattr(registered_tool, 'function') and hasattr(registered_tool.function, '__self__'):
                        instance = registered_tool.function.__self__
                        if isinstance(instance, tool_identifier):
                            found_instances.add(instance)

                for instance in found_instances:
                    if instance not in class_instances:
                        class_instances.append(instance)
                        original_objects_to_remove.add(instance)

                if not found_instances:
                    warning_log(
                        f"No instances of class {tool_identifier.__name__} found in registered tools. "
                        f"Pass the instance directly instead of the class.",
                        "ToolRegistry",
                    )

            # Regular class instance or other object
            else:
                instance_id = id(tool_identifier)

                if instance_id in self.class_instance_to_tools:
                    class_instances.append(tool_identifier)
                    original_objects_to_remove.add(tool_identifier)
                else:
                    found = False
                    for name, registered_tool in registered_tools.items():
                        # Direct match
                        if registered_tool is tool_identifier or id(registered_tool) == id(tool_identifier):
                            tool_names_to_remove.append(name)
                            found = True
                            break

                        # Agent match
                        if hasattr(registered_tool, 'agent') and (
                            registered_tool.agent is tool_identifier
                            or id(registered_tool.agent) == id(tool_identifier)
                        ):
                            tool_names_to_remove.append(name)
                            original_objects_to_remove.add(tool_identifier)
                            found = True
                            break

                        # Function match
                        if hasattr(registered_tool, 'function') and (
                            registered_tool.function is tool_identifier
                            or id(registered_tool.function) == id(tool_identifier)
                        ):
                            tool_names_to_remove.append(name)
                            original_objects_to_remove.add(tool_identifier)
                            found = True
                            break

                        # Handler match
                        if hasattr(registered_tool, 'handler') and (
                            registered_tool.handler is tool_identifier
                            or id(registered_tool.handler) == id(tool_identifier)
                        ):
                            tool_names_to_remove.append(name)
                            original_objects_to_remove.add(tool_identifier)
                            found = True
                            break

                    if not found and hasattr(tool_identifier, 'name'):
                        tool_names_to_remove.append(tool_identifier.name)
                    elif not found and hasattr(tool_identifier, '__name__'):
                        tool_names_to_remove.append(tool_identifier.__name__)

        all_removed_names: set = set(tool_names_to_remove)

        # Remove MCP handlers (cascade)
        if mcp_handlers:
            removed_names = self._unregister_mcp_handlers(mcp_handlers)
            all_removed_names.update(removed_names)

        # Remove class instances (cascade)
        if class_instances:
            removed_names = self._unregister_class_instances(class_instances)
            all_removed_names.update(removed_names)

        # Remove individual tools by name
        if tool_names_to_remove:
            self._unregister_tools(list(set(tool_names_to_remove)))

        # Remove from wrapped_tools
        for tool_name in all_removed_names:
            if tool_name in self.wrapped_tools:
                del self.wrapped_tools[tool_name]

        return (list(all_removed_names), list(original_objects_to_remove))

    # ------------------------------------------------------------------
    # Cascade-delete primitives (private)
    # ------------------------------------------------------------------

    def _unregister_tools(self, tool_names: List[str]) -> None:
        """Unregister tools by name. Cleans MCP and class-instance tracking."""
        if not tool_names:
            return

        for tool_name in tool_names:
            if tool_name in self.registered_tools:
                tool = self.registered_tools[tool_name]

                # MCP tool — remove from handler tracking
                if hasattr(tool, 'handler'):
                    handler = tool.handler
                    handler_id = id(handler)
                    if handler_id in self.mcp_handler_to_tools:
                        if tool_name in self.mcp_handler_to_tools[handler_id]:
                            self.mcp_handler_to_tools[handler_id].remove(tool_name)
                        if not self.mcp_handler_to_tools[handler_id]:
                            del self.mcp_handler_to_tools[handler_id]
                            if handler in self.mcp_handlers:
                                self.mcp_handlers.remove(handler)
                            self.raw_object_ids.discard(handler_id)

                # Bound method (class instance) — remove from instance tracking
                if hasattr(tool, 'function') and hasattr(tool.function, '__self__'):
                    instance = tool.function.__self__
                    instance_id = id(instance)
                    if instance_id in self.class_instance_to_tools:
                        if tool_name in self.class_instance_to_tools[instance_id]:
                            self.class_instance_to_tools[instance_id].remove(tool_name)
                        if not self.class_instance_to_tools[instance_id]:
                            del self.class_instance_to_tools[instance_id]
                            self.knowledge_base_instances.pop(instance_id, None)
                            self.toolkit_instances.pop(instance_id, None)
                            self.raw_object_ids.discard(instance_id)

                del self.registered_tools[tool_name]

    def _unregister_mcp_handlers(self, handlers: List[Any]) -> List[str]:
        """Unregister MCP handlers and ALL their tools."""
        if not handlers:
            return []

        from upsonic.tools.mcp import MCPHandler, MultiMCPHandler

        removed_tool_names: List[str] = []

        for handler in handlers:
            if not isinstance(handler, (MCPHandler, MultiMCPHandler)):
                continue

            handler_id = id(handler)
            tool_names = self.mcp_handler_to_tools.get(handler_id, [])

            for tool_name in tool_names:
                if tool_name in self.registered_tools:
                    del self.registered_tools[tool_name]
                    removed_tool_names.append(tool_name)

            if handler_id in self.mcp_handler_to_tools:
                del self.mcp_handler_to_tools[handler_id]

            if handler in self.mcp_handlers:
                self.mcp_handlers.remove(handler)

            self.raw_object_ids.discard(handler_id)

        return removed_tool_names

    def _unregister_class_instances(self, class_instances: List[Any]) -> List[str]:
        """Unregister class instances (ToolKit / regular / provider) and ALL their tools."""
        if not class_instances:
            return []

        removed_tool_names: List[str] = []

        for instance in class_instances:
            instance_id = id(instance)

            tool_names = self.class_instance_to_tools.get(instance_id, [])

            for tool_name in tool_names:
                if tool_name in self.registered_tools:
                    del self.registered_tools[tool_name]
                    removed_tool_names.append(tool_name)

            if instance_id in self.class_instance_to_tools:
                del self.class_instance_to_tools[instance_id]

            self.knowledge_base_instances.pop(instance_id, None)
            self.toolkit_instances.pop(instance_id, None)
            self.tool_provider_instances.pop(instance_id, None)
            self.raw_object_ids.discard(instance_id)

        return removed_tool_names

    # ------------------------------------------------------------------
    # Instructions
    # ------------------------------------------------------------------

    def collect_instructions(self) -> List[str]:
        """Collect all active instructions from toolkits, providers, and tools.

        Returns a deduplicated, ordered list of instruction strings.
        """
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

        # Tool-provider context (e.g. KnowledgeBase.build_context())
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
