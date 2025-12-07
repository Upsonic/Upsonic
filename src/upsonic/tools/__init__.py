"""
Upsonic Tools System

A comprehensive, modular tool handling system for AI agents that supports:
- Function tools with decorators
- Class-based tools and toolkits
- Agent-as-tool functionality
- MCP (Model Context Protocol) tools
- Deferred and external tool execution
- Tool orchestration and planning
- Rich behavioral configuration (caching, confirmation, hooks, etc.)
"""

from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task
    from upsonic.tools.base import (
        Tool,
        ToolKit,
        ToolDefinition,
        ToolResult,
        ToolMetadata,
        DocstringFormat,
        ObjectJsonSchema,
    )
    from upsonic.tools.config import (
        tool,
        ToolConfig,
        ToolHooks,
    )
    from upsonic.tools.metrics import (
        ToolMetrics,
    )
    from upsonic.tools.schema import (
        FunctionSchema,
        function_schema,
        SchemaGenerationError,
    )
    from upsonic.tools.processor import (
        ToolProcessor,
        ToolValidationError,
        ExternalExecutionPause,
    )
    from upsonic.tools.wrappers import (
        FunctionTool,
        AgentTool,
    )
    from upsonic.tools.orchestration import (
        PlanStep,
        AnalysisResult,
        Thought,
        ExecutionResult,
        plan_and_execute,
        Orchestrator,
    )
    from upsonic.tools.deferred import (
        ExternalToolCall,
        DeferredExecutionManager,
    )
    from upsonic.tools.mcp import (
        MCPTool,
        MCPHandler,
    )
    from upsonic.tools.builtin_tools import (
        AbstractBuiltinTool,
        WebSearchTool,
        WebSearchUserLocation,
        CodeExecutionTool,
        UrlContextTool,
        WebSearch,
        WebRead,
    )

def _get_base_classes() -> Dict[str, Any]:
    """Lazy import of base classes."""
    from upsonic.tools.base import (
        Tool,
        ToolKit,
        ToolDefinition,
        ToolResult,
        ToolMetadata,
        DocstringFormat,
        ObjectJsonSchema,
    )
    
    return {
        'Tool': Tool,
        'ToolKit': ToolKit,
        'ToolDefinition': ToolDefinition,
        'ToolResult': ToolResult,
        'ToolMetadata': ToolMetadata,
        'DocstringFormat': DocstringFormat,
        'ObjectJsonSchema': ObjectJsonSchema,
    }

def _get_config_classes() -> Dict[str, Any]:
    """Lazy import of config classes."""
    from upsonic.tools.config import (
        tool,
        ToolConfig,
        ToolHooks,
    )
    
    return {
        'tool': tool,
        'ToolConfig': ToolConfig,
        'ToolHooks': ToolHooks,
    }

def _get_metrics_classes() -> Dict[str, Any]:
    """Lazy import of metrics classes."""
    from upsonic.tools.metrics import (
        ToolMetrics,
    )
    
    return {
        'ToolMetrics': ToolMetrics,
    }

def _get_schema_classes() -> Dict[str, Any]:
    """Lazy import of schema classes."""
    from upsonic.tools.schema import (
        FunctionSchema,
        function_schema,
        SchemaGenerationError,
    )
    
    return {
        'FunctionSchema': FunctionSchema,
        'function_schema': function_schema,
        'SchemaGenerationError': SchemaGenerationError,
    }

def _get_processor_classes() -> Dict[str, Any]:
    """Lazy import of processor classes."""
    from upsonic.tools.processor import (
        ToolProcessor,
        ToolValidationError,
        ExternalExecutionPause,
    )
    
    return {
        'ToolProcessor': ToolProcessor,
        'ToolValidationError': ToolValidationError,
        'ExternalExecutionPause': ExternalExecutionPause,
    }

def _get_wrapper_classes() -> Dict[str, Any]:
    """Lazy import of wrapper classes."""
    from upsonic.tools.wrappers import (
        FunctionTool,
        AgentTool,
    )
    
    return {
        'FunctionTool': FunctionTool,
        'AgentTool': AgentTool,
    }

def _get_orchestration_classes() -> Dict[str, Any]:
    """Lazy import of orchestration classes."""
    from upsonic.tools.orchestration import (
        PlanStep,
        AnalysisResult,
        Thought,
        ExecutionResult,
        plan_and_execute,
        Orchestrator,
    )
    
    return {
        'PlanStep': PlanStep,
        'AnalysisResult': AnalysisResult,
        'Thought': Thought,
        'ExecutionResult': ExecutionResult,
        'plan_and_execute': plan_and_execute,
        'Orchestrator': Orchestrator,
    }

def _get_deferred_classes() -> Dict[str, Any]:
    """Lazy import of deferred classes."""
    from upsonic.tools.deferred import (
        ExternalToolCall,
        DeferredExecutionManager,
    )
    
    return {
        'ExternalToolCall': ExternalToolCall,
        'DeferredExecutionManager': DeferredExecutionManager,
    }

def _get_mcp_classes() -> Dict[str, Any]:
    """Lazy import of MCP classes."""
    from upsonic.tools.mcp import (
        MCPTool,
        MCPHandler,
        MultiMCPHandler,
        SSEClientParams,
        StreamableHTTPClientParams,
        prepare_command,
    )
    
    return {
        'MCPTool': MCPTool,
        'MCPHandler': MCPHandler,
        'MultiMCPHandler': MultiMCPHandler,
        'SSEClientParams': SSEClientParams,
        'StreamableHTTPClientParams': StreamableHTTPClientParams,
        'prepare_command': prepare_command,
    }

def _get_builtin_classes() -> Dict[str, Any]:
    """Lazy import of builtin classes."""
    from upsonic.tools.builtin_tools import (
        AbstractBuiltinTool,
        WebSearchTool,
        WebSearchUserLocation,
        CodeExecutionTool,
        UrlContextTool,
        WebSearch,
        WebRead,
    )
    
    return {
        'AbstractBuiltinTool': AbstractBuiltinTool,
        'WebSearchTool': WebSearchTool,
        'WebSearchUserLocation': WebSearchUserLocation,
        'CodeExecutionTool': CodeExecutionTool,
        'UrlContextTool': UrlContextTool,
        'WebSearch': WebSearch,
        'WebRead': WebRead,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    # Base classes
    base_classes = _get_base_classes()
    if name in base_classes:
        return base_classes[name]
    
    # Config classes
    config_classes = _get_config_classes()
    if name in config_classes:
        return config_classes[name]
    
    # Metrics classes
    metrics_classes = _get_metrics_classes()
    if name in metrics_classes:
        return metrics_classes[name]
    
    # Schema classes
    schema_classes = _get_schema_classes()
    if name in schema_classes:
        return schema_classes[name]
    
    # Processor classes
    processor_classes = _get_processor_classes()
    if name in processor_classes:
        return processor_classes[name]
    
    # Wrapper classes
    wrapper_classes = _get_wrapper_classes()
    if name in wrapper_classes:
        return wrapper_classes[name]
    
    # Orchestration classes
    orchestration_classes = _get_orchestration_classes()
    if name in orchestration_classes:
        return orchestration_classes[name]
    
    # Deferred classes
    deferred_classes = _get_deferred_classes()
    if name in deferred_classes:
        return deferred_classes[name]
    
    # MCP classes
    mcp_classes = _get_mcp_classes()
    if name in mcp_classes:
        return mcp_classes[name]
    
    # Builtin classes
    builtin_classes = _get_builtin_classes()
    if name in builtin_classes:
        return builtin_classes[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


class ToolManager:
    """High-level manager for all tool operations."""
    
    def __init__(self):
        from upsonic.tools.processor import ToolProcessor
        from upsonic.tools.deferred import DeferredExecutionManager
        
        self.processor = ToolProcessor()
        self.deferred_manager = DeferredExecutionManager()
        self.orchestrator = None
        self.wrapped_tools = {}
        self.current_task = None
        
    def register_tools(
        self,
        tools: list,
        task: Optional['Task'] = None,
        agent_instance: Optional[Any] = None
    ) -> Dict[str, Tool]:
        """
        Register tools with the manager.
        
        This is the MAIN registration method that:
        1. Filters out already registered tools (no re-processing)
        2. Processes new tools via ToolProcessor
        3. Creates behavioral wrappers
        4. Handles plan_and_execute orchestrator
        
        Args:
            tools: List of tools to register
            task: Optional task context
            agent_instance: Optional agent instance for orchestrator
            
        Returns:
            Dict of newly registered tools
        """
        self.current_task = task
        
        if not tools:
            return {}
        
        # Use processor's register_tools (which filters out already registered tools)
        newly_registered = self.processor.register_tools(tools)
        
        # Create wrappers for newly registered tools
        for name, tool in newly_registered.items():
            if name != 'plan_and_execute':
                self.wrapped_tools[name] = self.processor.create_behavioral_wrapper(tool)
        
        # Handle plan_and_execute if it was newly registered
        if 'plan_and_execute' in newly_registered:
            if agent_instance and agent_instance.enable_thinking_tool:
                # Update or create orchestrator
                if self.orchestrator:
                    # Update existing orchestrator with new tools
                    self.orchestrator.wrapped_tools = self.wrapped_tools
                    self.orchestrator.all_tools = {
                        name: func 
                        for name, func in self.wrapped_tools.items()
                        if name != 'plan_and_execute'
                    }
                    # Update task reference if provided (critical for task-level tool registration)
                    if task:
                        self.orchestrator.task = task
                        self.orchestrator.original_user_request = task.description
                else:
                    # Create new orchestrator
                    from upsonic.tools.orchestration import Orchestrator
                    self.orchestrator = Orchestrator(
                        agent_instance=agent_instance,
                        task=task,
                        wrapped_tools=self.wrapped_tools
                    )
                
                async def orchestrator_executor(thought) -> Any:
                    return await self.orchestrator.execute(thought)
                self.wrapped_tools['plan_and_execute'] = orchestrator_executor
            else:
                # Create regular wrapper
                self.wrapped_tools['plan_and_execute'] = self.processor.create_behavioral_wrapper(
                    newly_registered['plan_and_execute']
                )
        
        # Update existing orchestrator with task context if:
        # 1. Orchestrator exists (plan_and_execute was previously registered)
        # 2. Task is provided (task-level tool registration)
        # 3. plan_and_execute was not just newly registered (already handled above)
        elif self.orchestrator and task and 'plan_and_execute' not in newly_registered:
            self.orchestrator.task = task
            self.orchestrator.original_user_request = task.description
            # Also update tools in case new tools were added
            self.orchestrator.wrapped_tools = self.wrapped_tools
            self.orchestrator.all_tools = {
                name: func 
                for name, func in self.wrapped_tools.items()
                if name != 'plan_and_execute'
            }
        
        return newly_registered
    
    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        metrics: Optional['ToolMetrics'] = None,
        tool_call_id: Optional[str] = None
    ) -> ToolResult:
        """Execute a tool by name using pre-wrapped executor."""
        wrapped = self.wrapped_tools.get(tool_name)
        if not wrapped:
            raise ValueError(f"Tool '{tool_name}' not found or not wrapped")
        
        if not tool_call_id:
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
        
        try:
            start_time = time.time()
            
            if tool_name == 'plan_and_execute':
                from upsonic.tools.orchestration import Thought
                
                # Handle both wrapped and unwrapped thought data
                if 'thought' in args:
                    thought_data = args['thought']
                    if isinstance(thought_data, dict):
                        thought = Thought(**thought_data)
                    else:
                        thought = thought_data
                else:
                    # LLM sent fields directly (reasoning, plan, criticism, action)
                    thought = Thought(**args)
                
                result = await wrapped(thought)
            else:
                result = await wrapped(**args)
                
            execution_time = time.time() - start_time
            
            from upsonic.tools.base import ToolResult
            return ToolResult(
                tool_name=tool_name,
                content=result,
                tool_call_id=tool_call_id,
                success=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            from upsonic.tools.processor import ExternalExecutionPause
            if isinstance(e, ExternalExecutionPause):
                external_call = self.deferred_manager.create_external_call(
                    tool_name=tool_name,
                    args=args,
                    tool_call_id=tool_call_id
                )
                e.external_call = external_call
                raise e
            
            from upsonic.tools.base import ToolResult
            return ToolResult(
                tool_name=tool_name,
                content=str(e),
                tool_call_id=tool_call_id,
        success=False,
                error=str(e)
            )
    
    def get_tool_definitions(self) -> List['ToolDefinition']:
        """Get definitions for all registered tools."""
        from upsonic.tools.base import ToolDefinition
        
        definitions = []
        for tool in self.processor.registered_tools.values():
            config = getattr(tool, 'config', None)
            
            # Get JSON schema from tool.schema
            if tool.schema:
                json_schema = tool.schema.json_schema
            else:
                # Fallback if schema is not set
                json_schema = {'type': 'object', 'properties': {}}
            
            sequential = config.sequential if config else False
            
            definition = ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=json_schema,
                kind=tool.metadata.kind if hasattr(tool, 'metadata') else 'function',
                strict=tool.metadata.strict if hasattr(tool, 'metadata') else False,
                sequential=sequential,
                metadata=tool.metadata if tool.metadata else None
            )
            definitions.append(definition)
        return definitions
    
    def remove_tools(
        self,
        tools: Union[Any, List[Any]],
        registered_tools: Dict[str, Any]
    ) -> tuple[List[str], List[Any]]:
        """
        Remove tools from the manager.
        
        This is the MAIN removal method that handles ALL tool types:
        - Tool names (strings)
        - Function objects
        - Agent objects
        - MCP handlers
        - Class instances (ToolKit or regular classes)
        
        Args:
            tools: Single tool or list of tools to remove (any type)
            registered_tools: Agent's registered tools for reference
            
        Returns:
            Tuple of (removed_tool_names, removed_original_objects)
            - removed_tool_names: List of tool names that were removed
            - removed_original_objects: List of original objects to remove from agent.tools
        """
        from upsonic.tools.mcp import MCPHandler, MultiMCPHandler
        from upsonic.tools.base import ToolKit
        import inspect
        
        if not isinstance(tools, list):
            tools = [tools]
        
        if not tools:
            return ([], [])
        
        # Categorize tools by type
        mcp_handlers = []
        class_instances = []
        tool_names_to_remove = []
        original_objects_to_remove = set()
        
        for tool_identifier in tools:
            # String - it's a tool name
            if isinstance(tool_identifier, str):
                tool_names_to_remove.append(tool_identifier)
                
                # Find the original object for this tool name
                # ONLY add to original_objects_to_remove for 1:1 relationships (functions, agents)
                # NOT for 1:many relationships (MCP handlers, ToolKits, class instances)
                if tool_identifier in registered_tools:
                    registered_tool = registered_tools[tool_identifier]
                    
                    # Check if this tool comes from a 1:many relationship
                    is_one_to_many = False
                    
                    # MCP handler (1:many) - check if handler is tracked
                    if hasattr(registered_tool, 'handler'):
                        handler_id = id(registered_tool.handler)
                        if handler_id in self.processor.mcp_handler_to_tools:
                            handler_tools = self.processor.mcp_handler_to_tools[handler_id]
                            if len(handler_tools) > 1:  # Handler has multiple tools
                                is_one_to_many = True
                    
                    # ToolKit or class instance (1:many) - check if instance is tracked
                    elif hasattr(registered_tool, 'function') and hasattr(registered_tool.function, '__self__'):
                        # This is a method (has __self__), check if instance is tracked
                        instance = registered_tool.function.__self__
                        instance_id = id(instance)
                        if instance_id in self.processor.class_instance_to_tools:
                            instance_tools = self.processor.class_instance_to_tools[instance_id]
                            if len(instance_tools) > 1:  # Instance has multiple tools
                                is_one_to_many = True
                    
                    # Only add to original_objects_to_remove if it's a 1:1 relationship
                    if not is_one_to_many:
                        if hasattr(registered_tool, 'agent'):
                            original_objects_to_remove.add(registered_tool.agent)
                        elif hasattr(registered_tool, 'function'):
                            original_objects_to_remove.add(registered_tool.function)
                        # Note: Don't add handler here - it's handled above in the 1:many check
            
            # MCP Handler
            elif isinstance(tool_identifier, (MCPHandler, MultiMCPHandler)):
                mcp_handlers.append(tool_identifier)
                original_objects_to_remove.add(tool_identifier)
            
            # ToolKit instance
            elif isinstance(tool_identifier, ToolKit):
                class_instances.append(tool_identifier)
                original_objects_to_remove.add(tool_identifier)
            
            # Class (not instance)
            elif inspect.isclass(tool_identifier):
                # Find instance of this class in processor tracking
                for instance_id, tool_list in self.processor.class_instance_to_tools.items():
                    # This is a bit tricky - we need to find the actual instance
                    # Skip for now, user should pass instance
                    pass
            
            # Regular class instance or other object
            else:
                instance_id = id(tool_identifier)
                
                # Check if it's a tracked class instance
                if instance_id in self.processor.class_instance_to_tools:
                    class_instances.append(tool_identifier)
                    original_objects_to_remove.add(tool_identifier)
                else:
                    # It might be a function, agent, or wrapped tool
                    found = False
                    for name, registered_tool in registered_tools.items():
                        # Direct match
                        if registered_tool is tool_identifier or id(registered_tool) == id(tool_identifier):
                            tool_names_to_remove.append(name)
                            found = True
                            break
                        
                        # Agent match
                        if hasattr(registered_tool, 'agent') and (registered_tool.agent is tool_identifier or id(registered_tool.agent) == id(tool_identifier)):
                            tool_names_to_remove.append(name)
                            original_objects_to_remove.add(tool_identifier)
                            found = True
                            break
                        
                        # Function match
                        if hasattr(registered_tool, 'function') and (registered_tool.function is tool_identifier or id(registered_tool.function) == id(tool_identifier)):
                            tool_names_to_remove.append(name)
                            original_objects_to_remove.add(tool_identifier)
                            found = True
                            break
                        
                        # Handler match
                        if hasattr(registered_tool, 'handler') and (registered_tool.handler is tool_identifier or id(registered_tool.handler) == id(tool_identifier)):
                            tool_names_to_remove.append(name)
                            original_objects_to_remove.add(tool_identifier)
                            found = True
                            break
                    
                    if not found and hasattr(tool_identifier, 'name'):
                        tool_names_to_remove.append(tool_identifier.name)
                    elif not found and hasattr(tool_identifier, '__name__'):
                        tool_names_to_remove.append(tool_identifier.__name__)
        
        # Now remove everything
        all_removed_names = set(tool_names_to_remove)
        
        # Remove MCP handlers
        if mcp_handlers:
            removed_names = self.processor.unregister_mcp_handlers(mcp_handlers)
            all_removed_names.update(removed_names)
        
        # Remove class instances
        if class_instances:
            removed_names = self.processor.unregister_class_instances(class_instances)
            all_removed_names.update(removed_names)
        
        # Remove individual tools
        if tool_names_to_remove:
            self.processor.unregister_tools(list(set(tool_names_to_remove)))
        
        # Remove from wrapped_tools
        for tool_name in all_removed_names:
            if tool_name in self.wrapped_tools:
                del self.wrapped_tools[tool_name]
        
        # Update orchestrator
        if self.orchestrator and 'plan_and_execute' in all_removed_names:
            self.orchestrator = None
        elif self.orchestrator:
            self.orchestrator.wrapped_tools = self.wrapped_tools
            self.orchestrator.all_tools = {
                name: func 
                for name, func in self.wrapped_tools.items()
                if name != 'plan_and_execute'
            }
        
        return (list(all_removed_names), list(original_objects_to_remove))


__all__ = [
    'Tool',
    'ToolKit',
    'ToolDefinition',
    'ToolResult',
    'ToolMetadata',
    'DocstringFormat',
    'ObjectJsonSchema',
    
    'tool',
    'ToolConfig',
    'ToolHooks',
    
    'ToolMetrics',
    
    'FunctionSchema',
    'function_schema',
    'SchemaGenerationError',
    
    'ToolProcessor',
    'ToolValidationError',
    'ExternalExecutionPause',
    
    'FunctionTool',
    'AgentTool',
    
    
    'PlanStep',
    'AnalysisResult',
    'Thought',
    'ExecutionResult',
    'plan_and_execute',
    'Orchestrator',
    
    'ExternalToolCall',
    'DeferredExecutionManager',
    
    'MCPTool',
    'MCPHandler',
    'MultiMCPHandler',
    'SSEClientParams',
    'StreamableHTTPClientParams',
    'prepare_command',
    
    'ToolManager',
    
    'AbstractBuiltinTool',
    'WebSearchTool',
    'WebSearchUserLocation',
    'CodeExecutionTool',
    'UrlContextTool',
    'WebSearch',
    'WebRead',
]