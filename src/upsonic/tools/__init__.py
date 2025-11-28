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
from typing import Any, Dict, List, Optional, TYPE_CHECKING

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
        """Register a list of tools and create appropriate wrappers."""
        self.current_task = task
        
        # Track registered tool objects by their identity (object-level comparison)
        registered_tool_objects = set(id(t) for t in self.processor.registered_tools.values())
        
        # Filter out already registered tools (same object instance)
        tools_to_register = []
        for tool in tools:
            if tool is None:
                continue
            # Check if this exact object is already registered
            if id(tool) not in registered_tool_objects:
                tools_to_register.append(tool)
        
        # Process remaining tools
        registered_tools = self.processor.process_tools(tools_to_register)
        
        for name, tool in registered_tools.items():
            if name != 'plan_and_execute':
                self.wrapped_tools[name] = self.processor.create_behavioral_wrapper(tool)
        
        if 'plan_and_execute' in registered_tools and agent_instance and agent_instance.enable_thinking_tool:
            if not self.orchestrator and agent_instance:
                from upsonic.tools.orchestration import Orchestrator
                self.orchestrator = Orchestrator(
                    agent_instance=agent_instance,
                    task=task,
                    wrapped_tools=self.wrapped_tools
                )
            async def orchestrator_executor(thought) -> Any:
                return await self.orchestrator.execute(thought)
            self.wrapped_tools['plan_and_execute'] = orchestrator_executor
        elif 'plan_and_execute' in registered_tools:
            self.wrapped_tools ['plan_and_execute'] = self.processor.create_behavioral_wrapper(
                registered_tools['plan_and_execute']
            )
        
        return registered_tools
    
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
            
            if tool_name == 'plan_and_execute' and 'thought' in args:
                from upsonic.tools.orchestration import Thought
                thought_data = args['thought']
                if isinstance(thought_data, dict):
                    thought = Thought(**thought_data)
                else:
                    thought = thought_data
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