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
        ToolBase,
        ToolKit,
        ToolDefinition,
        ToolCall,
        ToolResult,
        ToolMetadata,
        ToolSchema,
        DocstringFormat,
        ObjectJsonSchema,
    )
    from upsonic.tools.config import (
        tool,
        ToolConfig,
        ToolHooks,
    )
    from upsonic.tools.context import (
        ToolContext,
        AgentDepsT,
    )
    from upsonic.tools.schema import (
        FunctionSchema,
        generate_function_schema,
        validate_tool_function,
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
        MethodTool,
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
        DeferredToolRequests,
        DeferredToolResults,
        ToolApproval,
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

def _get_base_classes():
    """Lazy import of base classes."""
    from upsonic.tools.base import (
        Tool,
        ToolBase,
        ToolKit,
        ToolDefinition,
        ToolCall,
        ToolResult,
        ToolMetadata,
        ToolSchema,
        DocstringFormat,
        ObjectJsonSchema,
    )
    
    return {
        'Tool': Tool,
        'ToolBase': ToolBase,
        'ToolKit': ToolKit,
        'ToolDefinition': ToolDefinition,
        'ToolCall': ToolCall,
        'ToolResult': ToolResult,
        'ToolMetadata': ToolMetadata,
        'ToolSchema': ToolSchema,
        'DocstringFormat': DocstringFormat,
        'ObjectJsonSchema': ObjectJsonSchema,
    }

def _get_config_classes():
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

def _get_context_classes():
    """Lazy import of context classes."""
    from upsonic.tools.context import (
        ToolContext,
        AgentDepsT,
    )
    
    return {
        'ToolContext': ToolContext,
        'AgentDepsT': AgentDepsT,
    }

def _get_schema_classes():
    """Lazy import of schema classes."""
    from upsonic.tools.schema import (
        FunctionSchema,
        generate_function_schema,
        validate_tool_function,
        SchemaGenerationError,
    )
    
    return {
        'FunctionSchema': FunctionSchema,
        'generate_function_schema': generate_function_schema,
        'validate_tool_function': validate_tool_function,
        'SchemaGenerationError': SchemaGenerationError,
    }

def _get_processor_classes():
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

def _get_wrapper_classes():
    """Lazy import of wrapper classes."""
    from upsonic.tools.wrappers import (
        FunctionTool,
        AgentTool,
        MethodTool,
    )
    
    return {
        'FunctionTool': FunctionTool,
        'AgentTool': AgentTool,
        'MethodTool': MethodTool,
    }

def _get_orchestration_classes():
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

def _get_deferred_classes():
    """Lazy import of deferred classes."""
    from upsonic.tools.deferred import (
        ExternalToolCall,
        DeferredToolRequests,
        DeferredToolResults,
        ToolApproval,
        DeferredExecutionManager,
    )
    
    return {
        'ExternalToolCall': ExternalToolCall,
        'DeferredToolRequests': DeferredToolRequests,
        'DeferredToolResults': DeferredToolResults,
        'ToolApproval': ToolApproval,
        'DeferredExecutionManager': DeferredExecutionManager,
    }

def _get_mcp_classes():
    """Lazy import of MCP classes."""
    from upsonic.tools.mcp import (
        MCPTool,
        MCPHandler,
    )
    
    return {
        'MCPTool': MCPTool,
        'MCPHandler': MCPHandler,
    }

def _get_builtin_classes():
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
    
    # Context classes
    context_classes = _get_context_classes()
    if name in context_classes:
        return context_classes[name]
    
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
        context: Optional[ToolContext] = None,
        task: Optional['Task'] = None,
        agent_instance: Optional[Any] = None
    ) -> Dict[str, Tool]:
        """Register a list of tools and create appropriate wrappers."""
        self.current_task = task
        
        registered_tools = self.processor.process_tools(tools, context)
        
        for name, tool in registered_tools.items():
            if name != 'plan_and_execute':
                if context is None:
                    from upsonic.tools.context import ToolContext
                    context = ToolContext(deps=None)
                self.wrapped_tools[name] = self.processor.create_behavioral_wrapper(
                    tool, context
                )
        
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
            if context is None:
                from upsonic.tools.context import ToolContext
                context = ToolContext(deps=None)
            self.wrapped_tools['plan_and_execute'] = self.processor.create_behavioral_wrapper(
                registered_tools['plan_and_execute'], 
                context
            )
        
        return registered_tools
    
    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[ToolContext] = None,
        tool_call_id: Optional[str] = None
    ) -> ToolResult:
        """Execute a tool by name using pre-wrapped executor."""
        wrapped = self.wrapped_tools.get(tool_name)
        if not wrapped:
            raise ValueError(f"Tool '{tool_name}' not found or not wrapped")
        
        if context:
            self.processor.current_context = context
        
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
                    tool_call_id=tool_call_id,
                    requires_approval=False
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
            strict = config.strict if config and config.strict is not None else tool.schema.strict
            sequential = config.sequential if config else False
            
            definition = ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool.schema.json_schema,
                kind='function',
                strict=strict,
                sequential=sequential,
                metadata=tool.metadata.custom if hasattr(tool, 'metadata') else None
            )
            definitions.append(definition)
        return definitions
    
    def has_deferred_requests(self) -> bool:
        """Check if there are pending deferred requests."""
        return self.deferred_manager.has_pending_requests()
    
    def get_deferred_requests(self) -> 'DeferredToolRequests':
        """Get pending deferred requests."""
        return self.deferred_manager.get_pending_requests()
    
    def process_deferred_results(
        self,
        results: 'DeferredToolResults'
    ) -> List['ToolResult']:
        """Process results from deferred execution."""
        return self.deferred_manager.process_results(results)


__all__ = [
    'Tool',
    'ToolBase', 
    'ToolKit',
    'ToolDefinition',
    'ToolCall',
    'ToolResult',
    'ToolMetadata',
    'ToolSchema',
    'DocstringFormat',
    'ObjectJsonSchema',
    
    'tool',
    'ToolConfig',
    'ToolHooks',
    
    'ToolContext',
    'AgentDepsT',
    
    'FunctionSchema',
    'generate_function_schema',
    'validate_tool_function',
    'SchemaGenerationError',
    
    'ToolProcessor',
    'ToolValidationError',
    'ExternalExecutionPause',
    
    'FunctionTool',
    'AgentTool',
    'MethodTool',
    
    
    'PlanStep',
    'AnalysisResult',
    'Thought',
    'ExecutionResult',
    'plan_and_execute',
    'Orchestrator',
    
    'ExternalToolCall',
    'DeferredToolRequests',
    'DeferredToolResults',
    'ToolApproval',
    'DeferredExecutionManager',
    
    'MCPTool',
    'MCPHandler',
    
    'ToolManager',
    
    'AbstractBuiltinTool',
    'WebSearchTool',
    'WebSearchUserLocation',
    'CodeExecutionTool',
    'UrlContextTool',
    'WebSearch',
    'WebRead',
]