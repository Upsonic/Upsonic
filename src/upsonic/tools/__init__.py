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
)


class ToolManager:
    """High-level manager for all tool operations."""
    
    def __init__(self):
        self.processor = ToolProcessor()
        self.deferred_manager = DeferredExecutionManager()
        self.orchestrator = None
        self.wrapped_tools = {}  # Cache for wrapped tools
        self.current_task = None  # Current task context
        
    def register_tools(
        self,
        tools: list,
        context: Optional[ToolContext] = None,
        task: Optional['Task'] = None,
        agent_instance: Optional[Any] = None
    ) -> Dict[str, Tool]:
        """Register a list of tools and create appropriate wrappers."""
        # Store current task
        self.current_task = task
        
        # Process raw tools
        registered_tools = self.processor.process_tools(tools, context)
        
        # First pass: Create behavioral wrappers for all non-orchestrator tools
        for name, tool in registered_tools.items():
            if name != 'plan_and_execute':
                # Create behavioral wrapper
                self.wrapped_tools[name] = self.processor.create_behavioral_wrapper(
                    tool, context or ToolContext(deps=None)
                )
        
        # Second pass: Handle orchestrator tool
        if 'plan_and_execute' in registered_tools and agent_instance and agent_instance.enable_thinking_tool:
            # Create Orchestrator instance with wrapped tools
            if not self.orchestrator and agent_instance:
                self.orchestrator = Orchestrator(
                    agent_instance=agent_instance,
                    task=task,
                    wrapped_tools=self.wrapped_tools  # Now contains all other tools
                )
            # Create wrapper that calls orchestrator
            async def orchestrator_executor(thought: Thought) -> Any:
                return await self.orchestrator.execute(thought)
            self.wrapped_tools['plan_and_execute'] = orchestrator_executor
        elif 'plan_and_execute' in registered_tools:
            # If thinking not enabled, wrap normally
            self.wrapped_tools['plan_and_execute'] = self.processor.create_behavioral_wrapper(
                registered_tools['plan_and_execute'], 
                context or ToolContext(deps=None)
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
        # Get the pre-wrapped tool
        wrapped = self.wrapped_tools.get(tool_name)
        if not wrapped:
            raise ValueError(f"Tool '{tool_name}' not found or not wrapped")
        
        # Update context if provided
        if context:
            # Store context for tools that need it
            self.processor.current_context = context
        
        # Use provided tool_call_id or generate one
        if not tool_call_id:
            tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
        
        try:
            # Execute using pre-wrapped tool
            start_time = time.time()
            
            # Special handling for plan_and_execute
            if tool_name == 'plan_and_execute' and 'thought' in args:
                # Convert dict to Thought object if needed
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
            
            return ToolResult(
                tool_name=tool_name,
                content=result,
                tool_call_id=tool_call_id,
                success=True,
                execution_time=execution_time
            )
            
        except ExternalExecutionPause as e:
            # Handle external execution
            external_call = self.deferred_manager.create_external_call(
                tool_name=tool_name,
                args=args,
                tool_call_id=tool_call_id,
                requires_approval=False
            )
            # Re-raise with the external call attached
            e.external_call = external_call
            raise e
            
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                content=str(e),
                tool_call_id=tool_call_id,
                success=False,
                error=str(e)
            )
    
    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Get definitions for all registered tools."""
        definitions = []
        for tool in self.processor.registered_tools.values():
            # Get strict and sequential flags from tool config
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
    
    def get_deferred_requests(self) -> DeferredToolRequests:
        """Get pending deferred requests."""
        return self.deferred_manager.get_pending_requests()
    
    def process_deferred_results(
        self,
        results: DeferredToolResults
    ) -> List[ToolResult]:
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
]
