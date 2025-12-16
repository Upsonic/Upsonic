"""
Concrete Step Implementations

This module contains all the concrete step implementations for the agent pipeline.
Each step handles a specific part of the agent execution flow. All steps must execute;
there is no skipping. If any error occurs, it's raised immediately to the user.

Steps emit events for streaming visibility, allowing users to track execution progress
and state changes in real-time.
"""

import asyncio
import time
from typing import Any, Optional, AsyncIterator, List
from .step import Step, StepResult, StepStatus
from .context import StepContext


class InitializationStep(Step):
    """Initialize agent state for execution."""
    
    @property
    def name(self) -> str:
        return "initialization"
    
    @property
    def description(self) -> str:
        return "Initialize agent for execution"
    
    def get_step_events(self, context: StepContext, result: StepResult) -> List[Any]:
        """Emit agent initialization event."""
        from upsonic.agent.events import AgentInitializedEvent
        
        agent_id = "unknown"
        if hasattr(context.agent, 'get_agent_id'):
            agent_id = context.agent.get_agent_id()
        
        return [AgentInitializedEvent(
            agent_id=agent_id,
            is_streaming=context.is_streaming
        )]
    
    async def execute(self, context: StepContext) -> StepResult:
        """Initialize agent state for new execution."""
        from upsonic.utils.printing import agent_started

        # Start task timing
        context.task.task_start(context.agent)

        agent_started(context.agent.get_agent_id())

        context.agent._tool_call_count = 0
        context.agent.current_task = context.task

        # Initialize appropriate result container based on mode
        if context.is_streaming:
            context.agent._stream_run_result.start_new_run()
        else:
            context.agent._run_result.start_new_run()

        return StepResult(
            status=StepStatus.SUCCESS,
            message="Agent initialized",
            execution_time=0.0
        )


class CacheCheckStep(Step):
    """Check if there's a cached response for the task."""
    
    @property
    def name(self) -> str:
        return "cache_check"
    
    @property
    def description(self) -> str:
        return "Check for cached responses"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Check cache for existing response."""
        from upsonic.agent.events import CacheCheckEvent, CacheHitEvent, CacheMissEvent
        
        if not context.task.enable_cache or context.task.is_paused:
            # Emit event for cache disabled
            if context.is_streaming:
                self._emit_event(CacheCheckEvent(
                    cache_enabled=False
                ), context)
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Caching not enabled or task paused",
                execution_time=0.0
            )
        
        # Set cache manager
        context.task.set_cache_manager(context.agent._cache_manager)
        
        if context.agent.debug:
            from upsonic.utils.printing import cache_configuration
            embedding_provider_name = None
            if context.task.cache_embedding_provider:
                embedding_provider_name = getattr(
                    context.task.cache_embedding_provider, 'model_name', 'Unknown'
                )
            
            cache_configuration(
                enable_cache=context.task.enable_cache,
                cache_method=context.task.cache_method,
                cache_threshold=context.task.cache_threshold if context.task.cache_method == "vector_search" else None,
                cache_duration_minutes=context.task.cache_duration_minutes,
                embedding_provider=embedding_provider_name
            )
        
        input_text = context.task._original_input or context.task.description
        input_preview = input_text[:100] if input_text else None
        cached_response = await context.task.get_cached_response(input_text, context.model)
        
        if cached_response is not None:
            similarity = None
            if hasattr(context.task, '_last_cache_entry') and 'similarity' in context.task._last_cache_entry:
                similarity = context.task._last_cache_entry['similarity']
            
            from upsonic.utils.printing import cache_hit
            cache_hit(
                cache_method=context.task.cache_method,
                similarity=similarity,
                input_preview=(context.task._original_input or context.task.description)[:100] 
                    if (context.task._original_input or context.task.description) else None
            )
            
            context.final_output = cached_response
            context.task._response = cached_response
            context.task.task_end()
            context.agent._run_result.output = cached_response
            
            # Early return flag - we'll handle this specially
            context.task._cached_result = True
            
            # Emit cache hit events
            if context.is_streaming:
                cached_preview = str(cached_response)[:100] if cached_response else None
                self._emit_event(CacheCheckEvent(
                    cache_enabled=True,
                    cache_method=context.task.cache_method,
                    cache_hit=True,
                    similarity=similarity,
                    input_preview=input_preview
                ), context)
                self._emit_event(CacheHitEvent(
                    cache_method=context.task.cache_method,
                    similarity=similarity,
                    cached_response_preview=cached_preview
                ), context)
            
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Cache hit - using cached response",
                execution_time=0.0
            )
        else:
            from upsonic.utils.printing import cache_miss
            cache_miss(
                cache_method=context.task.cache_method,
                input_preview=(context.task._original_input or context.task.description)[:100] 
                    if (context.task._original_input or context.task.description) else None
            )
            
            # Emit cache miss events
            if context.is_streaming:
                self._emit_event(CacheCheckEvent(
                    cache_enabled=True,
                    cache_method=context.task.cache_method,
                    cache_hit=False,
                    input_preview=input_preview
                ), context)
                self._emit_event(CacheMissEvent(
                    cache_method=context.task.cache_method,
                    reason="No matching cache entry found"
                ), context)
            
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Cache miss - will execute normally",
                execution_time=0.0
            )


class UserPolicyStep(Step):
    """Apply user policy to the task input."""
    
    @property
    def name(self) -> str:
        return "user_policy"
    
    @property
    def description(self) -> str:
        return "Apply user input safety policy"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Apply user policy to task input."""
        from upsonic.agent.events import PolicyCheckEvent
        
        if not context.agent.user_policy_manager.has_policies() or not context.task.description or context.task.is_paused:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="No user policy or task paused",
                execution_time=0.0
            )
        
        # Skip if we have cached result
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        
        policy_count = len(context.agent.user_policy_manager._policies) if hasattr(context.agent.user_policy_manager, '_policies') else 0
        original_content = context.task.description
        
        # Use the agent's _apply_user_policy method (consistent with AgentPolicyStep)
        processed_task, should_continue = await context.agent._apply_user_policy(context.task)
        context.task = processed_task
        
        if not should_continue:
            # Policy blocked the content
            context.final_output = context.task._response
            context.agent._run_result.output = context.final_output
            context.task._policy_blocked = True
            
            if context.is_streaming:
                self._emit_event(PolicyCheckEvent(
                    policy_type='user_policy',
                    action='block',
                    policies_checked=policy_count,
                    content_modified=False,
                    blocked_reason="Content blocked by user policy"
                ), context)
            
            return StepResult(
                status=StepStatus.SUCCESS,
                message="User input blocked by policy",
                execution_time=0.0
            )
        elif context.task.description != original_content:
            # Content was modified (REPLACE/ANONYMIZE)
            if context.is_streaming:
                self._emit_event(PolicyCheckEvent(
                    policy_type='user_policy',
                    action='modify',
                    policies_checked=policy_count,
                    content_modified=True
                ), context)
            
            return StepResult(
                status=StepStatus.SUCCESS,
                message="User input modified by policy",
                execution_time=0.0
            )
        
        # Content passed without modification
        if context.is_streaming:
            self._emit_event(PolicyCheckEvent(
                policy_type='user_policy',
                action='pass',
                policies_checked=policy_count,
                content_modified=False
            ), context)
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message="User policies passed",
            execution_time=0.0
        )


class ModelSelectionStep(Step):
    """Select the model to use for execution."""
    
    @property
    def name(self) -> str:
        return "model_selection"
    
    @property
    def description(self) -> str:
        return "Select model for execution"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Select the appropriate model."""
        from upsonic.agent.events import ModelSelectedEvent
        
        is_override = context.model is not None
        
        if context.model:
            from upsonic.models import infer_model
            context.model = infer_model(context.model)
        else:
            context.model = context.agent.model
        
        # Get provider name if available
        provider = getattr(context.model, 'system', None) or getattr(context.model, '_provider', None)
        provider_name = str(provider.name) if hasattr(provider, 'name') else str(provider) if provider else None
        
        if context.is_streaming:
            self._emit_event(ModelSelectedEvent(
                model_name=context.model.model_name,
                provider=provider_name,
                is_override=is_override
            ), context)
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message=f"Selected model: {context.model.model_name}",
            execution_time=0.0
        )


class ValidationStep(Step):
    """Validate task attachments and other requirements."""
    
    @property
    def name(self) -> str:
        return "validation"
    
    @property
    def description(self) -> str:
        return "Validate task requirements"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Validate task attachments."""
        from upsonic.utils.validators import validate_attachments_exist
        
        validate_attachments_exist(context.task)
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Validation passed",
            execution_time=0.0
        )


class ToolSetupStep(Step):
    """Setup tools for the task execution."""
    
    @property
    def name(self) -> str:
        return "tool_setup"
    
    @property
    def description(self) -> str:
        return "Setup tools for execution"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Setup tools for the task."""
        from upsonic.agent.events import ToolsConfiguredEvent
        
        # Setup task-specific tools
        context.agent._setup_task_tools(context.task)
        
        # Set current task on PlanningToolKit if it exists (for write_todos)
        if hasattr(context.agent, '_planning_toolkit') and context.agent._planning_toolkit:
            context.agent._planning_toolkit.set_current_task(context.task)
        
        # Get tool information for event
        tool_names = []
        has_mcp = False
        
        if hasattr(context.agent, '_tool_manager') and context.agent._tool_manager:
            tool_defs = context.agent._tool_manager.get_tool_definitions()
            tool_names = [t.name for t in tool_defs]
            
            # Check for MCP handlers
            from upsonic.tools.mcp import MCPHandler, MultiMCPHandler
            all_tools = context.agent.tools or []
            if context.task and hasattr(context.task, 'tools') and context.task.tools:
                all_tools = list(all_tools) + list(context.task.tools)
            has_mcp = any(isinstance(t, (MCPHandler, MultiMCPHandler)) for t in all_tools)
        
        if context.is_streaming:
            self._emit_event(ToolsConfiguredEvent(
                tool_count=len(tool_names),
                tool_names=tool_names,
                has_mcp_handlers=has_mcp
            ), context)
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Tools configured",
            execution_time=0.0
        )


class StorageConnectionStep(Step):
    """Setup storage connection for memory and database operations."""
    
    @property
    def name(self) -> str:
        return "storage_connection"
    
    @property
    def description(self) -> str:
        return "Setup storage connection"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Setup storage connection context manager."""
        # Storage connection is handled by the agent's _managed_storage_connection
        # which is wrapped around the entire pipeline execution
        # This step is just a marker for tracking
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Storage connection ready",
            execution_time=0.0
        )


class LLMManagerStep(Step):
    """Setup LLM manager for model selection and configuration."""
    
    @property
    def name(self) -> str:
        return "llm_manager"
    
    @property
    def description(self) -> str:
        return "Setup LLM manager"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Setup LLM manager and finalize model selection."""
        from upsonic.agent.context_managers.llm_manager import LLMManager
        
        # Create LLM manager with default and requested model
        llm_manager = LLMManager(
            default_model=context.agent.model,
            requested_model=context.model
        )
        
        # Use the LLM manager context
        async with llm_manager.manage_llm():
            selected_model = llm_manager.get_model()
            # The selected_model is a string identifier, we need to infer it
            if selected_model:
                from upsonic.models import infer_model
                context.model = infer_model(selected_model)
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message=f"LLM manager configured: {context.model.model_name}",
            execution_time=0.0
        )


class MessageBuildStep(Step):
    """Build the model request messages."""
    
    @property
    def name(self) -> str:
        return "message_build"
    
    @property
    def description(self) -> str:
        return "Build model request messages"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Build model request messages with memory manager."""
        from upsonic.agent.events import MessagesBuiltEvent
        
        # Skip if we have cached result or policy blocked
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(context.task, '_policy_blocked') and context.task._policy_blocked:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        
        from upsonic.agent.context_managers import MemoryManager
        
        # If this is a continuation, restore messages instead of building new ones
        if context.is_continuation and context.continuation_messages:
            context.messages = context.continuation_messages
            context.agent._current_messages = context.continuation_messages
            
            if context.is_streaming:
                self._emit_event(MessagesBuiltEvent(
                    message_count=len(context.messages),
                    has_system_prompt=False,
                    has_memory_messages=True,
                    is_continuation=True
                ), context)
            
            return StepResult(
                status=StepStatus.SUCCESS,
                message=f"Restored {len(context.messages)} messages from continuation",
                execution_time=0.0
            )
        
        # Use memory manager with async context
        memory_manager = MemoryManager(context.agent.memory)
        async with memory_manager.manage_memory() as memory_handler:
            messages = await context.agent._build_model_request(
                context.task,
                memory_handler,
                context.state
            )
            context.messages = messages
            context.agent._current_messages = messages
            
            # Store memory handler reference for later steps
            context._memory_handler = memory_handler
            
            # Determine message characteristics
            has_system = False
            has_memory = len(memory_handler.get_message_history()) > 0
            from upsonic.messages import ModelRequest, SystemPromptPart
            if messages:
                first_msg = messages[0]
                if isinstance(first_msg, ModelRequest):
                    has_system = any(isinstance(p, SystemPromptPart) for p in first_msg.parts)
            
            if context.is_streaming:
                self._emit_event(MessagesBuiltEvent(
                    message_count=len(messages),
                    has_system_prompt=has_system,
                    has_memory_messages=has_memory,
                    is_continuation=False
                ), context)
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message=f"Built {len(messages)} messages",
            execution_time=0.0
        )


class ModelExecutionStep(Step):
    """Execute the model request."""
    
    @property
    def name(self) -> str:
        return "model_execution"
    
    @property
    def description(self) -> str:
        return "Execute model request"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Execute model request with guardrail support and memory manager."""
        from upsonic.agent.events import (
            ModelRequestStartEvent, 
            ModelResponseEvent,
            ToolCallEvent,
            ToolResultEvent,
            ExternalToolPauseEvent
        )
        
        # Skip if we have cached result or policy blocked
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(context.task, '_policy_blocked') and context.task._policy_blocked:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        
        from upsonic.tools.processor import ExternalExecutionPause
        from upsonic.agent.context_managers import MemoryManager
        
        # Emit model request start event
        if context.is_streaming:
            has_tools = bool(context.agent.tools or (context.task and context.task.tools))
            tool_limit = getattr(context.agent, 'tool_call_limit', None)
            
            self._emit_event(ModelRequestStartEvent(
                model_name=context.model.model_name,
                is_streaming=False,
                has_tools=has_tools,
                tool_call_count=context.agent._tool_call_count,
                tool_call_limit=tool_limit
            ), context)
        
        try:
            if context.is_continuation and context.continuation_tool_results:
                from upsonic.messages import ModelRequest
                
                # Get the response with tool calls from context
                response_with_tool_calls = context.continuation_response_with_tool_calls
                
                if response_with_tool_calls:
                    context.messages.append(response_with_tool_calls)
                
                # Then add the tool results from external execution
                tool_results_message = ModelRequest(parts=context.continuation_tool_results)
                context.messages.append(tool_results_message)
                context.agent._current_messages = context.messages
            
            # Use memory manager with async context
            memory_manager = MemoryManager(context.agent.memory)
            async with memory_manager.manage_memory() as memory_handler:
                if context.task.guardrail:
                    final_response = await context.agent._execute_with_guardrail(
                        context.task,
                        memory_handler,
                        context.state
                    )
                else:
                    model_params = context.agent._build_model_request_parameters(context.task)
                    model_params = context.model.customize_request_parameters(model_params)
                    
                    response = await context.model.request(
                        messages=context.messages,
                        model_settings=context.model.settings,
                        model_request_parameters=model_params
                    )
                    
                    # Store response before calling _handle_model_response 
                    # so we can access it if ExternalExecutionPause is raised
                    context.response = response
                    
                    final_response = await context.agent._handle_model_response(
                        response,
                        context.messages
                    )
                
                context.response = final_response
                
                # Add the final response to messages for proper conversation history
                context.messages.append(final_response)
                context.agent._current_messages = context.messages
                
                # Store memory handler reference for later steps
                context._memory_handler = memory_handler
                
                # Emit model response event
                if context.is_streaming:
                    from upsonic.messages import TextPart, ToolCallPart
                    has_text = any(isinstance(p, TextPart) for p in final_response.parts)
                    tool_calls = [p for p in final_response.parts if isinstance(p, ToolCallPart)]
                    
                    self._emit_event(ModelResponseEvent(
                        model_name=context.model.model_name,
                        has_text=has_text,
                        has_tool_calls=len(tool_calls) > 0,
                        tool_call_count=len(tool_calls),
                        finish_reason=final_response.finish_reason
                    ), context)
            
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Model execution completed",
                execution_time=0.0
            )
            
        except ExternalExecutionPause as e:
            context.task.is_paused = True
            
            # IMPORTANT: Always use e.external_call (has tool_call_id from ToolManager)
            # We no longer create basic ToolCall without ID
            if not hasattr(e, 'external_call') or e.external_call is None:
                raise RuntimeError("ExternalExecutionPause must have external_call attached by ToolManager")
            
            context.task._tools_awaiting_external_execution.append(e.external_call)
            
            context.final_output = context.task.response
            context.agent._run_result.output = context.final_output
            
            # Emit external tool pause event
            if context.is_streaming:
                self._emit_event(ExternalToolPauseEvent(
                    tool_name=e.external_call.tool_name,
                    tool_call_id=e.external_call.tool_call_id,
                    tool_args=e.external_call.args_as_dict() if hasattr(e.external_call, 'args_as_dict') else {}
                ), context)
            
            # CRITICAL: We need to extract the response with tool_calls that triggered the pause
            # This response was generated but not yet added to messages when the exception was thrown
            # We need to save it so we can add it before injecting tool results on continuation
            model_response_with_tool_calls = None
            
            # The response should be in context.response if it was set before the pause
            # Or we need to look at the last element that might have been about to be added
            if hasattr(context, 'response') and context.response:
                model_response_with_tool_calls = context.response
            
            # Save continuation state in the task for resuming later
            # This allows continue_async to resume from exactly where we left off
            context.task._continuation_state = {
                'messages': list(context.messages) if context.messages else [],
                'response_with_tool_calls': model_response_with_tool_calls,  # Save the response with tool calls
                'tool_call_count': context.agent._tool_call_count,
                'tool_limit_reached': getattr(context.agent, '_tool_limit_reached', False),
                'current_messages': list(context.agent._current_messages) if context.agent._current_messages else [],
            }
            
            # This is a valid pause state, not an error - use PENDING status
            return StepResult(
                status=StepStatus.PENDING,
                message="Execution paused for external tool - waiting for external execution",
                execution_time=0.0
            )


class ResponseProcessingStep(Step):
    """Process the model response."""
    
    @property
    def name(self) -> str:
        return "response_processing"
    
    @property
    def description(self) -> str:
        return "Process model response"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Process model response and extract output."""
        # Skip if we have cached result, policy blocked, or external pause
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(context.task, '_policy_blocked') and context.task._policy_blocked:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        if context.task.is_paused:
            context.final_output = context.task.response
            context.agent._run_result.output = context.final_output
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to external pause",
                execution_time=0.0
            )
        
        output = context.agent._extract_output(context.task, context.response)
        context.task._response = output
        context.final_output = output
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Response processed",
            execution_time=0.0
        )


class ReflectionStep(Step):
    """Apply reflection processing to improve output."""
    
    @property
    def name(self) -> str:
        return "reflection"
    
    @property
    def description(self) -> str:
        return "Apply reflection processing"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Apply reflection to improve output."""
        from upsonic.agent.events import ReflectionEvent
        
        if not (context.agent.reflection_processor and context.agent.reflection):
            if context.is_streaming:
                self._emit_event(ReflectionEvent(
                    reflection_applied=False
                ), context)
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Reflection not enabled",
                execution_time=0.0
            )
        
        # Skip if cache hit, policy blocked, or external pause
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(context.task, '_policy_blocked') and context.task._policy_blocked:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        if context.task.is_paused:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to external pause",
                execution_time=0.0
            )
        
        original_output = context.final_output
        original_preview = str(original_output)[:100] if original_output else None
        
        improved_output = await context.agent.reflection_processor.process_with_reflection(
            context.agent,
            context.task,
            context.final_output
        )
        context.task._response = improved_output
        context.final_output = improved_output
        
        improved_preview = str(improved_output)[:100] if improved_output else None
        improvement_made = str(original_output) != str(improved_output)
        
        if context.is_streaming:
            self._emit_event(ReflectionEvent(
                reflection_applied=True,
                improvement_made=improvement_made,
                original_preview=original_preview,
                improved_preview=improved_preview
            ), context)
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Reflection applied",
            execution_time=0.0
        )


class CallManagementStep(Step):
    """Manage call processing and statistics."""
    
    @property
    def name(self) -> str:
        return "call_management"
    
    @property
    def description(self) -> str:
        return "Process call management"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Handle call management."""
        # Skip if no response or special states
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(context.task, '_policy_blocked') and context.task._policy_blocked:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        if context.task.is_paused:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to external pause",
                execution_time=0.0
            )
        
        from upsonic.agent.context_managers import CallManager
        
        if context.final_output is None and context.task:
            context.final_output = context.task.response
        context.agent._run_result.output = context.final_output
        
        call_manager = CallManager(
            context.model,
            context.task,
            debug=context.agent.debug,
            show_tool_calls=context.agent.show_tool_calls
        )
        
        async with call_manager.manage_call() as call_handler:
            call_handler.process_response(context.agent._run_result)
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Call management processed",
            execution_time=0.0
        )


class TaskManagementStep(Step):
    """Manage task processing and state."""
    
    @property
    def name(self) -> str:
        return "task_management"
    
    @property
    def description(self) -> str:
        return "Process task management"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Handle task management."""
        from upsonic.agent.context_managers import TaskManager
        
        task_manager = TaskManager(context.task, context.agent)
        
        async with task_manager.manage_task() as task_handler:
            task_handler.process_response(context.agent._run_result)
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Task management processed",
            execution_time=0.0
        )


class MemoryMessageTrackingStep(Step):
    """Track messages in memory."""
    
    @property
    def name(self) -> str:
        return "memory_message_tracking"
    
    @property
    def description(self) -> str:
        return "Track messages in memory"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Track messages in memory handler using async context."""
        from upsonic.agent.events import MemoryUpdateEvent
        
        # Skip if cache hit or policy blocked
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(context.task, '_policy_blocked') and context.task._policy_blocked:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        
        from upsonic.agent.context_managers import MemoryManager
        
        # Use memory manager with async context for tracking
        memory_manager = MemoryManager(context.agent.memory)
        async with memory_manager.manage_memory() as memory_handler:
            # Add new messages to run result
            # Note: context.response is already in context.messages (added in ModelExecutionStep)
            # So we only need to add the new messages slice, not the response separately
            history_length = len(memory_handler.get_message_history())
            new_messages_start = history_length
            
            messages_added = 0
            if context.messages and new_messages_start < len(context.messages):
                # This includes the final response since it was appended in ModelExecutionStep
                new_messages = context.messages[new_messages_start:]
                context.agent._run_result.add_messages(new_messages)
                messages_added = len(new_messages)
            
            # Process response in memory
            memory_handler.process_response(context.agent._run_result)
            
            # Get memory type
            if context.is_streaming:
                memory_type = None
                if context.agent.memory:
                    if hasattr(context.agent.memory, 'full_session_memory') and context.agent.memory.full_session_memory:
                        memory_type = 'full_session'
                    else:
                        memory_type = 'session'
                
                self._emit_event(MemoryUpdateEvent(
                    messages_added=messages_added,
                    memory_type=memory_type
                ), context)
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Memory tracking completed",
            execution_time=0.0
        )


class ReliabilityStep(Step):
    """Apply reliability layer processing."""
    
    @property
    def name(self) -> str:
        return "reliability"
    
    @property
    def description(self) -> str:
        return "Apply reliability layer"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Apply reliability layer with async context manager."""
        from upsonic.agent.events import ReliabilityEvent
        
        if not context.agent.reliability_layer:
            if context.is_streaming:
                self._emit_event(ReliabilityEvent(
                    reliability_applied=False
                ), context)
            return StepResult(
                status=StepStatus.SUCCESS,
                message="No reliability layer",
                execution_time=0.0
            )
        
        # Skip for special states
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(context.task, '_policy_blocked') and context.task._policy_blocked:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        
        from upsonic.agent.context_managers import ReliabilityManager
        
        original_output = context.final_output
        
        # Use reliability manager with async context
        reliability_manager = ReliabilityManager(
            context.task,
            context.agent.reliability_layer,
            context.model
        )
        
        async with reliability_manager.manage_reliability() as rel_handler:
            processed_task = await rel_handler.process_task(context.task)
            context.task = processed_task
            context.final_output = processed_task.response
        
        modifications_made = str(original_output) != str(context.final_output)
        
        if context.is_streaming:
            self._emit_event(ReliabilityEvent(
                reliability_applied=True,
                modifications_made=modifications_made
            ), context)
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Reliability applied",
            execution_time=0.0
        )


class AgentPolicyStep(Step):
    """Apply agent output policy with optional feedback loop.
    
    When agent_policy_feedback is enabled in the agent, this step will:
    1. Check the agent's response against policies
    2. If a violation occurs and retries are available, generate feedback
    3. Inject the feedback as a user message and re-execute the model
    4. Repeat until policy passes or loop count is exhausted
    5. Apply the final action (block/modify) if still failing after loops
    """
    
    @property
    def name(self) -> str:
        return "agent_policy"
    
    @property
    def description(self) -> str:
        return "Apply agent output safety policy"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Apply agent policy to output with feedback loop support."""
        from upsonic.agent.events import PolicyCheckEvent, PolicyFeedbackEvent
        
        if not context.agent.agent_policy_manager.has_policies() or not context.task.response:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="No agent policy or no response",
                execution_time=0.0
            )
        
        # Skip for special states
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(context.task, '_policy_blocked') and context.task._policy_blocked:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        
        policy_count = len(context.agent.agent_policy_manager.policies) if hasattr(context.agent.agent_policy_manager, 'policies') else 0
        original_response = context.task.response
        
        # Reset retry counter at start of new execution
        context.agent.agent_policy_manager.reset_retry_count()
        
        # Feedback loop: keep trying until policy passes or retries exhausted
        max_iterations = context.agent.agent_policy_manager.feedback_loop_count + 1  # +1 for initial check
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Apply agent policy - returns (task, feedback_message_or_none)
            processed_task, feedback_message = await context.agent._apply_agent_policy(context.task)
            
            if feedback_message is None:
                # Policy passed or final action applied (no retry needed)
                context.task = processed_task
                context.final_output = processed_task.response
                context.agent._run_result.output = context.final_output
                
                # Emit event for policy check result
                if context.is_streaming:
                    if context.task.response != original_response:
                        self._emit_event(PolicyCheckEvent(
                            policy_type='agent_policy',
                            action='modify',
                            policies_checked=policy_count,
                            content_modified=True
                        ), context)
                    else:
                        self._emit_event(PolicyCheckEvent(
                            policy_type='agent_policy',
                            action='pass',
                            policies_checked=policy_count,
                            content_modified=False
                        ), context)
                
                return StepResult(
                    status=StepStatus.SUCCESS,
                    message=f"Agent policies applied after {iteration} iteration(s)",
                    execution_time=0.0
                )
            
            # Feedback generated - need to re-execute the model
            if context.agent.debug:
                from upsonic.utils.printing import policy_feedback_retry
                policy_feedback_retry(
                    policy_type="agent_policy",
                    retry_count=iteration,
                    max_retries=max_iterations - 1
                )
            if context.is_streaming:
                self._emit_event(PolicyFeedbackEvent(
                    policy_type='agent_policy',
                    feedback_message=feedback_message,
                    retry_count=iteration,
                    max_retries=context.agent.agent_policy_manager.feedback_loop_count
                ), context)
            # Increment retry counter in policy manager
            context.agent.agent_policy_manager.increment_retry_count()
            
            # Re-execute the model with feedback message injected
            await self._rerun_model_with_feedback(context, feedback_message)
            
            # Task response is now updated with new model output
            # Loop back to check policy again
        
        # Should not reach here, but just in case
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Agent policies applied (exhausted retries)",
            execution_time=0.0
        )

    async def _rerun_model_with_feedback(self, context: StepContext, feedback_message: str) -> None:
        """Re-execute the model with the feedback message as a user prompt.
        
        This injects the feedback as a correction prompt and re-runs the model,
        updating the task response with the new output.
        """
        from upsonic.messages import UserPromptPart, ModelRequest, TextPart
        from upsonic.agent.context_managers import MemoryManager
        
        # Create a correction prompt from the feedback
        correction_prompt = (
            f"[POLICY VIOLATION FEEDBACK]\n\n"
            f"{feedback_message}\n\n"
            f"Please revise your response to comply with the policy requirements."
        )
        
        # Add the previous response and correction as messages
        if context.response:
            context.messages.append(context.response)
        
        correction_part = UserPromptPart(content=correction_prompt)
        correction_message = ModelRequest(parts=[correction_part])
        context.messages.append(correction_message)
        
        # Update current messages reference in agent
        context.agent._current_messages = context.messages
        
        # Re-execute model request
        model_params = context.agent._build_model_request_parameters(context.task)
        model_params = context.model.customize_request_parameters(model_params)
        
        response = await context.model.request(
            messages=context.messages,
            model_settings=context.model.settings,
            model_request_parameters=model_params
        )
        
        # Handle response (including any tool calls)
        final_response = await context.agent._handle_model_response(
            response,
            context.messages
        )
        
        context.response = final_response
        context.messages.append(final_response)
        context.agent._current_messages = context.messages
        
        # Extract and update task output
        output = context.agent._extract_output(context.task, final_response)
        context.task._response = output
        context.final_output = output

class CacheStorageStep(Step):
    """Store the response in cache."""
    
    @property
    def name(self) -> str:
        return "cache_storage"
    
    @property
    def description(self) -> str:
        return "Store response in cache"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Store response in cache."""
        from upsonic.agent.events import CacheStoredEvent
        
        if not (context.task.enable_cache and context.task.response):
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Caching not enabled or no response",
                execution_time=0.0
            )
        
        # Don't cache if it was a cache hit or policy blocked
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Already from cache",
                execution_time=0.0
            )
        if hasattr(context.task, '_policy_blocked') and context.task._policy_blocked:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Not caching blocked content",
                execution_time=0.0
            )
        
        input_text = context.task._original_input or context.task.description
        await context.task.store_cache_entry(input_text, context.task.response)
        
        if context.is_streaming:
            self._emit_event(CacheStoredEvent(
                cache_method=context.task.cache_method,
                duration_minutes=context.task.cache_duration_minutes
            ), context)
        
        if context.agent.debug:
            from upsonic.utils.printing import cache_stored
            cache_stored(
                cache_method=context.task.cache_method,
                input_preview=(context.task._original_input or context.task.description)[:100] 
                    if (context.task._original_input or context.task.description) else None,
                duration_minutes=context.task.cache_duration_minutes
            )
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Response cached",
            execution_time=0.0
        )


class StreamModelExecutionStep(Step):
    """Execute the model request in streaming mode."""
    
    @property
    def name(self) -> str:
        return "stream_model_execution"
    
    @property
    def description(self) -> str:
        return "Execute model request with streaming"
    
    @property
    def supports_streaming(self) -> bool:
        """This step supports streaming and yields events during execution."""
        return True
    
    async def execute(self, context: StepContext) -> StepResult:
        """This is not used in streaming mode - execute_stream is used instead."""
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Streaming execution - see execute_stream",
            execution_time=0.0
        )
    
    async def execute_stream(self, context: StepContext) -> AsyncIterator[Any]:
        """Execute model request in streaming mode and yield Agent events only."""
        from upsonic.messages import PartStartEvent, PartDeltaEvent, TextPart, ToolCallPart, ModelRequest
        from upsonic.agent.events import (
            ModelRequestStartEvent,
            TextDeltaEvent,
            TextCompleteEvent,
            ToolCallEvent,
            ToolResultEvent,
            FinalOutputEvent,
            convert_llm_event_to_agent_event,
        )
        
        start_time = time.time()
        accumulated_text = ""
        first_token_time = None
        
        # Emit model request start event
        has_tools = bool(context.agent.tools or (context.task and context.task.tools))
        tool_limit = getattr(context.agent, 'tool_call_limit', None)
        
        yield ModelRequestStartEvent(
            model_name=context.model.model_name,
            is_streaming=True,
            has_tools=has_tools,
            tool_call_count=context.agent._tool_call_count,
            tool_call_limit=tool_limit
        )
        
        # Skip if we have cached result or policy blocked
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            # Stream cached response as TextDeltaEvents
            cached_content = str(context.final_output)
            
            # Stream the cached content character by character as Agent events
            for char in cached_content:
                yield TextDeltaEvent(content=char)
                accumulated_text += char
                
                # Update stream result if available
                if context.stream_result:
                    context.stream_result._accumulated_text += char
                    if first_token_time is None:
                        first_token_time = time.time()
                        context.stream_result._first_token_time = first_token_time
            
            # Yield text complete and final output events
            yield TextCompleteEvent(content=cached_content)
            yield FinalOutputEvent(output=cached_content, output_type='cached')
            
            # Update final output and completion status
            if context.stream_result:
                context.stream_result._final_output = cached_content
                context.stream_result._is_complete = True
                context.stream_result._end_time = time.time()
            
            self._last_result = StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to cache hit",
                execution_time=time.time() - start_time
            )
            return
        
        if hasattr(context.task, '_policy_blocked') and context.task._policy_blocked:
            # Emit final output for blocked content
            yield FinalOutputEvent(output=None, output_type='blocked')
            
            self._last_result = StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to policy block",
                execution_time=time.time() - start_time
            )
            return
        
        # Build model parameters
        model_params = context.agent._build_model_request_parameters(context.task)
        model_params = context.model.customize_request_parameters(model_params)
        
        # Stream the model response with recursive tool call handling
        try:
            # Use recursive helper method to handle tool calls properly
            async for event in self._stream_with_tool_calls(context, model_params, accumulated_text, first_token_time):
                yield event
            
            # Extract output and update context
            output = context.agent._extract_output(context.task, context.response)
            context.task._response = output
            context.final_output = output
            
            # Yield final output event
            yield FinalOutputEvent(
                output=output,
                output_type='structured' if not isinstance(output, str) else 'text'
            )
            
            # Update stream result if available
            if context.stream_result:
                context.stream_result._final_output = output
                context.stream_result._is_complete = True
                context.stream_result._end_time = time.time()
            
            self._last_result = StepResult(
                status=StepStatus.SUCCESS,
                message="Streaming execution completed",
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self._last_result = StepResult(
                status=StepStatus.ERROR,
                message=f"Streaming execution failed: {str(e)}",
                execution_time=time.time() - start_time
            )
            raise

    async def _stream_with_tool_calls(self, context: StepContext, model_params, accumulated_text: str, first_token_time) -> AsyncIterator[Any]:
        """Recursively handle streaming with tool calls, yielding only Agent events."""
        from upsonic.messages import PartStartEvent, PartDeltaEvent, TextPart, ToolCallPart, ModelRequest
        from upsonic.agent.events import (
            TextDeltaEvent,
            TextCompleteEvent,
            ThinkingDeltaEvent,
            ToolCallDeltaEvent,
            ToolCallEvent,
            ToolResultEvent,
            convert_llm_event_to_agent_event,
        )
        
        # Check if we've reached tool call limit
        if hasattr(context.agent, '_tool_limit_reached') and context.agent._tool_limit_reached:
            return
        
        # Stream the model response
        async with context.model.request_stream(
            messages=context.messages,
            model_settings=context.model.settings,
            model_request_parameters=model_params
        ) as stream:
            async for event in stream:
                # Store raw LLM event in context for internal tracking
                context.streaming_events.append(event)
                
                # Convert LLM event to Agent event and yield
                agent_event = convert_llm_event_to_agent_event(event, accumulated_text=accumulated_text)
                
                if agent_event:
                    # Track text accumulation
                    if isinstance(agent_event, TextDeltaEvent):
                        accumulated_text += agent_event.content
                        if first_token_time is None:
                            first_token_time = time.time()
                            if context.stream_result:
                                context.stream_result._first_token_time = first_token_time
                        if context.stream_result:
                            context.stream_result._accumulated_text = accumulated_text
                    
                    yield agent_event
        
        # Get the final response from the stream
        final_response = stream.get()
        context.response = final_response
        
        # Emit text complete event if we accumulated text
        if accumulated_text:
            yield TextCompleteEvent(content=accumulated_text)
        
        # Check for tool calls
        tool_calls = [
            part for part in final_response.parts 
            if isinstance(part, ToolCallPart)
        ]
        
        if tool_calls:
            # Emit tool call events
            for i, tc in enumerate(tool_calls):
                yield ToolCallEvent(
                    tool_name=tc.tool_name,
                    tool_call_id=tc.tool_call_id,
                    tool_args=tc.args_as_dict(),
                    tool_index=i,
                    is_parallel=len(tool_calls) > 1
                )
            
            # Execute tool calls
            tool_results = await context.agent._execute_tool_calls(tool_calls)
            
            # Emit tool result events
            for tc, result in zip(tool_calls, tool_results):
                result_preview = str(result.content)[:100] if hasattr(result, 'content') else None
                is_error = hasattr(result, 'content') and isinstance(result.content, str) and 'error' in result.content.lower()
                
                yield ToolResultEvent(
                    tool_name=tc.tool_name,
                    tool_call_id=tc.tool_call_id,
                    result=result.content if hasattr(result, 'content') else None,
                    result_preview=result_preview,
                    is_error=is_error
                )
            
            # Check for tool limit reached
            if hasattr(context.agent, '_tool_limit_reached') and context.agent._tool_limit_reached:
                # Add tool calls and results to messages
                context.messages.append(final_response)
                context.messages.append(ModelRequest(parts=tool_results))
                
                # Add limit notification
                from upsonic.messages import UserPromptPart
                limit_notification = UserPromptPart(
                    content=f"[SYSTEM] Tool call limit of {context.agent.tool_call_limit} has been reached. "
                    f"No more tools are available. Please provide a final response based on the information you have."
                )
                limit_message = ModelRequest(parts=[limit_notification])
                context.messages.append(limit_message)
                
                # Reset accumulated_text for new streaming round
                accumulated_text = ""
                
                # Continue streaming with limit notification
                async for event in self._stream_with_tool_calls(context, model_params, accumulated_text, first_token_time):
                    yield event
                return
            
            # Check for stop execution flag
            should_stop = False
            for tool_result in tool_results:
                if hasattr(tool_result, 'content') and isinstance(tool_result.content, dict):
                    if tool_result.content.get('_stop_execution'):
                        should_stop = True
                        tool_result.content.pop('_stop_execution', None)
            
            if should_stop:
                # Create stop response
                final_text = ""
                for tool_result in tool_results:
                    if hasattr(tool_result, 'content'):
                        if isinstance(tool_result.content, dict):
                            final_text = str(tool_result.content.get('func', tool_result.content))
                        else:
                            final_text = str(tool_result.content)
                
                from upsonic.messages import TextPart, ModelResponse
                from upsonic._utils import now_utc
                from upsonic.usage import RequestUsage
                
                stop_response = ModelResponse(
                    parts=[TextPart(content=final_text)],
                    model_name=final_response.model_name,
                    timestamp=now_utc(),
                    usage=RequestUsage(),
                    provider_name=final_response.provider_name,
                    provider_response_id=final_response.provider_response_id,
                    provider_details=final_response.provider_details,
                    finish_reason="stop"
                )
                context.response = stop_response
                return
            
            # Add tool calls and results to messages
            context.messages.append(final_response)
            context.messages.append(ModelRequest(parts=tool_results))
            
            # Reset accumulated_text for new streaming round
            accumulated_text = ""
            
            # Recursively continue streaming with tool results
            async for event in self._stream_with_tool_calls(context, model_params, accumulated_text, first_token_time):
                yield event


class StreamMemoryMessageTrackingStep(Step):
    """Track messages in memory for streaming execution."""
    
    @property
    def name(self) -> str:
        return "stream_memory_message_tracking"
    
    @property
    def description(self) -> str:
        return "Track messages in memory during streaming"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Track messages in memory handler and stream result."""
        from upsonic.agent.events import MemoryUpdateEvent
        
        # Skip if cache hit or policy blocked
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(context.task, '_policy_blocked') and context.task._policy_blocked:
            return StepResult(
                status=StepStatus.SUCCESS,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        
        from upsonic.agent.context_managers import MemoryManager
        
        # Use memory manager with async context for tracking
        memory_manager = MemoryManager(context.agent.memory)
        async with memory_manager.manage_memory() as memory_handler:
            # Add new messages to stream result
            # Note: In streaming, context.response should already be in context.messages
            # So we only need to add the new messages slice, not the response separately
            history_length = len(memory_handler.get_message_history())
            new_messages_start = history_length
            
            messages_added = 0
            if context.messages and new_messages_start < len(context.messages):
                if context.stream_result:
                    # This includes the final response if it was appended to messages
                    new_messages = context.messages[new_messages_start:]
                    context.stream_result.add_messages(new_messages)
                    messages_added = len(new_messages)
            
            # Process response in memory
            if context.stream_result:
                memory_handler.process_response(context.stream_result)
            
            # Get memory type
            if context.is_streaming:
                memory_type = None
                if context.agent.memory:
                    if hasattr(context.agent.memory, 'full_session_memory') and context.agent.memory.full_session_memory:
                        memory_type = 'full_session'
                    else:
                        memory_type = 'session'
                
                self._emit_event(MemoryUpdateEvent(
                    messages_added=messages_added,
                    memory_type=memory_type
                ), context)
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Streaming memory tracking completed",
            execution_time=0.0
        )


class StreamFinalizationStep(Step):
    """Finalize the streaming execution."""
    
    @property
    def name(self) -> str:
        return "stream_finalization"
    
    @property
    def description(self) -> str:
        return "Finalize streaming execution"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Finalize streaming execution."""
        from upsonic.agent.events import ExecutionCompleteEvent
        
        # Ensure final_output is set from task response if not already set
        if context.final_output is None and context.task:
            context.final_output = context.task.response
        
        # Set final output in stream result if available
        if context.stream_result:
            if context.stream_result._final_output is None:
                context.stream_result._final_output = context.final_output
        
        # Determine output type
        output_type = 'text'
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            output_type = 'cached'
        elif hasattr(context.task, '_policy_blocked') and context.task._policy_blocked:
            output_type = 'blocked'
        elif context.final_output and not isinstance(context.final_output, str):
            output_type = 'structured'
        
        if context.is_streaming:
            output_preview = str(context.final_output)[:100] if context.final_output else None
            
            self._emit_event(ExecutionCompleteEvent(
                output_type=output_type,
                has_output=context.final_output is not None,
                output_preview=output_preview,
                total_tool_calls=context.agent._tool_call_count
            ), context)
        
        # End the task
        context.task.task_end()
        
        return StepResult(
            status=StepStatus.SUCCESS,
            message="Streaming finalized",
            execution_time=0.0
        )


class FinalizationStep(Step):
    """Finalize the execution."""
    
    @property
    def name(self) -> str:
        return "finalization"
    
    @property
    def description(self) -> str:
        return "Finalize execution"
    
    async def execute(self, context: StepContext) -> StepResult:
        """Finalize execution."""
        from upsonic.agent.events import ExecutionCompleteEvent
        
        # Ensure final_output is set from task response if not already set
        if context.final_output is None and context.task:
            context.final_output = context.task.response

        # Set final output in run result
        context.agent._run_result.output = context.final_output

        # End the task to calculate duration
        context.task.task_end()

        # Determine output type
        output_type = 'text'
        if hasattr(context.task, '_cached_result') and context.task._cached_result:
            output_type = 'cached'
        elif hasattr(context.task, '_policy_blocked') and context.task._policy_blocked:
            output_type = 'blocked'
        elif context.final_output and not isinstance(context.final_output, str):
            output_type = 'structured'
        
        if context.is_streaming:
            output_preview = str(context.final_output)[:100] if context.final_output else None
            total_duration = getattr(context.task, '_duration', None)
            
            self._emit_event(ExecutionCompleteEvent(
                output_type=output_type,
                has_output=context.final_output is not None,
                output_preview=output_preview,
                total_tool_calls=context.agent._tool_call_count,
                total_duration=total_duration
            ), context)

        # Print summary if needed
        if context.task and not context.task.not_main_task:
            from upsonic.utils.printing import print_price_id_summary, price_id_summary
            # Only print summary if price_id exists in summary (i.e., model was called)
            if context.task.price_id in price_id_summary:
                print_price_id_summary(context.task.price_id, context.task)

        # Cleanup task-level MCP handlers to prevent resource leaks
        # Only close handlers that are task-specific (not agent-level tools)
        try:
            from upsonic.tools.mcp import MCPHandler, MultiMCPHandler
            if context.task and hasattr(context.task, 'tools') and context.task.tools:
                agent_tools_set = set(context.agent.tools) if context.agent.tools else set()
                for tool in context.task.tools:
                    # Close handlers that are in task tools but not in agent tools
                    if isinstance(tool, (MCPHandler, MultiMCPHandler)):
                        if tool not in agent_tools_set:
                            try:
                                await tool.close()
                            except (RuntimeError, Exception) as e:
                                # Suppress event loop closed errors (common in threaded contexts)
                                error_msg = str(e).lower()
                                if "event loop is closed" not in error_msg and "loop" not in error_msg:
                                    # Only log non-loop-related errors in debug mode
                                    if context.agent.debug:
                                        from upsonic.utils.printing import console
                                        console.print(f"[yellow]Warning: Error closing task-level MCP handler: {e}[/yellow]")
        except Exception as e:
            # Don't let cleanup errors break execution
            if context.agent.debug:
                from upsonic.utils.printing import console
                console.print(f"[yellow]Warning: Error during MCP handler cleanup: {e}[/yellow]")

        return StepResult(
            status=StepStatus.SUCCESS,
            message="Execution finalized",
            execution_time=0.0
        )
