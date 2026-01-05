"""
Concrete Step Implementations

This module contains all the concrete step implementations for the agent pipeline.
Each step handles a specific part of the agent execution flow. All steps must execute;
there is no skipping. If any error occurs, it's raised immediately to the user.

Steps emit events for streaming visibility using utility functions from utils/agent/events.py.
PipelineManager passes task, agent, model to each step.
"""

import time
from typing import TYPE_CHECKING, AsyncIterator
from .step import Step, StepResult, StepStatus

if TYPE_CHECKING:
    from upsonic.run.agent.context import AgentRunContext
    from upsonic.tasks.tasks import Task
    from upsonic.models import Model
    from upsonic.agent.agent import Agent
    from upsonic.run.events.events import AgentEvent
else:
    AgentRunContext = "AgentRunContext"
    Task = "Task"
    Model = "Model"
    Agent = "Agent"
    AgentEvent = "AgentEvent"


class InitializationStep(Step):
    """Initialize agent state for execution."""
    
    @property
    def name(self) -> str:
        return "initialization"
    
    @property
    def description(self) -> str:
        return "Initialize agent for execution"
    
    @property
    def supports_streaming(self) -> bool:
        return True
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Initialize agent state for new execution."""
        from upsonic.utils.printing import agent_started

        # Start task timing
        task.task_start(agent)

        agent_started(agent.get_agent_id())

        context.tool_call_count = 0
        agent.current_task = task
        
        # Emit AgentInitializedEvent if streaming (RunStartedEvent is emitted by manager)
        if context.is_streaming:
            from upsonic.utils.agent.events import ayield_agent_initialized_event
            async for event in ayield_agent_initialized_event(
                run_id=context.run_id,
                agent_id=agent.agent_id,
                is_streaming=context.is_streaming
            ):
                context.events.append(event)
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Check cache for existing response."""
        if not task.enable_cache or task.is_paused:
            # Emit event for cache disabled
            if context.is_streaming:
                from upsonic.utils.agent.events import ayield_cache_check_event
                async for event in ayield_cache_check_event(
                    run_id=context.run_id,
                    cache_enabled=False
                ):
                    context.events.append(event)
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Caching not enabled or task paused",
                execution_time=0.0
            )
        
        # Set cache manager
        task.set_cache_manager(agent._cache_manager)
        
        if agent.debug:
            from upsonic.utils.printing import cache_configuration
            embedding_provider_name = None
            if task.cache_embedding_provider:
                embedding_provider_name = getattr(
                    task.cache_embedding_provider, 'model_name', 'Unknown'
                )
            
            cache_configuration(
                enable_cache=task.enable_cache,
                cache_method=task.cache_method,
                cache_threshold=task.cache_threshold if task.cache_method == "vector_search" else None,
                cache_duration_minutes=task.cache_duration_minutes,
                embedding_provider=embedding_provider_name
            )
        
        input_text = task._original_input or task.description
        input_preview = input_text[:100] if input_text else None
        cached_response = await task.get_cached_response(input_text, model)
        
        if cached_response is not None:
            similarity = None
            cache_key = None
            cache_entry = None
            if hasattr(task, '_last_cache_entry') and 'similarity' in task._last_cache_entry:
                cache_entry = task._last_cache_entry
                similarity = cache_entry.get('similarity')
                cache_key = cache_entry.get('key') or cache_entry.get('cache_key')
            
            from upsonic.utils.printing import cache_hit, debug_log_level2
            cache_hit(
                cache_method=task.cache_method,
                similarity=similarity,
                input_preview=(task._original_input or task.description)[:100] 
                    if (task._original_input or task.description) else None
            )
            
            # Level 2: Detailed cache hit information
            if agent.debug and agent.debug_level >= 2:
                debug_log_level2(
                    "Cache hit details",
                    "CacheCheckStep",
                    debug=agent.debug,
                    debug_level=agent.debug_level,
                    cache_method=task.cache_method,
                    similarity_score=similarity,
                    cache_key=cache_key,
                    input_text=(task._original_input or task.description)[:500],
                    cached_response_preview=str(cached_response)[:500] if cached_response else None,
                    cache_entry=cache_entry,
                    model_name=model.model_name if model else None
                )
            
            context.final_output = cached_response
            task._response = cached_response
            task.task_end()
            agent._agent_run_output.content = cached_response
            
            # Early return flag
            task._cached_result = True
            
            # Emit cache hit events
            if context.is_streaming:
                from upsonic.utils.agent.events import ayield_cache_check_event, ayield_cache_hit_event
                async for event in ayield_cache_check_event(
                    run_id=context.run_id,
                    cache_enabled=True,
                    cache_method=task.cache_method
                ):
                    context.events.append(event)
                async for event in ayield_cache_hit_event(
                    run_id=context.run_id,
                    cache_method=task.cache_method,
                    similarity=similarity,
                    cached_response_preview=str(cached_response)[:100] if cached_response else None
                ):
                    context.events.append(event)
            
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Cache hit - using cached response",
                execution_time=0.0
            )
        else:
            from upsonic.utils.printing import cache_miss
            cache_miss(
                cache_method=task.cache_method,
                input_preview=(task._original_input or task.description)[:100] 
                    if (task._original_input or task.description) else None
            )
            
            # Emit cache miss events
            if context.is_streaming:
                from upsonic.utils.agent.events import ayield_cache_check_event, ayield_cache_miss_event
                async for event in ayield_cache_check_event(
                    run_id=context.run_id,
                    cache_enabled=True,
                    cache_method=task.cache_method
                ):
                    context.events.append(event)
                async for event in ayield_cache_miss_event(
                    run_id=context.run_id,
                    cache_method=task.cache_method
                ):
                    context.events.append(event)
            return StepResult(
                status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Apply user policy to task input."""
        policy_count = len(agent.user_policy_manager._policies) if hasattr(agent.user_policy_manager, '_policies') else 0
        
        if not agent.user_policy_manager.has_policies() or not task.description or task.is_paused:
            if context.is_streaming:
                from upsonic.utils.agent.events import ayield_policy_check_event
                async for event in ayield_policy_check_event(
                    run_id=context.run_id or "",
                    policy_type='user_policy',
                    action='ALLOW',
                    policies_checked=policy_count,
                    content_modified=False,
                    blocked_reason=None
                ):
                    context.events.append(event)
            return StepResult(
                status=StepStatus.COMPLETED,
                message="No user policy or task paused",
                execution_time=0.0
            )
        
        # Skip if we have cached result
        if hasattr(task, '_cached_result') and task._cached_result:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        
        original_content = task.description
        
        # Use the agent's _apply_user_policy method (events are emitted inside)
        processed_task, should_continue = await agent._apply_user_policy(task, context)
        
        if not should_continue:
            # Policy blocked the content
            context.final_output = processed_task._response
            agent._agent_run_output.content = context.final_output
            processed_task._policy_blocked = True
            
            return StepResult(
                status=StepStatus.COMPLETED,
                message="User input blocked by policy",
                execution_time=0.0
            )
        elif processed_task.description != original_content:
            # Content was modified (REPLACE/ANONYMIZE)
            return StepResult(
                status=StepStatus.COMPLETED,
                message="User input modified by policy",
                execution_time=0.0
            )
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Select the appropriate model."""
        # Model is already determined by PipelineManager, just log it
        provider = getattr(model, 'system', None) or getattr(model, '_provider', None)
        provider_name = str(provider.name) if hasattr(provider, 'name') else str(provider) if provider else None
        
        # Populate model info on AgentRunOutput
        agent._agent_run_output.model = model.model_name if model else None
        agent._agent_run_output.model_provider = provider_name
        agent._agent_run_output.model_provider_profile = getattr(model, 'profile', None)
        
        # Level 2: Model selection details
        if agent.debug and agent.debug_level >= 2:
            from upsonic.utils.printing import debug_log_level2
            default_model_name = getattr(agent.model, 'model_name', 'Unknown') if agent.model else 'Unknown'
            selected_model_name = model.model_name if model else 'Unknown'
            debug_log_level2(
                "Model selection details",
                "ModelSelectionStep",
                debug=agent.debug,
                debug_level=agent.debug_level,
                default_model=default_model_name,
                selected_model=selected_model_name,
                provider=provider_name,
                model_settings=str(model.settings)[:300] if hasattr(model, 'settings') and model.settings else None
            )
        
        if context.is_streaming:
            from upsonic.utils.agent.events import ayield_model_selected_event
            async for event in ayield_model_selected_event(
                run_id=context.run_id,
                model_name=model.model_name,
                model_provider=provider_name or "unknown"
            ):
                context.events.append(event)
        
        return StepResult(
            status=StepStatus.COMPLETED,
            message=f"Selected model: {model.model_name}",
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Validate task attachments."""
        from upsonic.utils.validators import validate_attachments_exist
        
        attachments = task.attachments if hasattr(task, 'attachments') and task.attachments else []
        attachment_count = len(attachments)
        
        # Level 2: Validation details
        if agent.debug and agent.debug_level >= 2:
            from upsonic.utils.printing import debug_log_level2
            debug_log_level2(
                "Task validation",
                "ValidationStep",
                debug=agent.debug,
                debug_level=agent.debug_level,
                task_description=task.description[:300] if task.description else None,
                attachments_count=attachment_count,
                attachment_types=[type(att).__name__ for att in attachments] if attachments else [],
                response_format=str(task.response_format) if hasattr(task, 'response_format') and task.response_format else None
            )
        
        validate_attachments_exist(task)
        
        if context.is_streaming:
            from upsonic.utils.agent.events import ayield_validation_event
            async for event in ayield_validation_event(
                run_id=context.run_id or "",
                attachments_validated=attachment_count > 0,
                attachment_count=attachment_count,
                validation_passed=True
            ):
                context.events.append(event)
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Setup tools for the task."""
        # Setup task-specific tools
        agent._setup_task_tools(task)
        
        # Set current task on PlanningToolKit if it exists (for write_todos)
        if hasattr(agent, '_planning_toolkit') and agent._planning_toolkit:
            agent._planning_toolkit.set_current_task(task)
        
        # Get tool information for event
        tool_names = []
        has_mcp = False
        
        if hasattr(agent, '_tool_manager') and agent._tool_manager:
            tool_defs = agent._tool_manager.get_tool_definitions()
            tool_names = [t.name for t in tool_defs]
            
            # Level 2: Tool setup details
            if agent.debug and agent.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                tool_details = []
                for tool_def in tool_defs[:20]:  # First 20 tools
                    tool_details.append({
                        'name': tool_def.name,
                        'description': tool_def.description[:200] if tool_def.description else None,
                        'sequential': tool_def.sequential if hasattr(tool_def, 'sequential') else False,
                        'parameters_count': len(tool_def.parameters_json_schema.get('properties', {})) if hasattr(tool_def, 'parameters_json_schema') and tool_def.parameters_json_schema else 0
                    })
                
                debug_log_level2(
                    "Tool setup completed",
                    "ToolSetupStep",
                    debug=agent.debug,
                    debug_level=agent.debug_level,
                    total_tools=len(tool_names),
                    tool_names=tool_names[:20],  # First 20 tool names
                    tool_details=tool_details,
                    has_mcp=has_mcp,
                    task_tools_count=len(task.tools) if hasattr(task, 'tools') and task.tools else 0
                )
            
            # Check for MCP handlers
            from upsonic.tools.mcp import MCPHandler, MultiMCPHandler
            all_tools = agent.tools or []
            if task and hasattr(task, 'tools') and task.tools:
                all_tools = list(all_tools) + list(task.tools)
            has_mcp = any(isinstance(t, (MCPHandler, MultiMCPHandler)) for t in all_tools)
        
        if context.is_streaming:
            from upsonic.utils.agent.events import ayield_tools_configured_event
            async for event in ayield_tools_configured_event(
                run_id=context.run_id or "",
                tool_count=len(tool_names),
                tool_names=tool_names,
                has_mcp_handlers=has_mcp
            ):
                context.events.append(event)
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Setup storage connection context manager."""
        storage_type = None
        is_connected = False
        has_memory = agent.memory is not None
        session_id = None
        
        if agent.memory and agent.memory.storage:
            storage_type = type(agent.memory.storage).__name__
            is_connected = getattr(agent.memory.storage, '_connected', False)
            session_id = getattr(agent.memory, 'session_id', None)
        
        # Level 2: Storage connection details
        if agent.debug and agent.debug_level >= 2:
            from upsonic.utils.printing import debug_log_level2
            debug_log_level2(
                "Storage connection",
                "StorageConnectionStep",
                debug=agent.debug,
                debug_level=agent.debug_level,
                storage_type=storage_type,
                is_connected=is_connected,
                has_memory=has_memory,
                session_id=session_id,
                user_id=getattr(agent.memory, 'user_id', None) if agent.memory else None
            )
        
        if context.is_streaming:
            from upsonic.utils.agent.events import ayield_storage_connection_event
            async for event in ayield_storage_connection_event(
                run_id=context.run_id or "",
                storage_type=storage_type,
                is_connected=is_connected,
                has_memory=has_memory,
                session_id=session_id
            ):
                context.events.append(event)
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Setup LLM manager and finalize model selection."""
        from upsonic.agent.context_managers.llm_manager import LLMManager
        
        default_model_name = getattr(agent.model, 'model_name', 'Unknown') if agent.model else 'Unknown'
        
        # Create LLM manager with default and requested model
        llm_manager = LLMManager(
            default_model=agent.model,
            requested_model=model
        )
        
        # Prepare the LLM manager
        await llm_manager.aprepare()
        
        try:
            selected_model = llm_manager.get_model()
            # The selected_model is a string identifier, we need to infer it
            if selected_model:
                from upsonic.models import infer_model
                model = infer_model(selected_model)
        finally:
            await llm_manager.afinalize()
        
        requested_model_name = getattr(model, 'model_name', 'Unknown') if model else 'Unknown'
        model_changed = default_model_name != requested_model_name
        
        # Level 2: Model selection details
        if agent.debug and agent.debug_level >= 2:
            from upsonic.utils.printing import debug_log_level2
            debug_log_level2(
                "Model selection",
                "LLMManagerStep",
                debug=agent.debug,
                debug_level=agent.debug_level,
                default_model=default_model_name,
                requested_model=requested_model_name,
                selected_model=model.model_name if model else 'Unknown',
                model_changed=model_changed,
                use_llm_for_selection=getattr(agent, 'use_llm_for_selection', False),
                model_selection_criteria=getattr(agent, 'model_selection_criteria', None)
            )
        
        if context.is_streaming:
            from upsonic.utils.agent.events import ayield_llm_prepared_event
            async for event in ayield_llm_prepared_event(
                run_id=context.run_id or "",
                default_model=default_model_name,
                requested_model=requested_model_name,
                model_changed=model_changed
            ):
                context.events.append(event)
        
        return StepResult(
            status=StepStatus.COMPLETED,
            message=f"LLM manager configured: {model.model_name}",
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Build model request messages with memory manager."""
        from upsonic.run.events.events import MessagesBuiltEvent
        
        # Skip if we have cached result or policy blocked
        if hasattr(task, '_cached_result') and task._cached_result:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(task, '_policy_blocked') and task._policy_blocked:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        
        from upsonic.agent.context_managers import MemoryManager
        from upsonic.agent.context_managers.culture_manager_context import CultureContextManager
        
        # If this is a continuation for external_tool, chat_history is already injected
        # by _inject_external_tool_results, so skip restoration
        if context.requirements:
            for req in context.requirements:
                if req.pause_type == 'external_tool':
                    # Chat history already injected by _inject_external_tool_results
                    # which includes: saved_messages + response_with_tool_calls + tool_returns
                    if context.chat_history:  # Chat history already set up
                        if context.is_streaming:
                            from upsonic.utils.agent.events import ayield_messages_built_event
                            async for event in ayield_messages_built_event(
                                run_id=context.run_id or "",
                                message_count=len(context.chat_history),
                                has_system_prompt=False,
                                has_memory_messages=True,
                                is_continuation=True
                            ):
                                context.events.append(event)
                        
                        return StepResult(
                            status=StepStatus.COMPLETED,
                            message=f"Using {len(context.chat_history)} pre-injected messages for external tool continuation",
                            execution_time=0.0
                        )
                    break
        
        # Create memory manager (pass agent metadata for prompt injection)
        memory_manager = MemoryManager(agent.memory, agent_metadata=getattr(agent, 'metadata', None))
        
        # Create culture context manager if culture is enabled
        culture_manager_obj = None
        if agent.add_culture_to_context or agent.update_cultural_knowledge:
            culture_manager_obj = CultureContextManager(
                culture_manager=agent.culture_manager,
                update_cultural_knowledge=agent.update_cultural_knowledge,
            )
        
        # Prepare memory manager
        await memory_manager.aprepare()
        
        try:
            # Prepare culture manager if available
            if culture_manager_obj:
                await culture_manager_obj.aprepare()
            
            try:
                # Build messages
                if culture_manager_obj:
                    messages = await agent._build_model_request(
                        task,
                        memory_manager,
                        None,  # state not needed
                        culture_manager_obj if agent.add_culture_to_context else None,
                    )
                else:
                    messages = await agent._build_model_request(
                        task,
                        memory_manager,
                        None,  # state not needed
                    )
                # Set chat_history (full conversation for LLM), not messages (run-specific)
                context.chat_history = messages
                
                # Level 2: Message building details
                if agent.debug and agent.debug_level >= 2:
                    from upsonic.utils.printing import debug_log_level2
                    from upsonic.messages import ModelRequest, SystemPromptPart
                    message_details = []
                    total_parts = 0
                    for msg in messages:
                        if isinstance(msg, ModelRequest):
                            parts_count = len(msg.parts) if msg.parts else 0
                            total_parts += parts_count
                            has_system_part = any(isinstance(p, SystemPromptPart) for p in (msg.parts or []))
                            message_details.append({
                                'parts_count': parts_count,
                                'has_system': has_system_part
                            })
                    
                    debug_log_level2(
                        "Messages built",
                        "MessageBuildStep",
                        debug=agent.debug,
                        debug_level=agent.debug_level,
                        message_count=len(messages),
                        total_parts=total_parts,
                        message_details=message_details,
                        has_memory=len(memory_manager.get_message_history()) > 0,
                        memory_message_count=len(memory_manager.get_message_history()),
                        has_culture=culture_manager_obj is not None,
                        task_description=task.description[:300] if task else None
                    )
                
                # Determine message characteristics
                has_system = False
                has_memory = len(memory_manager.get_message_history()) > 0
                from upsonic.messages import ModelRequest, SystemPromptPart
                if messages:
                    first_msg = messages[0]
                    if isinstance(first_msg, ModelRequest):
                        has_system = any(isinstance(p, SystemPromptPart) for p in first_msg.parts)
                
                if context.is_streaming:
                    from upsonic.utils.agent.events import ayield_messages_built_event
                    async for event in ayield_messages_built_event(
                        run_id=context.run_id or "",
                        message_count=len(messages),
                        has_system_prompt=has_system,
                        has_memory_messages=has_memory,
                        is_continuation=False
                    ):
                        context.events.append(event)
            finally:
                # Finalize culture manager if available
                if culture_manager_obj:
                    await culture_manager_obj.afinalize()
        finally:
            # Sync context to output before memory finalization
            # This ensures agent_run_context is stored with the session
            if agent._agent_run_output:
                agent._agent_run_context = context
                agent._agent_run_output.sync_from_context(context)
            # Finalize memory manager
            await memory_manager.afinalize()
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Execute model request with guardrail support and memory manager."""
        
        # Skip if we have cached result or policy blocked
        if hasattr(task, '_cached_result') and task._cached_result:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(task, '_policy_blocked') and task._policy_blocked:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        
        from upsonic.agent.context_managers import MemoryManager
        
        # Emit model request start event
        if context.is_streaming:
            has_tools = bool(agent.tools or (task and task.tools))
            tool_limit = getattr(agent, 'tool_call_limit', None)
            from upsonic.utils.agent.events import ayield_model_request_start_event
            async for event in ayield_model_request_start_event(
                run_id=context.run_id or "",
                model_name=model.model_name,
                is_streaming=False,
                has_tools=has_tools,
                tool_call_count=context.tool_call_count,
                tool_call_limit=tool_limit
            ):
                context.events.append(event)
        
        # Note: External tool continuation is now handled by _inject_external_tool_results
        # in Agent.continue_run_async, so we don't need to inject messages here
        
        # Create memory manager (pass agent metadata for prompt injection)
        memory_manager = MemoryManager(agent.memory, agent_metadata=getattr(agent, 'metadata', None))
        await memory_manager.aprepare()
        
        try:
            if task.guardrail:
                final_response = await agent._execute_with_guardrail(
                    task,
                    memory_manager,
                    None  # state not needed
                )
            else:
                model_params = agent._build_model_request_parameters(task)
                model_params = model.customize_request_parameters(model_params)
                
                # Level 2: Log model request details before execution
                if agent.debug and agent.debug_level >= 2:
                    from upsonic.utils.printing import debug_log_level2
                    import json
                    messages_preview = []
                    for msg in context.chat_history[-3:]:  # Last 3 messages
                        if hasattr(msg, 'parts'):
                            msg_preview = []
                            for part in msg.parts[:2]:  # First 2 parts
                                if hasattr(part, 'content'):
                                    content = str(part.content)[:200]
                                    msg_preview.append(content)
                            messages_preview.append(" | ".join(msg_preview))
                    
                    debug_log_level2(
                        "Model request details",
                        "ModelExecutionStep",
                        debug=agent.debug,
                        debug_level=agent.debug_level,
                        model_name=model.model_name,
                        model_settings=json.dumps(model.settings.dict() if hasattr(model.settings, 'dict') else str(model.settings), default=str)[:500],
                        model_params=json.dumps(model_params, default=str)[:500],
                        message_count=len(context.chat_history),
                        messages_preview=messages_preview,
                        tool_count=len(agent.tools) if agent.tools else 0,
                        tool_call_count=context.tool_call_count
                    )
                
                import time
                model_start_time = time.time()
                response = await model.request(
                    messages=context.chat_history,
                    model_settings=model.settings,
                    model_request_parameters=model_params
                )
                model_execution_time = time.time() - model_start_time
                
                # Level 2: Log model response details
                if agent.debug and agent.debug_level >= 2:
                    from upsonic.utils.printing import debug_log_level2
                    usage_info = {}
                    if hasattr(response, 'usage') and response.usage:
                        usage_info = {
                            'input_tokens': response.usage.input_tokens,
                            'output_tokens': response.usage.output_tokens,
                            'total_tokens': getattr(response.usage, 'total_tokens', None)
                        }
                    
                    tool_calls_count = 0
                    if hasattr(response, 'parts'):
                        for part in response.parts:
                            if hasattr(part, 'tool_calls') and part.tool_calls:
                                tool_calls_count += len(part.tool_calls)
                    
                    debug_log_level2(
                        "Model response details",
                        "ModelExecutionStep",
                        debug=agent.debug,
                        debug_level=agent.debug_level,
                        model_name=model.model_name,
                        execution_time=model_execution_time,
                        usage=usage_info,
                        tool_calls_count=tool_calls_count,
                        response_preview=str(response)[:500] if response else None,
                        has_content=hasattr(response, 'content') and response.content is not None
                    )
                
                # Store response before calling _handle_model_response 
                # so we can access it if ExternalExecutionPause is raised
                context.response = response
                
                final_response = await agent._handle_model_response(
                    response,
                    context.chat_history
                )
            
            context.response = final_response
            
            # Add the final response to chat_history for proper conversation history
            context.chat_history.append(final_response)
            
            # Emit model response event
            if context.is_streaming:
                from upsonic.messages import TextPart, ToolCallPart
                has_text = any(isinstance(p, TextPart) for p in final_response.parts)
                tool_calls = [p for p in final_response.parts if isinstance(p, ToolCallPart)]
                from upsonic.utils.agent.events import ayield_model_response_event
                async for event in ayield_model_response_event(
                    run_id=context.run_id or "",
                    model_name=model.model_name,
                    has_text=has_text,
                    has_tool_calls=len(tool_calls) > 0,
                    tool_call_count=len(tool_calls),
                    finish_reason=final_response.finish_reason
                ):
                    context.events.append(event)
        finally:
            # Sync context to output before memory finalization
            if agent._agent_run_output:
                agent._agent_run_context = context
                agent._agent_run_output.sync_from_context(context)
            await memory_manager.afinalize()
        
        return StepResult(
            status=StepStatus.COMPLETED,
            message="Model execution completed",
            execution_time=0.0
        )
        # ExternalExecutionPause is caught and handled by PipelineManager


class ResponseProcessingStep(Step):
    """Process the model response."""
    
    @property
    def name(self) -> str:
        return "response_processing"
    
    @property
    def description(self) -> str:
        return "Process model response"
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Process model response and extract output."""
        # Skip if we have cached result, policy blocked, or external pause
        if hasattr(task, '_cached_result') and task._cached_result:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(task, '_policy_blocked') and task._policy_blocked:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        if task.is_paused:
            context.final_output = task.response
            agent._agent_run_output.content = context.final_output
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to external pause",
                execution_time=0.0
            )
        
        output = agent._extract_output(task, context.response)
        task._response = output
        context.final_output = output
        
        # Populate usage, thinking content, thinking parts, and images from the response
        if context.response:
            from upsonic.messages import ThinkingPart, BinaryContent
            
            # Set usage from response
            if hasattr(context.response, 'usage') and context.response.usage:
                agent._agent_run_output.usage = context.response.usage
            
            # Extract thinking parts
            thinking_parts = [part for part in context.response.parts if isinstance(part, ThinkingPart)]
            if thinking_parts:
                agent._agent_run_output.thinking_parts = thinking_parts
                # Set thinking_content to the last thinking part's content
                agent._agent_run_output.thinking_content = thinking_parts[-1].content
            
            # Extract images from response parts (if any)
            images = []
            for part in context.response.parts:
                if hasattr(part, 'content') and isinstance(part.content, BinaryContent):
                    # Check if it's an image type
                    if hasattr(part.content, 'media_type') and part.content.media_type and 'image' in part.content.media_type:
                        images.append(part.content)
            if images:
                agent._agent_run_output.images = images
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Apply reflection to improve output."""
        from upsonic.run.events.events import ReflectionEvent
        
        if not (agent.reflection_processor and agent.reflection):
            if context.is_streaming:
                from upsonic.utils.agent.events import ayield_reflection_event
                async for event in ayield_reflection_event(
                    run_id=context.run_id or "",
                    reflection_applied=False
                ):
                    context.events.append(event)
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Reflection not enabled",
                execution_time=0.0
            )
        
        # Skip if cache hit, policy blocked, or external pause
        if hasattr(task, '_cached_result') and task._cached_result:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(task, '_policy_blocked') and task._policy_blocked:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        if task.is_paused:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to external pause",
                execution_time=0.0
            )
        
        original_output = context.final_output
        original_preview = str(original_output)[:100] if original_output else None
        
        # Level 2: Reflection start details
        if agent.debug and agent.debug_level >= 2:
            from upsonic.utils.printing import debug_log_level2
            debug_log_level2(
                "Reflection processing starting",
                "ReflectionStep",
                debug=agent.debug,
                debug_level=agent.debug_level,
                original_output_preview=original_preview,
                reflection_config=str(agent.reflection_processor.config) if hasattr(agent.reflection_processor, 'config') else None
            )
        
        improved_output = await agent.reflection_processor.process_with_reflection(
            agent,
            task,
            context.final_output
        )
        task._response = improved_output
        context.final_output = improved_output
        
        improved_preview = str(improved_output)[:100] if improved_output else None
        improvement_made = str(original_output) != str(improved_output)
        
        # Level 2: Reflection completion details
        if agent.debug and agent.debug_level >= 2:
            from upsonic.utils.printing import debug_log_level2
            debug_log_level2(
                "Reflection processing completed",
                "ReflectionStep",
                debug=agent.debug,
                debug_level=agent.debug_level,
                improvement_made=improvement_made,
                original_output_preview=original_preview,
                improved_output_preview=improved_preview,
                original_length=len(str(original_output)) if original_output else 0,
                improved_length=len(str(improved_output)) if improved_output else 0
            )
        
        if context.is_streaming:
            from upsonic.utils.agent.events import ayield_reflection_event
            async for event in ayield_reflection_event(
                run_id=context.run_id or "",
                reflection_applied=True,
                improvement_made=improvement_made,
                original_preview=original_preview,
                improved_preview=improved_preview
            ):
                context.events.append(event)
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Handle call management."""
        # Skip if no response or special states
        if hasattr(task, '_cached_result') and task._cached_result:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(task, '_policy_blocked') and task._policy_blocked:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        if task.is_paused:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to external pause",
                execution_time=0.0
            )
        
        from upsonic.agent.context_managers import CallManager
        
        if context.final_output is None and task:
            context.final_output = task.response
        agent._agent_run_output.content = context.final_output
        
        call_manager = CallManager(
            model,
            task,
            debug=agent.debug,
            show_tool_calls=agent.show_tool_calls
        )
        
        await call_manager.aprepare()
        
        try:
            # AgentRunOutput has both new_messages() and output properties that CallManager needs
            call_manager.process_response(agent._agent_run_output)
        
            # Level 2: Call management details
            if agent.debug and agent.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                from upsonic.utils.llm_usage import llm_usage
                from upsonic.utils.tool_usage import tool_usage
                usage = llm_usage(agent._agent_run_output) if agent._agent_run_output else {}
                tool_usage_result = tool_usage(agent._agent_run_output, task) if agent.show_tool_calls and agent._agent_run_output else None
                
                debug_log_level2(
                    "Call management processed",
                    "CallManagementStep",
                    debug=agent.debug,
                    debug_level=agent.debug_level,
                    execution_time=call_manager.end_time - call_manager.start_time if call_manager.end_time and call_manager.start_time else None,
                    usage=usage,
                    tool_usage_count=len(tool_usage_result) if tool_usage_result else 0,
                    tool_calls=tool_usage_result[:10] if tool_usage_result else [],  # First 10 tools
                    model_name=model.model_name if model else None,
                    response_format=str(task.response_format) if hasattr(task, 'response_format') and task.response_format else None,
                    total_cost=getattr(task, 'total_cost', None)
                )
        finally:
            await call_manager.afinalize()
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Handle task management."""
        from upsonic.agent.context_managers import TaskManager
        
        task_manager = TaskManager(task, agent)
        
        await task_manager.aprepare()
        
        try:
            # AgentRunOutput has output property that TaskManager needs
            task_manager.process_response(agent._agent_run_output)
        finally:
            await task_manager.afinalize()
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Track messages in memory handler using async context."""
        from upsonic.run.events.events import MemoryUpdateEvent
        
        # Skip if cache hit or policy blocked
        if hasattr(task, '_cached_result') and task._cached_result:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(task, '_policy_blocked') and task._policy_blocked:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        
        from upsonic.agent.context_managers import MemoryManager
        
        # Create memory manager for tracking (pass agent metadata)
        memory_manager = MemoryManager(agent.memory, agent_metadata=getattr(agent, 'metadata', None))
        await memory_manager.aprepare()
        
        try:
            # Extract new messages from this run
            # context.chat_history contains: historical messages + new messages from this run
            # context.response is already appended to chat_history in ModelExecutionStep
            history_length = len(memory_manager.get_message_history())
            new_messages_start = history_length
            
            messages_added = 0
            new_messages = []
            if context.chat_history and new_messages_start < len(context.chat_history):
                # Extract only the new messages from this run
                new_messages = context.chat_history[new_messages_start:]
                agent._agent_run_output.add_messages(new_messages)
                messages_added = len(new_messages)
            
            # Set context.messages to only the new messages from this run
            # This keeps context.messages in sync with AgentRunOutput.messages
            context.messages = new_messages
            
            # Process response in memory - pass both model_response and agent_run_output
            # context.response is the raw model response with new_messages() method
            memory_manager.process_response(context.response, agent._agent_run_output)
            
            # Level 2: Memory tracking details
            if agent.debug and agent.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                memory_type = None
                if agent.memory:
                    if hasattr(agent.memory, 'full_session_memory_enabled') and agent.memory.full_session_memory_enabled:
                        memory_type = 'full_session'
                    elif hasattr(agent.memory, 'summary_memory_enabled') and agent.memory.summary_memory_enabled:
                        memory_type = 'summary'
                    else:
                        memory_type = 'session'
                
                debug_log_level2(
                    "Memory message tracking",
                    "MemoryMessageTrackingStep",
                    debug=agent.debug,
                    debug_level=agent.debug_level,
                    messages_added=messages_added,
                    total_chat_history=len(context.chat_history),
                    run_messages=len(context.messages),
                    history_length=history_length,
                    memory_type=memory_type,
                    session_id=getattr(agent.memory, 'session_id', None) if agent.memory else None,
                    user_id=getattr(agent.memory, 'user_id', None) if agent.memory else None,
                    full_session_memory=getattr(agent.memory, 'full_session_memory_enabled', False) if agent.memory else False,
                    summary_memory=getattr(agent.memory, 'summary_memory_enabled', False) if agent.memory else False,
                    user_analysis_memory=getattr(agent.memory, 'user_analysis_memory_enabled', False) if agent.memory else False
                )
            
            # Get memory type
            if context.is_streaming:
                memory_type = None
                if agent.memory:
                    if hasattr(agent.memory, 'full_session_memory') and agent.memory.full_session_memory:
                        memory_type = 'full_session'
                    else:
                        memory_type = 'session'
                
                from upsonic.utils.agent.events import ayield_memory_update_event
                async for event in ayield_memory_update_event(
                    run_id=context.run_id or "",
                    messages_added=messages_added,
                    memory_type=memory_type
                ):
                    context.events.append(event)
        finally:
            # Sync context to output before memory finalization
            if agent._agent_run_output:
                agent._agent_run_context = context
                agent._agent_run_output.sync_from_context(context)
            await memory_manager.afinalize()
        
        return StepResult(
            status=StepStatus.COMPLETED,
            message="Memory tracking completed",
            execution_time=0.0
        )


class CultureUpdateStep(Step):
    """
    Update cultural knowledge after agent execution.
    
    This step runs after model execution and memory tracking to extract
    and store cultural knowledge from the conversation.
    
    Notice: Culture is an experimental feature and is subject to change.
    """
    
    @property
    def name(self) -> str:
        return "culture_update"
    
    @property
    def description(self) -> str:
        return "Update cultural knowledge"
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Update cultural knowledge using CultureManager."""
        from upsonic.run.events.events import CultureUpdateEvent
        
        # Check if culture update is enabled
        if not agent.update_cultural_knowledge or not agent.culture_manager:
            if context.is_streaming:
                from upsonic.run.events.events import CultureUpdateEvent
                event = CultureUpdateEvent(
                    run_id=context.run_id,
                    culture_enabled=False,
                    extraction_triggered=False,
                    knowledge_updated=False
                )
                context.events.append(event)
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Culture update not enabled",
                execution_time=0.0
            )
        
        # Skip if cache hit or policy blocked
        if hasattr(task, '_cached_result') and task._cached_result:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(task, '_policy_blocked') and task._policy_blocked:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        
        # Focus only on user input for culture extraction
        # We extract cultural knowledge from the user's message/task description
        user_input = None
        
        # Get user input from task description
        if task and task.description:
            user_input = task.description
        
        # Skip if no user input
        if not user_input:
            if context.is_streaming:
                from upsonic.run.events.events import CultureUpdateEvent
                event = CultureUpdateEvent(
                    run_id=context.run_id,
                    culture_enabled=True,
                    extraction_triggered=False,
                    knowledge_updated=False
                )
                context.events.append(event)
            return StepResult(
                status=StepStatus.COMPLETED,
                message="No user input for culture extraction",
                execution_time=0.0
            )
        
        # Run culture extraction on user input only
        try:
            # Level 2: Culture extraction start
            if agent.debug and agent.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                debug_log_level2(
                    "Culture extraction starting",
                    "CultureUpdateStep",
                    debug=agent.debug,
                    debug_level=agent.debug_level,
                    user_input=user_input[:500],
                    culture_manager_debug=getattr(agent.culture_manager, 'debug', False),
                    model_name=getattr(agent.culture_manager, '_model_spec', None) if agent.culture_manager else None
                )
            
            await agent.culture_manager.acreate_cultural_knowledge(
                message=user_input
            )
            
            knowledge_updated = agent.culture_manager.knowledge_updated
            
            # Level 2: Culture extraction completion
            if agent.debug and agent.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                debug_log_level2(
                    "Culture extraction completed",
                    "CultureUpdateStep",
                    debug=agent.debug,
                    debug_level=agent.debug_level,
                    knowledge_updated=knowledge_updated,
                    user_input=user_input[:500]
                )
            
            if context.is_streaming:
                from upsonic.run.events.events import CultureUpdateEvent
                event = CultureUpdateEvent(
                    run_id=context.run_id,
                    culture_enabled=True,
                    extraction_triggered=True,
                    knowledge_updated=knowledge_updated
                )
                context.events.append(event)
            return StepResult(
                status=StepStatus.COMPLETED,
                message=f"Culture extraction completed, updated={knowledge_updated}",
                execution_time=0.0
            )
            
        except Exception as e:
            # Don't fail the agent if culture extraction fails
            from upsonic.utils.printing import culture_error
            culture_error(f"Culture extraction failed: {e}", debug=agent.culture_manager.debug)
            
            if context.is_streaming:
                from upsonic.run.events.events import CultureUpdateEvent
                event = CultureUpdateEvent(
                    run_id=context.run_id,
                    culture_enabled=True,
                    extraction_triggered=True,
                    knowledge_updated=False
                )
                context.events.append(event)
            return StepResult(
                status=StepStatus.COMPLETED,
                message=f"Culture extraction failed (non-fatal): {e}",
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Apply reliability layer with async context manager."""
        from upsonic.run.events.events import ReliabilityEvent
        
        if not agent.reliability_layer:
            if context.is_streaming:
                from upsonic.utils.agent.events import ayield_reliability_event
                async for event in ayield_reliability_event(
                    run_id=context.run_id or "",
                    reliability_applied=False
                ):
                    context.events.append(event)
            return StepResult(
                status=StepStatus.COMPLETED,
                message="No reliability layer",
                execution_time=0.0
            )
        
        # Skip for special states
        if hasattr(task, '_cached_result') and task._cached_result:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(task, '_policy_blocked') and task._policy_blocked:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        
        from upsonic.agent.context_managers import ReliabilityManager
        
        original_output = context.final_output
        
        # Create reliability manager
        reliability_manager = ReliabilityManager(
            task,
            agent.reliability_layer,
            model
        )
        
        await reliability_manager.aprepare()
        
        try:
            processed_task = await reliability_manager.process_task(task)
            task = processed_task
            context.final_output = processed_task.response
        finally:
            await reliability_manager.afinalize()

        modifications_made = str(original_output) != str(context.final_output)
        
        # Level 2: Reliability layer details
        if agent.debug and agent.debug_level >= 2:
            from upsonic.utils.printing import debug_log_level2
            debug_log_level2(
                "Reliability layer applied",
                "ReliabilityStep",
                debug=agent.debug,
                debug_level=agent.debug_level,
                modifications_made=modifications_made,
                original_output_preview=str(original_output)[:300] if original_output else None,
                processed_output_preview=str(context.final_output)[:300] if context.final_output else None,
                reliability_layer_type=type(agent.reliability_layer).__name__ if agent.reliability_layer else None
            )
        
        if context.is_streaming:
            from upsonic.utils.agent.events import ayield_reliability_event
            async for event in ayield_reliability_event(
                run_id=context.run_id or "",
                reliability_applied=True,
                modifications_made=modifications_made
            ):
                context.events.append(event)
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Apply agent policy to output with feedback loop support."""
        policy_count = len(agent.agent_policy_manager.policies) if hasattr(agent.agent_policy_manager, 'policies') else 0
        
        if not agent.agent_policy_manager.has_policies() or not task.response:
            if context.is_streaming:
                from upsonic.utils.agent.events import ayield_policy_check_event
                async for event in ayield_policy_check_event(
                    run_id=context.run_id or "",
                    policy_type='agent_policy',
                    action='ALLOW',
                    policies_checked=policy_count,
                    content_modified=False,
                    blocked_reason=None
                ):
                    context.events.append(event)
            return StepResult(
                status=StepStatus.COMPLETED,
                message="No agent policy or no response",
                execution_time=0.0
            )
        
        # Skip for special states
        if hasattr(task, '_cached_result') and task._cached_result:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(task, '_policy_blocked') and task._policy_blocked:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        
        original_response = task.response
        
        # Reset retry counter at start of new execution
        agent.agent_policy_manager.reset_retry_count()
        
        # Feedback loop: keep trying until policy passes or retries exhausted
        max_iterations = agent.agent_policy_manager.feedback_loop_count + 1  # +1 for initial check
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Apply agent policy - returns (task, feedback_message_or_none)
            # Events are emitted inside _apply_agent_policy
            processed_task, feedback_message = await agent._apply_agent_policy(task, context)
            
            # Level 2: Detailed policy check information
            if agent.debug and agent.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                response_changed = processed_task.response != original_response
                debug_log_level2(
                    f"Agent policy check (iteration {iteration}/{max_iterations})",
                    "AgentPolicyStep",
                    debug=agent.debug,
                    debug_level=agent.debug_level,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    policy_count=policy_count,
                    has_feedback=feedback_message is not None,
                    response_changed=response_changed,
                    original_response_preview=str(original_response)[:300] if original_response else None,
                    processed_response_preview=str(processed_task.response)[:300] if processed_task.response else None,
                    feedback_message=feedback_message[:500] if feedback_message else None
                )
            
            if feedback_message is None:
                # Policy passed or final action applied (no retry needed)
                # Events already emitted in _apply_agent_policy
                task = processed_task
                context.final_output = processed_task.response
                agent._agent_run_output.content = context.final_output
                
                return StepResult(
                    status=StepStatus.COMPLETED,
                    message=f"Agent policies applied after {iteration} iteration(s)",
                    execution_time=0.0
                )
            
            # Feedback generated - need to re-execute the model
            # PolicyFeedbackEvent already emitted in _apply_agent_policy
            if agent.debug:
                from upsonic.utils.printing import policy_feedback_retry
                policy_feedback_retry(
                    policy_type="agent_policy",
                    retry_count=iteration,
                    max_retries=max_iterations - 1
                )
            # Increment retry counter in policy manager
            agent.agent_policy_manager.increment_retry_count()
            
            # Re-execute the model with feedback message injected
            await self._rerun_model_with_feedback(context, task, agent, model, feedback_message)
            
            # Task response is now updated with new model output
            # Loop back to check policy again
        
        # Should not reach here, but just in case
        
        return StepResult(
            status=StepStatus.COMPLETED,
            message="Agent policies applied (exhausted retries)",
            execution_time=0.0
        )

    async def _rerun_model_with_feedback(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model", feedback_message: str) -> None:
        """Re-execute the model with the feedback message as a user prompt.
        
        This injects the feedback as a correction prompt and re-runs the model,
        updating the task response with the new output.
        """
        from upsonic.messages import UserPromptPart, ModelRequest
        
        # Create a correction prompt from the feedback
        correction_prompt = (
            f"[POLICY VIOLATION FEEDBACK]\n\n"
            f"{feedback_message}\n\n"
            f"Please revise your response to comply with the policy requirements."
        )
        
        # Add the previous response and correction as messages to chat_history
        if context.response:
            context.chat_history.append(context.response)
        
        correction_part = UserPromptPart(content=correction_prompt)
        correction_message = ModelRequest(parts=[correction_part])
        context.chat_history.append(correction_message)
        
        # Re-execute model request
        model_params = agent._build_model_request_parameters(task)
        model_params = model.customize_request_parameters(model_params)
        
        response = await model.request(
            messages=context.chat_history,
            model_settings=model.settings,
            model_request_parameters=model_params
        )
        
        # Handle response (including any tool calls)
        final_response = await agent._handle_model_response(
            response,
            context.chat_history
        )
        
        context.response = final_response
        context.chat_history.append(final_response)
        
        # Extract and update task output
        output = agent._extract_output(task, final_response)
        task._response = output
        context.final_output = output

class CacheStorageStep(Step):
    """Store the response in cache."""
    
    @property
    def name(self) -> str:
        return "cache_storage"
    
    @property
    def description(self) -> str:
        return "Store response in cache"
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Store response in cache."""
        from upsonic.run.events.events import CacheStoredEvent
        
        if not (task.enable_cache and task.response):
            if context.is_streaming:
                from upsonic.utils.agent.events import ayield_cache_stored_event
                async for event in ayield_cache_stored_event(
                    run_id=context.run_id or "",
                    cache_method='disabled',
                    duration_minutes=None
                ):
                    context.events.append(event)
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Caching not enabled or no response",
                execution_time=0.0
            )
        
        # Don't cache if it was a cache hit or policy blocked
        if hasattr(task, '_cached_result') and task._cached_result:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Already from cache",
                execution_time=0.0
            )
        if hasattr(task, '_policy_blocked') and task._policy_blocked:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Not caching blocked content",
                execution_time=0.0
            )
        
        input_text = task._original_input or task.description
        await task.store_cache_entry(input_text, task.response)
        
        if context.is_streaming:
            from upsonic.utils.agent.events import ayield_cache_stored_event
            async for event in ayield_cache_stored_event(
                run_id=context.run_id or "",
                cache_method=task.cache_method,
                duration_minutes=task.cache_duration_minutes
            ):
                context.events.append(event)
        
        if agent.debug:
            from upsonic.utils.printing import cache_stored, debug_log_level2
            cache_stored(
                cache_method=task.cache_method,
                input_preview=(task._original_input or task.description)[:100] 
                    if (task._original_input or task.description) else None,
                duration_minutes=task.cache_duration_minutes
            )
            
            # Level 2: Detailed cache storage information
            if agent.debug_level >= 2:
                response_preview = str(task.response)[:500] if task.response else None
                debug_log_level2(
                    "Cache storage details",
                    "CacheStorageStep",
                    debug=agent.debug,
                    debug_level=agent.debug_level,
                    cache_method=task.cache_method,
                    input_text=input_text[:500],
                    response_preview=response_preview,
                    response_length=len(str(task.response)) if task.response else 0,
                    duration_minutes=task.cache_duration_minutes,
                    cache_threshold=task.cache_threshold if task.cache_method == "vector_search" else None,
                    model_name=model.model_name if model else None
                )
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Execute model request (non-streaming fallback). Collects events in context.events."""
        from typing import AsyncIterator
        from upsonic.run.events.events import AgentEvent
        
        # Consume the streaming generator and collect events in context
        async for event in self.execute_stream(context, task, agent, model):
            context.events.append(event)
        
        # Return the result from context
        return context.current_step_result or StepResult(
            status=StepStatus.COMPLETED,
            message="Streaming execution completed",
            execution_time=0.0
        )
    
    async def execute_stream(
        self, 
        context: "AgentRunContext", 
        task: "Task", 
        agent: "Agent", 
        model: "Model"
    ) -> "AsyncIterator[AgentEvent]":
        """Execute model request in streaming mode, yielding events as they occur."""
        from typing import AsyncIterator
        from upsonic.run.events.events import (
            AgentEvent,
            ModelRequestStartEvent,
            TextDeltaEvent,
            TextCompleteEvent,
            FinalOutputEvent,
        )
        
        start_time = time.time()
        accumulated_text = ""
        first_token_time = None
        
        # Emit model request start event
        has_tools = bool(agent.tools or (task and task.tools))
        tool_limit = getattr(agent, 'tool_call_limit', None)
        run_id = context.run_id or ""
        
        yield ModelRequestStartEvent(
            run_id=run_id,
            model_name=model.model_name,
            is_streaming=True,
            has_tools=has_tools,
            tool_call_count=context.tool_call_count,
            tool_call_limit=tool_limit
        )
        
        # Skip if we have cached result or policy blocked
        if hasattr(task, '_cached_result') and task._cached_result:
            cached_content = str(context.final_output)
            
            # Stream the cached content character by character
            for char in cached_content:
                yield TextDeltaEvent(run_id=run_id, content=char)
                accumulated_text += char
            
            yield TextCompleteEvent(run_id=run_id, content=cached_content)
            yield FinalOutputEvent(run_id=run_id, output=cached_content, output_type='cached')
            
            context.final_output = cached_content
            context.current_step_result = StepResult(
                status=StepStatus.SKIPPED,
                message="Skipped due to cache hit",
                execution_time=time.time() - start_time
            )
            return
        
        if hasattr(task, '_policy_blocked') and task._policy_blocked:
            yield FinalOutputEvent(run_id=run_id, output=None, output_type='blocked')
            context.current_step_result = StepResult(
                status=StepStatus.SKIPPED,
                message="Skipped due to policy block",
                execution_time=time.time() - start_time
            )
            return
        
        # Build model parameters
        model_params = agent._build_model_request_parameters(task)
        model_params = model.customize_request_parameters(model_params)
        
        # Level 2: Streaming start details
        if agent.debug and agent.debug_level >= 2:
            from upsonic.utils.printing import debug_log_level2
            debug_log_level2(
                "Streaming execution starting",
                "StreamModelExecutionStep",
                debug=agent.debug,
                debug_level=agent.debug_level,
                model_name=model.model_name,
                has_tools=has_tools,
                tool_call_limit=tool_limit,
                current_tool_call_count=context.tool_call_count,
                message_count=len(context.chat_history)
            )
        
        try:
            chunk_count = 0
            total_chars = 0
            tool_calls_in_stream = 0
            
            # Use streaming helper method that yields events
            async for event in self._stream_with_tool_calls(context, task, agent, model, model_params, accumulated_text, first_token_time):
                yield event
                # Track statistics
                if isinstance(event, TextDeltaEvent):
                    chunk_count += 1
                    total_chars += len(event.content) if event.content else 0
                    accumulated_text += event.content
            
            # Level 2: Streaming completion details
            if agent.debug and agent.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                streaming_time = time.time() - start_time
                time_to_first_token = (first_token_time - start_time) if first_token_time else None
                debug_log_level2(
                    "Streaming execution completed",
                    "StreamModelExecutionStep",
                    debug=agent.debug,
                    debug_level=agent.debug_level,
                    total_streaming_time=streaming_time,
                    time_to_first_token=time_to_first_token,
                    chunks_received=chunk_count,
                    total_characters=total_chars,
                    accumulated_text_length=len(accumulated_text),
                    tool_calls_during_stream=tool_calls_in_stream,
                    final_output_preview=str(context.final_output)[:500] if context.final_output else None
                )
            
            # Check if execution was paused (streaming does not support HITL resumption)
            if task.is_paused:
                context.current_step_result = StepResult(
                    status=StepStatus.PAUSED,
                    message="Execution paused (use direct call mode for HITL continuation)",
                    execution_time=time.time() - start_time
                )
                return
            
            # Extract output and update context
            output = agent._extract_output(task, context.response)
            task._response = output
            context.final_output = output
            
            # Emit final output event
            yield FinalOutputEvent(
                run_id=run_id,
                output=output,
                output_type='structured' if not isinstance(output, str) else 'text'
            )
            
            # Update agent run output
            if agent._agent_run_output:
                agent._agent_run_output.content = output
                agent._agent_run_output.mark_completed()
            
            context.current_step_result = StepResult(
                status=StepStatus.COMPLETED,
                message="Streaming execution completed",
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            context.current_step_result = StepResult(
                status=StepStatus.ERROR,
                message=f"Streaming execution failed: {str(e)}",
                execution_time=time.time() - start_time
            )
            raise

    async def _stream_with_tool_calls(
        self, 
        context: "AgentRunContext", 
        task: "Task", 
        agent: "Agent", 
        model: "Model", 
        model_params: dict,
        accumulated_text: str, 
        first_token_time: float
    ) -> "AsyncIterator[AgentEvent]":
        """Recursively handle streaming with tool calls, yielding events as they occur."""
        from typing import AsyncIterator
        from upsonic.messages import TextPart, ToolCallPart, ModelRequest
        from upsonic.run.events.events import (
            AgentEvent,
            TextDeltaEvent,
            TextCompleteEvent,
            ToolCallEvent,
            ToolResultEvent,
            convert_llm_event_to_agent_event,
        )
        
        # Check if we've reached tool call limit
        if context.tool_limit_reached:
            return
        
        run_id = context.run_id or ""
        
        # Stream the model response
        async with model.request_stream(
            messages=context.chat_history,
            model_settings=model.settings,
            model_request_parameters=model_params
        ) as stream:
            async for event in stream:
                # Convert LLM event to Agent event and yield
                agent_event = convert_llm_event_to_agent_event(event, accumulated_text=accumulated_text)
                
                if agent_event:
                    # Track text accumulation
                    if isinstance(agent_event, TextDeltaEvent):
                        accumulated_text += agent_event.content
                        if first_token_time is None:
                            first_token_time = time.time()
                    
                    yield agent_event
        
        # Get the final response from the stream
        final_response = stream.get()
        context.response = final_response
        
        # Note: TextCompleteEvent is already yielded by convert_llm_event_to_agent_event
        # when PartEndEvent with TextPart is received, so we don't yield it again here
        
        # Check for tool calls
        tool_calls = [
            part for part in final_response.parts 
            if isinstance(part, ToolCallPart)
        ]
        
        if tool_calls:
            # Emit tool call events
            for i, tc in enumerate(tool_calls):
                yield ToolCallEvent(
                    run_id=run_id,
                    tool_name=tc.tool_name,
                    tool_args=tc.args_as_dict(),
                    tool_call_id=tc.tool_call_id,
                    tool_index=i
                )
            
            # Execute tool calls - ExternalExecutionPause will bubble up to PipelineManager
            tool_results = await agent._execute_tool_calls(tool_calls)
            
            # Emit tool result events
            for tc, result in zip(tool_calls, tool_results):
                result_preview = str(result.content)[:100] if hasattr(result, 'content') else None
                is_error = hasattr(result, 'content') and isinstance(result.content, str) and 'error' in result.content.lower()
                
                yield ToolResultEvent(
                    run_id=run_id,
                    tool_name=tc.tool_name,
                    tool_call_id=tc.tool_call_id,
                    result=result.content if hasattr(result, 'content') else None,
                    result_preview=result_preview,
                    is_error=is_error
                )
            
            # Check for tool limit reached
            if context.tool_limit_reached:
                # Add tool calls and results to chat_history
                context.chat_history.append(final_response)
                context.chat_history.append(ModelRequest(parts=tool_results))
                
                # Add limit notification
                from upsonic.messages import UserPromptPart
                limit_notification = UserPromptPart(
                    content=f"[SYSTEM] Tool call limit of {agent.tool_call_limit} has been reached. "
                    f"No more tools are available. Please provide a final response based on the information you have."
                )
                limit_message = ModelRequest(parts=[limit_notification])
                context.chat_history.append(limit_message)
                
                # Reset accumulated_text for new streaming round
                accumulated_text = ""
                
                # Continue streaming with limit notification
                async for event in self._stream_with_tool_calls(context, task, agent, model, model_params, accumulated_text, first_token_time):
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
            
            # Add tool calls and results to chat_history
            context.chat_history.append(final_response)
            context.chat_history.append(ModelRequest(parts=tool_results))
            
            # Reset accumulated_text for new streaming round
            accumulated_text = ""
            
            # Recursively continue streaming with tool results
            async for event in self._stream_with_tool_calls(context, task, agent, model, model_params, accumulated_text, first_token_time):
                yield event


class StreamMemoryMessageTrackingStep(Step):
    """Track messages in memory for streaming execution."""
    
    @property
    def name(self) -> str:
        return "stream_memory_message_tracking"
    
    @property
    def description(self) -> str:
        return "Track messages in memory during streaming"
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Track messages in memory handler and stream result."""
        from upsonic.run.events.events import MemoryUpdateEvent
        
        # Skip if cache hit or policy blocked
        if hasattr(task, '_cached_result') and task._cached_result:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to cache hit",
                execution_time=0.0
            )
        if hasattr(task, '_policy_blocked') and task._policy_blocked:
            return StepResult(
                status=StepStatus.COMPLETED,
                message="Skipped due to policy block",
                execution_time=0.0
            )
        
        from upsonic.agent.context_managers import MemoryManager
        
        # Create memory manager for tracking (pass agent metadata)
        memory_manager = MemoryManager(agent.memory, agent_metadata=getattr(agent, 'metadata', None))
        await memory_manager.aprepare()
        
        try:
            # Extract new messages from this run
            # context.chat_history contains: historical messages + new messages from this run
            history_length = len(memory_manager.get_message_history())
            new_messages_start = history_length
            
            messages_added = 0
            new_messages = []
            if context.chat_history and new_messages_start < len(context.chat_history):
                # Extract only the new messages from this run
                new_messages = context.chat_history[new_messages_start:]
                agent._agent_run_output.add_messages(new_messages)
                messages_added = len(new_messages)
            
            # Set context.messages to only the new messages from this run
            # This keeps context.messages in sync with AgentRunOutput.messages
            context.messages = new_messages
            
            # Process response in memory - pass both model_response and agent_run_output
            # context.response is the raw model response with new_messages() method
            memory_manager.process_response(context.response, agent._agent_run_output)
            
            # Get memory type and emit event
            memory_type = None
            if agent.memory:
                if hasattr(agent.memory, 'full_session_memory') and agent.memory.full_session_memory:
                    memory_type = 'full_session'
                else:
                    memory_type = 'session'
            
            if context.is_streaming:
                from upsonic.utils.agent.events import ayield_memory_update_event
                async for event in ayield_memory_update_event(
                    run_id=context.run_id or "",
                    memory_type=memory_type,
                    messages_added=messages_added
                ):
                    context.events.append(event)
                
        finally:
            # Sync context to output before memory finalization
            if agent._agent_run_output:
                agent._agent_run_context = context
                agent._agent_run_output.sync_from_context(context)
            await memory_manager.afinalize()
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Finalize streaming execution."""
        from upsonic.run.events.events import ExecutionCompleteEvent
        
        # Ensure final_output is set from task response if not already set
        if context.final_output is None and task:
            context.final_output = task.response
        
        # Set final output
        if agent._agent_run_output:
            agent._agent_run_output.content = context.final_output
        
        # Determine output type
        output_type = 'text'
        if hasattr(task, '_cached_result') and task._cached_result:
            output_type = 'cached'
        elif hasattr(task, '_policy_blocked') and task._policy_blocked:
            output_type = 'blocked'
        elif context.final_output and not isinstance(context.final_output, str):
            output_type = 'structured'
        # End the task
        task.task_end()

        if context.is_streaming:
            output_preview = str(context.final_output)[:100] if context.final_output else None
            from upsonic.utils.agent.events import ayield_execution_complete_event
            async for event in ayield_execution_complete_event(
                run_id=context.run_id or "",
                output_type=output_type,
                has_output=context.final_output is not None,
                output_preview=output_preview,
                total_tool_calls=context.tool_call_count,
                total_duration=task.duration if task.duration else None
            ):
                context.events.append(event)
            # RunCompletedEvent is emitted by manager after pipeline end
        
        return StepResult(
            status=StepStatus.COMPLETED,
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
    
    async def execute(self, context: "AgentRunContext", task: "Task", agent: "Agent", model: "Model") -> StepResult:
        """Finalize execution."""
        from upsonic.run.events.events import ExecutionCompleteEvent
        
        # Ensure final_output is set from task response if not already set
        if context.final_output is None and task:
            context.final_output = task.response

        # Set final output
        agent._agent_run_output.content = context.final_output

        # Mark run as completed
        agent._agent_run_output.mark_completed()

        # End the task to calculate duration
        task.task_end()

        # Determine output type
        output_type = 'text'
        if hasattr(task, '_cached_result') and task._cached_result:
            output_type = 'cached'
        elif hasattr(task, '_policy_blocked') and task._policy_blocked:
            output_type = 'blocked'
        elif context.final_output and not isinstance(context.final_output, str):
            output_type = 'structured'
        
        if context.is_streaming:
            output_preview = str(context.final_output)[:100] if context.final_output else None
            total_duration = task.duration if task.duration else None
            from upsonic.utils.agent.events import ayield_execution_complete_event
            async for event in ayield_execution_complete_event(
                run_id=context.run_id or "",
                output_type=output_type,
                has_output=context.final_output is not None,
                output_preview=output_preview,
                total_tool_calls=context.tool_call_count,
                total_duration=total_duration
            ):
                context.events.append(event)
            # RunCompletedEvent is emitted by manager after pipeline end
        # Print summary if needed
        if task and not task.not_main_task:
            from upsonic.utils.printing import print_price_id_summary, price_id_summary
            # Only print summary if price_id exists in summary (i.e., model was called)
            if task.price_id in price_id_summary:
                print_price_id_summary(task.price_id, task)

        # Cleanup task-level MCP handlers to prevent resource leaks
        # Only close handlers that are task-specific (not agent-level tools)
        try:
            from upsonic.tools.mcp import MCPHandler, MultiMCPHandler
            if task and hasattr(task, 'tools') and task.tools:
                agent_tools_set = set(agent.tools) if agent.tools else set()
                for tool in task.tools:
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
                                    if agent.debug:
                                        from upsonic.utils.printing import console
                                        console.print(f"[yellow]Warning: Error closing task-level MCP handler: {e}[/yellow]")
        except Exception as e:
            # Don't let cleanup errors break execution
            if agent.debug:
                from upsonic.utils.printing import console
                console.print(f"[yellow]Warning: Error during MCP handler cleanup: {e}[/yellow]")

        return StepResult(
            status=StepStatus.COMPLETED,
            message="Execution finalized",
            execution_time=0.0
        )
