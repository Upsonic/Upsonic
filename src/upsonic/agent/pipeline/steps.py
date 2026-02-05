"""
Concrete Step Implementations

This module contains all the concrete step implementations for the agent pipeline.
Each step handles a specific part of the agent execution flow using try-except pattern.

Steps emit events for streaming visibility using utility functions from utils/agent/events.py.
PipelineManager passes task, agent, model, and step_number to each step.

Each step is responsible for:
1. Checking cancellation via raise_if_cancelled
2. Executing business logic
3. Creating StepResult with all attributes set and returning it

The base class Step.run() method handles:
- Appending result to context.step_results
- Updating context.execution_stats
"""

import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional
from .step import Step, StepResult, StepStatus

if TYPE_CHECKING:
    from upsonic.run.agent.output import AgentRunOutput
    from upsonic.tasks.tasks import Task
    from upsonic.models import Model
    from upsonic.agent.agent import Agent
    from upsonic.run.events.events import AgentEvent
else:
    AgentRunOutput = "AgentRunOutput"
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

    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Initialize agent state for new execution."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            task.task_start(agent)
            
            # Check print flag from context (thread-safe, set per-run)
            should_print = context.print_flag
            if should_print:
                from upsonic.utils.printing import agent_started
                agent_started(agent.get_agent_id())

            context.tool_call_count = 0
            agent.current_task = context.task
            
            # Start usage timer for tracking run duration
            context.start_usage_timer()
            
            if context.is_streaming:
                from upsonic.utils.agent.events import ayield_agent_initialized_event
                async for event in ayield_agent_initialized_event(
                    run_id=context.run_id,
                    agent_id=agent.agent_id,
                    is_streaming=context.is_streaming
                ):
                    context.events.append(event)
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="Agent initialized",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
    


class CacheCheckStep(Step):
    """Check if there's a cached response for the task."""
    
    @property
    def name(self) -> str:
        return "cache_check"
    
    @property
    def description(self) -> str:
        return "Check for cached responses"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Check cache for existing response."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)

            if not task.enable_cache or task.is_paused:
                if context.is_streaming:
                    from upsonic.utils.agent.events import ayield_cache_check_event
                    async for event in ayield_cache_check_event(
                        run_id=context.run_id,
                        cache_enabled=False
                    ):
                        context.events.append(event)
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Caching not enabled or task paused",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            task.set_cache_manager(agent._cache_manager)
            
            if agent.debug:
                from upsonic.utils.printing import cache_configuration
                embedding_provider_name = None
                if task.cache_embedding_provider:
                    embedding_provider_name = task.cache_embedding_provider.get_model_name()
                
                cache_configuration(
                    enable_cache=task.enable_cache,
                    cache_method=task.cache_method,
                    cache_threshold=task.cache_threshold if task.cache_method == "vector_search" else None,
                    cache_duration_minutes=task.cache_duration_minutes,
                    embedding_provider=embedding_provider_name
                )
            
            input_text = task._original_input or task.description
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
                
                context.output = cached_response
                task._response = cached_response
                task.task_end()
                task._cached_result = True
                
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
                
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Cache hit - using cached response",
                    execution_time=time.time() - start_time,
                )
                return step_result
            else:
                from upsonic.utils.printing import cache_miss
                cache_miss(
                    cache_method=task.cache_method,
                    input_preview=(task._original_input or task.description)[:100] 
                        if (task._original_input or task.description) else None
                )
                
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
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Cache miss - will execute normally",
                    execution_time=time.time() - start_time,
                )
                return step_result
                
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
class UserPolicyStep(Step):
    """Apply user policy to the task input."""
    
    @property
    def name(self) -> str:
        return "user_policy"
    
    @property
    def description(self) -> str:
        return "Apply user input safety policy"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Apply user policy to task input."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
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
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="No user policy or task paused",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            if hasattr(task, '_cached_result') and task._cached_result:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to cache hit",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            original_content = task.description
            processed_task, should_continue = await agent._apply_user_policy(task, context)
            
            # Update AgentRunInput.user_prompt if task.description was modified by policy
            if hasattr(agent, '_agent_run_output') and agent._agent_run_output and agent._agent_run_output.input:
                if processed_task.description != original_content:
                    agent._agent_run_output.input.user_prompt = processed_task.description
            
            if not should_continue:
                context.output = processed_task._response
                processed_task._policy_blocked = True
                
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="User input blocked by policy",
                    execution_time=time.time() - start_time,
                )
                return step_result
            elif processed_task.description != original_content:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="User input modified by policy",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="User policies passed",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
class ModelSelectionStep(Step):
    """Select the model to use for execution."""
    
    @property
    def name(self) -> str:
        return "model_selection"
    
    @property
    def description(self) -> str:
        return "Select model for execution"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Select the appropriate model."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            context.model_name = model.model_name if model else None
            context.model_provider = model.system if model else None
            context.model_provider_profile = model.profile if model else None
            
            provider_name = model.system if model else None
            
            if agent.debug and agent.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                default_model_name = agent.model.model_name if agent.model else 'Unknown'
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
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message=f"Selected model: {model.model_name}",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
class ToolSetupStep(Step):
    """Setup tools for the task execution."""
    
    @property
    def name(self) -> str:
        return "tool_setup"
    
    @property
    def description(self) -> str:
        return "Setup tools for execution"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Setup tools for the task."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            agent._setup_task_tools(task)
            
            if hasattr(agent, '_planning_toolkit') and agent._planning_toolkit:
                agent._planning_toolkit.set_current_task(task)
            
            tool_names = []
            has_mcp = False
            
            if hasattr(agent, '_tool_manager') and agent._tool_manager:
                tool_defs = agent._tool_manager.get_tool_definitions()
                tool_names = [t.name for t in tool_defs]
                
                if agent.debug and agent.debug_level >= 2:
                    from upsonic.utils.printing import debug_log_level2
                    tool_details = []
                    for tool_def in tool_defs[:20]:
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
                        tool_names=tool_names[:20],
                        tool_details=tool_details,
                        has_mcp=has_mcp,
                        task_tools_count=len(task.tools) if hasattr(task, 'tools') and task.tools else 0
                    )
                
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
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="Tools configured",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
class StorageConnectionStep(Step):
    """Setup storage connection for memory and database operations."""
    
    @property
    def name(self) -> str:
        return "storage_connection"
    
    @property
    def description(self) -> str:
        return "Setup storage connection"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Setup storage connection context manager."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            storage_type = None
            is_connected = False
            has_memory = agent.memory is not None
            session_id = None
            
            if agent.memory and agent.memory.storage:
                storage_type = type(agent.memory.storage).__name__
                is_connected = getattr(agent.memory.storage, '_connected', False)
                session_id = getattr(agent.memory, 'session_id', None)
            
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
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="Storage connection ready",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
class LLMManagerStep(Step):
    """Setup LLM manager for model selection and configuration."""
    
    @property
    def name(self) -> str:
        return "llm_manager"
    
    @property
    def description(self) -> str:
        return "Setup LLM manager"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Setup LLM manager and finalize model selection."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        from upsonic.agent.context_managers.llm_manager import LLMManager
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            default_model_name = agent.model.model_name if agent.model else 'Unknown'
            
            llm_manager = LLMManager(
                default_model=agent.model,
                agent=agent,
                requested_model=model
            )
            
            await llm_manager.aprepare()
            await llm_manager.afinalize()
            
            requested_model_name = llm_manager.selected_model.model_name if llm_manager.selected_model else 'Unknown'
            model_changed = default_model_name != requested_model_name
            
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
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message=f"LLM manager configured: {model.model_name}",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
class MessageBuildStep(Step):
    """Build the model request messages."""
    
    @property
    def name(self) -> str:
        return "message_build"
    
    @property
    def description(self) -> str:
        return "Build model request messages"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Build model request messages with memory manager."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        messages = []
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            if hasattr(task, '_cached_result') and task._cached_result:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to cache hit",
                    execution_time=time.time() - start_time,
                )
                return step_result
            if hasattr(task, '_policy_blocked') and task._policy_blocked:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to policy block",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            from upsonic.agent.context_managers import MemoryManager
            
            memory_manager = MemoryManager(
                memory=agent.memory,
                agent_metadata=getattr(agent, 'metadata', None),
            )
            
            # Register memory manager in pipeline registry
            if pipeline_manager:
                pipeline_manager.set_manager('memory_manager', memory_manager)
            
            # Prepare memory (loads messages, user profile, and cultural knowledge)
            await memory_manager.aprepare()
            
            historical_message_count = len(memory_manager.get_message_history())
            
            # Check if agent has culture enabled (culture is now handled in SystemPromptManager, not MemoryManager)
            has_culture = bool(
                getattr(agent, '_culture_manager', None) and 
                getattr(agent._culture_manager, 'enabled', False) and
                getattr(agent._culture_manager, 'culture', None) and
                getattr(agent._culture_manager.culture, 'add_system_prompt', False)
            )
            
            # Set chat_history to historical messages FIRST and mark run start
            # This ensures new_messages() captures both request and response
            historical_messages = memory_manager.get_message_history()
            context.chat_history = list(historical_messages)  # Copy to avoid mutation
            
            # Mark the start of this run for message tracking
            # This records the current chat_history length (historical only) so new_messages()
            # knows where new messages from this run begin
            context.start_new_run()
            
            # Now build the full messages including the new request
            messages = await agent._build_model_request(
                task,
                memory_manager,
                None,
            )
            context.chat_history = messages
            
            if agent.debug and agent.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                from upsonic.utils.messages import analyze_model_request_messages
                message_details, total_parts = analyze_model_request_messages(messages)
                
                debug_log_level2(
                    "Messages built",
                    "MessageBuildStep",
                    debug=agent.debug,
                    debug_level=agent.debug_level,
                    message_count=len(messages),
                    total_parts=total_parts,
                    message_details=message_details,
                    has_memory=historical_message_count > 0,
                    historical_message_count=historical_message_count,
                    has_culture=has_culture,
                    task_description=task.description[:300] if task else None
                )
                
            has_system = False
            has_memory = historical_message_count > 0
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
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message=f"Built {len(messages)} messages",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
class ModelExecutionStep(Step):
    """Execute the model request."""
    
    @property
    def name(self) -> str:
        return "model_execution"
    
    @property
    def description(self) -> str:
        return "Execute model request"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Execute model request with guardrail support and memory manager."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        from upsonic.tools.processor import ExternalExecutionPause
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        memory_manager = None
        call_manager = None
        response = None  # Track response for usage update in finally
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            if hasattr(task, '_cached_result') and task._cached_result:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to cache hit",
                    execution_time=time.time() - start_time,
                )
                return step_result
            if hasattr(task, '_policy_blocked') and task._policy_blocked:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to policy block",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
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
            
            memory_manager = None
            if pipeline_manager:
                memory_manager = pipeline_manager.get_manager('memory_manager')
            
            # Fallback: Create memory_manager if not available (resumption case)
            if memory_manager is None and task.guardrail and agent.memory:
                from upsonic.agent.context_managers import MemoryManager
                memory_manager = MemoryManager(agent.memory, agent_metadata=getattr(agent, 'metadata', None))
                if pipeline_manager:
                    pipeline_manager.set_manager('memory_manager', memory_manager)
                await memory_manager.aprepare()
            
            # Create and register CallManager
            from upsonic.agent.context_managers import CallManager
            call_manager = CallManager(
                model,
                task,
                debug=agent.debug,
                print_output=context.print_flag,
                show_tool_calls=agent.show_tool_calls
            )
            if pipeline_manager:
                pipeline_manager.set_manager('call_manager', call_manager)
            
            if task.guardrail:
                await call_manager.aprepare()
                try:
                    final_response = await agent._execute_with_guardrail(
                        task,
                        memory_manager,
                        None
                    )
                    call_manager.process_response(final_response)
                finally:
                    await call_manager.afinalize()
            else:
                model_params = agent._build_model_request_parameters(task)
                model_params = model.customize_request_parameters(model_params)
                
                if agent.debug and agent.debug_level >= 2:
                    from upsonic.utils.printing import debug_log_level2
                    import json
                    messages_preview = []
                    for msg in context.chat_history[-3:]:
                        if hasattr(msg, 'parts'):
                            msg_preview = []
                            for part in msg.parts[:2]:
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
                
                await call_manager.aprepare()
                try:
                    model_start_time = time.time()
                    response = await model.request(
                        messages=context.chat_history,
                        model_settings=model.settings,
                        model_request_parameters=model_params
                    )
                    model_execution_time = time.time() - model_start_time
                finally:
                    await call_manager.afinalize()
                
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
                
                context.response = response
                
                final_response = await agent._handle_model_response(
                    response,
                    context.chat_history
                )
                
                # Store final_response in call_manager for logging
                call_manager.process_response(final_response)
            
            context.response = final_response
            context.chat_history.append(final_response)
            
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
            
            agent_tool_calls = getattr(agent, '_tool_call_count', 0)
            prev_tool_calls = context.tool_call_count
            context.tool_call_count = agent_tool_calls
            context.tool_limit_reached = getattr(agent, '_tool_limit_reached', False)
            
            # Update usage with new tool calls made in this model execution
            new_tool_calls = agent_tool_calls - prev_tool_calls
            if new_tool_calls > 0:
                context.increment_tool_calls(new_tool_calls)
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="Model execution completed",
                execution_time=time.time() - start_time,
            )
            return step_result
        
        except ExternalExecutionPause as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.PAUSED,
                message="Paused for external tool execution",
                execution_time=time.time() - start_time,
            )
            raise
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            # Always update usage from model response if available (even on error/cancel)
            # This ensures usage is tracked for durable execution recovery
            if response is not None and hasattr(response, 'usage') and response.usage:
                try:
                    context.update_usage_from_response(response.usage)
                    
                    # Calculate and set cost
                    from upsonic.utils.usage import calculate_cost_from_usage
                    cost_value = calculate_cost_from_usage(response.usage, model)
                    context.set_usage_cost(cost_value)
                except Exception:
                    pass  # Silently ignore usage update errors in finally
            
            if step_result:
                self._finalize_step_result(step_result, context)
            


class ResponseProcessingStep(Step):
    """Process the model response."""
    
    @property
    def name(self) -> str:
        return "response_processing"
    
    @property
    def description(self) -> str:
        return "Process model response"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Process model response and extract output."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            if hasattr(task, '_cached_result') and task._cached_result:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to cache hit",
                    execution_time=time.time() - start_time,
                )
                return step_result
            if hasattr(task, '_policy_blocked') and task._policy_blocked:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to policy block",
                    execution_time=time.time() - start_time,
                )
                return step_result
            if task.is_paused:
                context.output = task.response
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to external pause",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            output = agent._extract_output(task, context.response)
            task._response = output
            context.output = output
            
            if context.response:
                from upsonic.messages import ThinkingPart, BinaryContent
                
                thinking_parts = [part for part in context.response.parts if isinstance(part, ThinkingPart)]
                if thinking_parts:
                    context.thinking_parts = thinking_parts
                    context.thinking_content = thinking_parts[-1].content
                
                images = []
                for part in context.response.parts:
                    if hasattr(part, 'content') and isinstance(part.content, BinaryContent):
                        if hasattr(part.content, 'media_type') and part.content.media_type and 'image' in part.content.media_type:
                            images.append(part.content)
                if images:
                    context.images = images
            
            # Note: Message finalization is deferred to MemorySaveStep
            # This ensures all steps (including AgentPolicyStep feedback loop) 
            # can add their messages to chat_history before we extract new messages
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="Response processed",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
class ReflectionStep(Step):
    """Apply reflection processing to improve output."""
    
    @property
    def name(self) -> str:
        return "reflection"
    
    @property
    def description(self) -> str:
        return "Apply reflection processing"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Apply reflection to improve output."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            if not (agent.reflection_processor and agent.reflection):
                if context.is_streaming:
                    from upsonic.utils.agent.events import ayield_reflection_event
                    async for event in ayield_reflection_event(
                        run_id=context.run_id or "",
                        reflection_applied=False
                    ):
                        context.events.append(event)
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Reflection not enabled",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            if hasattr(task, '_cached_result') and task._cached_result:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to cache hit",
                    execution_time=time.time() - start_time,
                )
                return step_result
            if hasattr(task, '_policy_blocked') and task._policy_blocked:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to policy block",
                    execution_time=time.time() - start_time,
                )
                return step_result
            if task.is_paused:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to external pause",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            original_output = context.output
            original_preview = str(original_output)[:100] if original_output else None
            
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
            
            # Execute reflection processing - returns ReflectionResult with all info
            reflection_result = await agent.reflection_processor.process_with_reflection(
                agent,
                task,
                context.output
            )
            
            # Extract values from ReflectionResult
            improved_output = reflection_result.improved_output
            improvement_made = reflection_result.improvement_made
            improved_preview = str(improved_output)[:100] if improved_output else None
            
            # Track the reflection interaction in chat_history and update context attributes
            from upsonic.messages import UserPromptPart, TextPart, ModelRequest, ModelResponse
            
            # FIRST INPUT: Create ModelRequest with the evaluation prompt from ReflectionResult
            evaluation_request = ModelRequest(parts=[UserPromptPart(content=reflection_result.evaluation_prompt)])
            
            # LAST OUTPUT: Create ModelResponse with the improved output from ReflectionResult
            improved_text = str(improved_output) if improved_output else ""
            improved_response = ModelResponse(parts=[TextPart(content=improved_text)])
            
            # Add to chat_history (full session history)
            context.chat_history.append(evaluation_request)
            context.chat_history.append(improved_response)
            
            # Update context.response (last ModelResponse from LLM)
            context.response = improved_response
            
            # Update context.output (last output of the agent)
            task._response = improved_output
            context.output = improved_output
            
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
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="Reflection applied",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
class CallManagementStep(Step):
    """Manage call processing and statistics."""
    
    @property
    def name(self) -> str:
        return "call_management"
    
    @property
    def description(self) -> str:
        return "Process call management"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Handle call management."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        call_manager = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            if hasattr(task, '_cached_result') and task._cached_result:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to cache hit",
                    execution_time=time.time() - start_time,
                )
                return step_result
            if hasattr(task, '_policy_blocked') and task._policy_blocked:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to policy block",
                    execution_time=time.time() - start_time,
                )
                return step_result
            if task.is_paused:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to external pause",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            if context.output is None and task:
                context.output = task.response
            
            # Retrieve CallManager from pipeline registry
            if pipeline_manager:
                call_manager = pipeline_manager.get_manager('call_manager')
            
            if call_manager:
                await call_manager.alog_completion(context)
                
                if agent.debug and agent.debug_level >= 2:
                    from upsonic.utils.printing import debug_log_level2
                    from upsonic.utils.llm_usage import llm_usage
                    from upsonic.utils.tool_usage import tool_usage
                    usage = llm_usage(context) if context else None
                    tool_usage_result = tool_usage(context, task) if agent.show_tool_calls and context else None
                    
                    debug_log_level2(
                        "Call management processed",
                        "CallManagementStep",
                        debug=agent.debug,
                        debug_level=agent.debug_level,
                        execution_time=call_manager.end_time - call_manager.start_time if call_manager.end_time and call_manager.start_time else None,
                        usage=usage,
                        tool_usage_count=len(tool_usage_result) if tool_usage_result else 0,
                        tool_calls=tool_usage_result[:10] if tool_usage_result else [],
                        model_name=model.model_name if model else None,
                        response_format=str(task.response_format) if hasattr(task, 'response_format') and task.response_format else None,
                        total_cost=getattr(task, 'total_cost', None)
                    )
            elif context and context.output is not None:
                # Fallback for resumption case: call_manager not available but we have output
                # This happens when resuming from a cancelled/errored run where ModelExecutionStep was skipped
                from upsonic.utils.printing import call_end
                from upsonic.utils.llm_usage import llm_usage
                from upsonic.utils.tool_usage import tool_usage as get_tool_usage
                
                # Try to get usage from context.usage (RunUsage) first, then fallback to llm_usage
                if hasattr(context, 'usage') and context.usage and hasattr(context.usage, 'input_tokens'):
                    usage = {
                        "input_tokens": context.usage.input_tokens,
                        "output_tokens": context.usage.output_tokens
                    }
                else:
                    usage = llm_usage(context)
                
                tool_usage_result = get_tool_usage(context, task) if agent.show_tool_calls else None
                
                # Get timing from context.usage if available
                # We want current time minus duration to get start_time, and current time as end_time
                context_end_time = time.time()
                context_start_time = context_end_time  # Default: will show 0.0s execution time
                
                if hasattr(context, 'usage') and context.usage:
                    if hasattr(context.usage, 'duration') and context.usage.duration:
                        # Use duration to calculate start_time
                        context_start_time = context_end_time - context.usage.duration
                
                call_end(
                    context.output,
                    model,
                    task.response_format if hasattr(task, 'response_format') else str,
                    context_start_time,
                    context_end_time,
                    usage,
                    tool_usage_result,
                    agent.debug,
                    getattr(task, 'price_id', None),
                    print_output=context.print_flag
                )
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="Call management processed",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
class TaskManagementStep(Step):
    """Manage task processing and state."""
    
    @property
    def name(self) -> str:
        return "task_management"
    
    @property
    def description(self) -> str:
        return "Process task management"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Handle task management."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        from upsonic.agent.context_managers import TaskManager
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            task_manager = TaskManager(task, agent)
            
            await task_manager.aprepare()
            
            try:
                task_manager.process_response(context)
            finally:
                await task_manager.afinalize()
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="Task management processed",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
class MemorySaveStep(Step):
    """
    Save AgentSession to storage. This is the LAST step for completed runs.
    
    For HITL (errors, cancel, external tools), session saving is handled
    by PipelineManager's exception handlers.
    
    Message tracking is finalized HERE (not in ResponseProcessingStep) to ensure
    all steps that may add messages (e.g., AgentPolicyStep feedback loop) have
    completed before we extract the new messages.
    """
    
    @property
    def name(self) -> str:
        return "memory_save"
    
    @property
    def description(self) -> str:
        return "Save session to storage"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Save session via MemoryManager.afinalize() for completed runs."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)

            
            context.tool_call_count = getattr(agent, 'tool_call_count', 0)
            context.tool_limit_reached = getattr(agent, 'tool_limit_reached', False)
            
            # Finalize run messages BEFORE marking completed
            # This extracts new messages from chat_history (using _run_boundaries)
            # and sets them to context.messages
            context.finalize_run_messages()
            
            # Stop usage timer and finalize run-level usage
            context.stop_usage_timer()
            
            # Note: Session-level usage is updated in AgentSessionMemory.asave()
            # when the session is saved to storage
            
            # Mark completed BEFORE save so all state is captured (including resolved requirements)
            context.mark_completed()
                
            # Create step_result FIRST so we can include it in save
            messages_count = len(context.messages) if context.messages else 0
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message=f"Session saved ({messages_count} messages)",
                execution_time=time.time() - start_time,
            )
            

            if not agent.memory:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="No memory configured",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            # Get memory manager from pipeline registry and finalize it
            memory_manager = None
            if pipeline_manager:
                memory_manager = pipeline_manager.get_manager('memory_manager')
            
            # Set run output in memory manager before finalizing
            if memory_manager:
                memory_manager.set_run_output(context)
            
            # Add streaming events BEFORE save so they're included
            if context.is_streaming:
                memory_type = None
                if getattr(agent.memory, 'full_session_memory_enabled', False):
                    memory_type = 'full_session'
                else:
                    memory_type = 'session'
                
                from upsonic.utils.agent.events import ayield_memory_update_event
                messages_count = len(context.messages) if context.messages else 0
                async for event in ayield_memory_update_event(
                    run_id=context.run_id or "",
                    messages_added=messages_count,
                    memory_type=memory_type
                ):
                    context.events.append(event)
            
            
            # Finalize memory manager (this saves the session)
            if memory_manager:
                await memory_manager.afinalize()
                session_saved = True
            else:
                # Fallback to direct save if manager not found
                try:
                    await agent.memory.save_session_async(
                        output=context,
                        agent_id=agent.agent_id,
                    )
                    session_saved = True
                except Exception as save_error:
                    session_saved = False
                    if agent.debug:
                        from upsonic.utils.printing import warning_log
                        warning_log(f"Failed to save session: {save_error}", "MemorySaveStep")
            
            if agent.debug and agent.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                memory_type = None
                if getattr(agent.memory, 'full_session_memory_enabled', False):
                    memory_type = 'full_session'
                elif getattr(agent.memory, 'summary_memory_enabled', False):
                    memory_type = 'summary'
                else:
                    memory_type = 'session'
                
                debug_log_level2(
                    "Session saved to storage",
                    "MemorySaveStep",
                    debug=agent.debug,
                    debug_level=agent.debug_level,
                    session_saved=session_saved,
                    memory_type=memory_type,
                    session_id=getattr(agent.memory, 'session_id', None),
                    user_id=getattr(agent.memory, 'user_id', None),
                    run_id=context.run_id,
                    status=str(context.status) if context else None,
                    messages_count=len(context.messages) if context.messages else 0,
                )
            
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise


            
class ReliabilityStep(Step):
    """Apply reliability layer processing."""
    
    @property
    def name(self) -> str:
        return "reliability"
    
    @property
    def description(self) -> str:
        return "Apply reliability layer"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Apply reliability layer with async context manager."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        from upsonic.agent.context_managers import ReliabilityManager
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            if not agent.reliability_layer:
                if context.is_streaming:
                    from upsonic.utils.agent.events import ayield_reliability_event
                    async for event in ayield_reliability_event(
                        run_id=context.run_id or "",
                        reliability_applied=False
                    ):
                        context.events.append(event)
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="No reliability layer",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            if hasattr(task, '_cached_result') and task._cached_result:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to cache hit",
                    execution_time=time.time() - start_time,
                )
                return step_result
            if hasattr(task, '_policy_blocked') and task._policy_blocked:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to policy block",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            original_output = context.output
            
            reliability_manager = ReliabilityManager(
                task,
                agent.reliability_layer,
                model
            )
            
            await reliability_manager.aprepare()
            
            try:
                processed_task = await reliability_manager.process_task(task)
                task = processed_task
                context.output = processed_task.response
            finally:
                await reliability_manager.afinalize()

            modifications_made = str(original_output) != str(context.output)
            
            if agent.debug and agent.debug_level >= 2:
                from upsonic.utils.printing import debug_log_level2
                debug_log_level2(
                    "Reliability layer applied",
                    "ReliabilityStep",
                    debug=agent.debug,
                    debug_level=agent.debug_level,
                    modifications_made=modifications_made,
                    original_output_preview=str(original_output)[:300] if original_output else None,
                    processed_output_preview=str(context.output)[:300] if context.output else None,
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
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="Reliability applied",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
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
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Apply agent policy to output with feedback loop support."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        policy_count = len(agent.agent_policy_manager.policies) if hasattr(agent.agent_policy_manager, 'policies') else 0
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
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
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="No agent policy or no response",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            if hasattr(task, '_cached_result') and task._cached_result:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to cache hit",
                    execution_time=time.time() - start_time,
                )
                return step_result
            if hasattr(task, '_policy_blocked') and task._policy_blocked:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to policy block",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            original_response = task.response
            
            agent.agent_policy_manager.reset_retry_count()
            
            max_iterations = agent.agent_policy_manager.feedback_loop_count + 1
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                processed_task, feedback_message = await agent._apply_agent_policy(task, context)
                
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
                    task = processed_task
                    context.output = processed_task.response
                    
                    step_result = StepResult(
                        name=self.name,
                        step_number=step_number,
                        status=StepStatus.COMPLETED,
                        message=f"Agent policies applied after {iteration} iteration(s)",
                        execution_time=time.time() - start_time,
                    )
                    return step_result
                
                if agent.debug:
                    from upsonic.utils.printing import policy_feedback_retry
                    policy_feedback_retry(
                        policy_type="agent_policy",
                        retry_count=iteration,
                        max_retries=max_iterations - 1
                    )
                agent.agent_policy_manager.increment_retry_count()
                
                await self._rerun_model_with_feedback(context, task, agent, model, feedback_message)
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="Agent policies applied (exhausted retries)",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
    async def _rerun_model_with_feedback(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", feedback_message: str) -> None:
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
        if context.response and context.response not in context.chat_history:
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
        context.output = output

class CacheStorageStep(Step):
    """Store the response in cache."""
    
    @property
    def name(self) -> str:
        return "cache_storage"
    
    @property
    def description(self) -> str:
        return "Store response in cache"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Store response in cache."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            if not (task.enable_cache and task.response):
                if context.is_streaming:
                    from upsonic.utils.agent.events import ayield_cache_stored_event
                    async for event in ayield_cache_stored_event(
                        run_id=context.run_id or "",
                        cache_method='disabled',
                        duration_minutes=None
                    ):
                        context.events.append(event)
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Caching not enabled or no response",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
            if hasattr(task, '_cached_result') and task._cached_result:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Already from cache",
                    execution_time=time.time() - start_time,
                )
                return step_result
            if hasattr(task, '_policy_blocked') and task._policy_blocked:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Not caching blocked content",
                    execution_time=time.time() - start_time,
                )
                return step_result
            
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
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="Response cached",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
            
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
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model") -> StepResult:
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
        context: "AgentRunOutput", 
        task: "Task", 
        agent: "Agent", 
        model: "Model",
        step_number: int = 0,
        pipeline_manager: Optional[Any] = None
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
            cached_content = str(context.output)
            
            # Stream the cached content character by character
            for char in cached_content:
                yield TextDeltaEvent(run_id=run_id, content=char)
                accumulated_text += char
            
            yield TextCompleteEvent(run_id=run_id, content=cached_content)
            yield FinalOutputEvent(run_id=run_id, output=cached_content, output_type='cached')
            
            context.output = cached_content
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
                    final_output_preview=str(context.output)[:500] if context.output else None
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
            context.output = output
            
            # Emit final output event
            yield FinalOutputEvent(
                run_id=run_id,
                output=output,
                output_type='structured' if not isinstance(output, str) else 'text'
            )
            # Note: Message finalization is deferred to StreamMemoryMessageTrackingStep
            # This ensures all steps (including AgentPolicyStep feedback loop) 
            # can add their messages to chat_history before we extract new messages
            
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
        context: "AgentRunOutput", 
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
                            # Track time to first token in usage
                            context.set_usage_time_to_first_token()
                    
                    yield agent_event
        
        # Get the final response from the stream
        final_response = stream.get()
        context.response = final_response
        
        # Add the final response to chat_history for message tracking
        # This ensures the response is included in session memory
        context.chat_history.append(final_response)
        
        # Update usage from streaming response
        if hasattr(final_response, 'usage') and final_response.usage:
            context.update_usage_from_response(final_response.usage)
            
            # Calculate and set cost
            try:
                from upsonic.utils.usage import calculate_cost_from_usage
                cost_value = calculate_cost_from_usage(final_response.usage, model)
                context.set_usage_cost(cost_value)
            except Exception:
                pass  # Silently ignore cost calculation errors
        
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
                # Add tool results to chat_history (response already added above)
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
            
            # Add tool results to chat_history (response already added above)
            context.chat_history.append(ModelRequest(parts=tool_results))
            
            # Reset accumulated_text for new streaming round
            accumulated_text = ""
            
            # Recursively continue streaming with tool results
            async for event in self._stream_with_tool_calls(context, task, agent, model, model_params, accumulated_text, first_token_time):
                yield event


class StreamMemoryMessageTrackingStep(Step):
    """
    Save AgentSession to storage for streaming execution. This is the LAST step.
    
    For HITL (errors, cancel, external tools), session saving is handled
    by PipelineManager's exception handlers.
    
    Message finalization is done HERE (not in StreamModelExecutionStep) to ensure
    all steps that may add messages (e.g., AgentPolicyStep feedback loop) have
    completed before we extract the new messages.
    """
    
    @property
    def name(self) -> str:
        return "stream_memory_message_tracking"
    
    @property
    def description(self) -> str:
        return "Save session to storage"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int = 0, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Save session via Memory.save_session_async for completed streaming runs."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            # Skip if cache hit or policy blocked
            if hasattr(task, '_cached_result') and task._cached_result:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to cache hit",
                    execution_time=time.time() - start_time
                )
                return step_result
            
            if hasattr(task, '_policy_blocked') and task._policy_blocked:
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message="Skipped due to policy block",
                    execution_time=time.time() - start_time
                )
                return step_result
            
            # Finalize run messages BEFORE marking completed and saving
            # This extracts new messages from chat_history (using _run_boundaries)
            # and sets them to context.messages
            context.finalize_run_messages()
            
            # Stop usage timer and finalize run-level usage
            context.stop_usage_timer()
            
            # Note: Session-level usage is updated in AgentSessionMemory.asave()
            # when the session is saved to storage
            
            # Mark completed
            context.mark_completed()
            
            # Handle memory save
            session_saved = False
            memory_type = None
            messages_count = len(context.messages) if context.messages else 0
            
            if not agent.memory:
                # No memory configured - still emit event for visibility
                if context.is_streaming:
                    from upsonic.utils.agent.events import ayield_memory_update_event
                    async for event in ayield_memory_update_event(
                        run_id=context.run_id or "",
                        memory_type=None,
                        messages_added=messages_count
                    ):
                        context.events.append(event)
                
                step_result = StepResult(
                    name=self.name,
                    step_number=step_number,
                    status=StepStatus.COMPLETED,
                    message=f"No memory configured ({messages_count} messages finalized)",
                    execution_time=time.time() - start_time
                )
                return step_result
            
            # Save session to storage
            try:
                await agent.memory.save_session_async(
                    output=context,
                    agent_id=agent.agent_id,
                )
                session_saved = True
            except Exception as save_error:
                session_saved = False
                if agent.debug:
                    from upsonic.utils.printing import warning_log
                    warning_log(f"Failed to save session: {save_error}", "StreamMemoryMessageTrackingStep")
            
            # Get memory type and emit event
            if getattr(agent.memory, 'full_session_memory_enabled', False):
                memory_type = 'full_session'
            else:
                memory_type = 'session'
            
            if context.is_streaming:
                from upsonic.utils.agent.events import ayield_memory_update_event
                async for event in ayield_memory_update_event(
                    run_id=context.run_id or "",
                    memory_type=memory_type,
                    messages_added=messages_count
                ):
                    context.events.append(event)
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message=f"Session saved" if session_saved else "Session save failed",
                execution_time=time.time() - start_time
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time
            )
            raise
            
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)


class StreamFinalizationStep(Step):
    """Finalize the streaming execution."""
    
    @property
    def name(self) -> str:
        return "stream_finalization"
    
    @property
    def description(self) -> str:
        return "Finalize streaming execution"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int = 0, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Finalize streaming execution."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        from upsonic.run.events.events import ExecutionCompleteEvent
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            # Ensure final_output is set from task response if not already set
            if context.output is None and task:
                context.output = task.response
            
            # Determine output type
            output_type = 'text'
            if hasattr(task, '_cached_result') and task._cached_result:
                output_type = 'cached'
            elif hasattr(task, '_policy_blocked') and task._policy_blocked:
                output_type = 'blocked'
            elif context.output and not isinstance(context.output, str):
                output_type = 'structured'
            # End the task
            task.task_end()

            if context.is_streaming:
                output_preview = str(context.output)[:100] if context.output else None
                from upsonic.utils.agent.events import ayield_execution_complete_event
                async for event in ayield_execution_complete_event(
                    run_id=context.run_id or "",
                    output_type=output_type,
                    has_output=context.output is not None,
                    output_preview=output_preview,
                    total_tool_calls=context.tool_call_count,
                    total_duration=task.duration if task.duration else None
                ):
                    context.events.append(event)
                # RunCompletedEvent is emitted by manager after pipeline end
            
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="Streaming finalized",
                execution_time=time.time() - start_time
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time
            )
            raise
            
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)


class FinalizationStep(Step):
    """Finalize the execution."""
    
    @property
    def name(self) -> str:
        return "finalization"
    
    @property
    def description(self) -> str:
        return "Finalize execution"
    
    async def execute(self, context: "AgentRunOutput", task: "Task", agent: "Agent", model: "Model", step_number: int, pipeline_manager: Optional[Any] = None) -> StepResult:
        """Finalize execution."""
        from upsonic.run.cancel import raise_if_cancelled
        from upsonic.exceptions import RunCancelledException
        
        start_time = time.time()
        step_result: Optional[StepResult] = None
        
        try:
            if agent and hasattr(agent, 'run_id') and agent.run_id:
                raise_if_cancelled(agent.run_id)
            
            if context.output is None and task:
                context.output = task.response
            
            # De-anonymize response if anonymization was applied
            if task and hasattr(task, '_anonymization_map') and task._anonymization_map:
                from upsonic.safety_engine.anonymization import deanonymize_content
                
                if context.output is not None:
                    if isinstance(context.output, str):
                        context.output = deanonymize_content(context.output, task._anonymization_map)
                    elif hasattr(context.output, 'model_dump_json'):
                        # For Pydantic models, de-anonymize the JSON representation
                        # and reconstruct (this is a best-effort approach)
                        try:
                            import json
                            json_str = context.output.model_dump_json()
                            deanonymized_json = deanonymize_content(json_str, task._anonymization_map)
                            context.output = type(context.output).model_validate_json(deanonymized_json)
                        except Exception:
                            pass  # If it fails, keep original output
                
                # Also de-anonymize task._response if set
                if hasattr(task, '_response') and task._response:
                    if isinstance(task._response, str):
                        task._response = deanonymize_content(task._response, task._anonymization_map)
                
                if agent.debug:
                    from upsonic.utils.printing import debug_log
                    debug_log(
                        f"De-anonymized response using {len(task._anonymization_map)} mappings",
                        "FinalizationStep"
                    )
            
            task.task_end()

            output_type = 'text'
            if hasattr(task, '_cached_result') and task._cached_result:
                output_type = 'cached'
            elif hasattr(task, '_policy_blocked') and task._policy_blocked:
                output_type = 'blocked'
            elif context.output and not isinstance(context.output, str):
                output_type = 'structured'
            
            if context.is_streaming:
                output_preview = str(context.output)[:100] if context.output else None
                total_duration = task.duration if task.duration else None
                from upsonic.utils.agent.events import ayield_execution_complete_event
                async for event in ayield_execution_complete_event(
                    run_id=context.run_id or "",
                    output_type=output_type,
                    has_output=context.output is not None,
                    output_preview=output_preview,
                    total_tool_calls=context.tool_call_count,
                    total_duration=total_duration
                ):
                    context.events.append(event)

            if task and not task.not_main_task:
                from upsonic.utils.printing import print_price_id_summary, price_id_summary
                if task.price_id in price_id_summary:
                    print_price_id_summary(task.price_id, task, print_output=context.print_flag)

            try:
                from upsonic.tools.mcp import MCPHandler, MultiMCPHandler
                if task and hasattr(task, 'tools') and task.tools:
                    agent_tools_set = set(agent.tools) if agent.tools else set()
                    for tool in task.tools:
                        if isinstance(tool, (MCPHandler, MultiMCPHandler)):
                            if tool not in agent_tools_set:
                                try:
                                    await tool.close()
                                except (RuntimeError, Exception) as e:
                                    error_msg = str(e).lower()
                                    if "event loop is closed" not in error_msg and "loop" not in error_msg:
                                        if agent.debug:
                                            from upsonic.utils.printing import console
                                            console.print(f"[yellow]Warning: Error closing task-level MCP handler: {e}[/yellow]")
            except Exception as e:
                if agent.debug:
                    from upsonic.utils.printing import console
                    console.print(f"[yellow]Warning: Error during MCP handler cleanup: {e}[/yellow]")

            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.COMPLETED,
                message="Execution finalized",
                execution_time=time.time() - start_time,
            )
            return step_result
            
        except RunCancelledException as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.CANCELLED,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
            
        except Exception as e:
            step_result = StepResult(
                name=self.name,
                step_number=step_number,
                status=StepStatus.ERROR,
                message=str(e)[:500],
                execution_time=time.time() - start_time,
            )
            raise
        finally:
            if step_result:
                self._finalize_step_result(step_result, context)
