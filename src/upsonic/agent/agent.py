import asyncio
import copy
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union, TYPE_CHECKING

PromptCompressor = None

from upsonic import _utils
from upsonic.agent.base import BaseAgent
from upsonic.agent.run_result import RunResult, StreamRunResult
from upsonic._utils import now_utc
from upsonic.utils.retry import retryable
from upsonic.utils.validators import validate_attachments_for_model

if TYPE_CHECKING:
    from upsonic.models import Model, ModelRequest, ModelMessage, ModelRequestParameters, ModelResponse
    from upsonic.messages import ModelResponseStreamEvent, FinalResultEvent, PartStartEvent, PartDeltaEvent, TextPart, ToolCallPart, ToolReturnPart
    from upsonic.tasks.tasks import Task
    from upsonic.storage.memory.memory import Memory
    from upsonic.canvas.canvas import Canvas
    from upsonic.models.settings import ModelSettings
    from upsonic.profiles import ModelProfile
    from upsonic.reflection import ReflectionConfig, ReflectionProcessor
    from upsonic.safety_engine.base import Policy
    from upsonic.safety_engine.exceptions import DisallowedOperation
    from upsonic.safety_engine.models import PolicyInput, RuleOutput
    from upsonic.tools import ToolManager, ToolContext, ToolDefinition
    from upsonic.tools.processor import ExternalExecutionPause
    from upsonic.usage import RequestUsage
    from upsonic.agent.context_managers import (
        CallManager, ContextManager, ReliabilityManager, 
        MemoryManager, SystemPromptManager, TaskManager
    )
else:
    Model = "Model"
    ModelRequest = "ModelRequest"
    ModelMessage = "ModelMessage"
    ModelRequestParameters = "ModelRequestParameters"
    ModelResponse = "ModelResponse"
    Task = "Task"
    Memory = "Memory"
    Canvas = "Canvas"
    ModelSettings = "ModelSettings"
    ModelProfile = "ModelProfile"
    ReflectionConfig = "ReflectionConfig"
    ReflectionProcessor = "ReflectionProcessor"
    Policy = "Policy"
    DisallowedOperation = "DisallowedOperation"
    PolicyInput = "PolicyInput"
    RuleOutput = "RuleOutput"
    ToolManager = "ToolManager"
    ToolContext = "ToolContext"
    ToolDefinition = "ToolDefinition"
    ExternalExecutionPause = "ExternalExecutionPause"
    RequestUsage = "RequestUsage"
    validate_attachments_for_model = "validate_attachments_for_model"
    CallManager = "CallManager"
    ContextManager = "ContextManager"
    ReliabilityManager = "ReliabilityManager"
    MemoryManager = "MemoryManager"
    SystemPromptManager = "SystemPromptManager"
    TaskManager = "TaskManager"

RetryMode = Literal["raise", "return_false"]


class Agent(BaseAgent):
    """
    A comprehensive, high-level AI Agent that integrates all framework components.
    
    This Agent class provides:
    - Complete model abstraction through Model/Provider/Profile system
    - Advanced tool handling with ToolManager and Orchestrator
    - Streaming and non-streaming execution modes
    - Memory management and conversation history
    - Context management and prompt engineering
    - Caching capabilities
    - Safety policies and guardrails
    - Reliability layers
    - Canvas integration
    - External tool execution support
    
    Usage:
        Basic usage:
        ```python
        from upsonic import Agent, Task
        
        agent = Agent("openai/gpt-4o")
        task = Task("What is 1 + 1?")
        result = agent.do(task)
        ```
        
        Advanced usage:
        ```python
        agent = Agent(
            model="openai/gpt-4o",
            name="Math Teacher",
            memory=memory,
            enable_thinking_tool=True,
            user_policy=safety_policy
        )
        result = agent.stream(task)
        ```
    """
    
    def __init__(
        self,
        model: Union[str, "Model"] = "openai/gpt-4o",
        *,
        name: Optional[str] = None,
        memory: Optional["Memory"] = None,
        debug: bool = False,
        company_url: Optional[str] = None,
        company_objective: Optional[str] = None,
        company_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        reflection: Optional[str] = None,
        compression_strategy: Literal["none", "simple", "llmlingua"] = "none",
        compression_settings: Optional[Dict[str, Any]] = None,
        reliability_layer: Optional["ReliabilityProcessor"] = None,
        agent_id_: Optional[str] = None,
        canvas: Optional["Canvas"] = None,
        retry: int = 1,
        mode: RetryMode = "raise",
        role: Optional[str] = None,
        goal: Optional[str] = None,
        instructions: Optional[str] = None,
        education: Optional[str] = None,
        work_experience: Optional[str] = None,
        feed_tool_call_results: bool = False,
        show_tool_calls: bool = True,
        tool_call_limit: int = 5,
        enable_thinking_tool: bool = False,
        enable_reasoning_tool: bool = False,
        user_policy: Optional["Policy"] = None,
        agent_policy: Optional["Policy"] = None,
        settings: Optional["ModelSettings"] = None,
        profile: Optional["ModelProfile"] = None,
        reflection_config: Optional["ReflectionConfig"] = None,
    ):
        """
        Initialize the Agent with comprehensive configuration options.
        
        Args:
            model: Model identifier or Model instance
            name: Agent name for identification
            memory: Memory instance for conversation history
            debug: Enable debug logging
            company_url: Company URL for context
            company_objective: Company objective for context
            company_description: Company description for context
            system_prompt: Custom system prompt
            reflection: Reflection capabilities
            compression_strategy: The method for context compression ('none', 'simple', 'llmlingua').
            compression_settings: A dictionary of settings for the chosen strategy.
                - For "simple": {"max_length": 2000}
                - For "llmlingua": {"ratio": 0.5, "model_name": "...", "instruction": "..."}
            reliability_layer: Reliability layer for robustness
            agent_id_: Specific agent ID
            canvas: Canvas instance for visual interactions
            retry: Number of retry attempts
            mode: Retry mode behavior
            role: Agent role
            goal: Agent goal
            instructions: Specific instructions
            education: Agent education background
            work_experience: Agent work experience
            feed_tool_call_results: Include tool results in memory
            show_tool_calls: Display tool calls
            tool_call_limit: Maximum tool calls per execution
            enable_thinking_tool: Enable orchestrated thinking
            enable_reasoning_tool: Enable reasoning capabilities
            user_policy: User input safety policy
            agent_policy: Agent output safety policy
            settings: Model-specific settings
            profile: Model profile configuration
            reflection_config: Configuration for reflection and self-evaluation
        """
        from upsonic.models import infer_model
        self.model = infer_model(model)
        self.name = name
        self.agent_id_ = agent_id_
        
        self.role = role
        self.goal = goal
        self.instructions = instructions
        self.education = education
        self.work_experience = work_experience
        self.system_prompt = system_prompt
        
        self.company_url = company_url
        self.company_objective = company_objective
        self.company_description = company_description
        
        self.debug = debug
        self.reflection = reflection

        self.compression_strategy = compression_strategy
        self.compression_settings = compression_settings or {}
        self._prompt_compressor = None
        if self.compression_strategy == "llmlingua":
            try:
                from llmlingua import PromptCompressor
            except ImportError:
                raise ImportError(
                    "The 'llmlingua' package is required for the 'llmlingua' compression strategy. "
                    "Please install it using: pip install llmlingua"
                )
            model_name = self.compression_settings.get(
                "model_name", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
            )
            self._prompt_compressor = PromptCompressor(model_name=model_name, use_llmlingua2=True)

        self.reliability_layer = reliability_layer
        
        if retry < 1:
            raise ValueError("The 'retry' count must be at least 1.")
        if mode not in ("raise", "return_false"):
            raise ValueError(f"Invalid retry_mode '{mode}'. Must be 'raise' or 'return_false'.")
        
        self.retry = retry
        self.mode = mode
        
        self.show_tool_calls = show_tool_calls
        self.tool_call_limit = tool_call_limit
        self.enable_thinking_tool = enable_thinking_tool
        self.enable_reasoning_tool = enable_reasoning_tool
        
        self.memory = memory
        if self.memory:
            self.memory.feed_tool_call_results = feed_tool_call_results
        
        self.canvas = canvas
        
        self.user_policy = user_policy
        self.agent_policy = agent_policy
        
        self.reflection_config = reflection_config
        if reflection_config:
            from upsonic.reflection import ReflectionProcessor
            self.reflection_processor = ReflectionProcessor(reflection_config)
        else:
            self.reflection_processor = None
        
        if settings:
            self.model._settings = settings
        if profile:
            self.model._profile = profile
        
        from upsonic.cache import CacheManager
        from upsonic.tools import ToolManager
        
        self._cache_manager = CacheManager(session_id=f"agent_{self.agent_id}")
        self.tool_manager = ToolManager()
        
        self.tool_call_count = 0
        self._current_messages = []
        self._tool_call_count = 0
        
        self._run_result = RunResult(output=None)
        
        self._stream_run_result = StreamRunResult()
        
        self._setup_policy_models()
    
    def _setup_policy_models(self) -> None:
        """Setup model references for safety policies."""
        policies = [self.user_policy, self.agent_policy]
        
        for policy in policies:
            if policy is None:
                continue
            
            if hasattr(policy, 'base_llm') and policy.base_llm is None:
                from upsonic.safety_engine.llm.upsonic_llm import UpsonicLLMProvider
                policy.base_llm = UpsonicLLMProvider(
                    agent_name="Policy Base Agent",
                    model=self.model
                )
    
    @property
    def agent_id(self) -> str:
        """Get or generate agent ID."""
        if self.agent_id_ is None:
            self.agent_id_ = str(uuid.uuid4())
        return self.agent_id_
    
    def get_agent_id(self) -> str:
        """Get display-friendly agent ID."""
        if self.name:
            return self.name
        return f"Agent_{self.agent_id[:8]}"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for this agent's session."""
        return self._cache_manager.get_cache_stats()
    
    def clear_cache(self) -> None:
        """Clear the agent's session cache."""
        self._cache_manager.clear_cache()
    
    def get_run_result(self) -> RunResult:
        """
        Get the persistent RunResult that accumulates messages across all executions.
        
        Returns:
            RunResult: The agent's run result containing all messages and the last output
        """
        return self._run_result
    
    def reset_run_result(self) -> None:
        """
        Reset the RunResult to start fresh (clears all accumulated messages).
        
        Useful when you want to start a new conversation thread without creating a new agent.
        """
        self._run_result = RunResult(output=None)
    
    def get_stream_run_result(self) -> "StreamRunResult":
        """
        Get the persistent StreamRunResult that accumulates messages across all streaming executions.
        
        Returns:
            StreamRunResult: The agent's stream run result containing all messages and the last output
        """
        return self._stream_run_result
    
    def reset_stream_run_result(self) -> None:
        """
        Reset the StreamRunResult to start fresh (clears all accumulated messages).
        
        Useful when you want to start a new conversation thread without creating a new agent.
        """
        self._stream_run_result = StreamRunResult()
    
    def _setup_tools(self, task: "Task") -> None:
        """Setup tools with ToolManager for the current task."""
        self._tool_limit_reached = False
        
        if not task.tools:
            return
        
        from upsonic.tools import ToolContext
        self._context = ToolContext(
            deps=getattr(self, 'dependencies', None),
            agent_id=self.name,
            max_retries=self.retry,
            tool_call_limit=self.tool_call_limit
        )
        
        is_thinking_enabled = self.enable_thinking_tool
        if task.enable_thinking_tool is not None:
            is_thinking_enabled = task.enable_thinking_tool
        
        is_reasoning_enabled = self.enable_reasoning_tool
        if task.enable_reasoning_tool is not None:
            is_reasoning_enabled = task.enable_reasoning_tool

        if is_reasoning_enabled and not is_thinking_enabled:
            raise ValueError("Configuration error: 'enable_reasoning_tool' cannot be True if 'enable_thinking_tool' is False.")

        agent_for_this_run = copy.copy(self)
        agent_for_this_run.enable_thinking_tool = is_thinking_enabled
        agent_for_this_run.enable_reasoning_tool = is_reasoning_enabled

        final_tools = list(task.tools)
        if is_thinking_enabled:
            from upsonic.tools import plan_and_execute
            final_tools.append(plan_and_execute)
        
        self.tool_manager.register_tools(
            tools=final_tools,
            context=self._context,
            task=task,
            agent_instance=agent_for_this_run
        )
    
    async def _build_model_request(self, task: "Task", memory_handler: Optional["MemoryManager"]) -> List["ModelRequest"]:
        """Build the complete message history for the model request."""
        from upsonic.agent.context_managers import SystemPromptManager, ContextManager
        from upsonic.messages import SystemPromptPart, UserPromptPart, ModelRequest
        
        messages = []
        
        message_history = memory_handler.get_message_history()
        messages.extend(message_history)
        
        system_prompt_manager = SystemPromptManager(self, task)
        context_manager = ContextManager(self, task, getattr(task, 'state', None))
        
        async with system_prompt_manager.manage_system_prompt(memory_handler) as sp_handler, \
                   context_manager.manage_context(memory_handler) as ctx_handler:
            
            task_input = task.build_agent_input()
            user_part = UserPromptPart(content=task_input)
            
            parts = []
            
            if not messages:
                system_prompt = sp_handler.get_system_prompt()
                if system_prompt:
                    system_part = SystemPromptPart(content=system_prompt)
                    parts.append(system_part)
            
            parts.append(user_part)
            
            current_request = ModelRequest(parts=parts)
            messages.append(current_request)
            
            if self.compression_strategy != "none" and ctx_handler:
                context_prompt = ctx_handler.get_context_prompt()
                if context_prompt:
                    compressed_context = self._compress_context(context_prompt)
                    task.context_formatted = compressed_context
        return messages
    
    async def _build_model_request_with_input(
        self, 
        task: "Task", 
        memory_handler: Optional["MemoryManager"], 
        current_input: Any, 
        temporary_message_history: List["ModelRequest"]
    ) -> List["ModelRequest"]:
        """Build model request with custom input and message history for guardrail retries."""
        from upsonic.agent.context_managers import SystemPromptManager, ContextManager
        from upsonic.messages import SystemPromptPart, UserPromptPart, ModelRequest
        
        messages = list(temporary_message_history)
        
        system_prompt_manager = SystemPromptManager(self, task)
        context_manager = ContextManager(self, task, getattr(task, 'state', None))
        
        async with system_prompt_manager.manage_system_prompt(memory_handler) as sp_handler, \
                   context_manager.manage_context(memory_handler) as ctx_handler:
            
            user_part = UserPromptPart(content=current_input)
            
            parts = []
            
            if not messages:
                system_prompt = sp_handler.get_system_prompt()
                if system_prompt:
                    system_part = SystemPromptPart(content=system_prompt)
                    parts.append(system_part)
            
            parts.append(user_part)
            
            current_request = ModelRequest(parts=parts)
            messages.append(current_request)
            
            if self.compression_strategy != "none" and ctx_handler:
                context_prompt = ctx_handler.get_context_prompt()
                if context_prompt:
                    compressed_context = self._compress_context(context_prompt)
                    task.context_formatted = compressed_context
        
        return messages
    
    def _build_model_request_parameters(self, task: "Task") -> "ModelRequestParameters":
        """Build model request parameters including tools and structured output."""
        from pydantic import BaseModel
        from upsonic.output import OutputObjectDefinition
        from upsonic.models import ModelRequestParameters
        
        if hasattr(self, '_tool_limit_reached') and self._tool_limit_reached:
            tool_definitions = []
        elif self.tool_call_limit and self._tool_call_count >= self.tool_call_limit:
            tool_definitions = []
            self._tool_limit_reached = True
        else:
            tool_definitions = self.tool_manager.get_tool_definitions()
        
        output_mode = 'text'
        output_object = None
        allow_text_output = True
        
        if task.response_format and task.response_format != str and task.response_format is not str:
            if isinstance(task.response_format, type) and issubclass(task.response_format, BaseModel):
                output_mode = 'native'
                allow_text_output = False
                
                schema = task.response_format.model_json_schema()
                output_object = OutputObjectDefinition(
                    json_schema=schema,
                    name=task.response_format.__name__,
                    description=task.response_format.__doc__,
                    strict=True
                )
        
        return ModelRequestParameters(
            function_tools=tool_definitions,
            output_mode=output_mode,
            output_object=output_object,
            allow_text_output=allow_text_output
        )
    
    async def _execute_tool_calls(self, tool_calls: List["ToolCallPart"]) -> List["ToolReturnPart"]:
        """
        Execute tool calls and return results.
        
        Handles both sequential and parallel execution based on tool configuration.
        Tools marked as sequential will be executed one at a time.
        Other tools can be executed in parallel if multiple are called.
        """
        from upsonic.messages import ToolReturnPart
        from upsonic.tools import ToolContext
        
        if not tool_calls:
            return []
        
        if self.tool_call_limit and self._tool_call_count >= self.tool_call_limit:
            error_results = []
            for tool_call in tool_calls:
                error_results.append(ToolReturnPart(
                    tool_name=tool_call.tool_name,
                    content=f"Tool call limit of {self.tool_call_limit} reached. Cannot execute more tools.",
                    tool_call_id=tool_call.tool_call_id
                ))
            self._tool_limit_reached = True
            return error_results
        
        tool_defs = {td.name: td for td in self.tool_manager.get_tool_definitions()}
        
        sequential_calls = []
        parallel_calls = []
        
        for tool_call in tool_calls:
            tool_def = tool_defs.get(tool_call.tool_name)
            if tool_def and tool_def.sequential:
                sequential_calls.append(tool_call)
            else:
                parallel_calls.append(tool_call)
        
        results = []
        
        for tool_call in sequential_calls:
            try:
                task_id = None
                if hasattr(self, 'current_task') and self.current_task:
                    task_id = getattr(self.current_task, 'id', None) or getattr(self.current_task, 'price_id', None)
                
                result = await self.tool_manager.execute_tool(
                    tool_name=tool_call.tool_name,
                    args=tool_call.args_as_dict(),
                    context=ToolContext(
                        deps=getattr(self, 'dependencies', None),
                        agent_id=self.name,
                        task_id=task_id,
                        messages=self._current_messages,
                        retry=0,
                        max_retries=self.retry,
                        tool_call_count=self._tool_call_count,
                        tool_call_limit=self.tool_call_limit
                    ),
                    tool_call_id=tool_call.tool_call_id
                )
                
                self._tool_call_count += 1
                if hasattr(self, '_context') and self._context:
                    self._context.tool_call_count = self._tool_call_count
                
                tool_return = ToolReturnPart(
                    tool_name=result.tool_name,
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                    timestamp=now_utc()
                )
                results.append(tool_return)
                
            except ExternalExecutionPause as e:
                raise e
            except Exception as e:
                error_return = ToolReturnPart(
                    tool_name=tool_call.tool_name,
                    content=f"Error executing tool: {str(e)}",
                    tool_call_id=tool_call.tool_call_id,
                    timestamp=now_utc()
                )
                results.append(error_return)
        
        if parallel_calls:
            async def execute_single_tool(tool_call: "ToolCallPart") -> "ToolReturnPart":
                """Execute a single tool call and return the result."""
                try:
                    task_id = None
                    if hasattr(self, 'current_task') and self.current_task:
                        task_id = getattr(self.current_task, 'id', None) or getattr(self.current_task, 'price_id', None)
                    
                    result = await self.tool_manager.execute_tool(
                        tool_name=tool_call.tool_name,
                        args=tool_call.args_as_dict(),
                        context=ToolContext(
                            deps=getattr(self, 'dependencies', None),
                            agent_id=self.name,
                            task_id=task_id,
                            messages=self._current_messages,
                            retry=0,
                            max_retries=self.retry,
                            tool_call_count=self._tool_call_count,
                            tool_call_limit=self.tool_call_limit
                        ),
                        tool_call_id=tool_call.tool_call_id
                    )
                    
                    return ToolReturnPart(
                        tool_name=result.tool_name,
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                        timestamp=now_utc()
                    )
                    
                except ExternalExecutionPause:
                    raise
                except Exception as e:
                    return ToolReturnPart(
                        tool_name=tool_call.tool_name,
                        content=f"Error executing tool: {str(e)}",
                        tool_call_id=tool_call.tool_call_id,
                        timestamp=now_utc()
                    )
            
            parallel_results = await asyncio.gather(
                *[execute_single_tool(tc) for tc in parallel_calls],
                return_exceptions=False
            )
            
            self._tool_call_count += len(parallel_calls)
            if hasattr(self, '_context') and self._context:
                self._context.tool_call_count = self._tool_call_count
            
            results.extend(parallel_results)
        
        return results
    
    async def _handle_model_response(
        self, 
        response: "ModelResponse", 
        messages: List["ModelRequest"]
    ) -> "ModelResponse":
        """Handle model response including tool calls."""
        from upsonic.messages import ToolCallPart, ToolReturnPart, TextPart, UserPromptPart, ModelRequest, ModelResponse
        
        if hasattr(self, '_tool_limit_reached') and self._tool_limit_reached:
            return response
        
        tool_calls = [
            part for part in response.parts 
            if isinstance(part, ToolCallPart)
        ]
        
        if tool_calls:
            tool_results = await self._execute_tool_calls(tool_calls)
            
            if hasattr(self, '_tool_limit_reached') and self._tool_limit_reached:
                tool_request = ModelRequest(parts=tool_results)
                messages.append(response)
                messages.append(tool_request)
                
                limit_notification = UserPromptPart(
                    content=f"[SYSTEM] Tool call limit of {self.tool_call_limit} has been reached. "
                    f"No more tools are available. Please provide a final response based on the information you have."
                )
                limit_message = ModelRequest(parts=[limit_notification])
                messages.append(limit_message)
                
                model_params = self._build_model_request_parameters(getattr(self, 'current_task', None))
                model_params = self.model.customize_request_parameters(model_params)
                
                final_response = await self.model.request(
                    messages=messages,
                    model_settings=self.model.settings,
                    model_request_parameters=model_params
                )
                
                return final_response
            
            should_stop = False
            for tool_result in tool_results:
                if hasattr(tool_result, 'content') and isinstance(tool_result.content, dict):
                    if tool_result.content.get('_stop_execution'):
                        should_stop = True
                        tool_result.content.pop('_stop_execution', None)
            
            tool_request = ModelRequest(parts=tool_results)
            messages.append(response)
            messages.append(tool_request)
            
            if should_stop:
                final_text = ""
                for tool_result in tool_results:
                    if hasattr(tool_result, 'content'):
                        if isinstance(tool_result.content, dict):
                            final_text = str(tool_result.content.get('func', tool_result.content))
                        else:
                            final_text = str(tool_result.content)
                
                stop_response = ModelResponse(
                    parts=[TextPart(content=final_text)],
                    model_name=response.model_name,
                    timestamp=response.timestamp,
                    usage=response.usage,
                    provider_name=response.provider_name,
                    provider_response_id=response.provider_response_id,
                    provider_details=response.provider_details,
                    finish_reason="stop"
                )
                return stop_response
            
            model_params = self._build_model_request_parameters(getattr(self, 'current_task', None))
            model_params = self.model.customize_request_parameters(model_params)
            
            follow_up_response = await self.model.request(
                messages=messages,
                model_settings=self.model.settings,
                model_request_parameters=model_params
            )
            
            return await self._handle_model_response(follow_up_response, messages)
        
        return response
    
    async def _handle_cache(self, task: "Task") -> Optional[Any]:
        """Handle cache operations for the task."""
        if not task.enable_cache:
            return None
        
        if self.debug:
            from upsonic.utils.printing import cache_configuration
            embedding_provider_name = None
            if task.cache_embedding_provider:
                embedding_provider_name = getattr(task.cache_embedding_provider, 'model_name', 'Unknown')
            
            cache_configuration(
                enable_cache=task.enable_cache,
                cache_method=task.cache_method,
                cache_threshold=task.cache_threshold if task.cache_method == "vector_search" else None,
                cache_duration_minutes=task.cache_duration_minutes,
                embedding_provider=embedding_provider_name
            )
        
        input_text = task._original_input or task.description
        cached_response = await task.get_cached_response(input_text, self.model)
        
        if cached_response is not None:
            similarity = None
            if hasattr(task, '_last_cache_entry') and 'similarity' in task._last_cache_entry:
                similarity = task._last_cache_entry['similarity']
            
            from upsonic.utils.printing import cache_hit
            cache_hit(
                cache_method=task.cache_method,
                similarity=similarity,
                input_preview=(task._original_input or task.description)[:100] if (task._original_input or task.description) else None
            )
            
            return cached_response
        else:
            from upsonic.utils.printing import cache_miss
            cache_miss(
                cache_method=task.cache_method,
                input_preview=(task._original_input or task.description)[:100] if (task._original_input or task.description) else None
            )
            return None
    
    async def _apply_user_policy(self, task: "Task") -> tuple[Optional["Task"], bool]:
        """Apply user policy to task input."""
        if not (self.user_policy and task.description):
            return task, True
        
        from upsonic.safety_engine.models import PolicyInput, RuleOutput
        from upsonic.safety_engine.exceptions import DisallowedOperation
        
        policy_input = PolicyInput(input_texts=[task.description])
        try:
            rule_output, _action_output, policy_output = await self.user_policy.execute_async(policy_input)
            action_taken = policy_output.action_output.get("action_taken", "UNKNOWN")
            
            if self.debug and rule_output.confidence > 0.0:
                from upsonic.utils.printing import policy_triggered
                policy_triggered(
                    policy_name=self.user_policy.name,
                    check_type="User Input Check",
                    action_taken=action_taken,
                    rule_output=rule_output
                )
            
            if action_taken == "BLOCK":
                task.task_end()
                task._response = policy_output.output_texts[0] if policy_output.output_texts else "Content blocked by user policy."
                return task, False
            elif action_taken in ["REPLACE", "ANONYMIZE"]:
                task.description = policy_output.output_texts[0] if policy_output.output_texts else ""
                return task, True
                
        except DisallowedOperation as e:
            mock_rule_output = RuleOutput(
                confidence=1.0,
                content_type="DISALLOWED_OPERATION", 
                details=str(e)
            )
            if self.debug:
                from upsonic.utils.printing import policy_triggered
                policy_triggered(
                    policy_name=self.user_policy.name,
                    check_type="User Input Check",
                    action_taken="DISALLOWED_EXCEPTION",
                    rule_output=mock_rule_output
                )
            
            task.task_end()
            task._response = f"Operation disallowed by user policy: {e}"
            return task, False
        
        return task, True
    
    async def _execute_with_guardrail(self, task: "Task", memory_handler: Optional["MemoryManager"]) -> "ModelResponse":
        """
        Executes the agent's run method with a validation and retry loop based on a task guardrail.
        This method encapsulates the retry logic, hiding it from the main `do_async` pipeline.
        It returns a single, "clean" ModelResponse that represents the final, successful interaction.
        """
        from upsonic.messages import TextPart, ModelResponse
        retry_counter = 0
        validation_passed = False
        final_model_response = None
        last_error_message = ""
        
        temporary_message_history = copy.deepcopy(memory_handler.get_message_history())
        current_input = task.build_agent_input()

        if task.guardrail_retries is not None and task.guardrail_retries > 0:
            max_retries = task.guardrail_retries + 1
        else:
            max_retries = 1

        while not validation_passed and retry_counter < max_retries:
            messages = await self._build_model_request_with_input(task, memory_handler, current_input, temporary_message_history)
            self._current_messages = messages
            
            model_params = self._build_model_request_parameters(task)
            model_params = self.model.customize_request_parameters(model_params)
            
            response = await self.model.request(
                messages=messages,
                model_settings=self.model.settings,
                model_request_parameters=model_params
            )
            
            current_model_response = await self._handle_model_response(response, messages)
            
            if task.guardrail is None:
                validation_passed = True
                final_model_response = current_model_response
                break

            final_text_output = ""
            text_parts = [part.content for part in current_model_response.parts if isinstance(part, TextPart)]
            final_text_output = "".join(text_parts)

            if not final_text_output:
                validation_passed = True
                final_model_response = current_model_response
                break

            try:
                guardrail_result = task.guardrail(final_text_output)
                
                if isinstance(guardrail_result, tuple) and len(guardrail_result) == 2:
                    is_valid, result = guardrail_result
                elif isinstance(guardrail_result, bool):
                    is_valid = guardrail_result
                    result = final_text_output if guardrail_result else "Guardrail validation failed"
                else:
                    is_valid = bool(guardrail_result)
                    result = guardrail_result if guardrail_result else "Guardrail validation failed"

                if is_valid:
                    validation_passed = True
                    
                    if result != final_text_output:
                        updated_parts = []
                        found_and_updated = False
                        for part in current_model_response.parts:
                            if isinstance(part, TextPart) and not found_and_updated:
                                updated_parts.append(TextPart(content=str(result)))
                                found_and_updated = True
                            elif isinstance(part, TextPart):
                                updated_parts.append(TextPart(content=""))
                            else:
                                updated_parts.append(part)
                        
                        final_model_response = ModelResponse(
                            parts=updated_parts,
                            model_name=current_model_response.model_name,
                            timestamp=current_model_response.timestamp,
                            usage=current_model_response.usage,
                            provider_name=current_model_response.provider_name,
                            provider_response_id=current_model_response.provider_response_id,
                            provider_details=current_model_response.provider_details,
                            finish_reason=current_model_response.finish_reason
                        )
                    else:
                        final_model_response = current_model_response
                    break
                else:
                    retry_counter += 1
                    last_error_message = str(result)
                    
                    temporary_message_history.append(current_model_response)
                    
                    correction_prompt = f"Your previous response failed a validation check. Please review the reason and provide a corrected response. Failure Reason: {last_error_message}"
                    current_input = correction_prompt
                    
            except Exception as e:
                retry_counter += 1
                last_error_message = f"Guardrail execution error: {str(e)}"
                
                temporary_message_history.append(current_model_response)
                
                correction_prompt = f"Your previous response failed a validation check. Please review the reason and provide a corrected response. Failure Reason: {last_error_message}"
                current_input = correction_prompt

        if not validation_passed:
            error_msg = f"Task failed after {max_retries-1} retry(s). Last error: {last_error_message}"
            if self.mode == "raise":
                from upsonic.utils.package.exception import GuardrailValidationError
                raise GuardrailValidationError(error_msg)
            else:
                error_response = ModelResponse(
                    parts=[TextPart(content="Guardrail validation failed after retries")],
                    model_name=self.model.model_name,
                    timestamp=now_utc(),
                    usage=RequestUsage()
                )
                return error_response
                
        return final_model_response
    
    def _compress_context(self, context: str) -> str:
        """Compress context based on the selected strategy."""
        if self.compression_strategy == "simple":
            return self._compress_simple(context)
        elif self.compression_strategy == "llmlingua":
            return self._compress_llmlingua(context)
        return context

    def _compress_simple(self, context: str) -> str:
        """Compress context using simple whitespace removal and truncation."""
        if not context:
            return ""
        
        compressed = " ".join(context.split())
        
        max_length = self.compression_settings.get("max_length", 2000)
        
        if len(compressed) > max_length:
            part_size = max_length // 2 - 20
            compressed = compressed[:part_size] + " ... [COMPRESSED] ... " + compressed[-part_size:]
        
        return compressed
        

    def _compress_llmlingua(self, context: str) -> str:
        """Compress context using the LLMLingua library."""
        if not context or not self._prompt_compressor:
            return context

        ratio = self.compression_settings.get("ratio", 0.5)
        instruction = self.compression_settings.get("instruction", "")

        try:
            result = self._prompt_compressor.compress_prompt(
                context.split('\n'),
                instruction=instruction,
                ratio=ratio
            )
            return result['compressed_prompt']
        except Exception as e:
            if self.debug:
                print(f"LLMLingua compression failed, falling back to simple compression. Error: {e}")
            return self._compress_simple(context)
    

    async def _apply_agent_policy(self, task: "Task") -> "Task":
        """Apply agent policy to task output."""
        if not (self.agent_policy and task and task.response):
            return task
        
        from upsonic.safety_engine.models import PolicyInput, RuleOutput
        from upsonic.safety_engine.exceptions import DisallowedOperation
        
        response_text = ""
        if isinstance(task.response, str):
            response_text = task.response
        elif hasattr(task.response, 'model_dump_json'):
            response_text = task.response.model_dump_json()
        else:
            response_text = str(task.response)
        
        if response_text:
            agent_policy_input = PolicyInput(input_texts=[response_text])
            try:
                rule_output, _action_output, policy_output = await self.agent_policy.execute_async(agent_policy_input)
                action_taken = policy_output.action_output.get("action_taken", "UNKNOWN")
                
                if self.debug and rule_output.confidence > 0.0:
                    from upsonic.utils.printing import policy_triggered
                    policy_triggered(
                        policy_name=self.agent_policy.name,
                        check_type="Agent Output Check",
                        action_taken=action_taken,
                        rule_output=rule_output
                    )
                
                final_output = policy_output.output_texts[0] if policy_output.output_texts else "Response modified by agent policy."
                task._response = final_output
                
            except DisallowedOperation as e:
                mock_rule_output = RuleOutput(
                    confidence=1.0,
                    content_type="DISALLOWED_OPERATION",
                    details=str(e)
                )
                if self.debug:
                    from upsonic.utils.printing import policy_triggered
                    policy_triggered(
                        policy_name=self.agent_policy.name,
                        check_type="Agent Output Check", 
                        action_taken="DISALLOWED_EXCEPTION",
                        rule_output=mock_rule_output
                    )
                task._response = f"Agent response disallowed by policy: {e}"
        
        return task
    
    @asynccontextmanager
    async def _managed_storage_connection(self):
        """Manage storage connection lifecycle."""
        if not self.memory or not self.memory.storage:
            yield
            return
        
        storage = self.memory.storage
        was_connected_before = await storage.is_connected_async()
        try:
            if not was_connected_before:
                await storage.connect_async()
            yield
        finally:
            if not was_connected_before and await storage.is_connected_async():
                await storage.disconnect_async()
    
    
    @retryable()
    async def do_async(
        self, 
        task: "Task", 
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        state: Optional["State"] = None,
        *,
        graph_execution_id: Optional[str] = None
    ) -> "RunResult":
        """
        Execute a task asynchronously with complete framework integration.
        
        Args:
            task: Task to execute
            model: Override model for this execution
            debug: Enable debug mode
            retry: Number of retries
            state: Graph execution state
            graph_execution_id: Graph execution identifier
            
        Returns:
            RunResult: A result object containing:
                - output: The actual task response
                - all_messages(): All messages from all runs
                - new_messages(): Messages from the last run only
                
        Example:
            ```python
            result = await agent.do_async(task)
            print(result.output)  # Access the response
            print(result.new_messages())  # Get last run's messages
            ```
        """
        from upsonic.utils.printing import agent_started
        from upsonic.utils.validators import validate_attachments_for_model
        from upsonic.agent.context_managers import MemoryManager, CallManager, TaskManager, ReliabilityManager
        
        agent_started(self.get_agent_id())
        
        self.tool_call_count = 0
        self._tool_call_count = 0
        self.current_task = task
        
        self._run_result.start_new_run()
        
        async with self._managed_storage_connection():
            processed_task = None
            
            try:
                if not task.is_paused:
                    if task.enable_cache:
                        task.set_cache_manager(self._cache_manager)
                    
                    cached_response = await self._handle_cache(task)
                    if cached_response is not None:
                        processed_task = task
                        processed_task._response = cached_response
                        processed_task.task_end()
                        self._run_result.output = cached_response
                        return self._run_result
                    
                    task, should_continue = await self._apply_user_policy(task)
                    if not should_continue:
                        processed_task = task
                        self._run_result.output = processed_task.response
                        return self._run_result
                
                if model:
                    from upsonic.models import infer_model
                    current_model = infer_model(model)
                else:
                    current_model = self.model
                
                validate_attachments_for_model(current_model, task)
                
                self._setup_tools(task)
                
                memory_manager = MemoryManager(self.memory)
                reliability_manager = ReliabilityManager(task, self.reliability_layer, current_model) if self.reliability_layer else None
                
                async with memory_manager.manage_memory() as memory_handler:
                    call_manager = CallManager(current_model, task, debug=debug, show_tool_calls=self.show_tool_calls)
                    task_manager = TaskManager(task, self)
                    
                    async with call_manager.manage_call() as call_handler:
                        async with task_manager.manage_task() as task_handler:
                            try:
                                if task.guardrail:
                                    final_response = await self._execute_with_guardrail(task, memory_handler)
                                else:
                                    messages = await self._build_model_request(task, memory_handler)
                                    self._current_messages = messages
                                    
                                    model_params = self._build_model_request_parameters(task)
                                    model_params = current_model.customize_request_parameters(model_params)
                                    
                                    response = await current_model.request(
                                        messages=messages,
                                        model_settings=current_model.settings,
                                        model_request_parameters=model_params
                                    )
                                    
                                    final_response = await self._handle_model_response(response, messages)
                                
                                if self.reflection_processor:
                                    output = self._extract_output(final_response, task)
                                    improved_output = await self.reflection_processor.process_with_reflection(
                                        self, task, output
                                    )
                                    task._response = improved_output
                                else:
                                    output = self._extract_output(final_response, task)
                                    task._response = output
                                
                            except ExternalExecutionPause as e:
                                task_handler.task.is_paused = True
                                task_handler.task._tools_awaiting_external_execution.append(e.tool_call)
                                processed_task = task_handler.task
                                self._run_result.output = processed_task.response
                                return self._run_result
                            
                            self._run_result.output = task._response
                            
                            history_length = len(memory_handler.get_message_history()) if memory_handler else 0
                            new_messages_start = history_length
                            
                            if new_messages_start < len(messages):
                                self._run_result.add_messages(messages[new_messages_start:])
                            self._run_result.add_message(final_response)
                            
                            call_handler.process_response(self._run_result)
                            task_handler.process_response(self._run_result)
                            
                            memory_handler.process_response(self._run_result)
                            processed_task = task_handler.task
                
                if reliability_manager:
                    async with reliability_manager.manage_reliability() as rel_handler:
                        processed_task = await rel_handler.process_task(processed_task)
                
                processed_task = await self._apply_agent_policy(processed_task)
                
                if processed_task and processed_task.enable_cache and processed_task.response:
                    input_text = processed_task._original_input or processed_task.description
                    await processed_task.store_cache_entry(input_text, processed_task.response)
                    if self.debug:
                        from upsonic.utils.printing import cache_stored
                        cache_stored(
                            cache_method=processed_task.cache_method,
                            input_preview=(processed_task._original_input or processed_task.description)[:100] if (processed_task._original_input or processed_task.description) else None,
                            duration_minutes=processed_task.cache_duration_minutes
                        )
                
            except Exception as e:
                if self.debug:
                    print(f"Error in do_async: {e}")
                
                if self.mode == "raise":
                    raise
                else:
                    if processed_task:
                        processed_task._response = False
                    self._run_result.output = False
                    return self._run_result
        
        if processed_task and not processed_task.not_main_task:
            from upsonic.utils.printing import print_price_id_summary
            print_price_id_summary(processed_task.price_id, processed_task)
        
        if self._run_result.output is None and processed_task:
            self._run_result.output = processed_task.response
        
        return self._run_result
    
    def _extract_output(self, response: "ModelResponse", task: "Task") -> Any:
        """Extract output from model response based on task response format."""
        from upsonic.messages import TextPart
        
        text_parts = [part.content for part in response.parts if isinstance(part, TextPart)]
        
        if task.response_format == str or task.response_format is str:
            return "".join(text_parts)
        
        text_content = "".join(text_parts)
        if task.response_format != str and text_content:
            try:
                import json
                parsed = json.loads(text_content)
                if hasattr(task.response_format, 'model_validate'):
                    return task.response_format.model_validate(parsed)
                return parsed
            except:
                pass
        
        return text_content
    
    def do(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1
    ) -> "RunResult":
        """
        Execute a task synchronously.
        
        Args:
            task: Task to execute
            model: Override model for this execution
            debug: Enable debug mode
            retry: Number of retries
            
        Returns:
            RunResult: A result object with output and message tracking
        """
        task.price_id_ = None
        _ = task.price_id
        task._tool_calls = []
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.do_async(task, model, debug, retry))
                    return future.result()
            else:
                return loop.run_until_complete(self.do_async(task, model, debug, retry))
        except RuntimeError:
            return asyncio.run(self.do_async(task, model, debug, retry))
    
    def print_do(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1
    ) -> "RunResult":
        """
        Execute a task synchronously and print the result.
        
        Returns:
            RunResult: The result object (with output printed to console)
        """
        result = self.do(task, model, debug, retry)
        print(result.output)
        return result
    
    async def print_do_async(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1
    ) -> "RunResult":
        """
        Execute a task asynchronously and print the result.
        
        Returns:
            RunResult: The result object (with output printed to console)
        """
        result = await self.do_async(task, model, debug, retry)
        print(result.output)
        return result
    
    async def stream_async(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        state: Optional["State"] = None,
        *,
        graph_execution_id: Optional[str] = None
    ) -> "StreamRunResult":
        """
        Stream task execution asynchronously with StreamRunResult wrapper.
        
        Args:
            task: Task to execute
            model: Override model for this execution
            debug: Enable debug mode
            retry: Number of retries
            state: Graph execution state
            graph_execution_id: Graph execution identifier
            
        Returns:
            StreamRunResult: Advanced streaming result wrapper
            
        Example:
            ```python
            result = await agent.stream_async(task)
            async with result as stream:
                async for text in stream.stream_output():
                    print(text, end='', flush=True)
            ```
        """
        self._stream_run_result._agent = self
        self._stream_run_result._task = task
        self._stream_run_result._model = model
        self._stream_run_result._debug = debug
        self._stream_run_result._retry = retry
        
        self._stream_run_result._state = state
        self._stream_run_result._graph_execution_id = graph_execution_id
        
        return self._stream_run_result
    
    async def _stream_text_output(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        state: Optional["State"] = None,
        graph_execution_id: Optional[str] = None,
        stream_result: Optional[StreamRunResult] = None
    ) -> AsyncIterator[str]:
        """Stream text content from the model response.
        
        This method handles text extraction and accumulation from streaming events.
        """
        from upsonic.messages import FinalResultEvent
        
        accumulated_text = ""
        first_token_time = None
        
        async for event in self._create_stream_iterator(task, model, debug, retry, state, graph_execution_id):
            if stream_result:
                stream_result._streaming_events.append(event)
            
            text_content = self._extract_text_from_stream_event(event)
            if text_content:
                if first_token_time is None and stream_result:
                    first_token_time = time.time()
                    stream_result._first_token_time = first_token_time
                
                accumulated_text += text_content
                if stream_result:
                    stream_result._accumulated_text = accumulated_text
                    
                yield text_content
            
            if isinstance(event, FinalResultEvent) and stream_result:
                stream_result._is_complete = True
                stream_result._final_output = accumulated_text if accumulated_text else None
    
    async def _stream_events_output(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        state: Optional["State"] = None,
        graph_execution_id: Optional[str] = None,
        stream_result: Optional[StreamRunResult] = None
    ) -> AsyncIterator["ModelResponseStreamEvent"]:
        """Stream raw events from the model response.
        
        This method handles event streaming and text accumulation.
        """
        accumulated_text = ""
        first_token_time = None
        
        async for event in self._create_stream_iterator(task, model, debug, retry, state, graph_execution_id):
            if stream_result:
                stream_result._streaming_events.append(event)
            
            text_content = self._extract_text_from_stream_event(event)
            if text_content:
                if first_token_time is None and stream_result:
                    first_token_time = time.time()
                    stream_result._first_token_time = first_token_time
                
                accumulated_text += text_content
                if stream_result:
                    stream_result._accumulated_text = accumulated_text
            
            yield event
            
            if isinstance(event, FinalResultEvent) and stream_result:
                stream_result._is_complete = True
                stream_result._final_output = accumulated_text if accumulated_text else None
    
    def _extract_text_from_stream_event(self, event: "ModelResponseStreamEvent") -> Optional[str]:
        """Extract text content from a streaming event."""
        from upsonic.messages import PartStartEvent, PartDeltaEvent, TextPart
        
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            return event.part.content
        elif isinstance(event, PartDeltaEvent) and hasattr(event.delta, 'content_delta'):
            return event.delta.content_delta
        return None
    
    async def _create_stream_iterator(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        state: Optional["State"] = None,
        graph_execution_id: Optional[str] = None
    ) -> AsyncIterator["ModelResponseStreamEvent"]:
        """Create the actual stream iterator for streaming execution.
        
        This iterator yields all streaming events from the model, including the FinalResultEvent
        which now comes at the end of the stream (after all content has been received).
        """
        from upsonic.utils.printing import agent_started
        from upsonic.utils.validators import validate_attachments_for_model
        from upsonic.agent.context_managers import MemoryManager
        from upsonic.messages import PartStartEvent, TextPart, FinalResultEvent, ToolCallPart, ModelRequest
        
        agent_started(self.get_agent_id())
        
        self.tool_call_count = 0
        self._tool_call_count = 0
        
        current_task = task
        self.current_task = current_task
        
        self._stream_run_result.start_new_run()
        
        async with self._managed_storage_connection():
            try:
                if not current_task.is_paused:
                    if current_task.enable_cache:
                        current_task.set_cache_manager(self._cache_manager)
                    
                    cached_response = await self._handle_cache(current_task)
                    if cached_response is not None:
                        yield PartStartEvent(index=0, part=TextPart(content=str(cached_response)))
                        yield FinalResultEvent(tool_name=None, tool_call_id=None)
                        
                        current_task._response = cached_response
                        current_task.task_end()
                        return
                    
                    current_task, should_continue = await self._apply_user_policy(current_task)
                    if not should_continue:
                        yield FinalResultEvent(tool_name=None, tool_call_id=None)
                        return
            
                if model:
                    from upsonic.models import infer_model
                    current_model = infer_model(model)
                else:
                    current_model = self.model
                
                validate_attachments_for_model(current_model, current_task)
                
                self._setup_tools(current_task)
                
                memory_manager = MemoryManager(self.memory)
                
                async with memory_manager.manage_memory() as memory_handler:
                    messages = await self._build_model_request(current_task, memory_handler)
                    self._current_messages = messages
                    
                    model_params = self._build_model_request_parameters(current_task)
                    model_params = current_model.customize_request_parameters(model_params)
                    
                    final_result_received = False
                    try:
                        async with current_model.request_stream(
                            messages=messages,
                            model_settings=current_model.settings,
                            model_request_parameters=model_params
                        ) as stream:
                            event_count = 0
                            async for event in stream:
                                event_count += 1
                                yield event
                                
                                if isinstance(event, FinalResultEvent):
                                    final_result_received = True
                    except Exception as e:
                        raise
                        
                    final_response = stream.get()
                    
                    tool_calls = [
                        part for part in final_response.parts 
                        if isinstance(part, ToolCallPart)
                    ]
                    
                    if tool_calls:
                        tool_results = await self._execute_tool_calls(tool_calls)
                        
                        messages.append(final_response)
                        messages.append(ModelRequest(parts=tool_results))
                        
                        async with current_model.request_stream(
                            messages=messages,
                            model_settings=current_model.settings,
                            model_request_parameters=model_params
                        ) as continuation_stream:
                            async for event in continuation_stream:
                                yield event
                                
                                if isinstance(event, FinalResultEvent):
                                    final_result_received = True
                            
                            final_response = continuation_stream.get()
                    
                    output = self._extract_output(final_response, current_task)
                    current_task._response = output
                    current_task.task_end()
                    
                    self._stream_run_result._final_output = current_task._response
                    
                    history_length = len(memory_handler.get_message_history()) if memory_handler else 0
                    new_messages_start = history_length
                    
                    if new_messages_start < len(messages):
                        self._stream_run_result.add_messages(messages[new_messages_start:])
                    self._stream_run_result.add_message(final_response)
                    
                    memory_handler.process_response(self._stream_run_result)
                    
                    await self._apply_agent_policy(current_task)
                    
                    if current_task.enable_cache and current_task.response:
                        input_text = current_task._original_input or current_task.description
                        await current_task.store_cache_entry(input_text, current_task.response)
                
            except Exception as e:
                if self.debug:
                    print(f"Error in stream_async: {e}")
                raise
    
    def _create_stream_result(
        self,
        task: "Task", 
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1
    ) -> "StreamRunResult":
        """Create a StreamRunResult with deferred async execution."""
        stream_result = StreamRunResult()
        stream_result._agent = self
        stream_result._task = task
        
        stream_result._model = model
        stream_result._debug = debug
        stream_result._retry = retry
        
        return stream_result
    
    def stream(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1
    ) -> "StreamRunResult":
        """
        Stream task execution with StreamRunResult wrapper.
        
        Args:
            task: Task to execute
            model: Override model for this execution
            debug: Enable debug mode
            retry: Number of retries
            
        Returns:
            StreamRunResult: Advanced streaming result wrapper
            
        Example:
            ```python
            async with agent.stream(task) as result:
                async for text in result.stream_output():
                    print(text, end='', flush=True)
            ```
        """
        task.price_id_ = None
        _ = task.price_id
        task._tool_calls = []
        
        self._stream_run_result._agent = self
        self._stream_run_result._task = task
        self._stream_run_result._model = model
        self._stream_run_result._debug = debug
        self._stream_run_result._retry = retry
        
        return self._stream_run_result
    
    async def print_stream_async(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1
    ) -> Any:
        """Stream task execution asynchronously and print output."""
        result = await self.stream_async(task, model, debug, retry)
        async with result:
            async for text_chunk in result.stream_output():
                print(text_chunk, end='', flush=True)
            print()
            
            return result.get_final_output()
    
    def print_stream(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1
    ) -> Any:
        """Stream task execution synchronously and print output."""
        task.price_id_ = None
        _ = task.price_id
        task._tool_calls = []
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.print_stream_async(task, model, debug, retry))
                    return future.result()
            else:
                return loop.run_until_complete(self.print_stream_async(task, model, debug, retry))
        except RuntimeError:
            return asyncio.run(self.print_stream_async(task, model, debug, retry))
    
    # External execution support
    
    def continue_run(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1
    ) -> "RunResult":
        """Continue execution of a paused task."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.continue_async(task, model, debug, retry))
                    return future.result()
            else:
                return loop.run_until_complete(self.continue_async(task, model, debug, retry))
        except RuntimeError:
            return asyncio.run(self.continue_async(task, model, debug, retry))
    
    async def continue_async(
        self,
        task: "Task", 
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        state: Optional["State"] = None,
        *,
        graph_execution_id: Optional[str] = None
    ) -> "RunResult":
        """Continue execution of a paused task asynchronously."""
        if not task.is_paused or not task.tools_awaiting_external_execution:
            raise ValueError("The 'continue_async' method can only be called on a task that is currently paused for external execution.")
        
        tool_results_prompt = "\nThe following external tools were executed. Use their results to continue the task:\n"
        for tool_call in task.tools_awaiting_external_execution:
            tool_results_prompt += f"\n- Tool '{tool_call.tool_name}' was executed with arguments {tool_call.args}.\n"
            tool_results_prompt += f"  Result: {tool_call.result}\n"
        
        task.is_paused = False
        task.description += tool_results_prompt
        task._tools_awaiting_external_execution = []
        
        if task.enable_cache:
            task.set_cache_manager(self._cache_manager)
        
        return await self.do_async(task, model, debug, retry, state, graph_execution_id=graph_execution_id)


# Legacy alias for backwards compatibility
Direct = Agent
