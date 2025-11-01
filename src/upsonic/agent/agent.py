import asyncio
import copy
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union, TYPE_CHECKING

PromptCompressor = None

from upsonic.utils.logging_config import sentry_sdk
from upsonic import _utils
from upsonic.agent.base import BaseAgent
from upsonic.agent.run_result import RunResult, StreamRunResult
from upsonic._utils import now_utc
from upsonic.utils.retry import retryable
from upsonic.utils.validators import validate_attachments_exist
from upsonic.tools.processor import ExternalExecutionPause
from upsonic.messages import FinalResultEvent

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
    from upsonic.usage import RequestUsage
    from upsonic.agent.context_managers import (
        CallManager, ContextManager, ReliabilityManager, 
        MemoryManager, SystemPromptManager, TaskManager
    )
    from upsonic.graph.graph import State
    from ..db.database import DatabaseBase
    from upsonic.models.model_selector import ModelRecommendation
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
    RequestUsage = "RequestUsage"
    validate_attachments_exist = "validate_attachments_exist"
    CallManager = "CallManager"
    ContextManager = "ContextManager"
    ReliabilityManager = "ReliabilityManager"
    MemoryManager = "MemoryManager"
    SystemPromptManager = "SystemPromptManager"
    TaskManager = "TaskManager"
    State = "State"
    ModelRecommendation = "ModelRecommendation"

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
        db: Optional["DatabaseBase"] = None,
        debug: bool = False,
        company_url: Optional[str] = None,
        company_objective: Optional[str] = None,
        company_description: Optional[str] = None,
        company_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        reflection: bool = False,
        compression_strategy: Literal["none", "simple", "llmlingua"] = "none",
        compression_settings: Optional[Dict[str, Any]] = None,
        reliability_layer: Optional[Any] = None,
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
        user_policy: Optional[Union["Policy", List["Policy"]]] = None,
        agent_policy: Optional[Union["Policy", List["Policy"]]] = None,
        settings: Optional["ModelSettings"] = None,
        profile: Optional["ModelProfile"] = None,
        reflection_config: Optional["ReflectionConfig"] = None,
        model_selection_criteria: Optional[Dict[str, Any]] = None,
        use_llm_for_selection: bool = False,
        # Common reasoning/thinking attributes
        reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
        reasoning_summary: Optional[Literal["concise", "detailed"]] = None,
        thinking_enabled: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        thinking_include_thoughts: Optional[bool] = None,
        reasoning_format: Optional[Literal["hidden", "raw", "parsed"]] = None,
    ):
        """
        Initialize the Agent with comprehensive configuration options.
        
        Args:
            model: Model identifier or Model instance
            name: Agent name for identification
            memory: Memory instance for conversation history
            db: Database instance (overrides memory if provided)
            debug: Enable debug logging
            company_url: Company URL for context
            company_objective: Company objective for context
            company_description: Company description for context
            system_prompt: Custom system prompt
            reflection: Reflection capabilities (default is False)
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
            user_policy: User input safety policy (single policy or list of policies)
            agent_policy: Agent output safety policy (single policy or list of policies)
            settings: Model-specific settings
            profile: Model profile configuration
            reflection_config: Configuration for reflection and self-evaluation
            model_selection_criteria: Default criteria dictionary for recommend_model_for_task() (see SelectionCriteria)
            use_llm_for_selection: Default flag for whether to use LLM in recommend_model_for_task()
            
            # Common reasoning/thinking attributes (mapped to model-specific settings):
            reasoning_effort: Reasoning effort level for OpenAI models ("low", "medium", "high")
            reasoning_summary: Reasoning summary type for OpenAI models ("concise", "detailed")
            thinking_enabled: Enable thinking for Anthropic/Google models (True/False)
            thinking_budget: Token budget for thinking (Anthropic: budget_tokens, Google: thinking_budget)
            thinking_include_thoughts: Include thoughts in output (Google models)
            reasoning_format: Reasoning format for Groq models ("hidden", "raw", "parsed")
        """
        from upsonic.models import infer_model
        self.model = infer_model(model)
        self.name = name
        self.agent_id_ = agent_id_
        
        # Common reasoning/thinking attributes
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.thinking_enabled = thinking_enabled
        self.thinking_budget = thinking_budget
        self.thinking_include_thoughts = thinking_include_thoughts
        self.reasoning_format = reasoning_format
        
        self.role = role
        self.goal = goal
        self.instructions = instructions
        self.education = education
        self.work_experience = work_experience
        self.system_prompt = system_prompt
        
        self.company_url = company_url
        self.company_objective = company_objective
        self.company_description = company_description
        self.company_name = company_name
        
        self.debug = debug
        self.reflection = reflection
        
        # Model selection attributes
        self.model_selection_criteria = model_selection_criteria
        self.use_llm_for_selection = use_llm_for_selection
        self._model_recommendation: Optional[Any] = None  # Store last recommendation

        self.compression_strategy = compression_strategy
        self.compression_settings = compression_settings or {}
        self._prompt_compressor = None
        if self.compression_strategy == "llmlingua":
            try:
                from llmlingua import PromptCompressor
            except ImportError:
                from upsonic.utils.printing import import_error
                import_error(
                    package_name="llmlingua",
                    install_command="pip install llmlingua",
                    feature_name="llmlingua compression strategy"
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
        
        # Set db attribute
        self.db = db
        
        # Set memory attribute - override with db.memory if db is provided
        if db is not None:
            self.memory = db.memory
        else:
            self.memory = memory
            
        if self.memory:
            self.memory.feed_tool_call_results = feed_tool_call_results
        
        self.canvas = canvas
        
        # Initialize policy managers
        from upsonic.agent.policy_manager import PolicyManager
        self.user_policy_manager = PolicyManager(policies=user_policy, debug=self.debug)
        self.agent_policy_manager = PolicyManager(policies=agent_policy, debug=self.debug)
        
        # Keep backward compatibility - expose as single policy if only one
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
            
        self._apply_reasoning_settings()
        
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
        # Setup models for all policies in both managers
        self.user_policy_manager.setup_policy_models(self.model)
        self.agent_policy_manager.setup_policy_models(self.model)
    
    def _apply_reasoning_settings(self) -> None:
        """Apply common reasoning/thinking attributes to model-specific settings."""
        if not hasattr(self.model, '_settings') or self.model._settings is None:
            self.model._settings = {}
        
        try:
            current_settings = self.model._settings.copy()
        except (AttributeError, TypeError):
            current_settings = {}
            
        reasoning_settings = self._get_model_specific_reasoning_settings()
        
        try:
            self.model._settings = {**current_settings, **reasoning_settings}
        except TypeError:
            self.model._settings = current_settings
    
    def _get_model_specific_reasoning_settings(self) -> Dict[str, Any]:
        """Convert common reasoning attributes to model-specific settings."""
        settings = {}
        
        try:
            provider_name = getattr(self.model, 'system', '').lower()
        except (AttributeError, TypeError):
            provider_name = ''
        
        # OpenAI/OpenAI-compatible models
        if provider_name in ['openai', 'azure', 'deepseek', 'cerebras', 'fireworks', 'github', 'grok', 'heroku', 'moonshotai', 'openrouter', 'together', 'vercel', 'litellm']:
            # Apply reasoning_effort to all OpenAI models
            if self.reasoning_effort is not None:
                settings['openai_reasoning_effort'] = self.reasoning_effort
            
            # Only apply reasoning_summary to OpenAIResponsesModel
            if self.reasoning_summary is not None:
                from upsonic.models.openai import OpenAIResponsesModel
                if isinstance(self.model, OpenAIResponsesModel):
                    settings['openai_reasoning_summary'] = self.reasoning_summary
        
        # Anthropic models
        elif provider_name == 'anthropic':
            if self.thinking_enabled is not None or self.thinking_budget is not None:
                thinking_config = {}
                if self.thinking_enabled is not None:
                    thinking_config['type'] = 'enabled' if self.thinking_enabled else 'disabled'
                if self.thinking_budget is not None:
                    thinking_config['budget_tokens'] = self.thinking_budget
                settings['anthropic_thinking'] = thinking_config
        
        # Google models
        elif provider_name in ['google-gla', 'google-vertex']:
            if self.thinking_enabled is not None or self.thinking_budget is not None or self.thinking_include_thoughts is not None:
                thinking_config = {}
                if self.thinking_enabled is not None:
                    thinking_config['include_thoughts'] = self.thinking_include_thoughts if self.thinking_include_thoughts is not None else self.thinking_enabled
                if self.thinking_budget is not None:
                    thinking_config['thinking_budget'] = self.thinking_budget
                settings['google_thinking_config'] = thinking_config
        
        # Groq models
        elif provider_name == 'groq':
            if self.reasoning_format is not None:
                settings['groq_reasoning_format'] = self.reasoning_format
        
        return settings
    
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
        
        self._builtin_tools = self.tool_manager.processor.extract_builtin_tools(final_tools)
        
        self.tool_manager.register_tools(
            tools=final_tools,
            context=self._context,
            task=task,
            agent_instance=agent_for_this_run
        )
    
    async def _build_model_request(self, task: "Task", memory_handler: Optional["MemoryManager"], state: Optional["State"] = None) -> List["ModelRequest"]:
        """Build the complete message history for the model request."""
        from upsonic.agent.context_managers import SystemPromptManager, ContextManager
        from upsonic.messages import SystemPromptPart, UserPromptPart, ModelRequest
        
        messages = []
        
        message_history = memory_handler.get_message_history()
        messages.extend(message_history)
        
        system_prompt_manager = SystemPromptManager(self, task)
        context_manager = ContextManager(self, task, state)
        
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
        temporary_message_history: List["ModelRequest"],
        state: Optional["State"] = None
    ) -> List["ModelRequest"]:
        """Build model request with custom input and message history for guardrail retries."""
        from upsonic.agent.context_managers import SystemPromptManager, ContextManager
        from upsonic.messages import SystemPromptPart, UserPromptPart, ModelRequest
        
        messages = list(temporary_message_history)
        
        system_prompt_manager = SystemPromptManager(self, task)
        context_manager = ContextManager(self, task, state)
        
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
        
        builtin_tools = getattr(self, '_builtin_tools', [])
        
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
            builtin_tools=builtin_tools,
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
        """
        Apply user policy to task input.
        
        This method now uses PolicyManager to handle multiple policies.
        Kept for backward compatibility.
        """
        if not self.user_policy_manager.has_policies() or not task.description:
            return task, True
        
        from upsonic.safety_engine.models import PolicyInput
        
        policy_input = PolicyInput(input_texts=[task.description])
        result = await self.user_policy_manager.execute_policies_async(
            policy_input,
            check_type="User Input Check"
        )
        
        if result.should_block():
            # Re-raise DisallowedOperation if it was caught by PolicyManager
            if result.disallowed_exception:
                raise result.disallowed_exception
            
            task.task_end()
            task._response = result.get_final_message()
            return task, False
        elif result.action_taken in ["REPLACE", "ANONYMIZE"]:
            task.description = result.final_output or task.description
            return task, True
        
        return task, True
    
    async def _execute_with_guardrail(self, task: "Task", memory_handler: Optional["MemoryManager"], state: Optional["State"] = None) -> "ModelResponse":
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
            messages = await self._build_model_request_with_input(task, memory_handler, current_input, temporary_message_history, state)
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
                from upsonic.utils.printing import compression_fallback
                compression_fallback("llmlingua", "simple", str(e))
            return self._compress_simple(context)
    
    async def recommend_model_for_task_async(
        self,
        task: Union["Task", str],
        criteria: Optional[Dict[str, Any]] = None,
        use_llm: Optional[bool] = None
    ) -> "ModelRecommendation":
        """
        Get a model recommendation for a specific task.
        
        This method analyzes the task and returns a recommendation for the best model to use.
        The user can then decide whether to use the recommended model or stick with the default.
        
        Args:
            task: Task object or task description string
            criteria: Optional criteria dictionary for model selection (overrides agent's default)
            use_llm: Optional flag to use LLM for selection (overrides agent's default)
        
        Returns:
            ModelRecommendation: Object containing:
                - model_name: Recommended model identifier
                - reason: Explanation for the recommendation
                - confidence_score: Confidence level (0.0 to 1.0)
                - selection_method: "rule_based" or "llm_based"
                - estimated_cost_tier: Cost estimate (1-10)
                - estimated_speed_tier: Speed estimate (1-10)
                - alternative_models: List of alternative model names
        
        Example:
            ```python
            # Get recommendation
            recommendation = await agent.recommend_model_for_task_async(task)
            print(f"Recommended: {recommendation.model_name}")
            print(f"Reason: {recommendation.reason}")
            print(f"Confidence: {recommendation.confidence_score}")
            
            # Use it if you have credentials
            if user_has_credentials(recommendation.model_name):
                result = await agent.do_async(task, model=recommendation.model_name)
            else:
                result = await agent.do_async(task)  # Use default
            ```
        """
        try:
            from upsonic.models.model_selector import select_model_async, SelectionCriteria
            
            task_description = task.description if hasattr(task, 'description') else str(task)
            
            selection_criteria = None
            if criteria:
                selection_criteria = SelectionCriteria(**criteria)
            elif self.model_selection_criteria:
                selection_criteria = SelectionCriteria(**self.model_selection_criteria)
            
            use_llm_selection = use_llm if use_llm is not None else self.use_llm_for_selection
            
            recommendation = await select_model_async(
                task_description=task_description,
                criteria=selection_criteria,
                use_llm=use_llm_selection,
                agent=self if use_llm_selection else None,
                default_model=self.model.model_name
            )
            
            self._model_recommendation = recommendation
            
            if self.debug:
                from upsonic.utils.printing import model_recommendation_summary
                model_recommendation_summary(recommendation)
            
            return recommendation
            
        except Exception as e:
            if self.debug:
                from upsonic.utils.printing import model_recommendation_error
                model_recommendation_error(str(e))
            raise
    
    def recommend_model_for_task(
        self,
        task: Union["Task", str],
        criteria: Optional[Dict[str, Any]] = None,
        use_llm: Optional[bool] = None
    ) -> "ModelRecommendation":
        """
        Synchronous version of recommend_model_for_task_async.
        
        Get a model recommendation for a specific task.
        
        Args:
            task: Task object or task description string
            criteria: Optional criteria dictionary for model selection
            use_llm: Optional flag to use LLM for selection
        
        Returns:
            ModelRecommendation: Object containing recommendation details
        
        Example:
            ```python
            recommendation = agent.recommend_model_for_task("Write a sorting algorithm")
            print(f"Use: {recommendation.model_name}")
            ```
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.recommend_model_for_task_async(task, criteria, use_llm)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.recommend_model_for_task_async(task, criteria, use_llm)
                )
        except RuntimeError:
            return asyncio.run(self.recommend_model_for_task_async(task, criteria, use_llm))
    
    def get_last_model_recommendation(self) -> Optional[Any]:
        """
        Get the last model recommendation made by the agent.
        
        Returns:
            ModelRecommendation object or None if no recommendation was made
        """
        return self._model_recommendation
    

    async def _apply_agent_policy(self, task: "Task") -> "Task":
        """
        Apply agent policy to task output.
        
        This method now uses PolicyManager to handle multiple policies.
        Kept for backward compatibility.
        """
        if not self.agent_policy_manager.has_policies() or not task or not task.response:
            return task
        
        from upsonic.safety_engine.models import PolicyInput
        
        # Convert response to text
        response_text = ""
        if isinstance(task.response, str):
            response_text = task.response
        elif hasattr(task.response, 'model_dump_json'):
            response_text = task.response.model_dump_json()
        else:
            response_text = str(task.response)
        
        if response_text:
            agent_policy_input = PolicyInput(input_texts=[response_text])
            result = await self.agent_policy_manager.execute_policies_async(
                agent_policy_input,
                check_type="Agent Output Check"
            )
            
            # Apply the result
            if result.should_block():
                # Re-raise DisallowedOperation if it was caught by PolicyManager
                if result.disallowed_exception:
                    raise result.disallowed_exception
                
                task._response = result.get_final_message()
            elif result.action_taken in ["REPLACE", "ANONYMIZE"]:
                task._response = result.final_output or "Response modified by agent policy."
            elif result.final_output:
                task._response = result.final_output
        
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
    ) -> Any:
        """
        Execute a task asynchronously using the pipeline architecture.
        
        The execution is handled entirely by the pipeline - this method just
        creates the pipeline, creates the context, executes, and returns the output.
        All logic is in the pipeline steps.
        
        Args:
            task: Task to execute
            model: Override model for this execution
            debug: Enable debug mode
            retry: Number of retries
            state: Graph execution state
            graph_execution_id: Graph execution identifier
            
        Returns:
            The task output (any errors are raised immediately)
                
        Example:
            ```python
            result = await agent.do_async(task)
            print(result)  # Access the response
            ```
        """
        from upsonic.agent.pipeline import (
            PipelineManager, StepContext,
            InitializationStep, CacheCheckStep, UserPolicyStep,
            StorageConnectionStep, LLMManagerStep, ModelSelectionStep,
            ValidationStep, ToolSetupStep,
            MessageBuildStep, ModelExecutionStep, ResponseProcessingStep,
            ReflectionStep, CallManagementStep, TaskManagementStep,
            MemoryMessageTrackingStep,
            ReliabilityStep, AgentPolicyStep,
            CacheStorageStep, FinalizationStep
        )
        
        # Update policy managers debug flag if debug is enabled
        if debug:
            self.user_policy_manager.debug = True
            self.agent_policy_manager.debug = True
        
        async with self._managed_storage_connection():
            pipeline = PipelineManager(
                steps=[
                    InitializationStep(),
                    StorageConnectionStep(),
                    CacheCheckStep(),
                    UserPolicyStep(),
                    LLMManagerStep(),
                    ModelSelectionStep(),
                    ValidationStep(),
                    ToolSetupStep(),
                    MessageBuildStep(),
                    ModelExecutionStep(),
                    ResponseProcessingStep(),
                    ReflectionStep(),
                    MemoryMessageTrackingStep(),
                    AgentPolicyStep(),  # Move before CallManagementStep
                    CallManagementStep(),
                    TaskManagementStep(),
                    ReliabilityStep(),
                    CacheStorageStep(),
                    FinalizationStep(),
                ],
                debug=debug or self.debug
            )
            
            context = StepContext(
                task=task,
                agent=self,
                model=model,
                state=state
            )
            
            final_context = await pipeline.execute(context)
            sentry_sdk.flush()
            
            return self._run_result.output
    
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
    ) -> Any:
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
            loop = asyncio.get_running_loop()
            # If we get here, we're already in an async context with a running loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.do_async(task, model, debug, retry))
                return future.result()
        except RuntimeError:
            # No event loop is running, so we can safely use asyncio.run()
            return asyncio.run(self.do_async(task, model, debug, retry))
    
    def print_do(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1
    ) -> Any:
        """
        Execute a task synchronously and print the result.
        
        Returns:
            RunResult: The result object (with output printed to console)
        """
        result = self.do(task, model, debug, retry)
        from upsonic.utils.printing import success_log
        success_log(f"Task completed: {result}", "Agent")
        return result
    
    async def print_do_async(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1
    ) -> Any:
        """
        Execute a task asynchronously and print the result.
        
        Returns:
            RunResult: The result object (with output printed to console)
        """
        result = await self.do_async(task, model, debug, retry)
        from upsonic.utils.printing import success_log
        success_log(f"Task completed: {result}", "Agent")
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
        
        This method extracts and yields text from streaming events.
        Note: Event storage and accumulation are already handled by the pipeline.
        """
        from upsonic.messages import FinalResultEvent
        
        # The pipeline already handles event storage, text accumulation, and metrics
        # We just extract and yield text here
        async for event in self._create_stream_iterator(task, model, debug, retry, state, graph_execution_id):
            text_content = self._extract_text_from_stream_event(event)
            if text_content:
                yield text_content
    
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
        Note: Event storage and text accumulation are already handled by the pipeline.
        This method just yields the events.
        """
        # The pipeline already handles event storage, text accumulation, and metrics
        # We just yield the events here
        async for event in self._create_stream_iterator(task, model, debug, retry, state, graph_execution_id):
            yield event
    
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
        """Create the actual stream iterator for streaming execution using pipeline architecture.
        
        This iterator yields all streaming events from the model, including the FinalResultEvent
        which now comes at the end of the stream (after all content has been received).
        """
        from upsonic.agent.pipeline import (
            PipelineManager, StepContext,
            InitializationStep, CacheCheckStep, UserPolicyStep,
            StorageConnectionStep, LLMManagerStep, ModelSelectionStep,
            ValidationStep, ToolSetupStep, MessageBuildStep,
            StreamModelExecutionStep,
            AgentPolicyStep, CacheStorageStep,
            StreamMemoryMessageTrackingStep, StreamFinalizationStep
        )
        
        
        async with self._managed_storage_connection():
            # Create streaming pipeline with streaming-specific steps
            pipeline = PipelineManager(
                steps=[
                    InitializationStep(),
                    StorageConnectionStep(),
                    CacheCheckStep(),
                    UserPolicyStep(),
                    LLMManagerStep(),
                    ModelSelectionStep(),
                    ValidationStep(),
                    ToolSetupStep(),
                    MessageBuildStep(),
                    StreamModelExecutionStep(),  # Streaming-specific step
                    StreamMemoryMessageTrackingStep(),  # Streaming-specific memory tracking
                    AgentPolicyStep(),
                    CacheStorageStep(),
                    StreamFinalizationStep(),  # Streaming-specific finalization
                ],
                debug=debug or self.debug
            )
            
            # Create streaming context
            context = StepContext(
                task=task,
                agent=self,
                model=model,
                state=state,
                is_streaming=True,
                stream_result=self._stream_run_result
            )
            
            # Execute streaming pipeline and yield events
            async for event in pipeline.execute_stream(context):
                yield event
    
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
    ) -> Any:
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
    ) -> Any:
        """Continue execution of a paused task asynchronously."""
        if not task.is_paused or not task.tools_awaiting_external_execution:
            raise ValueError("The 'continue_async' method can only be called on a task that is currently paused for external execution.")
        
        from upsonic.agent.pipeline import (
            PipelineManager, StepContext,
            MessageBuildStep, ModelExecutionStep, ResponseProcessingStep,
            ReflectionStep, CallManagementStep, TaskManagementStep,
            MemoryMessageTrackingStep, ReliabilityStep, AgentPolicyStep,
            CacheStorageStep, FinalizationStep
        )
        from upsonic.messages import ToolReturnPart
        from upsonic._utils import now_utc
        
        # Convert external tool results to ToolReturnPart messages
        tool_return_parts = []
        for tool_call in task.tools_awaiting_external_execution:
            # Handle both ExternalToolCall (with tool_call_id) and ToolCall objects
            tool_call_id = None
            if hasattr(tool_call, 'tool_call_id'):
                tool_call_id = tool_call.tool_call_id
            
            # Get the result - ExternalToolCall uses 'result', ToolCall also uses 'result'
            result = getattr(tool_call, 'result', None)
            
            tool_return = ToolReturnPart(
                tool_name=tool_call.tool_name,
                content=result,
                tool_call_id=tool_call_id,
                timestamp=now_utc()
            )
            tool_return_parts.append(tool_return)
        
        # Get continuation state
        continuation_messages = []
        response_with_tool_calls = None
        if hasattr(task, '_continuation_state') and task._continuation_state:
            continuation_messages = task._continuation_state.get('messages', [])
            response_with_tool_calls = task._continuation_state.get('response_with_tool_calls')
        
        # Clear paused state
        task.is_paused = False
        task._tools_awaiting_external_execution = []
        
        if task.enable_cache:
            task.set_cache_manager(self._cache_manager)
        
        # Restore agent state from continuation
        if hasattr(task, '_continuation_state') and task._continuation_state:
            saved_state = task._continuation_state
            self._tool_call_count = saved_state.get('tool_call_count', 0)
            self.tool_call_count = saved_state.get('tool_call_count', 0)
            self._tool_limit_reached = saved_state.get('tool_limit_reached', False)
            self._current_messages = saved_state.get('current_messages', [])
        
        # Set current task (needed for pipeline steps)
        self.current_task = task
        
        # Determine model to use
        if model:
            from upsonic.models import infer_model
            current_model = infer_model(model)
        else:
            current_model = self.model
        
        async with self._managed_storage_connection():
            # Create a continuation pipeline - SKIP ALL STEPS UNTIL MessageBuildStep
            # 
            # The continuation flow:
            # 1. MessageBuildStep - restores saved messages
            # 2. ModelExecutionStep - injects response_with_tool_calls + tool_results, calls model
            # 3. All subsequent steps run normally
            #
            # We skip: InitializationStep, StorageConnectionStep, CacheCheckStep, UserPolicyStep,
            #          LLMManagerStep, ModelSelectionStep, ValidationStep, ToolSetupStep
            # 
            # These are already done in the initial run and state is preserved in continuation context
            
            pipeline = PipelineManager(
                steps=[
                    MessageBuildStep(),  # Restores saved messages
                    ModelExecutionStep(),  # Injects tool results and continues
                    ResponseProcessingStep(),
                    ReflectionStep(),
                    MemoryMessageTrackingStep(),
                    CallManagementStep(),
                    TaskManagementStep(),
                    ReliabilityStep(),
                    AgentPolicyStep(),
                    CacheStorageStep(),
                    FinalizationStep(),
                ],
                debug=debug or self.debug
            )
            
            # Create continuation context with all saved state
            context = StepContext(
                task=task,
                agent=self,
                model=current_model,
                state=state,
                is_continuation=True,  # This is the key flag
                continuation_messages=continuation_messages,
                continuation_tool_results=tool_return_parts,
                continuation_response_with_tool_calls=response_with_tool_calls  # The response that triggered pause
            )
            
            final_context = await pipeline.execute(context)
            sentry_sdk.flush()
            
            return self._run_result.output
    
    async def continue_durable_async(
        self,
        durable_execution_id: str,
        storage: Optional[Any] = None,
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1
    ) -> Any:
        """
        Continue execution from a durable execution checkpoint asynchronously.
        
        This method loads the saved execution state and resumes from the last
        successful checkpoint. It's used to recover from failures or interruptions.
        
        Args:
            durable_execution_id: The execution ID to resume
            storage: Storage backend (if different from the original)
            model: Override model for resumption
            debug: Enable debug mode
            retry: Number of retries
            
        Returns:
            The task output
            
        Raises:
            ValueError: If execution ID not found or state is invalid
            
        Example:
            ```python
            # After an error, resume from checkpoint
            result = await agent.continue_durable_async(task.durable_execution_id)
            ```
        """
        from upsonic.durable import DurableExecution
        from upsonic.agent.pipeline import PipelineManager, StepContext
        
        # Load the durable execution
        if storage is None:
            raise ValueError("Storage backend is required to continue durable execution")
        
        durable = await DurableExecution.load_by_id_async(durable_execution_id, storage)
        if durable is None:
            raise ValueError(f"Durable execution not found: {durable_execution_id}")
        
        checkpoint = await durable.load_checkpoint_async()
        if checkpoint is None:
            raise ValueError(f"No checkpoint found for execution: {durable_execution_id}")
        
        # Extract state from checkpoint
        task = checkpoint['task']
        context_data = checkpoint['context_data']  # Serialized context data
        step_index = checkpoint['step_index']
        step_name = checkpoint['step_name']
        agent_state = checkpoint.get('agent_state', {})
        
        from upsonic.utils.printing import info_log, warning_log
        checkpoint_status = checkpoint.get('status', 'unknown')
        
        info_log(
            f"🔄 RESUMING from checkpoint: {durable_execution_id}",
            "DurableRecovery"
        )
        
        if self.debug or debug:
            if checkpoint_status == "failed":
                info_log(
                    f"📍 Failed at step: {step_index} ({step_name})",
                    "DurableRecovery"
                )
                info_log(
                    f"🔄 Will RETRY step: {step_index} ({step_name})",
                    "DurableRecovery"
                )
            else:
                info_log(
                    f"📍 Last completed step: {step_index} ({step_name})",
                    "DurableRecovery"
                )
                info_log(
                    f"⏭️  Will resume from step: {step_index + 1}",
                    "DurableRecovery"
                )
        
        if agent_state:
            self._tool_call_count = agent_state.get('tool_call_count', 0)
            self._tool_limit_reached = agent_state.get('tool_limit_reached', False)
        
        if model:
            from upsonic.models import infer_model
            current_model = infer_model(model)
        else:
            current_model = self.model
        
        # Reconstruct context with current agent and model
        from upsonic.durable.serializer import DurableStateSerializer
        context = DurableStateSerializer.reconstruct_context(
            context_data,
            task=task,
            agent=self,
            model=current_model
        )
        
        from upsonic.agent.pipeline import (
            InitializationStep, CacheCheckStep, UserPolicyStep,
            StorageConnectionStep, LLMManagerStep, ModelSelectionStep,
            ValidationStep, ToolSetupStep, MessageBuildStep,
            ModelExecutionStep, ResponseProcessingStep,
            ReflectionStep, CallManagementStep, TaskManagementStep,
            MemoryMessageTrackingStep, ReliabilityStep, AgentPolicyStep,
            CacheStorageStep, FinalizationStep
        )
        
        all_steps = [
            InitializationStep(),
            StorageConnectionStep(),
            CacheCheckStep(),
            UserPolicyStep(),
            LLMManagerStep(),
            ModelSelectionStep(),
            ValidationStep(),
            ToolSetupStep(),
            MessageBuildStep(),
            ModelExecutionStep(),
            ResponseProcessingStep(),
            ReflectionStep(),
            MemoryMessageTrackingStep(),
            AgentPolicyStep(),  # Move before CallManagementStep
            CallManagementStep(),
            TaskManagementStep(),
            ReliabilityStep(),
            CacheStorageStep(),
            FinalizationStep(),
        ]
        
        # Determine resume point based on checkpoint status
        if checkpoint_status == "failed":
            # Retry the failed step
            resume_from_index = step_index
            completed_count = step_index
        else:
            # Continue from next step after successful checkpoint
            resume_from_index = step_index + 1
            completed_count = resume_from_index
        
        remaining_steps = all_steps[resume_from_index:]
        
        if self.debug or debug:
            info_log(
                f"📋 Total pipeline steps: {len(all_steps)}",
                "DurableRecovery"
            )
            info_log(
                f"⏳ Remaining to execute: {len(remaining_steps)} steps",
                "DurableRecovery"
            )
            
        if self.debug or debug:
            if remaining_steps:
                step_names = [s.name for s in remaining_steps[:5]]
                if len(remaining_steps) > 5:
                    step_names.append(f"... and {len(remaining_steps) - 5} more")
                info_log(
                    f"🎯 Steps to execute: {', '.join(step_names)}",
                    "DurableRecovery"
                )
        
        if not remaining_steps:
            if self.debug or debug:
                from upsonic.utils.printing import info_log
                info_log(
                    f"Execution {durable_execution_id} was already complete at checkpoint",
                    "Agent"
                )
            return task.response
        
        # Create pipeline with remaining steps
        async with self._managed_storage_connection():
            pipeline = PipelineManager(
                steps=remaining_steps,
                debug=debug or self.debug
            )
            
            final_context = await pipeline.execute(context)
            sentry_sdk.flush()
            
            return self._run_result.output
    
    def continue_durable(
        self,
        durable_execution_id: str,
        storage: Optional[Any] = None,
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1
    ) -> Any:
        """
        Continue execution from a durable execution checkpoint synchronously.
        
        This method loads the saved execution state and resumes from the last
        successful checkpoint. It's used to recover from failures or interruptions.
        
        Args:
            durable_execution_id: The execution ID to resume
            storage: Storage backend (if different from the original)
            model: Override model for resumption
            debug: Enable debug mode
            retry: Number of retries
            
        Returns:
            The task output
            
        Raises:
            ValueError: If execution ID not found or state is invalid
            
        Example:
            ```python
            from upsonic import Agent
            from upsonic.durable import FileDurableStorage
            
            storage = FileDurableStorage("./durable_state")
            agent = Agent("openai/gpt-4o")
            
            # Resume from checkpoint
            result = agent.continue_durable(
                durable_execution_id="20250127-abc123",
                storage=storage
            )
            ```
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.continue_durable_async(durable_execution_id, storage, model, debug, retry)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.continue_durable_async(durable_execution_id, storage, model, debug, retry)
                )
        except RuntimeError:
            return asyncio.run(
                self.continue_durable_async(durable_execution_id, storage, model, debug, retry)
            )

