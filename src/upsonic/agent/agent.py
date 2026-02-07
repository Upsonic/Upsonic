import asyncio
import copy
import uuid
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Literal, Optional, Union, TYPE_CHECKING


from upsonic.utils.logging_config import sentry_sdk, get_env_bool_optional
from upsonic.agent.base import BaseAgent
from upsonic.run.agent.output import AgentRunOutput
from upsonic.run.agent.input import AgentRunInput
from upsonic.run.base import RunStatus

from upsonic._utils import now_utc
from upsonic.utils.retry import retryable
from upsonic.tools.processor import ExternalExecutionPause
from upsonic.run.cancel import register_run, cleanup_run, raise_if_cancelled, cancel_run as cancel_run_func, is_cancelled
from upsonic.session.base import SessionType
from upsonic.output import DEFAULT_OUTPUT_TOOL_NAME

if TYPE_CHECKING:
    from upsonic.models import Model, ModelRequest, ModelRequestParameters, ModelResponse
    from upsonic.messages import ToolCallPart, ToolReturnPart
    from upsonic.tasks.tasks import Task
    from upsonic.storage.memory.memory import Memory
    from upsonic.canvas.canvas import Canvas
    from upsonic.models.settings import ModelSettings
    from upsonic.profiles import ModelProfile
    from upsonic.reflection import ReflectionConfig
    from upsonic.safety_engine.base import Policy
    from upsonic.tools import ToolDefinition
    from upsonic.usage import RequestUsage, RunUsage
    from upsonic.agent.context_managers import (
        MemoryManager
    )
    from upsonic.graph.graph import State
    from upsonic.run.events.events import AgentStreamEvent
    from upsonic.db.database import DatabaseBase
    from upsonic.models.model_selector import ModelRecommendation
    from upsonic.culture.culture import Culture
    from upsonic.culture.manager import CultureManager
    from upsonic.run.requirements import RunRequirement
    from upsonic.session.agent import RunData
else:
    Model = "Model"
    ModelRequest = "ModelRequest"
    ModelRequestParameters = "ModelRequestParameters"
    ModelResponse = "ModelResponse"
    Task = "Task"
    Memory = "Memory"
    Canvas = "Canvas"
    ModelSettings = "ModelSettings"
    ModelProfile = "ModelProfile"
    ReflectionConfig = "ReflectionConfig"
    Policy = "Policy"
    ToolDefinition = "ToolDefinition"
    RequestUsage = "RequestUsage"
    MemoryManager = "MemoryManager"
    CulturalKnowledge = "CulturalKnowledge"
    CultureManager = "CultureManager"
    State = "State"
    ModelRecommendation = "ModelRecommendation"
    DatabaseBase = "DatabaseBase"
    RunData = "RunData"


PromptCompressor = None

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
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        debug: bool = False,
        debug_level: int = 1,
        print: Optional[bool] = None,
        company_url: Optional[str] = None,
        company_objective: Optional[str] = None,
        company_description: Optional[str] = None,
        company_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        reflection: bool = False,
        context_management: bool = False,
        context_management_keep_recent: int = 5,
        context_management_model: Optional[str] = None,
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
        feed_tool_call_results: Optional[bool] = None,
        show_tool_calls: bool = True,
        tool_call_limit: int = 100,
        enable_thinking_tool: bool = False,
        enable_reasoning_tool: bool = False,
        tools: Optional[list] = None,
        user_policy: Optional[Union["Policy", List["Policy"]]] = None,
        agent_policy: Optional[Union["Policy", List["Policy"]]] = None,
        tool_policy_pre: Optional[Union["Policy", List["Policy"]]] = None,
        tool_policy_post: Optional[Union["Policy", List["Policy"]]] = None,
        # Policy feedback loop settings
        user_policy_feedback: bool = False,
        agent_policy_feedback: bool = False,
        user_policy_feedback_loop: int = 1,
        agent_policy_feedback_loop: int = 1,
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
        culture: Optional["Culture"] = None,
        # Agent metadata (passed to prompt)
        metadata: Optional[Dict[str, Any]] = None,
        # Workspace settings
        workspace: Optional[str] = None,
    ):
        """
        Initialize the Agent with comprehensive configuration options.
        
        Args:
            model: Model identifier or Model instance
            name: Agent name for identification
            memory: Memory instance for conversation history
            db: Database instance (overrides memory if provided)
            debug: Enable debug logging
            debug_level: Debug level (1 = standard, 2 = detailed). Only used when debug=True
            print: Enable printing of agent output and execution details. If None, reads from UPSONIC_AGENT_PRINT env variable. If set, overrides env variable.
            company_url: Company URL for context
            company_objective: Company objective for context
            company_description: Company description for context
            system_prompt: Custom system prompt
            reflection: Reflection capabilities (default is False)
            context_management: Enable automatic context window management (default True).
                When enabled, the middleware automatically prunes tool call history and
                summarizes old messages when the context approaches the model's limit.
            context_management_keep_recent: Number of recent tool-call events /
                messages to preserve when the context management middleware prunes or
                summarizes the history (default 5).
            context_management_model: Optional model identifier (e.g. 'openai/gpt-4.1')
                with a larger context window to use specifically for context compression /
                summarization. If None, the agent's primary model is used.
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
            tools: List of tools to register with this agent (can be functions, ToolKits, or other agents)
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
            tool_policy_pre: Tool safety policy for pre-execution validation (single policy or list of policies)
            tool_policy_post: Tool safety policy for post-execution validation (single policy or list of policies)
            user_policy_feedback: Enable feedback loop for user policy violations (returns helpful message instead of blocking)
            agent_policy_feedback: Enable feedback loop for agent policy violations (re-executes agent with feedback)
            user_policy_feedback_loop: Maximum retry count for user policy feedback (default 1)
            agent_policy_feedback_loop: Maximum retry count for agent policy feedback (default 1)
            
            culture: Culture instance defining agent behavior and communication guidelines.
                Includes description, add_system_prompt, repeat, and repeat_interval settings.
            workspace: Path to workspace folder containing Agents.md file with agent configuration.
                When set, the Agents.md content is included in system prompt and a greeting 
                message is generated before the first task/chat, integrated into message history.
        """
        from upsonic.models import infer_model
        self.model = infer_model(model)
        self.model_name=model
        self.name = name
        self.agent_id_ = agent_id_
        
        # Session/user overrides
        self._override_session_id = session_id
        self._override_user_id = user_id
        
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
        self.debug_level = debug_level if debug else 1
        self.reflection = reflection
        
        # Handle print flag with hierarchy:
        # 1. ENV variable (highest priority - overrides everything)
        # 2. Agent constructor print parameter
        # 3. Method name (print_do=True, do=False) - lowest priority (resolved per-method call)
        self._print_env: Optional[bool] = get_env_bool_optional("UPSONIC_AGENT_PRINT")
        self._print_param: Optional[bool] = print
        # Initialize with resolved default for introspection (ENV > param > True)
        # Note: This is the "default" value; per-method resolution uses _resolve_print_flag()
        self.print: bool = self._print_env if self._print_env is not None else (print if print is not None else True)

        # Set db attribute
        self.db = db
        
        # Set memory attribute - override with db.memory if db is provided
        if db is not None:
            self.memory = db.memory
        else:
            self.memory = memory
        
        # Model selection attributes
        self.model_selection_criteria = model_selection_criteria
        self.use_llm_for_selection = use_llm_for_selection
        self._model_recommendation: Optional[Any] = None  # Store last recommendation

        self.context_management: bool = context_management
        self.context_management_keep_recent: int = context_management_keep_recent
        self.context_management_model: Optional[str] = context_management_model
        self._context_management_middleware: Optional[Any] = None

        if self.context_management:
            from upsonic.agent.context_managers import ContextManagementMiddleware
            from upsonic.models import infer_model as _infer_model

            compression_model: Optional[Any] = None
            if self.context_management_model is not None:
                compression_model = _infer_model(self.context_management_model)

            self._context_management_middleware = ContextManagementMiddleware(
                model=self.model,
                keep_recent_count=self.context_management_keep_recent,
                context_compression_model=compression_model,
            )

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
        
        # Initialize agent-level tools
        self.tools = tools if tools is not None else []
            
        if self.memory and feed_tool_call_results is not None:
            self.memory.feed_tool_call_results = feed_tool_call_results
        
        self.canvas = canvas
        
        # Agent metadata (injected into prompts)
        self.metadata = metadata or {}
        
        # Store culture and create CultureManager if needed
        self._culture_input = culture
        self._culture_manager: Optional["CultureManager"] = None
        if culture is not None:
            from upsonic.culture.manager import CultureManager
            self._culture_manager = CultureManager(
                model=self.model_name,
                debug=self.debug,
                debug_level=self.debug_level,
                print=self.print,
            )
            self._culture_manager.set_culture(culture)
        
        # Initialize policy managers
        from upsonic.agent.policy_manager import PolicyManager
        self.user_policy_manager = PolicyManager(
            policies=user_policy,
            debug=self.debug,
            enable_feedback=user_policy_feedback,
            feedback_loop_count=user_policy_feedback_loop,
            policy_type="user_policy"
        )
        self.agent_policy_manager = PolicyManager(
            policies=agent_policy,
            debug=self.debug,
            enable_feedback=agent_policy_feedback,
            feedback_loop_count=agent_policy_feedback_loop,
            policy_type="agent_policy"
        )
        
        # Store feedback settings for reference
        self.user_policy_feedback = user_policy_feedback
        self.agent_policy_feedback = agent_policy_feedback
        self.user_policy_feedback_loop = user_policy_feedback_loop
        self.agent_policy_feedback_loop = agent_policy_feedback_loop
        
        # Keep backward compatibility - expose as single policy if only one
        self.user_policy = user_policy
        self.agent_policy = agent_policy
        
        # Initialize tool policy managers
        from upsonic.agent.tool_policy_manager import ToolPolicyManager
        self.tool_policy_pre_manager = ToolPolicyManager(policies=tool_policy_pre, debug=self.debug)
        self.tool_policy_post_manager = ToolPolicyManager(policies=tool_policy_post, debug=self.debug)
        
        # Keep references
        self.tool_policy_pre = tool_policy_pre
        self.tool_policy_post = tool_policy_post
        
        # Handle reflection configuration
        if reflection and not reflection_config:
            # Create default reflection config if reflection=True but no config provided
            from upsonic.reflection import ReflectionConfig
            reflection_config = ReflectionConfig()
        
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
        
        # Track registered agent tools
        self.registered_agent_tools = {}
        
        # Track agent-level builtin tools
        self.agent_builtin_tools = []
        
        # Register agent-level tools immediately
        self._register_agent_tools()
        
        # Tool tracking (deprecated - now tracked in AgentRunOutput)
        # Kept for backwards compatibility
        self._tool_call_count = 0
        self._tool_limit_reached = False
        
        # Run cancellation tracking
        self.run_id: Optional[str] = None
        
        self._setup_policy_models()

        self.session_type = SessionType.AGENT
        
        # Workspace settings
        self.workspace: Optional[str] = workspace
        self._workspace_greeting_executed: bool = False
        self._workspace_agents_md_content: Optional[str] = None
        
        # Pre-load workspace Agents.md content if workspace is set
        if self.workspace:
            self._workspace_agents_md_content = self._read_workspace_agents_md()
    
    def _read_workspace_agents_md(self) -> Optional[str]:
        """Read the Agents.md file from the workspace folder.
        
        Returns:
            Content of the Agents.md file, or None if not found.
        """
        import os
        
        if not self.workspace:
            return None
        
        agents_md_path = os.path.join(self.workspace, "Agents.md")
        
        try:
            with open(agents_md_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except FileNotFoundError:
            if self.debug:
                from upsonic.utils.printing import warning_log
                warning_log(
                    f"Agents.md not found at {agents_md_path}", 
                    "Workspace"
                )
            return None
        except Exception as e:
            if self.debug:
                from upsonic.utils.printing import error_log
                error_log(
                    f"Error reading Agents.md from {agents_md_path}: {str(e)}", 
                    "Workspace"
                )
            return None
    
    async def execute_workspace_greeting_async(
        self,
        return_output: bool = False,
    ) -> Any:
        """Execute the workspace greeting as a proper agent run.
        
        This method is called before the first task/chat when workspace is set.
        It makes an LLM request with a greeting prompt and returns the result
        just like do_async.
        
        Args:
            return_output: If True, return full AgentRunOutput. If False, return content only.
            
        Returns:
            Same as do_async - Task content or AgentRunOutput based on return_output.
        """
        if not self.workspace or self._workspace_greeting_executed:
            return None
        
        from upsonic.tasks.tasks import Task
        
        # Build the greeting prompt
        greeting_prompt = (
            "A new session was started. Say hi briefly (1-2 sentences) and ask what the user wants to do next. "
            "If the runtime model differs from default_model in the system prompt, mention the default model in the greeting. "
            "Do not mention internal steps, files, tools, or reasoning."
        )
        
        # Create a greeting task
        greeting_task = Task(description=greeting_prompt)
        
        # Mark greeting as executed BEFORE calling do_async to prevent recursion
        self._workspace_greeting_executed = True
        
        # Execute the greeting using do_async
        result = await self.do_async(
            task=greeting_task,
            return_output=return_output,
            _print_method_default=False
        )
        print("Greeting result: ", result)
        
        return result
    
    def execute_workspace_greeting(
        self,
        return_output: bool = False,
    ) -> Any:
        """Synchronous version of execute_workspace_greeting_async.
        
        Args:
            return_output: If True, return full AgentRunOutput. If False, return content only.
            
        Returns:
            Same as do - Task content or AgentRunOutput based on return_output.
        """
        import asyncio
        
        if not self.workspace or self._workspace_greeting_executed:
            return None
        
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    self.execute_workspace_greeting_async(return_output)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self.execute_workspace_greeting_async(return_output))
    
    def _setup_policy_models(self) -> None:
        """Setup model references for safety policies."""
        # Setup models for all policies in both managers
        self.user_policy_manager.setup_policy_models(self.model)
        self.agent_policy_manager.setup_policy_models(self.model)
        self.tool_policy_pre_manager.setup_policy_models(self.model)
        self.tool_policy_post_manager.setup_policy_models(self.model)
    
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
    
    @property
    def session_id(self) -> Optional[str]:
        """Get session_id from override, memory, or db."""
        if self._override_session_id:
            return self._override_session_id
        if self.memory and hasattr(self.memory, 'session_id'):
            return self.memory.session_id
        if self.db and hasattr(self.db, 'session_id'):
            return self.db.session_id
        if self.db and hasattr(self.db, 'memory') and hasattr(self.db.memory, 'session_id'):
            return self.db.memory.session_id
        return None
    
    @property
    def user_id(self) -> Optional[str]:
        """Get user_id from override, memory, or db."""
        if self._override_user_id:
            return self._override_user_id
        if self.memory and hasattr(self.memory, 'user_id'):
            return self.memory.user_id
        if self.db and hasattr(self.db, 'user_id'):
            return self.db.user_id
        if self.db and hasattr(self.db, 'memory') and hasattr(self.db.memory, 'user_id'):
            return self.db.memory.user_id
        return None
    
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
    
    def get_run_output(self) -> Optional[AgentRunOutput]:
        """
        Get the AgentRunOutput from the last execution.
        
        Returns:
            AgentRunOutput: The complete run output, or None if no run has been executed
        """
        return getattr(self, '_agent_run_output', None)
    
    def get_session_usage(self) -> "RunUsage":
        """
        Get the aggregated usage for the current session.
        
        Returns the session-level usage from the AgentSession stored in storage.
        If no memory is configured, returns an empty RunUsage.
        
        Usage:
            ```python
            agent = Agent("openai/gpt-4o", memory=memory)
            result = agent.do(task1)
            result = agent.do(task2)
            
            # Get aggregated usage for all runs in this session
            session_usage = agent.get_session_usage()
            print(session_usage.to_dict())
            ```
        
        Returns:
            RunUsage: Aggregated usage metrics for the session.
        """
        from upsonic.usage import RunUsage
        
        # Check if we have memory
        if not self.memory:
            return RunUsage()
        
        # Get session from storage
        session = self.memory.get_session()
        if session and hasattr(session, 'get_session_usage'):
            return session.get_session_usage()
        
        # Return empty usage if no session
        return RunUsage()
    
    async def aget_session_usage(self) -> "RunUsage":
        """
        Get the aggregated usage for the current session (async version).
        
        Returns the session-level usage from the AgentSession stored in storage.
        If no memory is configured, returns an empty RunUsage.
        
        Returns:
            RunUsage: Aggregated usage metrics for the session.
        """
        from upsonic.usage import RunUsage
        
        # Check if we have memory
        if not self.memory:
            return RunUsage()
        
        # Get session from storage asynchronously
        session = await self.memory.get_session_async()
        if session and hasattr(session, 'get_session_usage'):
            return session.get_session_usage()
        
        # Return empty usage if no session
        return RunUsage()
    
    def _create_agent_run_input(self, task: "Task") -> AgentRunInput:
        """
        Create AgentRunInput from Task, separating images and documents.
        
        Categorizes file attachments by mime type without processing:
        - Images: jpg, png, gif, webp, etc.
        - Documents: pdf, docx, txt, etc.
        
        Args:
            task: The task with attachments
            
        Returns:
            AgentRunInput with user_prompt, images (file paths), and documents (file paths)
        """
        import mimetypes
        
        images: List[str] = []
        documents: List[str] = []
        
        if task.attachments:
            for file_path in task.attachments:
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type is None:
                    mime_type = "application/octet-stream"
                
                if mime_type.startswith('image/'):
                    images.append(file_path)
                else:
                    documents.append(file_path)
        
        return AgentRunInput(
            user_prompt=task.description,
            images=images if images else None,
            documents=documents if documents else None
        )
    
    def _convert_to_task(self, task: Union[str, "Task"]) -> "Task":
        """
        Convert a string to a Task object if needed.
        
        Args:
            task: Task object or string description
            
        Returns:
            Task: Task object (converted if string was provided)
        """
        from upsonic.tasks.tasks import Task as TaskClass
        
        if isinstance(task, str):
            return TaskClass(description=task)
        return task
    
    def _resolve_print_flag(self, method_default: bool) -> bool:
        """
        Resolve the print flag based on hierarchy.
        
        Priority (highest to lowest):
        1. UPSONIC_AGENT_PRINT env variable (overrides everything)
        2. Agent constructor print parameter
        3. Method default (print_do=True, do=False)
        
        Args:
            method_default: Default value based on method name (True for print_do, False for do)
            
        Returns:
            bool: Whether to print output
        """
        # 1. ENV has highest priority - overrides everything
        if self._print_env is not None:
            return self._print_env
        
        # 2. Agent constructor parameter
        if self._print_param is not None:
            return self._print_param
        
        # 3. Method default (print_do=True, do=False)
        return method_default
    
    def _validate_task_for_new_run(
        self, 
        task: "Task", 
        is_resuming: bool = False
    ) -> Optional[AgentRunOutput]:
        """
        Validate task state before starting a new run.
        
        Checks if task is already completed or has problematic status.
        
        Args:
            task: Task to validate
            is_resuming: Whether this is a resume operation (skips problematic check)
            
        Returns:
            AgentRunOutput if task cannot be run (error/warning case), None if OK to proceed
        """
        from upsonic.utils.printing import warning_log
        
        # Check if task is already completed
        if task.is_completed:
            run_id = task.run_id or "unknown"
            warning_log(
                f"Task is already completed (run_id={run_id}). Cannot re-run a completed task.",
                "Agent"
            )
            return AgentRunOutput(
                run_id=run_id,
                agent_id=self.agent_id,
                agent_name=self.name,
                session_id=self.session_id,
                user_id=self.user_id,
                status=RunStatus.completed,
                output=f"Task is already completed (run_id={run_id}). Cannot re-run a completed task.",
            )
        
        # Check if task has a problematic status (only for non-resuming calls)
        if not is_resuming and task.is_problematic:
            status_str = task.status.value
            run_id = task.run_id
            warning_log(
                f"Task has a problematic run (run_id={run_id}, status={status_str}). "
                f"Use continue_run_async() to continue this run.",
                "Agent"
            )
            return AgentRunOutput(
                run_id=run_id,
                agent_id=self.agent_id,
                agent_name=self.name,
                session_id=self.session_id,
                user_id=self.user_id,
                status=task.status,
                output=f"Cannot start new run: Task has a {status_str} run (run_id={run_id}). "
                       f"Call continue_run_async() to continue this run.",
            )
        
        return None  # OK to proceed
    
    def _create_agent_run_output(
        self,
        run_id: str,
        task: "Task",
        run_input: AgentRunInput,
        is_streaming: bool = False
    ) -> AgentRunOutput:
        """
        Create AgentRunOutput with all required fields.
        
        This is a centralized factory method for creating AgentRunOutput instances,
        ensuring consistency across do_async and astream methods.
        
        Args:
            run_id: Unique run identifier
            task: Task being executed
            run_input: Agent run input data
            is_streaming: Whether this is a streaming execution
            
        Returns:
            AgentRunOutput: Initialized output context
        """
        from upsonic.schemas.kb_filter import KBFilterExpr
        
        kb_filter = KBFilterExpr.from_task(task) if hasattr(KBFilterExpr, 'from_task') else None
        
        return AgentRunOutput(
            run_id=run_id,
            agent_id=self.agent_id,
            agent_name=self.name,
            session_id=self.session_id,
            user_id=self.user_id,
            task=task,
            input=run_input,
            output=None,
            output_schema=task.response_format if hasattr(task, 'response_format') else None,
            thinking_content=None,
            thinking_parts=None,
            model_name=self.model.model_name if self.model else None,
            model_provider=self.model.system if self.model else None,
            model_provider_profile=self.model.profile if self.model else None,
            chat_history=[],
            messages=None,  # Will be set by finalize_run_messages()
            response=None,
            usage=None,
            additional_input_message=None,
            memory_message_count=0,
            tools=[],
            tool_call_count=0,
            tool_limit_reached=False,
            images=None,
            files=None,
            status=RunStatus.running,
            requirements=[],
            step_results=[],
            execution_stats=None,
            events=[],
            agent_knowledge_base_filter=kb_filter,
            metadata=None,
            session_state=None,
            is_streaming=is_streaming,
            accumulated_text="",
            pause_reason=None,
            error_details=None,
            created_at=int(now_utc().timestamp()),
            updated_at=None
        )
    
    def _apply_model_override(self, model: Optional[Union[str, "Model"]]) -> None:
        """
        Apply model override if provided.
        
        Args:
            model: Optional model override (string or Model instance)
        """
        if model:
            from upsonic.models import infer_model
            self.model = infer_model(model)
    
    
    def get_run_id(self) -> Optional[str]:
        """
        Get the current run ID.
        
        Returns:
            str: The current run ID, or None if no run is active.
        """
        return self.run_id
    
    def cancel_run(self, run_id: Optional[str] = None) -> bool:
        """
        Cancel a run by its ID.
        
        If no run_id is provided, cancels the current run.
        
        Args:
            run_id: The ID of the run to cancel. If None, cancels the current run.
            
        Returns:
            bool: True if the run was found and cancelled, False otherwise.
        """
        target_run_id = run_id or self.run_id
        if not target_run_id:
            return False
        return cancel_run_func(target_run_id)
    
    def _validate_tools_with_policy_pre(
        self, 
        context_description: str = "Tool Validation",
        registered_tools_dicts: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Validate all currently registered tools with tool_policy_pre before use.
        
        This is a centralized method for tool safety validation that can be called
        from different registration points (_register_agent_tools, add_tools, _setup_task_tools).
        
        Args:
            context_description: Description of where this validation is being called from
            registered_tools_dicts: List of registered tools dictionaries to check when removing tools.
                                   If None, defaults to [self.registered_agent_tools]
            
        Raises:
            DisallowedOperation: If any tool is blocked by the safety policy
        """
        if not hasattr(self, 'tool_policy_pre_manager') or not self.tool_policy_pre_manager.has_policies():
            return
        
        # Default to agent tools if not specified
        if registered_tools_dicts is None:
            registered_tools_dicts = [self.registered_agent_tools]
        
        import asyncio
        tool_definitions = self.tool_manager.get_tool_definitions()
        
        for tool_def in tool_definitions:
            # Skip built-in orchestration tools
            if tool_def.name == 'plan_and_execute':
                continue
                
            tool_info = {
                "name": tool_def.name,
                "description": tool_def.description or "",
                "parameters": tool_def.parameters_json_schema or {},
                "metadata": tool_def.metadata or {}
            }
            
            # Execute validation synchronously using nest_asyncio if needed
            try:
                # Check if we're in async context
                loop = asyncio.get_running_loop()
                # Already in event loop - use nest_asyncio
                import nest_asyncio
                nest_asyncio.apply()
                validation_result = asyncio.run(
                    self.tool_policy_pre_manager.execute_tool_validation_async(
                        tool_info=tool_info,
                        check_type=f"Pre-Execution Tool Validation ({context_description})"
                    )
                )
            except RuntimeError:
                # No event loop - safe to use asyncio.run()
                validation_result = asyncio.run(
                    self.tool_policy_pre_manager.execute_tool_validation_async(
                        tool_info=tool_info,
                        check_type=f"Pre-Execution Tool Validation ({context_description})"
                    )
                )
            
            if validation_result.should_block():
                # Handle blocking based on action type
                # If DisallowedOperation was raised by a RAISE action policy, re-raise it
                if validation_result.disallowed_exception:
                    raise validation_result.disallowed_exception
                
                # Otherwise it's a BLOCK action - skip this tool without raising exception
                # Remove the tool from the tool manager to prevent its use
                if self.debug:
                    from upsonic.utils.printing import warning_log
                    warning_log(
                        f"Tool '{tool_def.name}' blocked by safety policy: {validation_result.get_final_message()}",
                        "Tool Safety"
                    )
                
                # Find which registered_tools dict contains this tool and remove it
                # Try each dict in the provided list
                for registered_tools_dict in registered_tools_dicts:
                    self.tool_manager.remove_tools(
                        tools=[tool_def.name],
                        registered_tools=registered_tools_dict
                    )
                    
                    # Also remove from the tracking dict to keep it clean
                    if tool_def.name in registered_tools_dict:
                        del registered_tools_dict[tool_def.name]
    
    def _register_agent_tools(self) -> None:
        """
        Register agent-level tools with the ToolManager.
        
        This is called in __init__ to ensure agent tools are registered immediately.
        Automatically includes canvas tools if canvas is provided.
        """
        # Prepare tools list starting with user-provided tools
        final_tools = list(self.tools) if self.tools else []
        
        if self.canvas:
            canvas_functions = self.canvas.functions()
            for canvas_func in canvas_functions:
                if canvas_func not in final_tools:
                    final_tools.append(canvas_func)
            self.tools = final_tools
        
        if not final_tools:
            self.registered_agent_tools = {}
            self.agent_builtin_tools = []
            return
        
        # Add thinking tool if enabled
        if self.enable_thinking_tool:
            from upsonic.tools.orchestration import plan_and_execute
            if plan_and_execute not in final_tools:
                final_tools.append(plan_and_execute)
        
        # Separate builtin tools from regular tools
        from upsonic.tools.builtin_tools import AbstractBuiltinTool
        builtin_tools = []
        regular_tools = []
        
        for tool in final_tools:
            if tool is not None and isinstance(tool, AbstractBuiltinTool):
                builtin_tools.append(tool)
            else:
                regular_tools.append(tool)
        
        # Handle builtin tools separately - they don't need ToolManager/ToolProcessor
        self.agent_builtin_tools = builtin_tools
        
        # Register only regular tools with ToolManager
        if regular_tools:
            self.registered_agent_tools = self.tool_manager.register_tools(
                tools=regular_tools,
                task=None,  # Agent tools not task-specific
                agent_instance=self
            )
        else:
            self.registered_agent_tools = {}
        
        # PRE-EXECUTION TOOL VALIDATION
        # Validate all registered agent tools with tool_policy_pre
        self._validate_tools_with_policy_pre(
            context_description="Agent Tool Registration",
            registered_tools_dicts=[self.registered_agent_tools]
        )
    
    def add_tools(self, tools: Union[Any, List[Any]]) -> None:
        """
        Dynamically add tools to the agent and register them.
        
        This method:
        1. Separates builtin tools from regular tools
        2. For builtin tools: Updates self.tools and self.agent_builtin_tools directly
        3. For regular tools: Calls ToolManager to register them
        4. Updates self.registered_agent_tools with wrapped tools
        5. Validates tools with tool_policy_pre if configured
        
        Args:
            tools: A single tool or list of tools to add
            
        Raises:
            DisallowedOperation: If any tool is blocked by the safety policy
        """
        if not isinstance(tools, list):
            tools = [tools]
        
        # Prepare tools with plan_and_execute if needed
        tools_to_add = list(tools)
        
        # Add thinking tool if enabled and not already in the list
        if self.enable_thinking_tool:
            from upsonic.tools.orchestration import plan_and_execute
            if plan_and_execute not in tools_to_add and plan_and_execute not in self.tools:
                tools_to_add.append(plan_and_execute)
        
        # Separate builtin tools from regular tools
        from upsonic.tools.builtin_tools import AbstractBuiltinTool
        builtin_tools = []
        regular_tools = []
        
        for tool in tools_to_add:
            if tool is not None and isinstance(tool, AbstractBuiltinTool):
                builtin_tools.append(tool)
            else:
                regular_tools.append(tool)
        
        # Handle builtin tools separately - they don't need ToolManager/ToolProcessor
        if builtin_tools:
            if not hasattr(self, 'agent_builtin_tools'):
                self.agent_builtin_tools = []
            
            # Merge builtin tools (avoid duplicates based on unique_id)
            existing_ids = {tool.unique_id for tool in self.agent_builtin_tools}
            for tool in builtin_tools:
                if tool.unique_id not in existing_ids:
                    self.agent_builtin_tools.append(tool)
                    existing_ids.add(tool.unique_id)
        
        # Handle regular tools through ToolManager
        if regular_tools:
            # Call ToolManager to register new tools (filters already registered ones)
            newly_registered = self.tool_manager.register_tools(
                tools=regular_tools,
                task=None,  # Agent tools are not task-specific
                agent_instance=self
            )
            
            # Update self.registered_agent_tools with newly registered tools
            self.registered_agent_tools.update(newly_registered)
        
        # Update self.tools - add original tool objects (not plan_and_execute if auto-added)
        for tool in tools:
            if tool not in self.tools:
                self.tools.append(tool)
        
        # PRE-EXECUTION TOOL VALIDATION
        # Validate newly added tools with tool_policy_pre
        self._validate_tools_with_policy_pre(
            context_description="Dynamic Tool Addition (add_tools)",
            registered_tools_dicts=[self.registered_agent_tools]
        )
    
    def remove_tools(self, tools: Union[str, List[str], Any, List[Any]]) -> None:
        """
        Remove tools from the agent.
        
        Supports removing:
        - Tool names (strings)
        - Function objects
        - Agent objects
        - MCP handlers (and all their tools)
        - Class instances (ToolKit or regular classes, and all their tools)
        - Builtin tools (AbstractBuiltinTool instances)
        
        Args:
            tools: Single tool or list of tools to remove (any type)
        """
        if not isinstance(tools, list):
            tools = [tools]
        
        # Separate builtin tools from regular tools
        from upsonic.tools.builtin_tools import AbstractBuiltinTool
        builtin_tools_to_remove = []
        regular_tools_to_remove = []
        
        for tool in tools:
            if tool is not None and isinstance(tool, AbstractBuiltinTool):
                builtin_tools_to_remove.append(tool)
            else:
                regular_tools_to_remove.append(tool)
        
        # Handle regular tools through ToolManager
        removed_tool_names = []
        removed_objects = []
        
        if regular_tools_to_remove:
            # Call ToolManager to handle all removal logic for regular tools
            removed_tool_names, removed_objects = self.tool_manager.remove_tools(
                tools=regular_tools_to_remove,
                registered_tools=self.registered_agent_tools
            )
            
            # Update self.registered_agent_tools - remove the tool names
            for tool_name in removed_tool_names:
                if tool_name in self.registered_agent_tools:
                    del self.registered_agent_tools[tool_name]
        
        # Handle builtin tools separately - they don't use ToolManager/ToolProcessor
        if builtin_tools_to_remove and hasattr(self, 'agent_builtin_tools'):
            # Remove from agent_builtin_tools by unique_id
            builtin_ids_to_remove = {tool.unique_id for tool in builtin_tools_to_remove}
            self.agent_builtin_tools = [
                tool for tool in self.agent_builtin_tools 
                if tool.unique_id not in builtin_ids_to_remove
            ]
            # Add to removed_objects for self.tools cleanup
            removed_objects.extend(builtin_tools_to_remove)
        
        # Update self.tools - remove all removed objects (regular + builtin)
        if removed_objects:
            self.tools = [t for t in self.tools if t not in removed_objects]
    
    def get_tool_defs(self) -> List["ToolDefinition"]:
        """
        Get the tool definitions for all currently registered tools.
        
        Returns:
            List[ToolDefinition]: List of tool definitions from the ToolManager
        """
        return self.tool_manager.get_tool_definitions()
    
    def _setup_task_tools(self, task: "Task") -> None:
        """Setup tools with ToolManager for the current task (task tools only)."""
        self._tool_limit_reached = False
        
        # Always initialize tool metrics (needed for both agent and task tools)
        from upsonic.tools import ToolMetrics
        self._tool_metrics = ToolMetrics(
            tool_call_count=self._tool_call_count,
            tool_call_limit=self.tool_call_limit
        )
        
        # Only process task-level tools (agent tools already registered in __init__)
        task_tools = task.tools if task.tools else []
        
        # Determine thinking/reasoning settings (Task overrides Agent)
        is_thinking_enabled = self.enable_thinking_tool
        if task.enable_thinking_tool is not None:
            is_thinking_enabled = task.enable_thinking_tool
        
        is_reasoning_enabled = self.enable_reasoning_tool
        if task.enable_reasoning_tool is not None:
            is_reasoning_enabled = task.enable_reasoning_tool

        if is_reasoning_enabled and not is_thinking_enabled:
            raise ValueError("Configuration error: 'enable_reasoning_tool' cannot be True if 'enable_thinking_tool' is False.")

        # If thinking is enabled at task level, add plan_and_execute to task tools
        # (unless it's already explicitly added as a regular tool)
        from upsonic.tools.orchestration import plan_and_execute
        
        tools_to_register = list(task_tools) if task_tools else []
        
        if is_thinking_enabled and plan_and_execute not in tools_to_register:
            tools_to_register.append(plan_and_execute)
        
        # If no tools to register, return early
        if not tools_to_register:
            return

        agent_for_this_run = copy.copy(self)
        agent_for_this_run.enable_thinking_tool = is_thinking_enabled
        agent_for_this_run.enable_reasoning_tool = is_reasoning_enabled

        # Separate builtin tools from regular tools
        from upsonic.tools.builtin_tools import AbstractBuiltinTool
        builtin_tools = []
        regular_tools = []
        
        for tool in tools_to_register:
            if tool is not None and isinstance(tool, AbstractBuiltinTool):
                builtin_tools.append(tool)
            else:
                regular_tools.append(tool)
        
        # Handle builtin tools separately - they don't need ToolManager/ToolProcessor
        task.task_builtin_tools = builtin_tools
        
        # Register only regular task tools and store them in task.registered_task_tools
        if regular_tools:
            newly_registered = self.tool_manager.register_tools(
                tools=regular_tools,
                task=task,
                agent_instance=agent_for_this_run
            )
        else:
            newly_registered = {}
        
        # Update task's registered_task_tools with newly registered tools
        task.registered_task_tools.update(newly_registered)
        
        # PRE-EXECUTION TOOL VALIDATION
        # Validate all registered tools (agent + task) with tool_policy_pre before execution
        self._validate_tools_with_policy_pre(
            context_description="Task Tool Setup",
            registered_tools_dicts=[self.registered_agent_tools, task.registered_task_tools]
        )
    
    async def _build_model_request(
        self, 
        task: "Task", 
        memory_handler: Optional["MemoryManager"], 
        state: Optional["State"] = None,
    ) -> tuple[List["ModelRequest"], Optional["ModelResponse"]]:
        """Build the complete message history for the model request.

        Returns:
            A tuple of (messages, context_full_response).
            context_full_response is None when the context fits within the
            model's window, or a ModelResponse with a fixed message when
            the context is full after all reduction strategies.
        """
        from upsonic.agent.context_managers import SystemPromptManager, ContextManager
        from upsonic.messages import SystemPromptPart, UserPromptPart, ModelRequest
        
        messages: List["ModelRequest"] = []
        message_history = memory_handler.get_message_history()
        messages.extend(message_history)
        
        system_prompt_manager = SystemPromptManager(self, task)
        context_manager = ContextManager(self, task, state)
        
        async with system_prompt_manager.manage_system_prompt(memory_handler) as sp_handler, \
                   context_manager.manage_context(memory_handler) as _ctx_handler:
            
            if hasattr(self, '_agent_run_output') and self._agent_run_output and self._agent_run_output.input:
                run_input = self._agent_run_output.input
                if run_input.input is None:
                    run_input.build_input(context_formatted=task.context_formatted)
                task_input = run_input.input
                if task.context_formatted:
                    task.context_formatted = None
            else:
                raise RuntimeError("AgentRunInput not available. This should not happen.")
            
            user_part = UserPromptPart(content=task_input)
            
            parts = []
            
            if sp_handler.should_include_system_prompt(messages):
                system_prompt = sp_handler.get_system_prompt()
                if system_prompt:
                    system_part = SystemPromptPart(content=system_prompt)
                    parts.append(system_part)
            
            parts.append(user_part)
            
            current_request = ModelRequest(parts=parts)
            messages.append(current_request)

        # Apply context management middleware
        context_full_response: Optional["ModelResponse"] = None
        if self.context_management and self._context_management_middleware:
            managed_msgs, ctx_full = await self._context_management_middleware.apply(messages)
            messages = managed_msgs
            if ctx_full:
                context_full_response = self._context_management_middleware._build_context_full_response(
                    model_name=self.model.model_name
                )

        return messages, context_full_response
    
    async def _build_model_request_with_input(
        self, 
        task: "Task", 
        memory_handler: Optional["MemoryManager"], 
        current_input: Any, 
        temporary_message_history: List["ModelRequest"],
        state: Optional["State"] = None,
    ) -> tuple[List["ModelRequest"], Optional["ModelResponse"]]:
        """Build model request with custom input and message history for guardrail retries.

        Returns:
            A tuple of (messages, context_full_response).
            context_full_response is None when the context fits within the
            model's window, or a ModelResponse with a fixed message when
            the context is full after all reduction strategies.
        """
        from upsonic.agent.context_managers import SystemPromptManager, ContextManager
        from upsonic.messages import SystemPromptPart, UserPromptPart, ModelRequest
        
        messages: List["ModelRequest"] = list(temporary_message_history)
        
        system_prompt_manager = SystemPromptManager(self, task)
        context_manager = ContextManager(self, task, state)
        
        async with system_prompt_manager.manage_system_prompt(memory_handler) as sp_handler, \
                   context_manager.manage_context(memory_handler) as _ctx_handler:
            
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

        # Apply context management middleware
        context_full_response: Optional["ModelResponse"] = None
        if self.context_management and self._context_management_middleware:
            managed_msgs, ctx_full = await self._context_management_middleware.apply(messages)
            messages = managed_msgs
            if ctx_full:
                context_full_response = self._context_management_middleware._build_context_full_response(
                    model_name=self.model.model_name
                )

        return messages, context_full_response
    
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
        
        # Combine agent-level and task-level builtin tools
        agent_builtin_tools = getattr(self, 'agent_builtin_tools', [])
        task_builtin_tools = getattr(task, 'task_builtin_tools', [])

        # Merge builtin tools, avoiding duplicates based on unique_id
        builtin_tools_dict = {}
        for tool in agent_builtin_tools:
            builtin_tools_dict[tool.unique_id] = tool
        for tool in task_builtin_tools:
            builtin_tools_dict[tool.unique_id] = tool
        builtin_tools = list(builtin_tools_dict.values())
        
        output_mode = 'text'
        output_object = None
        output_tools = []
        allow_text_output = True
        
        if task.response_format and task.response_format != str and task.response_format is not str:
            if isinstance(task.response_format, type) and issubclass(task.response_format, BaseModel):
                output_mode = 'auto'
                allow_text_output = False
                
                schema = task.response_format.model_json_schema()
                output_object = OutputObjectDefinition(
                    json_schema=schema,
                    name=task.response_format.__name__,
                    description=task.response_format.__doc__,
                    strict=True
                )
                
                # Create output tool for tool-based structured output
                output_tools = self._build_output_tools(task.response_format, schema)
        
        return ModelRequestParameters(
            function_tools=tool_definitions,
            builtin_tools=builtin_tools,
            output_mode=output_mode,
            output_object=output_object,
            output_tools=output_tools,
            allow_text_output=allow_text_output
        )
    
    def _build_output_tools(self, response_format: type, schema: dict) -> list:
        """Build output tools for tool-based structured output.
        
        Creates a ToolDefinition that the model can use to return structured data
        when native JSON schema output is not supported.
        
        Args:
            response_format: The Pydantic model class for the response
            schema: The JSON schema for the response format
            
        Returns:
            List containing a single ToolDefinition for structured output
        """
        from upsonic.tools import ToolDefinition
        
        return [ToolDefinition(
            name=DEFAULT_OUTPUT_TOOL_NAME,
            parameters_json_schema=schema,
            description=response_format.__doc__ or f"Return the final result as a {response_format.__name__}",
            kind='output',
            strict=True
        )]
    
    async def _execute_tool_calls(self, tool_calls: List["ToolCallPart"]) -> List["ToolReturnPart"]:
        """
        Execute tool calls and return results.
        
        Handles both sequential and parallel execution based on tool configuration.
        Tools marked as sequential will be executed one at a time.
        Other tools can be executed in parallel if multiple are called.
        """
        from upsonic.messages import ToolReturnPart
        
        if not tool_calls:
            return []
        
        # Check for cancellation before executing tools
        if self.run_id:
            raise_if_cancelled(self.run_id)
        
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
            # POST-EXECUTION TOOL CALL VALIDATION
            if hasattr(self, 'tool_policy_post_manager') and self.tool_policy_post_manager.has_policies():
                tool_def = tool_defs.get(tool_call.tool_name)
                tool_call_info = {
                    "name": tool_call.tool_name,
                    "description": tool_def.description if tool_def else "",
                    "parameters": tool_def.parameters_json_schema if tool_def else {},
                    "arguments": tool_call.args_as_dict(),
                    "call_id": tool_call.tool_call_id
                }
                
                validation_result = await self.tool_policy_post_manager.execute_tool_call_validation_async(
                    tool_call_info=tool_call_info,
                    check_type="Post-Execution Tool Call Validation"
                )
                
                if validation_result.should_block():
                    # Handle blocking based on action type
                    # If DisallowedOperation was raised by a RAISE action policy, re-raise it
                    if validation_result.disallowed_exception:
                        raise validation_result.disallowed_exception
                    
                    # Otherwise it's a BLOCK action - return error message without raising
                    results.append(ToolReturnPart(
                        tool_name=tool_call.tool_name,
                        content=validation_result.get_final_message(),
                        tool_call_id=tool_call.tool_call_id,
                        timestamp=now_utc()
                    ))
                    continue  # Skip execution
            
            try:
                import time
                tool_start_time = time.time()
                result = await self.tool_manager.execute_tool(
                    tool_name=tool_call.tool_name,
                    args=tool_call.args_as_dict(),
                    metrics=self._tool_metrics,
                    tool_call_id=tool_call.tool_call_id
                )
                tool_execution_time = time.time() - tool_start_time
                
                self._tool_call_count += 1
                if hasattr(self, '_tool_metrics') and self._tool_metrics:
                    self._tool_metrics.tool_call_count = self._tool_call_count
                
                tool_return = ToolReturnPart(
                    tool_name=result.tool_name,
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                    timestamp=now_utc()
                )
                results.append(tool_return)
                
                # Track tool execution in AgentRunOutput
                if hasattr(self, '_agent_run_output') and self._agent_run_output:
                    from upsonic.run.tools.tools import ToolExecution
                    tool_exec = ToolExecution(
                        tool_call_id=tool_call.tool_call_id,
                        tool_name=tool_call.tool_name,
                        tool_args=tool_call.args_as_dict(),
                        result=str(result.content) if result.content else None,
                    )
                    if self._agent_run_output.tools is None:
                        self._agent_run_output.tools = []
                    self._agent_run_output.tools.append(tool_exec)
                
                # Level 2: Detailed tool execution logging
                if self.debug and self.debug_level >= 2:
                    from upsonic.utils.printing import debug_log_level2
                    tool_def = tool_defs.get(tool_call.tool_name)
                    debug_log_level2(
                        f"Tool executed: {tool_call.tool_name}",
                        "Agent",
                        debug=self.debug,
                        debug_level=self.debug_level,
                        tool_name=tool_call.tool_name,
                        tool_description=tool_def.description if tool_def else "Unknown",
                        tool_parameters=tool_call.args_as_dict(),
                        tool_result=str(result.content)[:1000] if result.content else None,  # Truncate very long results
                        tool_execution_time=tool_execution_time,
                        tool_call_id=tool_call.tool_call_id,
                        total_tool_calls=self._tool_call_count,
                        tool_call_limit=self.tool_call_limit,
                        tool_sequential=tool_def.sequential if tool_def else False
                    )
                
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
                
                # Level 2: Tool execution error details
                if self.debug and self.debug_level >= 2:
                    from upsonic.utils.printing import debug_log_level2
                    import traceback
                    error_traceback = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    debug_log_level2(
                        f"Tool execution error: {tool_call.tool_name}",
                        "Agent",
                        debug=self.debug,
                        debug_level=self.debug_level,
                        tool_name=tool_call.tool_name,
                        tool_parameters=tool_call.args_as_dict(),
                        error_type=type(e).__name__,
                        error_message=str(e),
                        error_traceback=error_traceback[-1500:],  # Last 1500 chars
                        tool_call_id=tool_call.tool_call_id
                    )
        
        if parallel_calls:
            async def execute_single_tool(tool_call: "ToolCallPart") -> "ToolReturnPart":
                """Execute a single tool call and return the result."""
                # POST-EXECUTION TOOL CALL VALIDATION (for parallel execution)
                if hasattr(self, 'tool_policy_post_manager') and self.tool_policy_post_manager.has_policies():
                    tool_def = tool_defs.get(tool_call.tool_name)
                    tool_call_info = {
                        "name": tool_call.tool_name,
                        "description": tool_def.description if tool_def else "",
                        "parameters": tool_def.parameters_json_schema if tool_def else {},
                        "arguments": tool_call.args_as_dict(),
                        "call_id": tool_call.tool_call_id
                    }
                    
                    validation_result = await self.tool_policy_post_manager.execute_tool_call_validation_async(
                        tool_call_info=tool_call_info,
                        check_type="Post-Execution Tool Call Validation"
                    )
                    
                    if validation_result.should_block():
                        # Handle blocking based on action type
                        # If DisallowedOperation was raised by a RAISE action policy, re-raise it
                        if validation_result.disallowed_exception:
                            raise validation_result.disallowed_exception
                        
                        # Otherwise it's a BLOCK action - return error message without raising
                        return ToolReturnPart(
                            tool_name=tool_call.tool_name,
                            content=validation_result.get_final_message(),
                            tool_call_id=tool_call.tool_call_id,
                            timestamp=now_utc()
                        )
                
                try:
                    result = await self.tool_manager.execute_tool(
                        tool_name=tool_call.tool_name,
                        args=tool_call.args_as_dict(),
                        metrics=self._tool_metrics,
                        tool_call_id=tool_call.tool_call_id
                    )
                    
                    # Track tool execution in AgentRunOutput (parallel)
                    if hasattr(self, '_agent_run_output') and self._agent_run_output:
                        from upsonic.run.tools.tools import ToolExecution
                        tool_exec = ToolExecution(
                            tool_call_id=tool_call.tool_call_id,
                            tool_name=tool_call.tool_name,
                            tool_args=tool_call.args_as_dict(),
                            result=str(result.content) if result.content else None,
                        )
                        if self._agent_run_output.tools is None:
                            self._agent_run_output.tools = []
                        self._agent_run_output.tools.append(tool_exec)
                    
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
            
            # Execute all tools in parallel, capturing exceptions
            parallel_results = await asyncio.gather(
                *[execute_single_tool(tc) for tc in parallel_calls],
                return_exceptions=True  # Capture ALL exceptions including ExternalExecutionPause
            )
            
            # Separate successful results from external execution pauses
            external_pauses: List[ExternalExecutionPause] = []
            successful_results: List["ToolReturnPart"] = []
            other_errors: List[Exception] = []
            
            for tc, result in zip(parallel_calls, parallel_results):
                if isinstance(result, ExternalExecutionPause):
                    external_pauses.append(result)
                elif isinstance(result, Exception):
                    # Other exceptions - convert to error result
                    other_errors.append(result)
                    successful_results.append(ToolReturnPart(
                        tool_name=tc.tool_name,
                        content=f"Error executing tool: {str(result)}",
                        tool_call_id=tc.tool_call_id,
                        timestamp=now_utc()
                    ))
                else:
                    successful_results.append(result)
            
            # If ANY tools need external execution, combine them into ONE exception
            if external_pauses:
                # Collect all external_calls from all pauses (standardized on external_calls list)
                all_external_calls = []
                for pause in external_pauses:
                    if pause.external_calls:
                        all_external_calls.extend(pause.external_calls)
                
                # Raise a single exception with ALL external calls
                combined_pause = ExternalExecutionPause(external_calls=all_external_calls)
                raise combined_pause
            
            self._tool_call_count += len(parallel_calls)
            if hasattr(self, '_tool_metrics') and self._tool_metrics:
                self._tool_metrics.tool_call_count = self._tool_call_count
            
            results.extend(successful_results)
        
        return results
    
    async def _handle_model_response(
        self, 
        response: "ModelResponse", 
        messages: List["ModelRequest"]
    ) -> "ModelResponse":
        """Handle model response including tool calls."""
        from upsonic.messages import ToolCallPart, TextPart, UserPromptPart, ModelRequest, ModelResponse
        from upsonic._utils import now_utc
        
        if hasattr(self, '_tool_limit_reached') and self._tool_limit_reached:
            return response
        
        # Handle culture repeat logic
        if self._culture_manager and self._culture_manager.enabled:
            culture = self._culture_manager.culture
            if culture and culture.repeat:
                if self._culture_manager.should_repeat():
                    # Ensure culture is prepared
                    if not self._culture_manager.prepared:
                        await self._culture_manager.aprepare()
                    
                    culture_formatted = self._culture_manager.format_for_system_prompt()
                    if culture_formatted:
                        # Create mock ModelRequest with culture guidelines
                        culture_system_part = UserPromptPart(content=culture_formatted)
                        culture_request = ModelRequest(parts=[culture_system_part])
                        
                        # Create mock ModelResponse acknowledging culture
                        culture_response = ModelResponse(
                            parts=[TextPart(content="Culture guidelines acknowledged.")],
                            model_name=response.model_name if response else None,
                            timestamp=now_utc(),
                            usage=response.usage if response else None,
                            provider_name=response.provider_name if response else None,
                        )
                        
                        # Insert culture into messages before the current response
                        messages.append(culture_request)
                        messages.append(culture_response)
        
        tool_calls = [
            part for part in response.parts 
            if isinstance(part, ToolCallPart)
        ]
        
        # Filter out output tool calls - these are used for structured output
        # and should not be executed as regular tools
        output_tool_names = {DEFAULT_OUTPUT_TOOL_NAME}
        regular_tool_calls = [tc for tc in tool_calls if tc.tool_name not in output_tool_names]
        
        # If all tool calls are output tools, return response directly (structured output)
        if tool_calls and not regular_tool_calls:
            return response
        
        if regular_tool_calls:
            tool_results = await self._execute_tool_calls(regular_tool_calls)
            
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
                
                # Apply context management middleware before model request
                if self.context_management and self._context_management_middleware:
                    managed_msgs, ctx_full = await self._context_management_middleware.apply(messages)
                    messages.clear()
                    messages.extend(managed_msgs)
                    if ctx_full:
                        return self._context_management_middleware._build_context_full_response(
                            model_name=self.model.model_name
                        )
                
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
            
            # Apply context management middleware before follow-up model request
            if self.context_management and self._context_management_middleware:
                managed_msgs, ctx_full = await self._context_management_middleware.apply(messages)
                messages.clear()
                messages.extend(managed_msgs)
                if ctx_full:
                    return self._context_management_middleware._build_context_full_response(
                        model_name=self.model.model_name
                    )
            
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
            from upsonic.utils.printing import cache_configuration, debug_log_level2
            embedding_provider_name = None
            if task.cache_embedding_provider:
                embedding_provider_name = task.cache_embedding_provider.model_name
            
            cache_configuration(
                enable_cache=task.enable_cache,
                cache_method=task.cache_method,
                cache_threshold=task.cache_threshold if task.cache_method == "vector_search" else None,
                cache_duration_minutes=task.cache_duration_minutes,
                embedding_provider=embedding_provider_name
            )
            
            # Level 2: Detailed cache information
            if self.debug_level >= 2:
                debug_log_level2(
                    "Cache check details",
                    "Agent",
                    debug=self.debug,
                    debug_level=self.debug_level,
                    task_description=task.description[:100] if task.description else None,
                    cache_enabled=task.enable_cache,
                    cache_method=task.cache_method,
                    cache_threshold=task.cache_threshold,
                    cache_duration_minutes=task.cache_duration_minutes,
                    embedding_provider=embedding_provider_name,
                    model_name=self.model.model_name
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
    
    async def _apply_user_policy(
        self, 
        task: "Task", 
        context: Optional[AgentRunOutput] = None
    ) -> tuple[Optional["Task"], bool]:
        """
        Apply user policy to task input.
        
        This method now uses PolicyManager to handle multiple policies.
        When feedback is enabled, returns a helpful message to the user instead
        of hard blocking, explaining what was wrong and how to correct it.
        
        Args:
            task: The task to apply policy to
            context: Optional AgentRunOutput for event emission
        
        Returns:
            tuple: (task, should_continue)
                - task: The task (possibly modified with feedback response)
                - should_continue: False if task should stop (blocked or feedback given)
        """
        if not self.user_policy_manager.has_policies() or not task.description:
            # Emit ALLOW event if no policies
            if context and context.is_streaming:
                from upsonic.utils.agent.events import ayield_policy_check_event
                async for event in ayield_policy_check_event(
                    run_id=context.run_id or "",
                    policy_type='user_policy',
                    action='ALLOW',
                    policies_checked=0
                ):
                    context.events.append(event)
            return task, True
        
        from upsonic.safety_engine.models import PolicyInput
        
        policy_input = PolicyInput(input_texts=[task.description])
        result = await self.user_policy_manager.execute_policies_async(
            policy_input,
            check_type="User Input Check"
        )
        
        # Get policies checked count
        policies_checked = len(self.user_policy_manager.policies)
        
        # Map action_taken to event action
        action_mapping = {
            "ALLOW": "ALLOW",
            "BLOCK": "BLOCK",
            "REPLACE": "REPLACE",
            "ANONYMIZE": "ANONYMIZE",
            "DISALLOWED_EXCEPTION": "RAISE ERROR"
        }
        event_action = action_mapping.get(result.action_taken, "ALLOW")
        
        # Emit PolicyCheckEvent
        if context and context.is_streaming:
            from upsonic.utils.agent.events import ayield_policy_check_event
            content_modified = result.action_taken in ["REPLACE", "ANONYMIZE"]
            blocked_reason = result.message if result.action_taken == "BLOCK" else None
            
            async for event in ayield_policy_check_event(
                run_id=context.run_id or "",
                policy_type='user_policy',
                action=event_action,
                policies_checked=policies_checked,
                content_modified=content_modified,
                blocked_reason=blocked_reason
            ):
                context.events.append(event)
        
        # Emit PolicyFeedbackEvent if feedback was generated
        if context and context.is_streaming and result.feedback_message:
            from upsonic.utils.agent.events import ayield_policy_feedback_event
            async for event in ayield_policy_feedback_event(
                run_id=context.run_id or "",
                policy_type='user_policy',
                feedback_message=result.feedback_message,
                retry_count=self.user_policy_manager._current_retry_count,
                max_retries=self.user_policy_manager.feedback_loop_count,
                violated_policy=result.violated_policy_name
            ):
                context.events.append(event)
        
        if result.should_block():
            # Re-raise DisallowedOperation if it was caught by PolicyManager
            # (unless feedback was generated - then we want to return the feedback)
            if result.disallowed_exception and not result.feedback_message:
                raise result.disallowed_exception
            
            task.task_end()
            # Use feedback message if available (gives helpful guidance to user)
            task._response = result.get_final_message()
            
            # Print feedback info if debug mode and feedback was generated
            if self.debug and result.feedback_message:
                from upsonic.utils.printing import user_policy_feedback_returned, debug_log_level2
                user_policy_feedback_returned(
                    policy_name=result.violated_policy_name or "Unknown Policy",
                    feedback_message=result.feedback_message
                )
                # Level 2: Detailed policy feedback information
                if self.debug_level >= 2:
                    debug_log_level2(
                        "User policy feedback details",
                        "Agent",
                        debug=self.debug,
                        debug_level=self.debug_level,
                        policy_name=result.violated_policy_name or "Unknown Policy",
                        feedback_message=result.feedback_message,
                        action_taken=result.action_taken,
                        confidence=result.confidence if hasattr(result, 'confidence') else None,
                        original_input=task.description[:200] if task.description else None
                    )
            return task, False
        elif result.action_taken in ["REPLACE", "ANONYMIZE"]:
            task.description = result.final_output or task.description
            # Store transformation map for de-anonymization of LLM response
            if result.transformation_map:
                task._anonymization_map = result.transformation_map
                # Add anonymization notice to the prompt - prepend for visibility
                anonymization_notice = (
                    "[PRIVACY MODE ACTIVE: Personal data has been anonymized with random placeholders. "
                    "Answer the question directly using the placeholder values shown. "
                    "Do NOT comment on, question, or mention the format of any data.]\n\n"
                )
                task.description = anonymization_notice + task.description
                if self.debug:
                    from upsonic.utils.printing import debug_log, anonymization_debug_panel
                    debug_log(
                        f"Stored anonymization map with {len(result.transformation_map)} entries for de-anonymization",
                        "Agent"
                    )
                    # Show detailed anonymization info - what LLM will see vs original
                    anonymization_debug_panel(
                        original_values=[entry.get("original", "") for entry in result.transformation_map.values()],
                        anonymized_values=[entry.get("anonymous", "") for entry in result.transformation_map.values()],
                        llm_input=task.description
                    )
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
        
        if hasattr(self, '_agent_run_output') and self._agent_run_output and self._agent_run_output.input:
            run_input = self._agent_run_output.input
            if run_input.input is None:
                run_input.build_input(context_formatted=task.context_formatted)
            current_input = run_input.input
            if task.context_formatted:
                task.context_formatted = None
        else:
            raise RuntimeError("AgentRunInput not available. This should not happen.")

        if task.guardrail_retries is not None and task.guardrail_retries > 0:
            max_retries = task.guardrail_retries + 1
        else:
            max_retries = 1

        while not validation_passed and retry_counter < max_retries:
            messages, context_full_response = await self._build_model_request_with_input(task, memory_handler, current_input, temporary_message_history, state)
            
            if context_full_response is not None:
                return context_full_response
            
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
                # Parse structured output if response_format is a Pydantic model
                guardrail_input = final_text_output
                if task.response_format and task.response_format != str:
                    try:
                        import json
                        parsed = json.loads(final_text_output)
                        if hasattr(task.response_format, 'model_validate'):
                            guardrail_input = task.response_format.model_validate(parsed)
                    except:
                        # If parsing fails, use the text output
                        guardrail_input = final_text_output
                
                guardrail_result = task.guardrail(guardrail_input)
                
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
    

    async def _apply_agent_policy(
        self, 
        task: "Task", 
        context: Optional[AgentRunOutput] = None
    ) -> tuple["Task", Optional[str]]:
        """
        Apply agent policy to task output.
        
        This method uses PolicyManager to handle multiple policies.
        When feedback is enabled and a violation occurs, it returns the feedback
        message along with the task so the caller can decide to retry.
        
        Args:
            task: The task to apply policy to
            context: Optional AgentRunOutput for event emission
        
        Returns:
            tuple: (task, feedback_message_or_none)
                - task: The task (possibly modified with blocked response)
                - feedback_message: If not None, agent should retry with this feedback
        """
        if not self.agent_policy_manager.has_policies() or not task or not task.response:
            # Emit ALLOW event if no policies
            if context and context.is_streaming:
                from upsonic.utils.agent.events import ayield_policy_check_event
                async for event in ayield_policy_check_event(
                    run_id=context.run_id or "",
                    policy_type='agent_policy',
                    action='ALLOW',
                    policies_checked=0
                ):
                    context.events.append(event)
            return task, None
        
        from upsonic.safety_engine.models import PolicyInput
        
        # Convert response to text
        response_text = ""
        if isinstance(task.response, str):
            response_text = task.response
        elif hasattr(task.response, 'model_dump_json'):
            response_text = task.response.model_dump_json()
        else:
            response_text = str(task.response)
        
        if not response_text:
            return task, None
        
        agent_policy_input = PolicyInput(input_texts=[response_text])
        result = await self.agent_policy_manager.execute_policies_async(
            agent_policy_input,
            check_type="Agent Output Check"
        )
        
        # Get policies checked count
        policies_checked = len(self.agent_policy_manager.policies)
        original_response = task.response
        
        # Map action_taken to event action
        action_mapping = {
            "ALLOW": "ALLOW",
            "BLOCK": "BLOCK",
            "REPLACE": "REPLACE",
            "ANONYMIZE": "ANONYMIZE",
            "DISALLOWED_EXCEPTION": "RAISE ERROR"
        }
        event_action = action_mapping.get(result.action_taken, "ALLOW")
        
        # Emit PolicyCheckEvent
        if context and context.is_streaming:
            from upsonic.utils.agent.events import ayield_policy_check_event
            content_modified = result.action_taken in ["REPLACE", "ANONYMIZE"] or (
                result.final_output and str(result.final_output) != str(original_response)
            )
            blocked_reason = result.message if result.action_taken == "BLOCK" else None
            
            async for event in ayield_policy_check_event(
                run_id=context.run_id or "",
                policy_type='agent_policy',
                action=event_action,
                policies_checked=policies_checked,
                content_modified=content_modified,
                blocked_reason=blocked_reason
            ):
                context.events.append(event)
        
        # Check if retry with feedback should be attempted
        if result.should_retry_with_feedback() and self.agent_policy_manager.can_retry():
            # Emit PolicyFeedbackEvent
            if context and context.is_streaming:
                from upsonic.utils.agent.events import ayield_policy_feedback_event
                async for event in ayield_policy_feedback_event(
                    run_id=context.run_id or "",
                    policy_type='agent_policy',
                    feedback_message=result.feedback_message,
                    retry_count=self.agent_policy_manager._current_retry_count,
                    max_retries=self.agent_policy_manager.feedback_loop_count,
                    violated_policy=result.violated_policy_name
                ):
                    context.events.append(event)
            # Return feedback message for retry - don't modify task yet
            return task, result.feedback_message
        
        # Apply the result (no retry - either passed or exhausted retries)
        if result.should_block():
            # Re-raise DisallowedOperation if it was caught by PolicyManager
            if result.disallowed_exception and not result.feedback_message:
                raise result.disallowed_exception
            
            task._response = result.get_final_message()
        elif result.action_taken in ["REPLACE", "ANONYMIZE"]:
            task._response = result.final_output or "Response modified by agent policy."
        elif result.final_output:
            task._response = result.final_output
        
        return task, None

    
    
    @retryable()
    async def do_async(
        self, 
        task: Union[str, "Task"], 
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        return_output: bool = False,
        state: Optional["State"] = None,
        *,
        graph_execution_id: Optional[str] = None,
        _resume_output: Optional[AgentRunOutput] = None,
        _resume_step_index: Optional[int] = None,
        _print_method_default: bool = False,
    ) -> Any:
        """
        Execute a task asynchronously using the pipeline architecture.
        
        The execution is handled entirely by the pipeline - this method just
        creates the pipeline, creates the output context, executes, and returns the output.
        All logic is in the pipeline steps.
        
        Args:
            task: Task to execute
            model: Override model for this execution
            debug: Enable debug mode
            retry: Number of retries
            return_output: If True, return full AgentRunOutput. If False (default), return content only.
            state: Graph execution state
            graph_execution_id: Graph execution identifier
            _resume_output: Internal - output for HITL resumption
            _resume_step_index: Internal - step index to resume from
            _print_method_default: Internal - default print value based on method (do=False, print_do=True)
            
        Returns:
            Task content (str, BaseModel, etc.) if return_output=False
            Full AgentRunOutput if return_output=True
                
        Example:
            ```python
            # Get content directly (default)
            result = await agent.do_async(task)
            print(result)  # Prints the response content
            
            # Get full output object
            output = await agent.do_async(task, return_output=True)
            print(output.content)  # Access content
            print(output.messages)  # Access messages
            ```
        """
        from upsonic.agent.pipeline import PipelineManager
        
        # Resolve print flag based on hierarchy (ENV > param > method default)
        # Store locally - don't mutate self.print to ensure thread-safety
        resolved_print_flag = self._resolve_print_flag(_print_method_default)
        
        # Convert string to Task if needed
        task = self._convert_to_task(task)
        
        # Determine start step index (0 for new runs, specified index for resumption)
        start_step_index = _resume_step_index if _resume_step_index is not None else 0
        is_resuming = _resume_output is not None
        
        # Validate task state (completed/problematic checks)
        validation_error = self._validate_task_for_new_run(task, is_resuming)
        if validation_error is not None:
            if return_output:
                return validation_error
            return validation_error.output
        
        # Update policy managers debug flag if debug is enabled
        if debug or self.debug:
            self.user_policy_manager.debug = True
            self.agent_policy_manager.debug = True
        
        # For resumption, use existing run_id and output
        if is_resuming:
            run_id = _resume_output.run_id
            self.run_id = run_id
            self._agent_run_output = _resume_output
            self._agent_run_output.is_streaming = False
            # Set resolved print flag on output for thread-safe access
            self._agent_run_output.print_flag = resolved_print_flag
        else:
            run_id = str(uuid.uuid4())
            self.run_id = run_id
            register_run(run_id)
        
        try:
            if not is_resuming:
                # 1. Create AgentRunInput
                run_input = self._create_agent_run_input(task)

                # 2. Create AgentRunOutput using centralized factory method
                self._agent_run_output = self._create_agent_run_output(
                    run_id=run_id,
                    task=task,
                    run_input=run_input,
                    is_streaming=False
                )
                
                # Set resolved print flag on output for thread-safe access
                self._agent_run_output.print_flag = resolved_print_flag
                
                # Set run_id on task for cross-process continuation support
                if task is not None:
                    task.run_id = run_id
            
            
            # Apply model override if provided
            self._apply_model_override(model)
            
            # 3. Create pipeline with direct (non-streaming) steps
            pipeline = PipelineManager(
                steps=self._create_direct_pipeline_steps(),
                task=self._agent_run_output.task,
                agent=self,
                model=self.model,
                debug=debug or self.debug
            )
            
            # 4. Execute pipeline (with optional start_step_index for resumption)
            try:
                await pipeline.execute(self._agent_run_output, start_step_index=start_step_index)
                cleanup_run(run_id)
            except Exception as pipeline_error:

                sentry_sdk.flush()
                raise pipeline_error
            
            sentry_sdk.flush()
            
            # Return based on return_output parameter
            if return_output:
                return self._agent_run_output
            return self._agent_run_output.output
        finally:
            self.run_id = None
    
    def _extract_output(self, task: "Task", response: "ModelResponse") -> Any:
        """Extract the output from a model response."""
        from upsonic.messages import TextPart, ToolCallPart
        
        # Check for image outputs first
        images = response.images
        if images:
            # If there are multiple images, return a list; if single, return the image data
            if len(images) == 1:
                return images[0].data
            else:
                return [img.data for img in images]
        
        # Check for tool call output from structured output tool
        if task.response_format and task.response_format != str and task.response_format is not str:
            tool_call_parts = [part for part in response.parts if isinstance(part, ToolCallPart)]
            for tool_call in tool_call_parts:
                # Look for the output tool
                if tool_call.tool_name == DEFAULT_OUTPUT_TOOL_NAME:
                    try:
                        args = tool_call.args_as_dict()
                        if hasattr(task.response_format, 'model_validate'):
                            return task.response_format.model_validate(args)
                        return args
                    except Exception:
                        pass
        
        # Extract text parts for non-image responses
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
        task: Union[str, "Task"],
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        return_output: bool = False
    ) -> Any:
        """
        Execute a task synchronously.
        
        Args:
            task: Task to execute (can be a Task object or a string description)
            model: Override model for this execution
            debug: Enable debug mode
            retry: Number of retries
            return_output: If True, return full AgentRunOutput. If False (default), return content only.
            
        Returns:
            Task content (str, BaseModel, etc.) if return_output=False
            Full AgentRunOutput if return_output=True
        """
        # Auto-convert string to Task object if needed
        from upsonic.tasks.tasks import Task as TaskClass
        if isinstance(task, str):
            task = TaskClass(description=task)
        
        task.price_id_ = None
        _ = task.price_id
        task._tool_calls = []

        try:
            loop = asyncio.get_running_loop()
            # If we get here, we're already in an async context with a running loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.do_async(task, model, debug, retry, return_output, _print_method_default=False))
                return future.result()
        except RuntimeError:
            # No event loop is running, so we can safely use asyncio.run()
            return asyncio.run(self.do_async(task, model, debug, retry, return_output, _print_method_default=False))
    
    def print_do(
        self,
        task: Union[str, "Task"],
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        return_output: bool = False
    ) -> Any:
        """
        Execute a task synchronously and print the result.
        
        Returns:
            Task content (str, BaseModel, etc.) if return_output=False
            Full AgentRunOutput if return_output=True
        """
        # Auto-convert string to Task object if needed
        from upsonic.tasks.tasks import Task as TaskClass
        if isinstance(task, str):
            task = TaskClass(description=task)
        
        task.price_id_ = None
        _ = task.price_id
        task._tool_calls = []

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.do_async(task, model, debug, retry, return_output, _print_method_default=True))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.do_async(task, model, debug, retry, return_output, _print_method_default=True))
    
    async def print_do_async(
        self,
        task: Union[str, "Task"],
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        return_output: bool = False
    ) -> Any:
        """
        Execute a task asynchronously with print output enabled (unless overridden by ENV or Agent param).
        
        Print hierarchy (highest to lowest priority):
        1. UPSONIC_AGENT_PRINT env variable
        2. Agent constructor print parameter
        3. Method default (print_do_async=True)
        
        Returns:
            Task content (str, BaseModel, etc.) if return_output=False
            Full AgentRunOutput if return_output=True
        """
        return await self.do_async(task, model, debug, retry, return_output, _print_method_default=True)
    
    def stream(
        self,
        task: Union[str, "Task"],
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        events: bool = False,
        state: Optional["State"] = None,
        *,
        event: Optional[bool] = None,
    ) -> Iterator[Union[str, "AgentStreamEvent"]]:
        """
        Stream task execution synchronously - yields events/text as they arrive.
        
        For async streaming, use `astream()` instead.
        
        Args:
            task: Task to execute
            model: Override model for this execution
            debug: Enable debug mode
            retry: Number of retries
            events: If True, yield AgentEvent objects. If False (default), yield text chunks.
            state: Graph execution state
            event: Deprecated, use 'events' instead.
            
        Yields:
            AgentEvent if events=True, str if events=False
            
        Example:
            ```python
            # Stream text synchronously
            for text in agent.stream(task):
                print(text, end='', flush=True)
            
            # Stream events
            from upsonic.run.events import RunEvent
            for chunk in agent.stream(task, events=True):
                if chunk.event_kind == RunEvent.run_content:
                    print(chunk.content)
            ```
        """
        import queue
        import threading
        
        if event is not None:
            events = event
        
        result_queue: queue.Queue = queue.Queue()
        error_holder: List[Exception] = []
        
        async def stream_to_queue():
            try:
                async for item in self.astream(task, model, debug, retry, events, state):
                    result_queue.put(item)
            except Exception as e:
                error_holder.append(e)
            finally:
                result_queue.put(None)
        
        def run_async_stream():
            asyncio.run(stream_to_queue())
        
        try:
            asyncio.get_running_loop()
            thread = threading.Thread(target=run_async_stream, daemon=True)
            thread.start()
        except RuntimeError:
            thread = threading.Thread(target=run_async_stream, daemon=True)
            thread.start()
        
        while True:
            item = result_queue.get()
            if item is None:
                if error_holder:
                    raise error_holder[0]
                break
            yield item
    
    async def astream(
        self,
        task: Union[str, "Task"],
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        events: bool = False,
        state: Optional["State"] = None,
        *,
        event: Optional[bool] = None,
    ) -> AsyncIterator[Union[str, "AgentStreamEvent"]]:
        """
        Stream task execution asynchronously - yields events or text as they arrive.
        
        Note: HITL (Human-in-the-Loop) features are not supported in streaming mode.
        Use do_async() for HITL functionality.
        
        Args:
            task: Task to execute
            model: Override model for this execution
            debug: Enable debug mode
            retry: Number of retries
            events: If True, yield AgentEvent objects. If False (default), yield text chunks.
            state: Graph execution state
            event: Deprecated, use 'events' instead.
            
        Yields:
            AgentEvent if events=True, str if events=False
            
        Example:
            ```python
            # Stream text
            async for text in agent.astream(task):
                print(text, end='', flush=True)
            
            # Stream events
            from upsonic.run.events import RunEvent
            async for evt in agent.astream(task, events=True):
                if evt.event_kind == RunEvent.run_content:
                    print(evt.content, end='')
            ```
        """
        if event is not None:
            events = event
        
        from upsonic.agent.pipeline import PipelineManager
        from upsonic.utils.printing import warning_log
        
        # Convert string to Task if needed
        task = self._convert_to_task(task)
        
        # Validate task state (completed/problematic checks)
        # For streaming, we just return early instead of yielding error output
        validation_error = self._validate_task_for_new_run(task, is_resuming=False)
        if validation_error is not None:
            # For streaming with problematic status, add specific warning
            if task.is_problematic:
                warning_log(
                    "Streaming does not support HITL continuation. Use continue_run_async() instead.",
                    "Agent"
                )
            return
        
        run_id = str(uuid.uuid4())
        self.run_id = run_id
        register_run(run_id)
        
        try:
            # 1. Create AgentRunInput
            run_input = self._create_agent_run_input(task)
            
            # 2. Create AgentRunOutput using centralized factory method
            self._agent_run_output = self._create_agent_run_output(
                run_id=run_id,
                task=task,
                run_input=run_input,
                is_streaming=True
            )
            
            # Set resolved print flag on output for thread-safe access
            # Streaming defaults to False since user handles output themselves
            self._agent_run_output.print_flag = self._resolve_print_flag(False)
            
            # Set run_id on task for cross-process continuation support
            if task is not None:
                task.run_id = run_id
            
            # Apply model override if provided
            self._apply_model_override(model)
            
            # 3. Create pipeline with streaming steps
            pipeline = PipelineManager(
                steps=self._create_streaming_pipeline_steps(),
                task=self._agent_run_output.task,
                agent=self,
                model=self.model,
                debug=debug or self.debug
            )
            
            # 4. Stream events from pipeline and filter based on events parameter
            try:
                async for pipeline_event in pipeline.execute_stream(context=self._agent_run_output, start_step_index=0):
                    if events:
                        yield pipeline_event
                    else:
                        text_content = self._extract_text_from_stream_event(pipeline_event)
                        if text_content:
                            self._agent_run_output.accumulated_text += text_content
                            yield text_content
            except Exception as stream_error:
                raise stream_error
            
            
        finally:
            cleanup_run(run_id)
            self.run_id = None
    
    def _extract_text_from_stream_event(self, event: Any) -> Optional[str]:
        """Extract text content from a streaming event.
        
        Handles both Agent events (TextDeltaEvent) and raw LLM events
        (PartStartEvent, PartDeltaEvent).
        """
        from upsonic.messages import PartStartEvent, PartDeltaEvent, TextPart, TextPartDelta
        from upsonic.run.events.events import TextDeltaEvent
        
        # Handle Agent events (new event system)
        if isinstance(event, TextDeltaEvent):
            return event.content
        
        # Handle raw LLM events (legacy/internal)
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            return event.part.content
        elif isinstance(event, PartDeltaEvent):
            # Check if delta is a TextPartDelta specifically
            if isinstance(event.delta, TextPartDelta):
                return event.delta.content_delta
            # Fallback to hasattr check for compatibility
            elif hasattr(event.delta, 'content_delta'):
                return event.delta.content_delta
        return None
    
    
    # HITL checkpoint loading - delegates to Memory class
    
    async def _load_incomplete_run_data_from_storage(self, run_id: str) -> Optional["RunData"]:
        """
        Load a resumable RunData from storage. Delegates to Memory class.
        """
        if not self.memory:
            raise ValueError("No memory configured. Agent must have memory configured to load paused runs.")
        
        return await self.memory.load_resumable_run_async(run_id=run_id, session_type=self.session_type, agent_id=self.agent_id)
    
    async def _check_if_run_is_problematic(
        self,
        task: Optional["Task"],
        run_id: Optional[str]
    ) -> tuple[bool, Optional[RunStatus]]:
        """
        Check if a run is problematic (paused, cancelled, or error).
        
        Uses task.is_problematic if task provided, otherwise checks in-memory output,
        and only loads from storage if run_id is provided and no matching in-memory output.
        
        Args:
            task: Task instance (if provided, uses task.status directly - fast path)
            run_id: Run ID to check (if provided without task, may load from storage)
            
        Returns:
            Tuple of (is_problematic, run_status)
        """
        if task is not None:
            # Task provided - use task.status directly (fast path, no storage call)
            return task.is_problematic, task.status
        
        if run_id is not None:
            # Only run_id provided - check in-memory output first, then storage if needed
            _output = getattr(self, '_agent_run_output', None)
            if _output and _output.run_id == run_id:
                # In-memory output matches - use it
                return _output.is_problematic, _output.status
            else:
                # No matching in-memory output - must load from storage
                run_data = await self._load_incomplete_run_data_from_storage(run_id)
                if run_data and run_data.output:
                    return run_data.output.is_problematic, run_data.output.status
        
        return False, None
    
    async def _check_if_run_is_completed(
        self,
        task: Optional["Task"],
        run_id: Optional[str]
    ) -> tuple[bool, Optional[RunStatus]]:
        """
        Check if a run is already completed.
        
        Uses task.is_completed if task provided, otherwise checks in-memory output,
        and only loads from storage if run_id is provided and no matching in-memory output.
        
        Args:
            task: Task instance (if provided, uses task.status directly - fast path)
            run_id: Run ID to check (if provided without task, may load from storage)
            
        Returns:
            Tuple of (is_completed, run_status)
        """
        if task is not None:
            # Task provided - use task.is_completed directly (fast path, no storage call)
            return task.is_completed, task.status
        
        if run_id is not None:
            # Only run_id provided - check in-memory output first, then storage if needed
            _output = getattr(self, '_agent_run_output', None)
            if _output and _output.run_id == run_id:
                # In-memory output matches - use it
                return _output.is_complete, _output.status
            else:
                # No matching in-memory output - load ANY run from storage (not just resumable)
                if self.memory:
                    run_data = await self.memory.load_run_async(run_id=run_id, session_type=self.session_type, agent_id=self.agent_id)
                    if run_data and run_data.output:
                        return run_data.output.is_complete, run_data.output.status
        
        return False, None
    
    def _create_direct_pipeline_steps(self) -> List[Any]:
        """
        Create pipeline steps for direct call mode (do_async).
        
        Returns:
            List of all pipeline steps for direct execution
        """
        from upsonic.agent.pipeline import (
            InitializationStep, CacheCheckStep, UserPolicyStep,
            StorageConnectionStep, LLMManagerStep, ModelSelectionStep,
            ToolSetupStep, MessageBuildStep,
            ModelExecutionStep, ResponseProcessingStep,
            ReflectionStep, CallManagementStep, TaskManagementStep,
            MemorySaveStep,
            ReliabilityStep, AgentPolicyStep,
            CacheStorageStep, FinalizationStep
        )
        
        return [
            InitializationStep(),          # 0
            StorageConnectionStep(),       # 1
            CacheCheckStep(),              # 2
            UserPolicyStep(),              # 3
            LLMManagerStep(),              # 4
            ModelSelectionStep(),          # 5
            ToolSetupStep(),               # 6
            MessageBuildStep(),            # 7
            ModelExecutionStep(),          # 8 <-- External tool resumes here
            ResponseProcessingStep(),      # 9 <-- Tracks messages to AgentRunOutput
            ReflectionStep(),              # 10
            CallManagementStep(),          # 11 <-- Displays tool calls & usage
            TaskManagementStep(),          # 12
            ReliabilityStep(),             # 13
            AgentPolicyStep(),             # 14
            CacheStorageStep(),            # 15
            FinalizationStep(),            # 16
            MemorySaveStep(),              # 17 <-- LAST: Saves AgentSession to storage
        ]
    
    def _get_step_index_by_name(self, step_name: str, is_streaming: bool = False) -> int:
        """
        Get the index of a pipeline step by its name.
        
        This allows dynamic lookup of step indices instead of hardcoding,
        which is important for HITL resumption logic that needs to know
        which steps set _run_boundaries.
        
        Args:
            step_name: The name of the step (e.g., "message_build")
            is_streaming: Whether to use streaming pipeline (default: direct)
            
        Returns:
            The index of the step in the pipeline
            
        Raises:
            ValueError: If the step is not found
        """
        steps = self._create_streaming_pipeline_steps() if is_streaming else self._create_direct_pipeline_steps()
        
        for i, step in enumerate(steps):
            if step.name == step_name:
                return i
        
        raise ValueError(f"Step '{step_name}' not found in {'streaming' if is_streaming else 'direct'} pipeline")
    
    def _create_streaming_pipeline_steps(self) -> List[Any]:
        """
        Create pipeline steps for streaming mode (stream).
        
        Returns:
            List of all pipeline steps for streaming execution
        """
        from upsonic.agent.pipeline import (
            InitializationStep, CacheCheckStep, UserPolicyStep,
            StorageConnectionStep, LLMManagerStep, ModelSelectionStep,
            ToolSetupStep, MessageBuildStep,
            StreamModelExecutionStep,
            AgentPolicyStep, CacheStorageStep,
            StreamMemoryMessageTrackingStep, StreamFinalizationStep
        )
        
        return [
            InitializationStep(),              # 0
            StorageConnectionStep(),           # 1
            CacheCheckStep(),                  # 2
            UserPolicyStep(),                  # 3
            LLMManagerStep(),                  # 4
            ModelSelectionStep(),              # 5
            ToolSetupStep(),                   # 6
            MessageBuildStep(),                # 7
            StreamModelExecutionStep(),        # 8 <-- External tool resumes here (tracks messages internally)
            AgentPolicyStep(),                 # 9
            CacheStorageStep(),                # 10
            StreamFinalizationStep(),          # 11
            StreamMemoryMessageTrackingStep(), # 12 <-- LAST: Saves AgentSession to storage
        ]
    
    def _create_full_pipeline_steps(self, is_streaming: bool = False) -> List[Any]:
        """
        Create complete pipeline steps based on execution mode.
        
        Args:
            is_streaming: If True, return streaming pipeline steps.
                         If False, return direct call pipeline steps.
        
        Returns:
            List of all pipeline steps in order
        """
        if is_streaming:
            return self._create_streaming_pipeline_steps()
        return self._create_direct_pipeline_steps()
    
    async def _inject_external_tool_results(
        self, 
        output: AgentRunOutput, 
        requirements: list
    ) -> None:
        """
        Inject external tool results from ALL ToolExecutions into output chat_history.
        
        The chat_history is already preserved in output. We just need to add the 
        response with tool calls (if there) and then the tool return parts.
        
        Also adds ToolExecution objects to output.tools for proper tool_usage tracking
        so that call_end can display the tool calls table.
        
        Args:
            output: The agent run output (single source of truth)
            requirements: List of RunRequirements with ToolExecution containing results
        """
        from upsonic.messages import ModelRequest, ToolReturnPart
        from upsonic._utils import now_utc
        
        if not requirements:
            return
        
        # Add response with tool calls if present and not already in chat_history
        if output.response:
            # Check if response is already the last message
            if not output.chat_history or output.chat_history[-1] != output.response:
                output.chat_history.append(output.response)
        
        # Initialize output.tools if needed
        if output.tools is None:
            output.tools = []
        
        # Inject tool results for ALL requirements
        tool_return_parts = []
        for requirement in requirements:
            if requirement.tool_execution and requirement.tool_execution.result:
                te = requirement.tool_execution
                tool_return_parts.append(ToolReturnPart(
                    tool_name=te.tool_name,
                    content=te.result,
                    tool_call_id=te.tool_call_id,
                    timestamp=now_utc()
                ))
                output.tools.append(te)
        
        # Add all tool returns in a single ModelRequest
        if tool_return_parts:
            output.chat_history.append(ModelRequest(parts=tool_return_parts))
    
    # External execution support
    
    def continue_run(
        self,
        task: Optional["Task"] = None,
        run_id: Optional[str] = None,
        requirements: Optional[List["RunRequirement"]] = None,
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        return_output: bool = False,
        *,
        streaming: Optional[bool] = None,
        event: bool = False,
        external_tool_executor: Optional[Callable[["RunRequirement"], str]] = None
    ) -> Any:
        """
        Continue a paused agent run (synchronous wrapper).
        
        Automatically detects if the original run was streaming and continues
        in the same mode, or you can override with the streaming parameter.
        
        Supports all HITL continuation scenarios:
        1. External tool execution: Pass task object with external results filled
        2. Durable execution (error recovery): Pass run_id to load from storage
        3. Cancel run resumption: Pass run_id to load from storage
        
        Args:
            task: Task object (for external tool execution with results)
            run_id: Run ID to load from storage (for durable/cancel)
            model: Override model
            debug: Enable debug mode
            retry: Number of retries
            return_output: If True, return full AgentRunOutput. If False (default), return content only.
            streaming: If True, return list of events/text. If False, return result. 
                      If None (default), auto-detect from original run.
            event: If True (with streaming), return list of AgentEvent objects.
                   If False (with streaming), return list of text chunks.
                external tool resumption.
            external_tool_executor: Optional function that executes external tools.
                When provided, if the agent pauses again with NEW external tool requirements,
                the executor is called automatically for each requirement.
                Signature: (requirement: RunRequirement) -> str
                This allows handling of sequential tool calls without a while loop.
            
        Returns:
            - For direct mode: Task content if return_output=False, AgentRunOutput if return_output=True
            - For streaming mode: List of events (if event=True) or text chunks (if event=False)
            
        Example:
            # Force direct mode
            result = agent.continue_run(run_id=result.run_id, streaming=False, return_output=True)
            
            # With external tool executor (handles sequential tool calls automatically)
            def my_executor(req):
                return execute_my_tool(req.tool_execution.tool_args)
            result = agent.continue_run(run_id=result.run_id, external_tool_executor=my_executor)
        """
        if not task and not run_id:
            raise ValueError("Either 'task' or 'run_id' must be provided")
        
        # Check if we need to auto-detect streaming mode
        use_streaming = streaming
        if use_streaming is None:
            # Auto-detect from in-memory output
            _output = getattr(self, '_agent_run_output', None)
            if _output:
                use_streaming = _output.is_streaming
            else:
                use_streaming = False  # Default to direct mode
        
        if use_streaming:
            # Streaming mode: collect all items (events or text) into a list
            async def collect_stream():
                results = []
                async_gen = await self.continue_run_async(
                    task, run_id, requirements, model, debug, retry, return_output,
                    streaming=True,
                    event=event,
                    external_tool_executor=external_tool_executor
                )
                async for item in async_gen:
                    results.append(item)
                return results
            
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, collect_stream())
                    return future.result()
            except RuntimeError:
                return asyncio.run(collect_stream())
        else:
            # Direct mode
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, 
                            self.continue_run_async(
                                task, run_id, requirements, model, debug, retry, return_output,
                                streaming=False,
                                event=event,
                                external_tool_executor=external_tool_executor
                            )
                        )
                        return future.result()
                else:
                    return loop.run_until_complete(
                        self.continue_run_async(
                            task, run_id, requirements, model, debug, retry, return_output,
                            streaming=False,
                            event=event,
                            external_tool_executor=external_tool_executor
                        )
                    )
            except RuntimeError:
                return asyncio.run(
                    self.continue_run_async(
                        task, run_id, requirements, model, debug, retry, return_output,
                        streaming=False,
                        event=event,
                        external_tool_executor=external_tool_executor
                    )
                )
    
    async def _prepare_continuation_context(
        self,
        task: Optional["Task"],
        run_id: Optional[str],
        model: Optional[Union[str, "Model"]],
        debug: bool,
        requirements: Optional[List["RunRequirement"]] = None,
    ) -> tuple:
        """
        Prepare output context and determine resume point for continue_run_async.
        
        Status-based processing:
        - PAUSED status: External tool execution - inject tool results
        - ERROR/CANCELLED status: Resume from the problematic step
        
        All state needed for resumption is in AgentRunOutput (chat_history, step_results, etc.)
        
        Args:
            requirements: Optional list of RunRequirement with tool results set.
                         For new agents, pass the requirements from original output
                         (with results set) to copy results to the loaded state.
            
        Returns:
            Tuple of (output, task, resume_step_index)
        """
        if not task and not run_id:
            raise ValueError("Either 'task' or 'run_id' must be provided")
        
        output: Optional[AgentRunOutput] = None
        
        # Step 1: Determine if we should load from storage
        need_storage_load = False
        
        # Get in-memory output (may not exist for new agents)
        _in_memory_output = getattr(self, '_agent_run_output', None)
        
        if run_id:
            # run_id provided - check if we already have matching in-memory output
            if _in_memory_output:
                in_memory_run_id = _in_memory_output.run_id
                in_memory_agent_id = _in_memory_output.agent_id
                
                if in_memory_run_id != run_id:
                    need_storage_load = True
                elif in_memory_agent_id != self.agent_id:
                    need_storage_load = True
                else:
                    output = _in_memory_output
            else:
                need_storage_load = True
        else:
            # No run_id provided - use in-memory output if available
            if _in_memory_output:
                in_memory_agent_id = _in_memory_output.agent_id
                if in_memory_agent_id != self.agent_id:
                    # Different agent - try to load from storage using task.run_id
                    if task and hasattr(task, 'run_id') and task.run_id:
                        need_storage_load = True
                        run_id = task.run_id
                    else:
                        raise ValueError(
                            "In-memory output belongs to different agent. "
                            "Provide run_id to load from storage."
                        )
                else:
                    output = _in_memory_output
            else:
                # No in-memory checkpoint - try to load from storage using task.run_id
                if task and hasattr(task, 'run_id') and task.run_id:
                    need_storage_load = True
                    run_id = task.run_id
                else:
                    raise ValueError(
                        "No in-memory checkpoint found. Provide run_id to load from storage."
                    )
        
        # Step 2: Load from storage if needed
        if need_storage_load:
            run_data = await self._load_incomplete_run_data_from_storage(run_id)
            if not run_data:
                raise ValueError(f"No resumable run found with run_id: {run_id}")
            
            if not run_data.output:
                raise ValueError("Run has no output (checkpoint not found)")
            
            self._agent_run_output = run_data.output
            output = self._agent_run_output
            
            # Step 2b: Copy tool results from passed requirements to loaded output
            # This is needed for new agents - they load state from storage (without results)
            # and the user passes requirements (with results) to inject
            if requirements:
                # Build a map of tool_call_id -> result from passed requirements
                results_map = {}
                for req in requirements:
                    if req.tool_execution and req.tool_execution.result:
                        results_map[req.tool_execution.tool_call_id] = req.tool_execution.result
                
                # Copy results to loaded output's requirements using set_external_execution_result
                for req in output.requirements or []:
                    if req.tool_execution and req.tool_execution.tool_call_id in results_map:
                        req.set_external_execution_result(results_map[req.tool_execution.tool_call_id])
            
        # Step 3: Validate we have what we need
        if output is None:
            raise ValueError(
                "No checkpoint found. Either provide run_id to load from storage, "
                "or call continue_run_async while agent still has in-memory output."
            )
        
        # Get task from output if not provided
        if task is None:
            task = output.task
        self.current_task = task
        if task is None:
            raise ValueError("Cannot extract task from checkpoint")
        
        run_status = output.status
        
        # Centralized: All problematic runs use get_problematic_step() to find resume point
        if run_status not in (RunStatus.paused, RunStatus.error, RunStatus.cancelled):
            raise ValueError(
                f"Cannot continue run with status '{run_status}'. "
                "Only paused, error, or cancelled runs can be continued."
            )
        
        # Get the problematic step for resume point (works for paused, error, cancelled)
        problematic_step = output.get_problematic_step()
        if not problematic_step:
            raise ValueError(f"Run has {run_status.value} status but no problematic step found in output")
        
        resume_step_index = problematic_step.step_number
        
        # MessageBuildStep sets _run_boundaries when building messages.
        # Only call start_new_run() if we're resuming AFTER MessageBuildStep.
        # If we resume AT or BEFORE MessageBuildStep, it will rebuild chat_history and
        # set the correct boundary itself.
        message_build_step_index = self._get_step_index_by_name("message_build")
        if resume_step_index > message_build_step_index:
            # CRITICAL: Mark the start of this resumed run for message tracking BEFORE any modifications.
            # When resuming AFTER MessageBuildStep, it's skipped (we resume from a later step),
            # so we need to record the current chat_history length here.
            # This ensures finalize_run_messages() includes ALL messages from THIS resumed run,
            # including injected tool results.
            output.start_new_run()
        
        # For paused runs (external tool execution), inject tool results
        if run_status == RunStatus.paused:
            external_tool_reqs = output.get_external_tool_requirements_with_results()
            
            if not external_tool_reqs:
                raise ValueError(
                    "Run is paused but no external tool requirements with results found. "
                    "Set tool results using requirement.set_external_execution_result(result)."
                )
            
            # Inject tool results (chat_history already preserved in output)
            await self._inject_external_tool_results(output, external_tool_reqs)
        
        # Clear paused state and set up for continuation
        task.is_paused = False
        output.task = task
        
        if task.enable_cache:
            task.set_cache_manager(self._cache_manager)
        
        return output, task, resume_step_index
    
    async def continue_run_async(
        self,
        task: Optional["Task"] = None,
        run_id: Optional[str] = None,
        requirements: Optional[List["RunRequirement"]] = None,
        model: Optional[Union[str, "Model"]] = None,
        debug: bool = False,
        retry: int = 1,
        return_output: bool = False,
        state: Optional["State"] = None,
        *,
        streaming: bool = False,
        event: bool = False,
        external_tool_executor: Optional[Callable[["RunRequirement"], str]] = None,
        graph_execution_id: Optional[str] = None
    ) -> Any:
        """
        Continue a paused agent run using StepResult-based intelligent resumption.
        
        Note: HITL continuation is only supported in direct call mode (streaming=False).
        
        Supports HITL continuation for external tool execution:
        - PAUSED status: External tool execution - inject tool results and resume
        - ERROR/CANCELLED status: Resume from the problematic step
        
        If the run is NOT problematic (not paused, cancelled, or error), this method
        will log an informative message and call do_async to start a fresh run.
        
        Args:
            task: Task object with external results (for external tool continuation)
            run_id: Run ID to load from storage (for durable/cancel continuation)
            model: Override model
            debug: Enable debug mode
            retry: Number of retries
            return_output: If True, return full AgentRunOutput. If False, return content only.
            state: Graph execution state
            streaming: Must be False. Streaming mode not supported for HITL continuation.
            event: Ignored (streaming not supported)
            external_tool_executor: Optional function that executes external tools.
                When provided, if the agent pauses again with NEW external tool requirements,
                the executor is called automatically for each requirement.
                Signature: (requirement: RunRequirement) -> str
                This allows handling of sequential tool calls without a while loop.
            graph_execution_id: Graph execution identifier
            
        Returns:
            Task content or AgentRunOutput (if return_output=True)
            
        Raises:
            ValueError: If streaming=True is passed
        """
        from upsonic.utils.printing import info_log, warning_log
        
        # HITL continuation is only supported in direct call mode
        if streaming:
            raise ValueError(
                "Streaming mode is not supported for HITL continuation. "
                "Use streaming=False (default) for continue_run_async."
            )
        
        # Check if task is already completed - cannot continue a completed task
        is_completed, run_status = await self._check_if_run_is_completed(task, run_id)
        if is_completed:
            resolved_run_id = (task.run_id if task else run_id) or "unknown"
            warning_log(
                f"Task is already completed (run_id={resolved_run_id}). Cannot continue a completed task.",
                "Agent"
            )
            completed_output = AgentRunOutput(
                run_id=resolved_run_id,
                agent_id=self.agent_id,
                agent_name=self.name,
                session_id=self.session_id,
                user_id=self.user_id,
                status=RunStatus.completed,
                output=f"Task is already completed (run_id={resolved_run_id}). Cannot continue a completed task.",
            )
            if return_output:
                return completed_output
            return completed_output.output
        
        # Check if run is problematic (paused, cancelled, error) before preparing continuation
        is_problematic, run_status = await self._check_if_run_is_problematic(task, run_id)
        
        if not is_problematic:
            # Run is not problematic - log and call do_async for a fresh run
            status_str = run_status.value if run_status else "unknown"
            info_log(
                f"Run status is '{status_str}' (not paused, cancelled, or error). "
                f"Starting fresh run via do_async.",
                "Agent"
            )
            return await self.do_async(
                task=task,
                model=model,
                debug=debug,
                retry=retry,
                return_output=return_output,
                state=state,
                graph_execution_id=graph_execution_id,
            )
        
        # Prepare context and determine resume point for problematic runs
        output, task, resume_step_index = await self._prepare_continuation_context(
            task, run_id, model, debug, requirements
        )
        
        # Execute and handle any subsequent external tool calls automatically
        return await self._continue_run_direct_impl(
            task, model, debug, retry, return_output, output, resume_step_index, external_tool_executor
        )
    
    async def _continue_run_direct_impl(
        self,
        task: "Task",
        model: Optional[Union[str, "Model"]],
        debug: bool,
        retry: int,
        return_output: bool,
        output: AgentRunOutput,
        resume_step_index: int,
        external_tool_executor: Optional[Callable[["RunRequirement"], str]] = None,
    ) -> Any:
        """
        Internal direct call implementation for continue_run_async.
        
        Handles the loop for sequential external tool calls automatically when
        external_tool_executor is provided.
        
        For durable execution and cancel run, marks requirements as resolved when
        the whole run completes successfully. For cancelled runs, calls cleanup_run
        on successful completion.
        """
        max_rounds = 10  # Safety limit
        rounds = 0
        result = None
        
        while rounds < max_rounds:
            rounds += 1
            
            # Execute the agent
            result = await self.do_async(
                task,
                model=model,
                debug=debug,
                retry=retry,
                return_output=True,
                _resume_output=output,
                _resume_step_index=resume_step_index,
            )
            
            # Check if run completed successfully
            # Note: MemorySaveStep marks durable/cancel requirements as resolved and saves session
            if result.is_complete:
                if return_output:
                    return result
                return result.output if hasattr(result, 'output') else result
            
            # Check if there are external tool requirements that need handling
            external_tool_active = [r for r in result.active_requirements if r.needs_external_execution]
            
            if not external_tool_active:
                # No external tools - return result (might be error or paused for other reasons)
                if return_output:
                    return result
                return result.output if hasattr(result, 'output') else result
            
            # If no executor provided, return paused result with external tool requirements
            if not external_tool_executor:
                if return_output:
                    return result
                return result.output if hasattr(result, 'output') else result
            
            # Execute ALL new external tools using the provided executor
            for requirement in result.active_requirements:
                if requirement.is_external_tool_execution:
                    tool_result = external_tool_executor(requirement)
                    requirement.tool_execution.result = tool_result
            
            # Prepare output for next round - use get_problematic_step() for resume point
            output = getattr(self, '_agent_run_output', None)
            problematic_step = output.get_problematic_step()
            if problematic_step:
                resume_step_index = problematic_step.step_number
            
            # Clear task.is_paused so ResponseProcessingStep extracts output properly
            task.is_paused = False
            
            # Inject the new external tool results into output
            external_tool_reqs = output.get_external_tool_requirements_with_results()
            if external_tool_reqs:
                await self._inject_external_tool_results(output, external_tool_reqs)
            
            # Loop continues to execute the next round with the injected results
        
        # If we hit max_rounds, return what we have
        if return_output:
            return result
        return result.output if hasattr(result, 'output') else result
