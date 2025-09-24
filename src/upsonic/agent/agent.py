import asyncio
import os
import uuid
from typing import Any, List, Union, Optional, Literal, TYPE_CHECKING, Dict
import time
from contextlib import asynccontextmanager
import copy

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.agent import AgentRunResult


from upsonic.canvas.canvas import Canvas

from upsonic.utils.printing import print_price_id_summary, cache_hit, cache_miss, cache_stored, cache_configuration, agent_started
from upsonic.cache import CacheManager
from upsonic.agent.base import BaseAgent
from upsonic.tools.processor import ToolProcessor, ExternalExecutionPause
from upsonic.storage.base import Storage
from upsonic.utils.retry import retryable
from upsonic.utils.validators import validate_attachments_for_model
from upsonic.storage.memory.memory import Memory
from upsonic.models.base import BaseModelProvider
from upsonic.models.factory import ModelFactory
from upsonic.utils.package.exception import GuardrailValidationError
from upsonic.safety_engine.base import Policy
from upsonic.safety_engine.models import PolicyInput
from upsonic.safety_engine.exceptions import DisallowedOperation
from upsonic.utils.printing import policy_triggered

from upsonic.agent.context_managers import (
    CallManager,
    ContextManager,
    ReliabilityManager,
    MemoryManager,
    SystemPromptManager,
    TaskManager,
)

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task

RetryMode = Literal["raise", "return_false"]

class Direct(BaseAgent):
    """Static methods for making direct LLM calls using the Upsonic."""

    def __init__(self, 
                 name: str | None = None, 
                 model: Union[str, BaseModelProvider] | None = "openai/gpt-4o",
                 memory: Optional[Memory] = None,
                 debug: bool = False, 
                 company_url: str | None = None, 
                 company_objective: str | None = None,
                 company_description: str | None = None,
                 system_prompt: str | None = None,
                 reflection: str | None = None,
                 compress_context: bool = False,
                 reliability_layer = None,
                 agent_id_: str | None = None,
                 canvas: Canvas | None = None,
                 retry: int = 1,
                 mode: RetryMode = "raise",
                 role: str | None = None,
                 goal: str | None = None,
                 instructions: str | None = None,
                 education: str | None = None,
                 work_experience: str | None = None,
                 feed_tool_call_results: bool = False,
                 show_tool_calls: bool = True,
                 tool_call_limit: int = 5,
                 enable_thinking_tool: bool = False,
                 enable_reasoning_tool: bool = False,
                 user_policy: Optional[Policy] = None,
                 agent_policy: Optional[Policy] = None,
                 ):

        self.canvas = canvas
        self.memory = memory


        if self.memory:
            self.memory.feed_tool_call_results = feed_tool_call_results

        
        self.debug = debug
        if model is not None:
            # Use ModelFactory to handle both string and provider instances
            self.model_provider = ModelFactory.create(model)
        else:
            self.model_provider = None

        # Setup LLM models for UpsonicLLMProvider agents if policies use them
        self._setup_policy_llm_models(user_policy, agent_policy) 
        self.agent_id_ = agent_id_
        self.name = name
        self.company_url = company_url
        self.company_objective = company_objective
        self.company_description = company_description
        self.system_prompt = system_prompt

        self.reliability_layer = reliability_layer


        self.role = role
        self.goal = goal
        self.instructions = instructions
        self.education = education
        self.work_experience = work_experience
        
        if retry < 1:
            raise ValueError("The 'retry' count must be at least 1.")
        if mode not in ("raise", "return_false"):
            raise ValueError(f"Invalid retry_mode '{mode}'. Must be 'raise' or 'return_false'.")

        self.retry = retry
        self.mode = mode
        
        self.show_tool_calls = show_tool_calls
        self.tool_call_limit = tool_call_limit

        self.tool_call_count = 0


        self.enable_thinking_tool = enable_thinking_tool
        self.enable_reasoning_tool = enable_reasoning_tool

        self.user_policy = user_policy
        self.agent_policy = agent_policy
        
        self._cache_manager = CacheManager(session_id=f"agent_{self.agent_id}")
    
    def _setup_policy_llm_models(self, user_policy, agent_policy):
        """Setup LLM models for agents in UpsonicLLMProvider objects used by policies"""
        from upsonic.safety_engine.llm.upsonic_llm import UpsonicLLMProvider
        
        policies = [user_policy, agent_policy]
        
        for policy in policies:
            if policy is None:
                continue
                
            # If the policy doesn't have a base_llm and we have a model_provider, create one
            if policy.base_llm is None and self.model_provider is not None:
                policy.base_llm = UpsonicLLMProvider(
                    agent_name="Policy Base Agent",
                    model=None  # Will be set below
                )
                policy.base_llm.agent.model_provider = self.model_provider
            
            # Check if policy.base_llm is an UpsonicLLMProvider
            elif isinstance(policy.base_llm, UpsonicLLMProvider):
                # Set the model for the UpsonicLLMProvider's agent if we have a model_provider
                if self.model_provider is not None:
                    policy.base_llm.agent.model_provider = self.model_provider
            
            # Also check other LLM providers in the policy
            if policy.language_identify_llm is None and self.model_provider is not None:
                policy.language_identify_llm = UpsonicLLMProvider(
                    agent_name="Policy Language Detection Agent",
                    model=None
                )
                policy.language_identify_llm.agent.model_provider = self.model_provider
            elif isinstance(policy.language_identify_llm, UpsonicLLMProvider):
                if self.model_provider is not None:
                    policy.language_identify_llm.agent.model_provider = self.model_provider
                    
            if policy.text_finder_llm is None and self.model_provider is not None:
                policy.text_finder_llm = UpsonicLLMProvider(
                    agent_name="Policy Text Finder Agent", 
                    model=None
                )
                policy.text_finder_llm.agent.model_provider = self.model_provider
            elif isinstance(policy.text_finder_llm, UpsonicLLMProvider):
                if self.model_provider is not None:
                    policy.text_finder_llm.agent.model_provider = self.model_provider



    @property
    def agent_id(self):
        if self.agent_id_ is None:
            self.agent_id_ = str(uuid.uuid4())
        return self.agent_id_
    
    def get_agent_id(self):
        if self.name:
            return self.name
        return f"Agent_{self.agent_id[:8]}"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for this agent's session."""
        return self._cache_manager.get_cache_stats()
    
    def clear_cache(self):
        """Clear the agent's session cache."""
        self._cache_manager.clear_cache()
    




    async def print_do_async(self, task: Union["Task", List["Task"]], model: Optional[Union[str, BaseModelProvider]] = None, debug: bool = False, retry: int = 1):
        """
        Execute a direct LLM call and print the result asynchronously.
        
        Args:
            task: The task to execute or list of tasks
            model: The LLM model to use
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls 
            
        Returns:
            The response from the LLM
        """
        result = await self.do_async(task, model, debug, retry)
        return result


    def do(self, task: Union["Task", List["Task"]], model: Optional[Union[str, BaseModelProvider]] = None, debug: bool = False, retry: int = 1):
        """
        Execute a direct LLM call with the given task and model synchronously.
        
        Args:
            task: The task to execute or list of tasks
            model: The LLM model to use
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls
            
        Returns:
            The response from the LLM
        """
        # Refresh price_id and tool call history at the start for each task
        if isinstance(task, list):
            for each_task in task:
                each_task.price_id_ = None  # Reset to generate new price_id
                _ = each_task.price_id  # Trigger price_id generation
                each_task._tool_calls = []  # Clear tool call history
        else:
            task.price_id_ = None  # Reset to generate new price_id
            _ = task.price_id  # Trigger price_id generation
            task._tool_calls = []  # Clear tool call history
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Event loop is already running, we need to run in a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.do_async(task, model, debug, retry))
                    return future.result()
            else:
                # Event loop exists but not running, we can use it
                return loop.run_until_complete(self.do_async(task, model, debug, retry))
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self.do_async(task, model, debug, retry))


    def print_do(self, task: Union["Task", List["Task"]], model: Optional[Union[str, BaseModelProvider]] = None, debug: bool = False, retry: int = 1):
        """
        Execute a direct LLM call and print the result synchronously.
        
        Args:
            task: The task to execute or list of tasks
            model: The LLM model to use
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls
            
        Returns:
            The response from the LLM
        """
        result = self.do(task, model, debug, retry)
        print(result)
        return result



    async def agent_create(self, provider: BaseModelProvider, single_task: "Task", system_prompt: str):
        """
        Creates and configures the underlying PydanticAgent, processing and wrapping
        all tools with the advanced behavioral logic from ToolProcessor.
        """
        validate_attachments_for_model(provider, single_task)

        agent_model, agent_settings = await provider._provision(single_task.tools)

        is_thinking_enabled = self.enable_thinking_tool
        if single_task.enable_thinking_tool is not None:
            is_thinking_enabled = single_task.enable_thinking_tool

        is_reasoning_enabled = self.enable_reasoning_tool
        if single_task.enable_reasoning_tool is not None:
            is_reasoning_enabled = single_task.enable_reasoning_tool

        # Sanity Check: Reasoning requires Thinking.
        if is_reasoning_enabled and not is_thinking_enabled:
            raise ValueError("Configuration error: 'enable_reasoning_tool' cannot be True if 'enable_thinking_tool' is False.")
        
        agent_for_this_run = copy.copy(self)
        agent_for_this_run.enable_thinking_tool = is_thinking_enabled
        agent_for_this_run.enable_reasoning_tool = is_reasoning_enabled

        tool_processor = ToolProcessor(agent=agent_for_this_run)
        
        final_tools_for_pydantic_ai = []
        mcp_servers = []
        
        processed_tools_generator = tool_processor.normalize_and_process(single_task.tools)

        for original_tool, config in processed_tools_generator:
            if callable(original_tool):
                if hasattr(original_tool, '_is_orchestrator'):
                    wrapped_tool = tool_processor.generate_orchestrator_wrapper(self, single_task)
                else:
                    wrapped_tool = tool_processor.generate_behavioral_wrapper(original_tool, config)
                
                final_tools_for_pydantic_ai.append(wrapped_tool)
            elif original_tool is None and config is not None:
                mcp_server = config
                mcp_servers.append(mcp_server)

        the_agent = PydanticAgent(
            agent_model,
            output_type=single_task.response_format,
            system_prompt=system_prompt,
            end_strategy="exhaustive",
            retries=5,
            mcp_servers=mcp_servers,
            model_settings=agent_settings,
        )

        if not hasattr(the_agent, '_registered_tools'):
            the_agent._registered_tools = set()
        
        for tool_func in final_tools_for_pydantic_ai:
            tool_id = id(tool_func)
            if tool_id not in the_agent._registered_tools:
                the_agent.tool_plain(tool_func)
                the_agent._registered_tools.add(tool_id)
        
        if not hasattr(self, '_upsonic_wrapped_tools'):
            self._upsonic_wrapped_tools = {}
        if not hasattr(agent_for_this_run, '_upsonic_wrapped_tools'):
            agent_for_this_run._upsonic_wrapped_tools = {}
        
        # Store a reference to the final wrapped tools for the orchestrator to access.
        self._upsonic_wrapped_tools = {
            tool_func.__name__: tool_func for tool_func in final_tools_for_pydantic_ai
        }
        agent_for_this_run._upsonic_wrapped_tools = self._upsonic_wrapped_tools

        return the_agent



    @asynccontextmanager
    async def _managed_storage_connection(self):
        """
        A robust async context manager that correctly manages the lifecycle of
        the fully asynchronous storage connection using the _async API.
        """
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

    async def _execute_with_guardrail(self, agent: PydanticAgent, task: "Task", memory_handler: MemoryManager) -> AgentRunResult:
        """
        Executes the agent's run method with a validation and retry loop based on a task guardrail.
        This method encapsulates the retry logic, hiding it from the main `do_async` pipeline.
        It returns a single, "clean" ModelResponse that represents the final, successful interaction.
        """
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
            current_model_response = await agent.run(
                current_input,
                message_history=temporary_message_history
            )
            
            if task.guardrail is None:
                validation_passed = True
                final_model_response = current_model_response
                break

            final_text_output = ""
            new_messages = current_model_response.new_messages()
            if new_messages and isinstance(new_messages[-1], ModelResponse):
                final_response_message = new_messages[-1]
                text_parts = [part.content for part in final_response_message.parts if isinstance(part, TextPart)]
                final_text_output = "".join(text_parts)

            if not final_text_output:
                validation_passed = True
                final_model_response = current_model_response
                break

            is_valid, result = task.guardrail(final_text_output)

            if is_valid:
                validation_passed = True
                
                if new_messages and isinstance(new_messages[-1], ModelResponse):
                    final_response_message = new_messages[-1]
                    found_and_updated = False
                    for part in final_response_message.parts:
                        if isinstance(part, TextPart):
                            if not found_and_updated:
                                part.content = result
                                found_and_updated = True
                            else:
                                part.content = ""
                
                final_model_response = current_model_response
                break
            else:
                retry_counter += 1
                last_error_message = str(result)
                temporary_message_history.extend(current_model_response.new_messages())
                correction_prompt = f"Your previous response failed a validation check. Please review the reason and provide a corrected response. Failure Reason: {last_error_message}"
                current_input = correction_prompt

        if not validation_passed:
            raise GuardrailValidationError(f"Task failed after {max_retries-1} retry(s). Last error: {last_error_message}")
        return final_model_response

    async def _handle_task_cache(self, task: "Task") -> Optional[Any]:
        """
        Handle cache operations for the task.
        
        Args:
            task: The task to check cache for
            
        Returns:
            Cached response if found, None otherwise
        """
        if not task.enable_cache:
            return None
        
        # Show cache configuration
        if self.debug:
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
        
        # Get cached response using original input
        input_text = task._original_input or task.description
        cached_response = await task.get_cached_response(input_text, self.model_provider)
        
        if cached_response is not None:
            # Cache hit
            similarity = None
            if hasattr(task, '_last_cache_entry') and 'similarity' in task._last_cache_entry:
                similarity = task._last_cache_entry['similarity']
            
            cache_hit(
                cache_method=task.cache_method,
                similarity=similarity,
                input_preview=(task._original_input or task.description)[:100] if (task._original_input or task.description) else None
            )
            
            # Set the response and mark task as completed
            task._response = cached_response
            task.task_end()
            return cached_response
        else:
            # Cache miss
            cache_miss(
                cache_method=task.cache_method,
                input_preview=(task._original_input or task.description)[:100] if (task._original_input or task.description) else None
            )
            return None



    async def _apply_user_policy_async(self, task: "Task") -> (Optional["Task"], bool):
        """Applies the user policy to the task description, returning the modified task and a flag to continue."""
        if not (self.user_policy and task.description):
            return task, True

        policy_input = PolicyInput(input_texts=[task.description])
        try:
            rule_output, _action_output, policy_output = await self.user_policy.execute_async(policy_input)
            action_taken = policy_output.action_output.get("action_taken", "UNKNOWN")

            if self.debug and rule_output.confidence > 0.0:
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
            from upsonic.safety_engine.models import RuleOutput
            mock_rule_output = RuleOutput(
                confidence=1.0, 
                content_type="DISALLOWED_OPERATION", 
                details=str(e)
            )
            if self.debug:
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

    async def _apply_agent_policy_async(self, processed_task: "Task") -> "Task":
        """Applies the agent policy to the final response, returning the modified task."""
        if not (self.agent_policy and processed_task and processed_task.response):
            return processed_task

        response_text = ""
        if isinstance(processed_task.response, str):
            response_text = processed_task.response
        elif hasattr(processed_task.response, 'model_dump_json'):
            response_text = processed_task.response.model_dump_json()
        else:
            response_text = str(processed_task.response)

        if response_text:
            agent_policy_input = PolicyInput(input_texts=[response_text])
            try:
                rule_output, _action_output, policy_output = await self.agent_policy.execute_async(agent_policy_input)
                action_taken = policy_output.action_output.get("action_taken", "UNKNOWN")

                if self.debug and rule_output.confidence > 0.0:
                    policy_triggered(
                        policy_name=self.agent_policy.name,
                        check_type="Agent Output Check",
                        action_taken=action_taken,
                        rule_output=rule_output
                    )

                final_output = policy_output.output_texts[0] if policy_output.output_texts else "Response modified by agent policy."
                processed_task._response = final_output
            
            except DisallowedOperation as e:
                from upsonic.safety_engine.models import RuleOutput
                mock_rule_output = RuleOutput(
                    confidence=1.0, 
                    content_type="DISALLOWED_OPERATION", 
                    details=str(e)
                )
                if self.debug:
                    policy_triggered(
                        policy_name=self.agent_policy.name,
                        check_type="Agent Output Check",
                        action_taken="DISALLOWED_EXCEPTION",
                        rule_output=mock_rule_output
                    )
                processed_task._response = f"Agent response disallowed by policy: {e}"
        
        return processed_task



    @retryable()
    async def do_async(self, task: "Task", model: Optional[Union[str, BaseModelProvider]] = None, debug: bool = False, retry: int = 1, state: Any = None, *, graph_execution_id: Optional[str] = None):
        """
        Execute a direct LLM call with robust, context-managed storage connections
        and agent-level control over history management.
        """
        # Print agent started message
        agent_started(self.get_agent_id())
        
        self.tool_call_count = 0
        async with self._managed_storage_connection():
            processed_task = None
            exception_caught = None
            model_response = None

            try:
                if not task.is_paused:
                    # Set the cache manager for the task
                    if task.enable_cache:
                        task.set_cache_manager(self._cache_manager)
                    
                    cached_response = await self._handle_task_cache(task)
                    if cached_response is not None:
                        processed_task = task
                        return cached_response
                    
                    task, should_continue = await self._apply_user_policy_async(task)
                    if not should_continue:
                        processed_task = task
                        return processed_task.response 

                if model is not None:
                    provider_for_this_run = ModelFactory.create(model)
                else:
                    provider_for_this_run = self.model_provider
                    
                if not provider_for_this_run:
                    raise ValueError("No model provider configured. Please pass a model object to the Direct agent constructor or to the do/do_async method.")
                
                memory_manager = MemoryManager(self.memory)
                async with memory_manager.manage_memory() as memory_handler:

                    system_prompt_manager = SystemPromptManager(self, task)
                    context_manager = ContextManager(self, task, state)
                    
                    async with system_prompt_manager.manage_system_prompt(memory_handler) as sp_handler, \
                                context_manager.manage_context(memory_handler) as ctx_handler:

                        call_manager = CallManager(provider_for_this_run, task, debug=debug, show_tool_calls=self.show_tool_calls)
                        task_manager = TaskManager(task, self)
                        reliability_manager = ReliabilityManager(task, self.reliability_layer, provider_for_this_run)

                        agent = await self.agent_create(provider_for_this_run, task, sp_handler.get_system_prompt())

                        async with reliability_manager.manage_reliability() as reliability_handler:
                            async with call_manager.manage_call() as call_handler:
                                async with task_manager.manage_task() as task_handler:
                                    try:
                                        async with agent.run_mcp_servers():
                                            model_response = await self._execute_with_guardrail(agent, task, memory_handler)
                                    except ExternalExecutionPause as e:
                                        # Agent paused for external execution
                                        task_handler.task.is_paused = True
                                        task_handler.task._tools_awaiting_external_execution.append(e.tool_call)
                                        processed_task = task_handler.task

                                        return processed_task.response
                                        
                                    model_response = call_handler.process_response(model_response)
                                    model_response = task_handler.process_response(model_response)
                                    model_response = memory_handler.process_response(model_response)
                                    processed_task = await reliability_handler.process_task(task_handler.task)

                processed_task = await self._apply_agent_policy_async(processed_task)
                
                if processed_task and processed_task.enable_cache and processed_task.response:
                    input_text = processed_task._original_input or processed_task.description
                    await processed_task.store_cache_entry(input_text, processed_task.response)
                    if self.debug:
                        cache_stored(
                            cache_method=processed_task.cache_method,
                            input_preview=(processed_task._original_input or processed_task.description)[:100] if (processed_task._original_input or processed_task.description) else None,
                            duration_minutes=processed_task.cache_duration_minutes
                        )

            except StopIteration as e:
                if self.debug: print(f"Execution stopped gracefully by policy: {e}")
            except DisallowedOperation as e:

                if self.debug: print(f"Caught DisallowedOperation at do_async level. Finalizing response.")

            except Exception as e:
                exception_caught = e
                raise

        if processed_task and not processed_task.not_main_task:
            print_price_id_summary(processed_task.price_id, processed_task)

        return processed_task.response if processed_task else None



    def continue_run(self, task: "Task", model: Optional[Union[str, BaseModelProvider]] = None, debug: bool = False, retry: int = 1):
        """
        Continues the execution of a paused task after external tool results have been provided.
        
        Args:
            task: The Task object, which was previously paused and now has the results for the
                  tools that were awaiting external execution.
            model: The LLM model to use for continuation.
            debug: Whether to enable debug mode.
            retry: Number of retries for failed calls.
            
        Returns:
            The final response from the LLM after continuation.
        """
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


    async def continue_async(self, task: "Task", model: Optional[Union[str, BaseModelProvider]] = None, debug: bool = False, retry: int = 1, state: Any = None, *, graph_execution_id: Optional[str] = None):
        """
        Asynchronously continues the execution of a paused task.
        """
        if not task.is_paused or not task.tools_awaiting_external_execution:
            raise ValueError("The 'continue_async' method can only be called on a task that is currently paused for external execution.")

        tool_results_prompt = "\nThe following external tools were executed. Use their results to continue the task:\n"
        for tool_call in task.tools_awaiting_external_execution:
            tool_results_prompt += f"\n- Tool '{tool_call.tool_name}' was executed with arguments {tool_call.tool_args}.\n"
            tool_results_prompt += f"  Result: {tool_call.result}\n"
        
        task.is_paused = False
        task.description += tool_results_prompt
        task._tools_awaiting_external_execution = []
        
        if task.enable_cache:
            task.set_cache_manager(self._cache_manager)

        return await self.do_async(task, model, debug, retry, state, graph_execution_id=graph_execution_id)