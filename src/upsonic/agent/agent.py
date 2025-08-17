import asyncio
import os
import uuid
from typing import Any, List, Union, Optional, Literal
import time
from contextlib import asynccontextmanager
import copy

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.agent import AgentRunResult


from upsonic.canvas.canvas import Canvas
from upsonic.tasks.tasks import Task
from upsonic.utils.error_wrapper import upsonic_error_handler
from upsonic.utils.printing import print_price_id_summary
from upsonic.agent.base import BaseAgent
from upsonic.tools.processor import ToolProcessor
from upsonic.storage.base import Storage
from upsonic.utils.retry import retryable
from upsonic.utils.validators import validate_attachments_for_model
from upsonic.storage.memory.memory import Memory
from upsonic.models.base import BaseModelProvider
from upsonic.utils.package.exception import GuardrailValidationError

from upsonic.agent.context_managers import (
    CallManager,
    ContextManager,
    ReliabilityManager,
    MemoryManager,
    SystemPromptManager,
    TaskManager,
)

RetryMode = Literal["raise", "return_false"]

class Direct(BaseAgent):
    """Static methods for making direct LLM calls using the Upsonic."""

    def __init__(self, 
                 name: str | None = None, 
                 model: BaseModelProvider | None = None,
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
                 retry: int = 3,
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
                 ):

        self.canvas = canvas
        self.memory = memory


        if self.memory:
            print(f"Using existing Memory instance feed_tool_call_results: {self.memory.feed_tool_call_results}")
            self.memory.feed_tool_call_results = feed_tool_call_results
            print("Updated Memory feed_tool_call_results:", self.memory.feed_tool_call_results)

        
        self.debug = debug
        if model is not None and not isinstance(model, BaseModelProvider):
            raise TypeError("The `model` parameter must be an instance of a BaseModelProvider subclass (e.g., OpenAI, Anthropic).")

        self.model_provider = model
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


    @property
    def agent_id(self):
        if self.agent_id_ is None:
            self.agent_id_ = str(uuid.uuid4())
        return self.agent_id_
    
    def get_agent_id(self):
        if self.name:
            return self.name
        return f"Agent_{self.agent_id[:8]}"
    



    @upsonic_error_handler(max_retries=3, show_error_details=True)
    async def print_do_async(self, task: Union[Task, List[Task]], model: Optional[BaseModelProvider] = None, debug: bool = False, retry: int = 3):
        """
        Execute a direct LLM call and print the result asynchronously.
        
        Args:
            task: The task to execute or list of tasks
            model: The LLM model to use
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls (default: 3)
            
        Returns:
            The response from the LLM
        """
        result = await self.do_async(task, model, debug, retry)
        print(result)
        return result

    @upsonic_error_handler(max_retries=3, show_error_details=True)
    def do(self, task: Union[Task, List[Task]], model: Optional[BaseModelProvider] = None, debug: bool = False, retry: int = 3):
        """
        Execute a direct LLM call with the given task and model synchronously.
        
        Args:
            task: The task to execute or list of tasks
            model: The LLM model to use
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls (default: 3)
            
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
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self.do_async(task, model, debug, retry))
        
        if loop.is_running():
            # Event loop is already running, we need to run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.do_async(task, model, debug, retry))
                return future.result()
        else:
            # Event loop exists but not running, we can use it
            return loop.run_until_complete(self.do_async(task, model, debug, retry))

    @upsonic_error_handler(max_retries=3, show_error_details=True)
    def print_do(self, task: Union[Task, List[Task]], model: Optional[BaseModelProvider] = None, debug: bool = False, retry: int = 3):
        """
        Execute a direct LLM call and print the result synchronously.
        
        Args:
            task: The task to execute or list of tasks
            model: The LLM model to use
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls (default: 3)
            
        Returns:
            The response from the LLM
        """
        result = self.do(task, model, debug, retry)
        print(result)
        return result


    @upsonic_error_handler(max_retries=2, show_error_details=True)
    async def agent_create(self, provider: BaseModelProvider, single_task: Task, system_prompt: str):
        """
        Creates and configures the underlying PydanticAgent, processing and wrapping
        all tools with the advanced behavioral logic from ToolProcessor.
        """
        validate_attachments_for_model(provider, single_task)

        agent_model, agent_settings = await provider._provision()

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

    async def _execute_with_guardrail(self, agent: PydanticAgent, task: Task, memory_handler: MemoryManager) -> AgentRunResult:
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


    @upsonic_error_handler(max_retries=3, show_error_details=True)
    @retryable()
    async def do_async(self, task: Task, model: Optional[BaseModelProvider] = None, debug: bool = False, retry: int = 3, state: Any = None, *, graph_execution_id: Optional[str] = None):
        """
        Execute a direct LLM call with robust, context-managed storage connections
        and agent-level control over history management.
        """
        self.tool_call_count = 0
        async with self._managed_storage_connection():
            processed_task = None
            exception_caught = None
            model_response = None

            try:
                provider_for_this_run = model or self.model_provider
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
                                    async with agent.run_mcp_servers():
                                        model_response = await self._execute_with_guardrail(agent, task, memory_handler)

                                    model_response = call_handler.process_response(model_response)
                                    model_response = task_handler.process_response(model_response)
                                    model_response = memory_handler.process_response(model_response)
                                    processed_task = await reliability_handler.process_task(task_handler.task)
            except Exception as e:
                exception_caught = e
                raise

        if processed_task and not processed_task.not_main_task:
            print_price_id_summary(processed_task.price_id, processed_task)

        return processed_task.response if processed_task else None