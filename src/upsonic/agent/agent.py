import asyncio
import os
import uuid
from typing import Any, List, Union, Optional
import time
from contextlib import asynccontextmanager

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import BinaryContent
from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio
from pydantic_ai.messages import ModelMessagesTypeAdapter


from upsonic.canvas.canvas import Canvas
from upsonic.models.model import get_agent_model
from upsonic.models.model_registry import ModelNames
from upsonic.tasks.tasks import Task
from upsonic.utils.error_wrapper import upsonic_error_handler
from upsonic.utils.printing import print_price_id_summary
from upsonic.agent.base import BaseAgent
from upsonic.tools.processor import ToolProcessor
from upsonic.storage.base import Storage
from upsonic.storage.session.sessions import AgentSession

from upsonic.agent.context_managers import (
    CallManager,
    ContextManager,
    LLMManager,
    ReliabilityManager,
    SystemPromptManager,
    TaskManager,
)




class Direct(BaseAgent):
    """Static methods for making direct LLM calls using the Upsonic."""

    def __init__(self, 
                 name: str | None = None, 
                 model: ModelNames | None = None, 
                 debug: bool = False, 
                 company_url: str | None = None, 
                 company_objective: str | None = None,
                 company_description: str | None = None,
                 system_prompt: str | None = None,
                 memory: str | None = None,
                 reflection: str | None = None,
                 compress_context: bool = False,
                 reliability_layer = None,
                 agent_id_: str | None = None,
                 storage: Optional[Storage] = None,
                 canvas: Canvas | None = None,
                 session_id: Optional[str] = None,
                 add_history_to_messages: bool = True,
                 num_history_runs: Optional[int] = None,
                 ):
        self.canvas = canvas

        
        self.debug = debug
        self.default_llm_model = model
        self.agent_id_ = agent_id_
        self.name = name
        self.company_url = company_url
        self.company_objective = company_objective
        self.company_description = company_description
        self.system_prompt = system_prompt

        self.reliability_layer = reliability_layer

        self.storage = storage
        self.session_id_ = session_id
        self.add_history_to_messages = add_history_to_messages
        self.num_history_runs = num_history_runs
        
        


    @property
    def agent_id(self):
        if self.agent_id_ is None:
            self.agent_id_ = str(uuid.uuid4())
        return self.agent_id_
    
    def get_agent_id(self):
        if self.name:
            return self.name
        return f"Agent_{self.agent_id[:8]}"
    

    @property
    def session_id(self):
        """
        Provides a unique session ID, generating one if not already set.
        This will be the primary identifier for storing agent state/history.
        """
        if self.session_id_ is None:
            self.session_id_ = str(uuid.uuid4())
        return self.session_id_



    @upsonic_error_handler(max_retries=3, show_error_details=True)
    async def print_do_async(self, task: Union[Task, List[Task]], model: ModelNames | None = None, debug: bool = False, retry: int = 3):
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
    def do(self, task: Union[Task, List[Task]], model: ModelNames | None = None, debug: bool = False, retry: int = 3):
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
    def print_do(self, task: Union[Task, List[Task]], model: ModelNames | None = None, debug: bool = False, retry: int = 3):
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
    async def agent_create(self, llm_model, single_task, system_prompt: str):
        """
        Creates and configures the underlying PydanticAgent, processing and wrapping
        all tools with the advanced behavioral logic from ToolProcessor.
        """
        agent_model = get_agent_model(llm_model)

        # --- NEW: Instantiate our ToolProcessor ---
        tool_processor = ToolProcessor()
        
        # --- NEW: Lists to hold the processed tools ---
        final_tools_for_pydantic_ai = []
        mcp_servers = []
        
        # --- NEW: Process tools using the new engine ---
        # The normalize_and_process method handles validation and configuration resolution.
        # It yields tuples of (original_function_or_method, resolved_config).
        processed_tools_generator = tool_processor.normalize_and_process(single_task.tools)

        for original_tool, config in processed_tools_generator:
            # For each validated and configured tool, generate the final behavioral wrapper.
            # This wrapper contains all the logic for caching, confirmation, etc.
            wrapped_tool = tool_processor.generate_behavioral_wrapper(original_tool, config)
            final_tools_for_pydantic_ai.append(wrapped_tool)

        # --- MODIFIED: The MCP server logic is now separate and clearer ---
        # Note: This assumes MCP server "tools" are class types, as in your original code.
        # We process them separately from the standard callable tools.
        tools_to_remove_from_task = []
        for tool in single_task.tools:
            if isinstance(tool, type):
                # Check if it's an MCP SSE server (has url property)
                if hasattr(tool, 'url'):
                    url = getattr(tool, 'url')
                    the_mcp_server = MCPServerSSE(url)
                    mcp_servers.append(the_mcp_server)
                    tools_to_remove_from_task.append(tool)
                
                # Check if it's a normal MCP server (has command property)
                elif hasattr(tool, 'command'):
                    env = getattr(tool, 'env', {}) if hasattr(tool, 'env') and isinstance(getattr(tool, 'env', None), dict) else {}
                    command = getattr(tool, 'command', None)
                    args = getattr(tool, 'args', [])

                    the_mcp_server = MCPServerStdio(command, args=args, env=env)
                    mcp_servers.append(the_mcp_server)
                    tools_to_remove_from_task.append(tool)

        # It's good practice to not modify the list while iterating.
        for tool in tools_to_remove_from_task:
            single_task.tools.remove(tool)


        # --- MODIFIED: Instantiate the PydanticAgent with the *wrapped* tools ---
        the_agent = PydanticAgent(
            agent_model,
            output_type=single_task.response_format,
            system_prompt=system_prompt,
            end_strategy="exhaustive",
            retries=5,
            mcp_servers=mcp_servers
        )

        if not hasattr(the_agent, '_registered_tools'):
            the_agent._registered_tools = set()

        for tool_func in final_tools_for_pydantic_ai:
            tool_id = id(tool_func) # Get a unique ID for the function object
            if tool_id not in the_agent._registered_tools:
                the_agent.tool_plain(tool_func)
                the_agent._registered_tools.add(tool_id)

        if not hasattr(the_agent, '_upsonic_wrapped_tools'):
            the_agent._upsonic_wrapped_tools = {}

        the_agent._upsonic_wrapped_tools = {
            tool_func.__name__: tool_func for tool_func in final_tools_for_pydantic_ai
        }

        return the_agent



    @asynccontextmanager
    async def _managed_storage_connection(self):
        """
        A highly robust async context manager that ensures the storage connection
        is active during an operation and cleans up after itself safely.

        This version includes checks to ensure the `connect` and `disconnect`
        methods exist before attempting to call them, making it compatible with
        any storage provider, regardless of its specific implementation.
        """
        if not self.storage:
            yield
            return

        can_connect = hasattr(self.storage, 'connect') and callable(self.storage.connect)
        can_disconnect = hasattr(self.storage, 'disconnect') and callable(self.storage.disconnect)

        was_connected_before = self.storage.is_connected()

        try:
            if can_connect and not was_connected_before:
                self.storage.connect()
            yield

        finally:
            if can_disconnect and not was_connected_before and self.storage.is_connected():
                self.storage.disconnect()



    @upsonic_error_handler(max_retries=3, show_error_details=True)
    async def do_async(self, task: Task, model: ModelNames | None = None, debug: bool = False, retry: int = 3, state: Any = None, *, graph_execution_id: Optional[str] = None):
        """
        Execute a direct LLM call with robust, context-managed storage connections
        and agent-level control over history management.
        """
        async with self._managed_storage_connection():
            full_history = []
            message_history = []
            current_session = None

            if self.storage:
                current_session = self.storage.read(session_id=self.session_id)

                if self.add_history_to_messages and current_session and current_session.memory:
                    
                    for history in current_session.memory:
                        print(f"[Storage] EVERY HISTORY IN MEMORY KEYWORDDDDDD: {history}")

                        full_history.append(history)
                    history_to_process = []
                    
                    if self.num_history_runs is not None and self.num_history_runs > 0:
                        history_to_process = full_history[-self.num_history_runs:]
                        
                        print(f"[Storage] Loading the last {self.num_history_runs} runs ({len(history_to_process)} messages) from session '{self.session_id}'.")
                    else:
                        history_to_process = full_history
                        print(f"[Storage] Loading full history ({len(history_to_process)} messages) from session '{self.session_id}'.")

                    try:
                        for history in history_to_process:
                            print("HISTORY IN PROCESSING: ", history)
                            message_history = ModelMessagesTypeAdapter.validate_python(history)
                        print("VALIDATED MESSAGE HISTORY: ", message_history)
                        print("LENGTH OF MESSAGE HISTORY: ", len(message_history))
                    except Exception as e:
                        print(f"Warning: Could not validate stored history. Starting fresh. Error: {e}")
                        message_history = []

            processed_task = None
            exception_caught = None
            model_response = None

            try:
                llm_manager = LLMManager(self.default_llm_model, model)
                async with llm_manager.manage_llm() as llm_handler:
                    selected_model = llm_handler.get_model()

                    system_prompt_manager = SystemPromptManager(self, task)
                    context_manager = ContextManager(self, task, state)
                    async with system_prompt_manager.manage_system_prompt() as sp_handler, \
                                context_manager.manage_context() as ctx_handler:

                        call_manager = CallManager(selected_model, task, debug=debug)
                        task_manager = TaskManager(task, self)
                        reliability_manager = ReliabilityManager(task, self.reliability_layer, selected_model)

                        agent = await self.agent_create(selected_model, task, sp_handler.get_system_prompt())

                        async with reliability_manager.manage_reliability() as reliability_handler:
                            async with call_manager.manage_call() as call_handler:
                                async with task_manager.manage_task() as task_handler:
                                    async with agent.run_mcp_servers():
                                        model_response = await agent.run(
                                            task.build_agent_input(),
                                            message_history=message_history
                                        )

                                    model_response = call_handler.process_response(model_response)
                                    model_response = task_handler.process_response(model_response)
                                    processed_task = await reliability_handler.process_task(task_handler.task)
            except Exception as e:
                exception_caught = e
                raise

            finally:
                if self.storage and (processed_task or exception_caught or model_response):
                    updated_session_data = {}
                    if current_session:
                        updated_session_data = current_session.model_dump()
                    else:
                        updated_session_data['memory'] = []
                        updated_session_data['extra_data'] = {}

                    updated_session_data.setdefault('memory', [])
                    updated_session_data.update({
                        "session_id": self.session_id,
                        "agent_id": self.agent_id,
                        "user_id": self.get_agent_id(),
                        "updated_at": time.time(),
                    })

                    if exception_caught:
                        error_info = { "error_type": type(exception_caught).__name__, "error_message": str(exception_caught) }
                        updated_session_data.setdefault("extra_data", {}).update({"last_error": error_info})

                    elif model_response and processed_task:
                        from pydantic_core import to_jsonable_python
                        # Always saves the FULL history to the database
                        all_messages_as_dicts = to_jsonable_python(model_response.all_messages())
                        print("ALL_MESAGES_AS_DICTS: ", [all_messages_as_dicts])
                        print("TYPE OF ALL_MESAGES_AS_DICTS: ", type(all_messages_as_dicts))
                        print("LENGTH OF ALL_MESAGES_AS_DICTS: ", len([all_messages_as_dicts]))
                        updated_session_data['memory'] = [all_messages_as_dicts]

                    final_session = AgentSession.model_validate(updated_session_data)
                    
                    self.storage.upsert(final_session)
                    print(f"[Storage] Session '{self.session_id}' has been successfully upserted.")

        if processed_task and not processed_task.not_main_task:
            print_price_id_summary(processed_task.price_id, processed_task)

        return processed_task.response if processed_task else None