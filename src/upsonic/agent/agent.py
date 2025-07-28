import asyncio
import os
import uuid
from typing import Any, List, Union, Optional, Dict
from types import SimpleNamespace
import time
from contextlib import asynccontextmanager
import json

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import BinaryContent
from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio
from pydantic_ai import CallToolsNode
from pydantic_graph import End
from pydantic_ai.messages import TextPart, ToolCallPart


from upsonic.canvas.canvas import Canvas
from upsonic.models.model import get_agent_model
from upsonic.models.model_registry import ModelNames
from upsonic.tasks.tasks import Task
from upsonic.utils.error_wrapper import upsonic_error_handler
from upsonic.utils.printing import call_end, print_price_id_summary
from upsonic.agent.base import BaseAgent
from upsonic.tools.processor import ToolProcessor
from upsonic.utils.tool_usage import tool_usage
from upsonic.storage.base import Storage
from upsonic.storage.session.llm import LLMConversation, LLMTurn

from upsonic.agent.context_managers import (
    CallManager,
    ContextManager,
    LLMManager,
    MemoryManager,
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
                 conversation_id: Optional[str] = None,
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
        self.memory = memory

        self.reliability_layer = reliability_layer

        self.storage = storage
        self.conversation_id = conversation_id
        


    @property
    def agent_id(self):
        if self.agent_id_ is None:
            self.agent_id_ = str(uuid.uuid4())
        return self.agent_id_
    
    def get_agent_id(self):
        if self.name:
            return self.name
        return f"Agent_{self.agent_id[:8]}"


    async def _assemble_and_save_turn(self, turn_data: Dict[str, Any], processed_task: Optional[Task], exception: Optional[Exception] = None):
        """
        Assembles the final 'assistant' LLMTurn object and persists it.
        """
        if not self.storage or not self.conversation_id:
            return

        # --- Handle Failure Case ---
        metadata = {"status": "failed" if exception else "success"}
        if turn_data.get("graph_execution_id"):
            metadata["graph_execution_id"] = turn_data["graph_execution_id"]

        if exception:
            assistant_turn = LLMTurn(
                role="assistant",
                timestamp_start=turn_data.get('timestamp_start', time.time()),
                timestamp_end=turn_data.get('timestamp_end', time.time()),
                llm_config=turn_data.get('llm_config'),
                final_prompt=turn_data.get('final_prompt'),
                error=f"{type(exception).__name__}: {str(exception)}",
                metadata=metadata
            )
            self.storage.append_turn(self.conversation_id, assistant_turn)
            print(f"[Storage] Persisted FAILED 'assistant' turn for conversation: {self.conversation_id}")
            return

        if not processed_task:
            return

        try:
            assistant_turn = LLMTurn(
                role="assistant",
                timestamp_start=turn_data.get('timestamp_start', time.time()),
                timestamp_end=turn_data.get('timestamp_end', time.time()),
                content=turn_data.get('content'),
                final_prompt=turn_data.get('final_prompt'),
                llm_config=turn_data.get('llm_config'),
                llm_response=processed_task.response,
                usage_stats=turn_data.get('usage_stats'),
                tool_calls=turn_data.get('tool_calls'),
                cost=turn_data.get('cost'),
                metadata=metadata
            )
            self.storage.append_turn(self.conversation_id, assistant_turn)
            print(f"[Storage] Persisted SUCCEEDED 'assistant' turn for conversation: {self.conversation_id}")
        except Exception as e:
            print(f"[Storage] CRITICAL: Failed to assemble or save turn data. Error: {e}")

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

    def agent_tool_register(self,agent, tasks):

        # If tasks is not a list
        if not isinstance(tasks, list):
            tasks = [tasks]

        # Keep track of already registered tools to prevent duplicates
        if not hasattr(agent, '_registered_tools'):
            agent._registered_tools = set()

        for task in tasks:
            for tool in task.tools:
                # Create a unique identifier for the tool
                # Using id() to get unique object identifier
                tool_id = id(tool)
                
                # Only register if not already registered
                if tool_id not in agent._registered_tools:
                    agent.tool_plain(tool)
                    agent._registered_tools.add(tool_id)

        return agent


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
        An internal async context manager to ensure the storage connection is
        active during an operation.
        """
        if not self.storage:
            yield
            return

        was_connected_before = self.storage.is_connected()

        try:
            if not was_connected_before:

                self.storage.connect()
            
            yield

        finally:
            if not was_connected_before:
                self.storage.disconnect()



    @upsonic_error_handler(max_retries=3, show_error_details=True)
    async def do_async(self, task: Task, model: ModelNames | None = None, debug: bool = False, retry: int = 3, state: Any = None, *, graph_execution_id: Optional[str] = None):
        """
        Execute a direct LLM call and orchestrate the persistence of the interaction.
        """
        async with self._managed_storage_connection():
            turn_data = {}
            processed_task = None
            exception_caught = None

            if self.storage:
                turn_data["timestamp_start"] = time.time()
                if graph_execution_id:
                    turn_data["graph_execution_id"] = graph_execution_id

                if not self.conversation_id:
                    self.conversation_id = str(uuid.uuid4())
                    new_conversation = LLMConversation(
                        conversation_id=self.conversation_id,
                        user_id=self.get_agent_id(),
                        entity_id=self.agent_id,
                        created_at=int(turn_data["timestamp_start"]),
                        updated_at=int(turn_data["timestamp_start"]),
                    )
                    self.storage.start_conversation(new_conversation)
                    print(f"[Storage] Started new conversation: {self.conversation_id}")

            try:
                llm_manager = LLMManager(self.default_llm_model, model, turn_data=turn_data)
                
                async with llm_manager.manage_llm() as llm_handler:
                    selected_model = llm_handler.get_model()
                    
                    system_prompt_manager = SystemPromptManager(self, task, turn_data=turn_data)
                    context_manager = ContextManager(self, task, state, turn_data=turn_data)

                    memory_manager = MemoryManager(self, task)
                    
                    async with system_prompt_manager.manage_system_prompt() as sp_handler, \
                                context_manager.manage_context() as ctx_handler:

                        if self.storage and self.conversation_id:
                            user_turn_metadata = {}
                            if turn_data.get("graph_execution_id"):
                                user_turn_metadata["graph_execution_id"] = turn_data["graph_execution_id"]

                            user_turn = LLMTurn(
                                role="user",
                                timestamp_start=turn_data.get('timestamp_start', time.time()),
                                content=task.description,
                                final_prompt=turn_data.get('final_prompt', ''),
                                metadata=user_turn_metadata,
                            )
                            self.storage.append_turn(self.conversation_id, user_turn)
                            print(f"[Storage] Persisted 'user' turn for conversation: {self.conversation_id}")


                        call_manager = CallManager(selected_model, task, debug, turn_data=turn_data)
                        task_manager = TaskManager(task, self)
                        reliability_manager = ReliabilityManager(task, self.reliability_layer, selected_model)
                        
                        agent = await self.agent_create(selected_model, task, sp_handler.get_system_prompt())

                        async with reliability_manager.manage_reliability() as reliability_handler:
                            async with memory_manager.manage_memory() as memory_handler:
                                async with call_manager.manage_call(memory_handler) as call_handler:
                                    async with task_manager.manage_task() as task_handler:
                                        should_stop_early = False
                                        final_output = None
                                        show_result = False
                                        
                                        async with agent.run_mcp_servers():
                                            model_response = await agent.run(task.build_agent_input(), message_history=memory_handler.historical_messages)

                                                    
                                        model_response = memory_handler.process_response(model_response)
                                        model_response = call_handler.process_response(model_response)
                                        model_response = task_handler.process_response(model_response)
                                        processed_task = await reliability_handler.process_task(task_handler.task)

            except Exception as e:
                exception_caught = e
                raise
            finally:
                if self.storage:
                    turn_data["timestamp_end"] = time.time()
                    await self._assemble_and_save_turn(turn_data, processed_task, exception=exception_caught)

            if processed_task and not processed_task.not_main_task:
                print_price_id_summary(processed_task.price_id, processed_task)
                
            return processed_task.response if processed_task else None
