import uuid
from ..canvas.canvas import Canvas
from ..tasks.tasks import Task
from ..models.model_registry import ModelNames
from ..utils.printing import print_price_id_summary, call_end


from .tool_usage import tool_usage
from .llm_usage import llm_usage
from .agent_tool_register import agent_tool_register
from .model import get_agent_model
from .agent_creation import agent_create
from .context_managers import CallManager, TaskManager, ReliabilityManager, MemoryManager, LLMManager




from ..utils.error_wrapper import upsonic_error_handler
import time
import asyncio
from typing import Any, List, Union
from pydantic_ai import Agent as PydanticAgent, BinaryContent
import os

from ..memory.memory import get_agent_memory, save_agent_memory



class Direct:
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
                 canvas: Canvas | None = None,
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












    @upsonic_error_handler(max_retries=3, show_error_details=True)
    async def do_async(self, task: Task, model: ModelNames | None = None, debug: bool = False, retry: int = 3):
        """
        Execute a direct LLM call with the given task and model asynchronously.
        
        Args:
            task: The task to execute or list of tasks
            model: The LLM model to use

            debug: Whether to enable debug mode
            retry: Number of retries for failed calls (default: 3)
            
        Returns:
            The response from the LLM
        """

        llm_manager = LLMManager(self.default_llm_model, model)
        
        async with llm_manager.manage_llm() as llm_handler:
            selected_model = llm_handler.get_model()
            
            agent = await agent_create(selected_model, task)
            memory_manager = MemoryManager(self, task)
            call_manager = CallManager(selected_model, task, debug)
            task_manager = TaskManager(task, self)
            reliability_manager = ReliabilityManager(task, self.reliability_layer, selected_model)

            async with reliability_manager.manage_reliability() as reliability_handler:
                async with memory_manager.manage_memory() as memory_handler:
                    async with call_manager.manage_call(memory_handler) as call_handler:
                        async with task_manager.manage_task() as task_handler:
                            async with agent.run_mcp_servers():
                                model_response = await agent.run(task.build_agent_input(), message_history=memory_handler.historical_messages)
                            
                            # Save the response to all contexts
                            model_response = memory_handler.process_response(model_response)
                            model_response = call_handler.process_response(model_response)
                            model_response = task_handler.process_response(model_response)
                            processed_task = await reliability_handler.process_task(task_handler.task)
        
        # Print the price ID summary if the task has a price ID
        if not processed_task.not_main_task:
            print_price_id_summary(processed_task.price_id, processed_task)
            
        return processed_task.response
