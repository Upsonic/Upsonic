import uuid
from ..canvas.canvas import Canvas
from ..tasks.tasks import Task
from ..models.model_registry import ModelNames
from ..utils.printing import print_price_id_summary, call_end
from ..utils.direct_llm_call.tool_usage import tool_usage
from ..utils.direct_llm_call.llm_usage import llm_usage
from ..utils.direct_llm_call.task_end import task_end
from ..utils.direct_llm_call.task_start import task_start
from ..utils.direct_llm_call.task_response import task_response
from ..utils.direct_llm_call.agent_tool_register import agent_tool_register
from ..utils.direct_llm_call.model import get_agent_model
from ..utils.direct_llm_call.agent_creation import agent_create
from ..utils.error_wrapper import upsonic_error_handler
import time
import asyncio
from typing import Any, List, Union
from pydantic_ai import Agent as PydanticAgent, BinaryContent
import os
from ..utils.model_set import model_set
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
        model = model_set(model)
        self.canvas = canvas

        self.model = model
        self.debug = debug
        self.default_llm_model = model
        self.agent_id_ = agent_id_
        self.name = name
        self.company_url = company_url
        self.company_objective = company_objective
        self.company_description = company_description
        self.system_prompt = system_prompt
        self.memory = memory
        

    @property
    def agent_id(self):
        if self.agent_id_ is None:
            self.agent_id_ = str(uuid.uuid4())
        return self.agent_id_
    
    def get_agent_id(self):
        if self.name:
            return self.name
        return f"Agent_{self.agent_id[:8]}"

    def _build_agent_input(self, task: Task):
        """
        Build the input for the agent run function, including images if present.
        
        Args:
            task: The task containing description and potentially images
            
        Returns:
            Either a string (description only) or a list containing description and BinaryContent objects
        """
        if not task.images:
            return task.description
            
        # Build input list with description and images
        input_list = [task.description]
        
        for image_path in task.images:
            try:
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                
                # Determine media type based on file extension
                file_extension = image_path.lower().split('.')[-1]
                if file_extension in ['jpg', 'jpeg']:
                    media_type = 'image/jpeg'
                else:
                    media_type = f'image/{file_extension}'
                    
                input_list.append(BinaryContent(data=image_data, media_type=media_type))
                
            except Exception as e:
                # Log error but continue with other images
                if self.debug:
                    print(f"Warning: Could not load image {image_path}: {e}")
                continue
                
        return input_list

    @upsonic_error_handler(max_retries=3, show_error_details=True)
    async def do_async(self, task: Union[Task, List[Task]], model: ModelNames | None = None, debug: bool = False, retry: int = 3):
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
        start_time = time.time()
        
        @upsonic_error_handler(max_retries=retry, show_error_details=debug)
        async def _execute_single_task(single_task: Task, llm_model: ModelNames | None, task_start_time: float, task_debug: bool, task_retry: int):
            """
            Execute a single task with the LLM.
            
            Args:
                single_task: The task to execute
                llm_model: The LLM model to use
                task_start_time: Start time for timing
                task_debug: Whether to enable debug mode
                task_retry: Number of retries for failed calls
            """
            # LLM Selection
            if llm_model is None:
                llm_model = self.default_llm_model

            # Start Time For Task
            task_start(single_task, self)

            # Get the model from registry
            agent_model, error = get_agent_model(llm_model)
            if error:
                return error

            # Create agent
            agent = await agent_create(agent_model, single_task)
            agent_tool_register(None, agent, single_task)

            # Get historical messages count before making the call
            historical_messages = get_agent_memory(self) if self.memory else []
            historical_message_count = len(historical_messages)

            # Make request to the model using MCP servers context manager
            async with agent.run_mcp_servers():
                model_response = await agent.run(self._build_agent_input(single_task), message_history=historical_messages)

            if self.memory:
                save_agent_memory(self, model_response)

            # Setting Task Response
            task_response(model_response, single_task)

            # End Time For Task
            task_end(single_task)
            
            # Calculate usage and tool usage only for current interaction
            usage = llm_usage(model_response, historical_message_count)
            tool_usage_result = tool_usage(model_response, single_task, historical_message_count)
            
            # Call end logging
            call_end(model_response.output, llm_model, single_task.response_format, task_start_time, time.time(), usage, tool_usage_result, task_debug, single_task.price_id)
        
        # Handle single task or list of tasks
        if isinstance(task, list):
            for each_task in task:
                await _execute_single_task(each_task, model, start_time, debug, retry)
        else:
            await _execute_single_task(task, model, start_time, debug, retry)
            
        # Print the price ID summary if the task has a price ID
        if not isinstance(task, list) and not task.not_main_task:
            print_price_id_summary(task.price_id, task)
            
        return task.response if not isinstance(task, list) else [t.response for t in task]

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




