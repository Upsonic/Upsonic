

from ..tasks.tasks import Task
from typing import Any, TypeVar

T = TypeVar('T')
from ...model_registry import ModelNames
from ..printing import print_price_id_summary



from .tool_usage import tool_usage
from .llm_usage import llm_usage
from .task_end import task_end
from .task_start import task_start
from .task_response import task_response
from .agent_tool_register import agent_tool_register
from .model import get_agent_model


import time
import cloudpickle
import asyncio

from ..knowledge_base.knowledge_base import KnowledgeBase
cloudpickle.DEFAULT_PROTOCOL = 2


from typing import Any, List, Union
from pydantic import BaseModel

from ..tasks.tasks import Task

from ..printing import call_end





from pydantic_ai.direct import model_request
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest


class Direct:
    """Static methods for making direct LLM calls using the Upsonic client."""

    def __init__(self, model: ModelNames | None = None, client: Any = None, debug: bool = False):
        self.model = model
        self.client = client
        self.debug = debug
        self.default_llm_model = model

    async def call_async(
        self,
        task: Union[Task, List[Task]],
        llm_model: str = None,
        retry: int = 3
    ) -> Any:
        """
        Asynchronous version of the call method.
        """
        start_time = time.time()

        if isinstance(task, list):
            for each in task:
                the_result = await self.call_async_(each, llm_model, retry)
                call_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_usage"], self.debug, each.price_id)
        else:
            the_result = await self.call_async_(task, llm_model, retry)
            print(the_result)
            call_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_usage"], self.debug, task.price_id)

        end_time = time.time()

        return task.response

    async def call_async_(
        self,
        task: Task,
        llm_model: str = None,
        retry: int = 3
    ) -> Any:
        """
        Asynchronous version of the call_ method.
        """

        # LLM Selection
        if llm_model is None:
            llm_model = self.default_llm_model

        # Start Time For Task
        task_start(task)    



        # Get the model from registry
        model, error = get_agent_model(llm_model)
        if error:
            return error
        



        agent = Agent(model, output_type=task.response_format, system_prompt="", end_strategy="exhaustive", retries=5, )

        agent_tool_register(None, agent, task)
        

        # Make a synchronous request to the model
        model_response = await agent.run(task.description)







        # Setting Task Response
        task_response(model_response, task)

        # End Time For Task
        task_end(task)
        return {
            "result": model_response.output,
            "llm_model": llm_model,
            "response_format": task.response_format,
            "usage": llm_usage(model_response),
            "tool_usage": tool_usage(model_response, task)
        }

    async def do_async(self, task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int = 3):
        """
        Execute a direct LLM call with the given task and model asynchronously.
        
        Args:
            task: The task to execute
            model: The LLM model to use (default: "openai/gpt-4")
            client: Optional custom client to use instead of creating a new one
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls (default: 3)
            
        Returns:
            The response from the LLM
        """

        # Execute the direct call asynchronously with retry parameter
        await self.call_async(task, self.model, retry=retry)
        
        # Print the price ID summary if the task has a price ID
        if not task.not_main_task:
            print_price_id_summary(task.price_id, task)
            
        return task.response

    async def print_do_async(self, task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int = 3):
        """
        Execute a direct LLM call and print the result asynchronously.
        
        Args:
            task: The task to execute
            model: The LLM model to use (default: "openai/gpt-4")
            client: Optional custom client to use instead of creating a new one
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls (default: 3)
            
        Returns:
            The response from the LLM
        """
        result = await self.do_async(task, model, client, debug, retry)
        print(result)
        return result

