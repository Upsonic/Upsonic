import copy
import time
import cloudpickle
import asyncio

from ..knowledge_base.knowledge_base import KnowledgeBase
cloudpickle.DEFAULT_PROTOCOL = 2

import dill
import base64
import httpx
from typing import Any, List, Dict, Optional, Type, Union
from pydantic import BaseModel

from ..tasks.tasks import Task

from ..printing import call_end


from ..tasks.task_response import ObjectResponse

from ..language import Language

from ..level_utilized.utility import context_serializer, response_format_serializer, tools_serializer, response_format_deserializer, error_handler

class Call:


    def call(
        self,
        task: Union[Task, List[Task]],
        llm_model: str = None,
        retry: int = 3
    ) -> Any:
        
        start_time = time.time()

        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # If there's a running loop, run the async function in that loop
                    if isinstance(task, list):
                        for each in task:
                            the_result = asyncio.run_coroutine_threadsafe(self.call_async_(each, llm_model, retry), loop).result()
                            call_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_usage"], self.debug, each.price_id)
                    else:
                        the_result = asyncio.run_coroutine_threadsafe(self.call_async_(task, llm_model, retry), loop).result()
                        call_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_usage"], self.debug, task.price_id)
                else:
                    # If there's a loop but it's not running, use asyncio.run
                    if isinstance(task, list):
                        for each in task:
                            the_result = asyncio.run(self.call_async_(each, llm_model, retry))
                            call_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_usage"], self.debug, each.price_id)
                    else:
                        the_result = asyncio.run(self.call_async_(task, llm_model, retry))
                        call_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_usage"], self.debug, task.price_id)
            except RuntimeError:
                # No event loop exists, create one with asyncio.run
                if isinstance(task, list):
                    for each in task:
                        the_result = asyncio.run(self.call_async_(each, llm_model, retry))
                        call_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_usage"], self.debug, each.price_id)
                else:
                    the_result = asyncio.run(self.call_async_(task, llm_model, retry))
                    call_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_usage"], self.debug, task.price_id)
        except Exception as outer_e:
            try:
                from ...server import stop_dev_server, stop_main_server, is_tools_server_running, is_main_server_running

                if is_tools_server_running() or is_main_server_running():
                    stop_dev_server()

            except Exception:
                pass

            raise outer_e

        end_time = time.time()

        return task.response

    def call_(
        self,
        task: Task,
        llm_model: str = None,
        retry: int = 3
    ) -> Any:
        """
        Call GPT-4 with optional tools and MCP servers.

        Args:
            prompt: The input prompt for GPT-4
            response_format: The expected response format (can be a type or Pydantic model)
            tools: Optional list of tool names to use
            retry: Number of retries for failed calls (default: 3)

        Returns:
            The response in the specified format
        """
        # Try to get the current event loop
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the async function in that loop
                return asyncio.run_coroutine_threadsafe(self.call_async_(task, llm_model, retry), loop).result()
            else:
                # If there's a loop but it's not running, use asyncio.run
                return asyncio.run(self.call_async_(task, llm_model, retry))
        except RuntimeError:
            # No event loop exists, create one with asyncio.run
            return asyncio.run(self.call_async_(task, llm_model, retry))

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

        try:
            if isinstance(task, list):
                for each in task:
                    the_result = await self.call_async_(each, llm_model, retry)
                    call_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_usage"], self.debug, each.price_id)
            else:
                the_result = await self.call_async_(task, llm_model, retry)
                call_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_usage"], self.debug, task.price_id)
        except Exception as outer_e:
            try:
                from ...server import stop_dev_server, stop_main_server, is_tools_server_running, is_main_server_running

                if is_tools_server_running() or is_main_server_running():
                    stop_dev_server()
            except Exception:
                pass
            raise outer_e

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
        task.start_time = time.time()
        from ..trace import sentry_sdk
        from ..level_utilized.utility import CallErrorException
        
        # Use the provided model or default to the client's default
        if llm_model is None:
            llm_model = self.default_llm_model
            
        tools = tools_serializer(task.tools)

        response_format = task.response_format
        with sentry_sdk.start_transaction(op="task", name="Call.call_async") as transaction:
            with sentry_sdk.start_span(op="serialize"):
                # Serialize the response format if it's a type or BaseModel
                response_format_str = response_format_serializer(task.response_format)

                new_context = []
                if task.context:
                    for each in task.context:
                        if isinstance(each, KnowledgeBase):
                            if not each.rag:
                                new_context.append(each.markdown(self))
                        else:
                            new_context.append(each)

                context = context_serializer(new_context, self)

            with sentry_sdk.start_span(op="prepare_request"):
                # Prepare the request data
                data = {
                    "prompt": task.description + await task.additional_description(self), 
                    "images": task.images_base_64,
                    "response_format": response_format_str,
                    "tools": tools or [],
                    "context": context,
                    "llm_model": llm_model,
                    "system_prompt": None
                }

            retry_count = 0
            while True:
                try:
                    with sentry_sdk.start_span(op="send_request"):
                        result = await self.send_request_async("/level_one/gpt4o", data)
                        original_result = result
                        
                        # Extract the tool_usage from the result['result'] before changing 'result'
                        tool_usage_value = []
                        if isinstance(result, dict) and 'result' in result and isinstance(result['result'], dict) and 'tool_usage' in result['result']:
                            tool_usage_value = result['result']['tool_usage']
                            
                            # Store tool calls in the task
                            for tool_call in tool_usage_value:
                                task.add_tool_call(tool_call)
                        
                        result = result["result"]
                        
                        if error_handler(result):  # If it's a retriable error
                            if retry > 0 and retry_count < retry:  # Check if retries are enabled and we can retry
                                retry_count += 1
                                from ..printing import agent_retry
                                agent_retry(retry_count, retry)
                                continue  # Try again
                            else:
                                raise CallErrorException(result)  # No more retries, raise the error
                        
                        break  # If no error or non-retriable error, break the loop

                except Exception as e:
                    if retry > 0 and retry_count < retry:  # Check if retries are enabled and we can retry
                        retry_count += 1
                        from ..printing import agent_retry
                        agent_retry(retry_count, retry)
                        continue  # Try again
                    raise e  # No more retries, raise the error

            with sentry_sdk.start_span(op="deserialize"):
                deserialized_result = response_format_deserializer(response_format_str, result)

        task._response = deserialized_result["result"]
        

        if task.response_lang:
            language = Language(task.response_lang, task, llm_model)
            processed_result = await language.transform()
            task._response = processed_result


        response_format_req = None
        if response_format_str == "str":
            response_format_req = response_format_str
        else:
            # Class name
            response_format_req = response_format.__name__
        
        task.end_time = time.time()
        
        # Make sure all necessary fields are extracted properly
        result_value = deserialized_result["result"]
        usage_value = deserialized_result.get("usage", {"input_tokens": 0, "output_tokens": 0})
        
        return {
            "result": result_value,
            "llm_model": llm_model,
            "response_format": response_format_req,
            "usage": usage_value,
            "tool_usage": tool_usage_value
        }



