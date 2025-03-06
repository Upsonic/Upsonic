import copy
import time
import cloudpickle

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


from ..level_utilized.utility import context_serializer, response_format_serializer, tools_serializer, response_format_deserializer, error_handler

class Call:


    def call(
        self,
        task: Union[Task, List[Task]],
        llm_model: str = None,
    ) -> Any:
        
        start_time = time.time()


        try:
            if isinstance(task, list):
                for each in task:
                    the_result = self.call_(each, llm_model)
                    call_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], self.debug)
            else:
                the_result = self.call_(task, llm_model)
                call_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], self.debug)
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
    ) -> Any:
        task.start_time = time.time()
        from ..trace import sentry_sdk
        """
        Call GPT-4 with optional tools and MCP servers.

        Args:
            prompt: The input prompt for GPT-4
            response_format: The expected response format (can be a type or Pydantic model)
            tools: Optional list of tool names to use


        Returns:
            The response in the specified format
        """

        if llm_model is None:
            llm_model = self.default_llm_model



        tools = tools_serializer(task.tools)

        response_format = task.response_format
        with sentry_sdk.start_transaction(op="task", name="Call.call") as transaction:
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
                    "prompt": task.description + task.additional_description(self), 
                    "images": task.images_base_64,
                    "response_format": response_format_str,
                    "tools": tools or [],
                    "context": context,
                    "llm_model": llm_model,
                    "system_prompt": None
                }



            with sentry_sdk.start_span(op="send_request"):
                result = self.send_request("/level_one/gpt4o", data)
                original_result = result

                
                result = result["result"]
            
                


                error_handler(result)

                

                


            with sentry_sdk.start_span(op="deserialize"):
                deserialized_result = response_format_deserializer(response_format_str, result)





        task._response = deserialized_result["result"]


        response_format_req = None
        if response_format_str == "str":
            response_format_req = response_format_str
        else:
            # Class name
            response_format_req = response_format.__name__


        
        task.end_time = time.time()
        return {"result": deserialized_result["result"], "llm_model": llm_model, "response_format": response_format_req, "usage": deserialized_result["usage"]}



