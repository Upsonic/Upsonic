import inspect
import traceback
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel

from openai import AsyncAzureOpenAI

from pydantic import BaseModel
from pydantic_ai.result import ResultData
from fastapi import HTTPException, status
from functools import wraps
from typing import Any, Callable, Optional
from pydantic_ai import RunContext, Tool
from anthropic import AsyncAnthropicBedrock


from ...storage.configuration import Configuration

from ...tools_server.function_client import FunctionToolManager

def tool_wrapper(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Log the tool call
        tool_name = getattr(func, "__name__", str(func))

        
        try:
            # Call the original function
            result = func(*args, **kwargs)

            return result
        except Exception as e:
            print("Tool call failed:", e)
            return {"status_code": 500, "detail": f"Tool call failed: {e}"}
    
    return wrapper


def agent_creator(
        response_format: BaseModel = str,
        tools: list[str] = [],
        context: Any = None,
        llm_model: str = "gpt-4o",
        system_prompt: Optional[Any] = None 
    ) -> ResultData:

        if llm_model == "gpt-4o":
            openai_api_key = Configuration.get("OPENAI_API_KEY")
            if not openai_api_key:

                return {"status_code": 401, "detail": "No API key provided. Please set OPENAI_API_KEY in your configuration."}
            model = OpenAIModel(llm_model, api_key=openai_api_key)
        elif llm_model == "claude-3-5-sonnet":
            anthropic_api_key = Configuration.get("ANTHROPIC_API_KEY")
            if not anthropic_api_key:

                return {"status_code": 401, "detail": "No API key provided. Please set ANTHROPIC_API_KEY in your configuration."}
            model = AnthropicModel("claude-3-5-sonnet-latest", api_key=anthropic_api_key)

        
        elif llm_model == "claude-3-5-sonnet-aws":
            aws_access_key_id = Configuration.get("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = Configuration.get("AWS_SECRET_ACCESS_KEY")
            aws_region = Configuration.get("AWS_REGION")


            if not aws_access_key_id or not aws_secret_access_key or not aws_region:
                return {"status_code": 401, "detail": "No AWS credentials provided. Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION in your configuration."}
            
            model = AsyncAnthropicBedrock(
                aws_access_key=aws_access_key_id,
                aws_secret_key=aws_secret_access_key,
                aws_region=aws_region
            )

            model = AnthropicModel("us.anthropic.claude-3-5-sonnet-20241022-v2:0", anthropic_client=model)

        elif llm_model == "gpt-4o-azure":
            azure_endpoint = Configuration.get("AZURE_OPENAI_ENDPOINT")
            azure_api_version = Configuration.get("AZURE_OPENAI_API_VERSION")
            azure_api_key = Configuration.get("AZURE_OPENAI_API_KEY")

            missing_keys = []
            if not azure_endpoint:
                missing_keys.append("AZURE_OPENAI_ENDPOINT")
            if not azure_api_version:
                missing_keys.append("AZURE_OPENAI_API_VERSION")
            if not azure_api_key:
                missing_keys.append("AZURE_OPENAI_API_KEY")

            if missing_keys:
                return {
                    "status_code": 401,
                    "detail": f"No API key provided. Please set {', '.join(missing_keys)} in your configuration."
                }

            model = AsyncAzureOpenAI(api_version=azure_api_version, azure_endpoint=azure_endpoint, api_key=azure_api_key)
            model = OpenAIModel('gpt-4o', openai_client=model)

        else:

            return {"status_code": 400, "detail": f"Unsupported LLM model: {llm_model}"}

        context_string = ""
        if context is not None:
            if not isinstance(context, list):
                context = [context]

            if isinstance(context, list):
                for each in context:

                    from ...client.level_two.agent import Characterization
                    from ...client.level_two.agent import OtherTask
                    from ...client.tasks.tasks import Task
                    from ...client.knowledge_base.knowledge_base import KnowledgeBase
                    type_string = type(each).__name__

                    if type_string == Characterization.__name__:
                        context_string += f"\n\nThis is your character ```character {each.model_dump()}```"
                    elif type_string == OtherTask.__name__:
                        context_string += f"\n\nContexts from old tasks: ```old_task {each.task} {each.result}```"
                    elif type_string == Task.__name__:
                        response = None
                        try:
                            response = each.response.dict()
                        except:
                            try:
                                response = each.response.model_json_schema()
                            except:
                                response = each.response
                                
                        context_string += f"\n\nContexts from old tasks: ```old_task {each.description} {response}```   "
                    else:
                        context_string += f"\n\nContexts ```context {each}```"







        system_prompt_ = ()

        if system_prompt is not None:
            system_prompt_ = system_prompt + f"The context is: {context_string}"
        elif context_string != "":
            system_prompt_ = f"You are a helpful assistant. User want to add an old task context to the task. The context is: {context_string}"
        
        print("system_prompt", system_prompt_)


        roulette_agent = Agent(
            model,
            result_type=response_format,
            retries=5,
            system_prompt=system_prompt_

        )



        the_wrapped_tools = []

        with FunctionToolManager() as function_client:
            the_list_of_tools = function_client.get_tools_by_name(tools)

            for each in the_list_of_tools:
                # Wrap the tool with our wrapper
  
    
                wrapped_tool = tool_wrapper(each)

                the_wrapped_tools.append(wrapped_tool)
            



        for each in the_wrapped_tools:

            # İnspect signature of the tool
            signature = inspect.signature(each)

            roulette_agent.tool_plain(each, retries=5)




        # Computer use

        if "claude-3-5-sonnet" in llm_model:
            print("Tools", tools)
            if "ComputerUse.*" in tools:
                try:
                    from .cu import ComputerUse_tools
                    for each in ComputerUse_tools:
                        roulette_agent.tool_plain(each, retries=5)
                except Exception as e:
                    print("Error", e)

        


        return roulette_agent

