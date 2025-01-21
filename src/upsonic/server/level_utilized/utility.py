import inspect
import traceback
import types
from itertools import chain
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel, OpenAIAgentModel
from pydantic_ai.models.anthropic import AnthropicModel
from openai import AsyncOpenAI, NOT_GIVEN
from openai import AsyncAzureOpenAI

from pydantic import BaseModel
from pydantic_ai.result import ResultData
from fastapi import HTTPException, status
from functools import wraps
from typing import Any, Callable, Optional
from pydantic_ai import RunContext, Tool
from anthropic import AsyncAnthropicBedrock
from dataclasses import dataclass
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import chat
from collections.abc import AsyncIterator
from typing import Literal
from openai import AsyncStream


from ...storage.configuration import Configuration

from ...tools_server.function_client import FunctionToolManager

class CustomOpenAIAgentModel(OpenAIAgentModel):
    async def _completions_create(
        self, messages: list[Any], stream: bool, model_settings: Any | None
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        if not self.tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not self.allow_text_result:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        openai_messages = list(chain(*(self._map_message(m) for m in messages)))
        model_settings = model_settings or {}

        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            n=1,
            parallel_tool_calls=False,  # Force parallel_tool_calls to False
            tools=self.tools or NOT_GIVEN,
            tool_choice=tool_choice or NOT_GIVEN,
            stream=stream,
            stream_options={'include_usage': True} if stream else NOT_GIVEN,
            max_tokens=model_settings.get('max_tokens', NOT_GIVEN),
            temperature=model_settings.get('temperature', NOT_GIVEN),
            top_p=model_settings.get('top_p', NOT_GIVEN),
            timeout=model_settings.get('timeout', NOT_GIVEN),
        )

class CustomOpenAIModel(OpenAIModel):
    async def agent_model(
        self,
        *,
        function_tools: list[Any],
        allow_text_result: bool,
        result_tools: list[Any],
    ) -> Any:
        tools = [self._map_tool_definition(r) for r in function_tools]
        if result_tools:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return CustomOpenAIAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
        )

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


def summarize_text(text: str, llm_model: Any, chunk_size: int = 100000, max_size: int = 300000) -> str:
    """Base function to summarize any text by splitting into chunks and summarizing each."""
    # Return early if text is None or empty
    if text is None:
        return ""
    
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return ""

    if not text:
        return ""

    # If text is already under max_size, return it
    if len(text) <= max_size:
        return text

    # Adjust chunk size based on model
    if "gpt" in str(llm_model).lower():
        # OpenAI has a 1M character limit, we'll use a much smaller chunk size to be safe
        chunk_size = min(chunk_size, 100000)  # 100K per chunk for OpenAI
    elif "claude" in str(llm_model).lower():
        chunk_size = min(chunk_size, 200000)  # 200K per chunk for Claude
    
    try:
        print(f"Original text length: {len(text)}")
        
        # If text is extremely long, do an initial aggressive truncation
        if len(text) > 2000000:  # If over 2M characters
            text = text[:2000000]  # Take first 2M characters
            print("Text was extremely long, truncated to 2M characters")
        
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        print(f"Number of chunks: {len(chunks)}")
        
        model = agent_creator(response_format=str, tools=[], context=None, llm_model=llm_model, system_prompt=None)
        if isinstance(model, dict) and "status_code" in model:
            print(f"Error creating model: {model}")
            return text[:max_size]
        
        # Process chunks in smaller batches if there are too many
        batch_size = 5
        summarized_chunks = []
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]
            
            for i, chunk in enumerate(batch):
                chunk_num = batch_start + i + 1
                try:
                    print(f"Processing chunk {chunk_num}/{len(chunks)}, length: {len(chunk)}")
                    
                    # Create a more focused prompt for better summarization
                    prompt = (
                        "Please provide an extremely concise summary of the following text. "
                        "Focus only on the most important points and key information. "
                        "Be as brief as possible while retaining critical meaning:\n\n"
                    )
                    
                    message = [{"type": "text", "text": prompt + chunk}]
                    result = model.run_sync(message)
                    
                    if result and hasattr(result, 'data') and result.data:
                        # Ensure the summary isn't too long
                        summary = result.data[:max_size//len(chunks)]
                        summarized_chunks.append(summary)
                    else:
                        print(f"Warning: Empty or invalid result for chunk {chunk_num}")
                        # Include a shorter truncated version as fallback
                        summarized_chunks.append(chunk[:500] + "...")
                except Exception as e:
                    print(f"Error summarizing chunk {chunk_num}: {str(e)}")
                    # Include a shorter truncated version as fallback
                    summarized_chunks.append(chunk[:500] + "...")

        # Combine all summarized chunks
        combined_summary = "\n\n".join(summarized_chunks)
        
        # If still too long, recursively summarize with smaller chunks
        if len(combined_summary) > max_size:
            print(f"Combined summary still too long ({len(combined_summary)} chars), recursively summarizing...")
            return summarize_text(
                combined_summary, 
                llm_model, 
                chunk_size=max(5000, chunk_size//4),  # Reduce chunk size more aggressively
                max_size=max_size
            )
            
        print(f"Final summary length: {len(combined_summary)}")
        return combined_summary
    except Exception as e:
        traceback.print_exc()
        print(f"Error in summarize_text: {str(e)}")
        # If all else fails, return a truncated version
        return text[:max_size]

def summarize_message_prompt(message_prompt: str, llm_model: Any) -> str:
    """Summarizes the message prompt to reduce its length while preserving key information."""
    print("\n\n\n****************Summarizing message prompt****************\n\n\n")
    if message_prompt is None:
        return ""
    
    try:
        # Use a smaller max size for message prompts
        max_size = 50000  # 100K for messages
        summarized_message_prompt = summarize_text(message_prompt, llm_model, max_size=max_size)
        if summarized_message_prompt is None:
            return ""
        print(f"Summarized message prompt length: {len(summarized_message_prompt)}")
        return summarized_message_prompt
    except Exception as e:
        print(f"Error in summarize_message_prompt: {str(e)}")
        try:
            return str(message_prompt)[:50000] if message_prompt else ""
        except:
            return ""

def summarize_system_prompt(system_prompt: str, llm_model: Any) -> str:
    """Summarizes the system prompt to reduce its length while preserving key information."""
    print("\n\n\n****************Summarizing system prompt****************\n\n\n")
    if system_prompt is None:
        return ""
    
    try:
        # Use a smaller max size for system prompts
        max_size = 50000  # 100K for system prompts
        summarized_system_prompt = summarize_text(system_prompt, llm_model, max_size=max_size)
        if summarized_system_prompt is None:
            return ""
        print(f"Summarized system prompt length: {len(summarized_system_prompt)}")
        return summarized_system_prompt
    except Exception as e:
        print(f"Error in summarize_system_prompt: {str(e)}")
        try:
            return str(system_prompt)[:50000] if system_prompt else ""
        except:
            return ""

def summarize_context_string(context_string: str, llm_model: Any) -> str:
    """Summarizes the context string to reduce its length while preserving key information."""
    print("\n\n\n****************Summarizing context string****************\n\n\n")
    if context_string is None or context_string == "":
        return ""
    
    try:
        # Use a smaller max size for context strings
        max_size = 50000  # 50K for context strings
        summarized_context = summarize_text(context_string, llm_model, max_size=max_size)
        if summarized_context is None:
            return ""
        print(f"Summarized context string length: {len(summarized_context)}")
        return summarized_context
    except Exception as e:
        print(f"Error in summarize_context_string: {str(e)}")
        try:
            return str(context_string)[:50000] if context_string else ""
        except:
            return ""

def agent_creator(
        response_format: BaseModel = str,
        tools: list[str] = [],
        context: Any = None,
        llm_model: str = "gpt-4o",
        system_prompt: Optional[Any] = None,
        context_compress: bool = False
    ) -> ResultData:

        if llm_model == "gpt-4o":
            openai_api_key = Configuration.get("OPENAI_API_KEY")
            if not openai_api_key:
                return {"status_code": 401, "detail": "No API key provided. Please set OPENAI_API_KEY in your configuration."}
            client = AsyncOpenAI(
                api_key=openai_api_key,  # This is the default and can be omitted
            )
            model = CustomOpenAIModel(llm_model, openai_client=client)
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
            model = CustomOpenAIModel('gpt-4o', openai_client=model)

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
                        description = each.description
                        try:
                            response = each.response.dict()
                        except:
                            try:
                                response = each.response.model_dump()
                            except:
                                response = each.response
                                
                        context_string += f"\n\nContexts from old tasks: ```old_task {description} {response}```   "
                    else:
                        context_string += f"\n\nContexts ```context {each}```"

        # Compress context string if enabled
        if context_compress and context_string:
            context_string = summarize_context_string(context_string, llm_model)

        system_prompt_ = ()

        if system_prompt is not None:
            system_prompt_ = system_prompt + f"The context is: {context_string}"
        elif context_string != "":
            system_prompt_ = f"You are a helpful assistant. User want to add an old task context to the task. The context is: {context_string}"
        



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

