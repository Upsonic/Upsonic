import inspect
import traceback
import types
from itertools import chain
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from openai import AsyncOpenAI, NOT_GIVEN
from openai import AsyncAzureOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google_gla import GoogleGLAProvider
import hashlib
from pydantic_ai.messages import ImageUrl
from pydantic_ai import BinaryContent

from pydantic import BaseModel
from fastapi import HTTPException, status
from functools import wraps
from typing import Any, Callable, Optional, Dict
from pydantic_ai import RunContext, Tool
from anthropic import AsyncAnthropicBedrock
from dataclasses import dataclass
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import chat
from collections.abc import AsyncIterator
from typing import Literal
from openai import AsyncStream


from ...storage.configuration import Configuration
from ...storage.caching import save_to_cache_with_expiry, get_from_cache_with_expiry

from ...tools_server.function_client import FunctionToolManager

# Import from the centralized model registry
from ...model_registry import (
    MODEL_SETTINGS,
    MODEL_REGISTRY,
    OPENAI_MODELS,
    ANTHROPIC_MODELS,
    get_model_registry_entry,
    get_model_settings,
    has_capability
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

    # Generate a cache key based on text content and parameters
    cache_key = hashlib.md5(f"{text}{llm_model}{chunk_size}{max_size}".encode()).hexdigest()
    
    # Try to get from cache first
    cached_result = get_from_cache_with_expiry(cache_key)
    if cached_result is not None:
        print("Using cached summary")
        return cached_result

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
        
        # Cache the result for 1 hour (3600 seconds)
        save_to_cache_with_expiry(combined_summary, cache_key, 3600)
        
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
        print("Before summarize_message_prompt length: ", len(message_prompt))
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
        print("Before summarize_system_prompt length: ", len(system_prompt))
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
        print("Before summarize_context_string length: ", len(context_string))
        print(f"Summarized context string length: {len(summarized_context)}")
        return summarized_context
    except Exception as e:
        print(f"Error in summarize_context_string: {str(e)}")
        try:
            return str(context_string)[:50000] if context_string else ""
        except:
            return ""

def process_error_traceback(e):
    """Extract and format error traceback information consistently."""
    tb = traceback.extract_tb(e.__traceback__)
    file_path = tb[-1].filename
    if "pydantic_ai" in file_path:
        return {"status_code": 500, "detail": str(e)}
    if "Upsonic/src/" in file_path:
        file_path = file_path.split("Upsonic/src/")[1]
    line_number = tb[-1].lineno
    return {"status_code": 500, "detail": f"Error processing request in {file_path} at line {line_number}: {str(e)}"}

def prepare_message_history(prompt, images=None, llm_model=None, tools=None):
    """Prepare message history with prompt and images, adding screenshot for models with computer use capability."""
    message_history = [prompt]
    
    if images:
        for image in images:
            message_history.append(ImageUrl(url=f"data:image/jpeg;base64,{image}"))

    # Add screenshot for models with computer_use capability when ComputerUse tools are requested

    if llm_model and tools and ("ComputerUse.*" in tools or "Screenshot.*" in tools) and has_capability(llm_model, "computer_use"):
        try:
            from .cu import ComputerUse_screenshot_tool_bytes
            result_of_screenshot = ComputerUse_screenshot_tool_bytes()
            message_history.append(BinaryContent(data=result_of_screenshot, media_type='image/png'))
            print(f"Added screenshot for model {llm_model} with computer_use capability")
        except Exception as e:
            print(f"Error adding screenshot for {llm_model}: {e}")
            
    return message_history

def format_response(result):
    """Format the successful response consistently."""
    messages = result.all_messages()
    
    # Track tool usage
    tool_usage = []
    current_tool = None
    
    for msg in messages:
        if msg.kind == 'request':
            for part in msg.parts:
                if part.part_kind == 'tool-return':
                    if current_tool and current_tool['tool_name'] != 'final_result':
                        current_tool['tool_result'] = part.content
                        tool_usage.append(current_tool)
                    current_tool = None
                    
        elif msg.kind == 'response':
            for part in msg.parts:
                if part.part_kind == 'tool-call' and part.tool_name != 'final_result':
                    current_tool = {
                        'tool_name': part.tool_name,
                        'params': part.args,
                        'tool_result': None
                    }

    usage = result.usage()
    return {
        "status_code": 200,
        "result": result.data,
        "usage": {
            "input_tokens": usage.request_tokens,
            "output_tokens": usage.response_tokens
        },
        "tool_usage": tool_usage
    }

async def handle_compression_retry(prompt, images, tools, llm_model, response_format, context, system_prompt=None, agent_memory=None):
    """Handle compression and retry when facing token limit issues."""
    try:
        # Compress prompts
        compressed_system_prompt = summarize_system_prompt(system_prompt, llm_model) if system_prompt else None
        compressed_message = summarize_message_prompt(prompt, llm_model)
        
        # Prepare new message history
        message_history = prepare_message_history(compressed_message, images, llm_model, tools)
        
        # Create new agent with compressed prompts
        roulette_agent = agent_creator(
            response_format=response_format,
            tools=tools,
            context=context,
            llm_model=llm_model,
            system_prompt=compressed_system_prompt,
            context_compress=False
        )
        
        # Run the agent with compressed inputs
        print("Sending request with compressed prompts")
        if agent_memory:
            result = await roulette_agent.run(message_history, message_history=agent_memory)
        else:
            result = await roulette_agent.run(message_history)
        print("Received response with compressed prompts")
        
        return result
    except Exception as e:
        raise e  # Re-raise for consistent error handling

def _create_openai_client(api_key_name="OPENAI_API_KEY"):
    """Helper function to create an OpenAI client with the specified API key."""
    api_key = Configuration.get(api_key_name)
    if not api_key:
        return None, {"status_code": 401, "detail": f"No API key provided. Please set {api_key_name} in your configuration."}
    
    client = AsyncOpenAI(api_key=api_key)
    return client, None

def _create_azure_openai_client():
    """Helper function to create an Azure OpenAI client."""
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
        return None, {
            "status_code": 401,
            "detail": f"No API key provided. Please set {', '.join(missing_keys)} in your configuration."
        }

    client = AsyncAzureOpenAI(
        api_version=azure_api_version, 
        azure_endpoint=azure_endpoint, 
        api_key=azure_api_key
    )
    return client, None

def _create_openai_model(model_name: str, api_key_name: str = "OPENAI_API_KEY"):
    """Helper function to create an OpenAI model with specified model name and API key."""
    client, error = _create_openai_client(api_key_name)
    if error:
        return None, error
    return OpenAIModel(model_name, provider=OpenAIProvider(openai_client=client)), None

def _create_azure_openai_model(model_name: str):
    """Helper function to create an Azure OpenAI model with specified model name."""
    client, error = _create_azure_openai_client()
    if error:
        return None, error
    return OpenAIModel(model_name, provider=OpenAIProvider(openai_client=client)), None

def _create_deepseek_model():
    """Helper function to create a Deepseek model."""
    deepseek_api_key = Configuration.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        return None, {"status_code": 401, "detail": "No API key provided. Please set DEEPSEEK_API_KEY in your configuration."}

    return OpenAIModel(
        'deepseek-chat',
        provider=OpenAIProvider(
            base_url='https://api.deepseek.com',
            api_key=deepseek_api_key
        )
    ), None

def _create_ollama_model(model_name: str):
    """Helper function to create an Ollama model with specified model name."""
    # Ollama runs locally, so we don't need API keys
    base_url = Configuration.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return OpenAIModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url)
    ), None

def _create_openrouter_model(model_name: str):
    """Helper function to create an OpenRouter model with specified model name."""
    api_key = Configuration.get("OPENROUTER_API_KEY")
    if not api_key:
        return None, {"status_code": 401, "detail": "No API key provided. Please set OPENROUTER_API_KEY in your configuration."}
    
    # If model_name starts with openrouter/, remove it
    if model_name.startswith("openrouter/"):
        model_name = model_name.split("openrouter/", 1)[1]
    
    return OpenAIModel(
        model_name,
        provider=OpenAIProvider(
            base_url='https://openrouter.ai/api/v1',
            api_key=api_key
        )
    ), None

def _create_gemini_model(model_name: str):
    """Helper function to create a Gemini model with specified model name."""
    api_key = Configuration.get("GOOGLE_GLA_API_KEY")
    if not api_key:
        return None, {"status_code": 401, "detail": "No API key provided. Please set GOOGLE_GLA_API_KEY in your configuration."}
    
    return GeminiModel(
        model_name,
        provider=GoogleGLAProvider(api_key=api_key)
    ), None

def _create_anthropic_model(model_name: str):
    """Helper function to create an Anthropic model with specified model name."""
    anthropic_api_key = Configuration.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        return None, {"status_code": 401, "detail": "No API key provided. Please set ANTHROPIC_API_KEY in your configuration."}
    return AnthropicModel(model_name, provider=AnthropicProvider(api_key=anthropic_api_key)), None

def _create_bedrock_anthropic_model(model_name: str):
    """Helper function to create an AWS Bedrock Anthropic model with specified model name."""
    aws_access_key_id = Configuration.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = Configuration.get("AWS_SECRET_ACCESS_KEY")
    aws_region = Configuration.get("AWS_REGION")

    if not aws_access_key_id or not aws_secret_access_key or not aws_region:
        return None, {"status_code": 401, "detail": "No AWS credentials provided. Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION in your configuration."}
    
    bedrock_client = AsyncAnthropicBedrock(
        aws_access_key=aws_access_key_id,
        aws_secret_key=aws_secret_access_key,
        aws_region=aws_region
    )

    return AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=bedrock_client)), None

def _process_context(context):
    """Process context data into a formatted context string."""
    if context is None:
        return ""
        
    if not isinstance(context, list):
        context = [context]
        
    context_string = ""
    for each in context:
        from ...client.level_two.agent import Characterization
        from ...client.level_two.agent import OtherTask
        from ...client.tasks.tasks import Task
        from ...client.tasks.task_response import ObjectResponse
        from ...client.knowledge_base.knowledge_base import KnowledgeBase
        type_string = type(each).__name__
        the_class_string = None
        try:
            the_class_string = each.__bases__[0].__name__
        except:
            pass
        
        if type_string == Characterization.__name__:
            context_string += f"\n\nThis is your character ```character {each.model_dump()}```"
        elif type_string == OtherTask.__name__:
            context_string += f"\n\nContexts from question answering: ```question_answering question: {each.task} answer: {each.result}```"
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
                    
            context_string += f"\n\nContexts from question answering: ```question_answering question: {description} answer: {response}```   "
        elif the_class_string == ObjectResponse.__name__ or the_class_string == BaseModel.__name__:
            context_string += f"\n\nContexts from object response: ```Requested Output {each.model_fields}```"
        else:
            context_string += f"\n\nContexts ```context {each}```"
            
    return context_string

def _setup_tools(roulette_agent, tools, llm_model):
    """Set up the tools for the agent."""
    the_wrapped_tools = []

    # First check for ComputerUse tools compatibility
    if "ComputerUse.*" in tools:
        if not has_capability(llm_model, "computer_use"):
            return {
                "status_code": 405,
                "detail": f"ComputerUse tools are not supported by the model {llm_model}. Please use a model that supports computer_use capability."
            }

    # Set up function tools
    with FunctionToolManager() as function_client:
        the_list_of_tools = function_client.get_tools_by_name(tools)

        for each in the_list_of_tools:
            wrapped_tool = tool_wrapper(each)
            the_wrapped_tools.append(wrapped_tool)
        
    for each in the_wrapped_tools:
        signature = inspect.signature(each)
        roulette_agent.tool_plain(each, retries=5)

    # Set up ComputerUse tools for models with that capability
    if "ComputerUse.*" in tools:
        try:
            from .cu import ComputerUse_tools
            for each in ComputerUse_tools:
                roulette_agent.tool_plain(each, retries=5)
        except Exception as e:
            print(f"Error setting up ComputerUse tools: {e}")

    # Set up BrowserUse tools
    if "BrowserUse.*" in tools:
        try:
            from .bu import BrowserUse_tools
            from .bu.browseruse import LLMManager
            LLMManager.set_model(llm_model)

            for each in BrowserUse_tools:
                roulette_agent.tool_plain(each, retries=5)
        except Exception as e:
            print(f"Error setting up BrowserUse tools: {e}")
            
    return roulette_agent

def _create_model_from_registry(llm_model: str):
    """Create a model instance based on the registry entry."""
    registry_entry = get_model_registry_entry(llm_model)
    if not registry_entry:
        return None, {"status_code": 400, "detail": f"Unsupported LLM model: {llm_model}"}
    
    provider = registry_entry["provider"]
    model_name = registry_entry["model_name"]
    
    if provider == "openai":
        api_key = registry_entry.get("api_key", "OPENAI_API_KEY")
        return _create_openai_model(model_name, api_key)
    elif provider == "azure_openai":
        return _create_azure_openai_model(model_name)
    elif provider == "deepseek":
        return _create_deepseek_model()
    elif provider == "anthropic":
        return _create_anthropic_model(model_name)
    elif provider == "bedrock_anthropic":
        return _create_bedrock_anthropic_model(model_name)
    elif provider == "ollama":
        return _create_ollama_model(model_name)
    elif provider == "openrouter":
        return _create_openrouter_model(model_name)
    elif provider == "gemini":
        return _create_gemini_model(model_name)
    else:
        return None, {"status_code": 400, "detail": f"Unsupported provider: {provider}"}

def agent_creator(
        response_format: BaseModel = str,
        tools: list[str] = [],
        context: Any = None,
        llm_model: str = None,
        system_prompt: Optional[Any] = None,
        context_compress: bool = False
    ):
        # Use default model if none provided
        if llm_model is None:
            llm_model = "openai/gpt-4o"
            print(f"No model specified, using default: {llm_model}")
        
        # Get the model from registry
        model, error = _create_model_from_registry(llm_model)
        if error:
            return error

        # Process context
        context_string = _process_context(context)

        # Compress context string if enabled
        if context_compress and context_string:
            context_string = summarize_context_string(context_string, llm_model)

        # Prepare system prompt
        system_prompt_ = ()
        if system_prompt is not None:
            system_prompt_ = system_prompt + f"The context is: {context_string}"
        elif context_string != "":
            system_prompt_ = f"You are a helpful assistant. User want to add an context to the task. The context is: {context_string}"
        
        # Get the appropriate model settings based on the model type
        model_settings = get_model_settings(llm_model, tools)

        # Create the agent
        roulette_agent = Agent(
            model,
            result_type=response_format,
            retries=5,
            system_prompt=system_prompt_,
            model_settings=model_settings
        )

        # Set up tools and check for errors
        result = _setup_tools(roulette_agent, tools, llm_model)
        
        # If result is a dict, it means there was an error
        if isinstance(result, dict) and "status_code" in result:
            return result

        return result

