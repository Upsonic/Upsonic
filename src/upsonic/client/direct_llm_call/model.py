
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from openai import AsyncOpenAI, NOT_GIVEN
from openai import AsyncAzureOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google_gla import GoogleGLAProvider


from anthropic import AsyncAnthropicBedrock



from ...storage.configuration import Configuration


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



def get_agent_model(llm_model: str):
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

