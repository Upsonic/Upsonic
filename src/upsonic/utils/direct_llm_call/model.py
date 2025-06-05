import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from openai import AsyncOpenAI, NOT_GIVEN
from openai import AsyncAzureOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from ..error_wrapper import upsonic_error_handler


from anthropic import AsyncAnthropicBedrock

# Load environment variables from .env file
load_dotenv()

# Import from the centralized model registry
from ...models.model_registry import (
    MODEL_SETTINGS,
    MODEL_REGISTRY,
    OPENAI_MODELS,
    ANTHROPIC_MODELS,
    get_model_registry_entry,
    get_model_settings,
    has_capability
)


class ModelCreationStrategy(ABC):
    """Abstract base class for model creation strategies."""
    
    @abstractmethod
    def create_model(self, model_name: str, **kwargs) -> Tuple[Optional[Any], Optional[dict]]:
        """Create a model instance. Returns (model, error_dict)."""
        pass


class OpenAIStrategy(ModelCreationStrategy):
    """Strategy for creating OpenAI models."""
    
    def create_model(self, model_name: str, **kwargs) -> Tuple[Optional[Any], Optional[dict]]:
        api_key_name = kwargs.get("api_key", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_name)
        if not api_key:
            return None, {"status_code": 401, "detail": f"No API key provided. Please set {api_key_name} in your configuration."}
        
        client = AsyncOpenAI(api_key=api_key)
        return OpenAIModel(model_name, provider=OpenAIProvider(openai_client=client)), None


class AzureOpenAIStrategy(ModelCreationStrategy):
    """Strategy for creating Azure OpenAI models."""
    
    def create_model(self, model_name: str, **kwargs) -> Tuple[Optional[Any], Optional[dict]]:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

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
        return OpenAIModel(model_name, provider=OpenAIProvider(openai_client=client)), None


class DeepseekStrategy(ModelCreationStrategy):
    """Strategy for creating Deepseek models."""
    
    def create_model(self, model_name: str, **kwargs) -> Tuple[Optional[Any], Optional[dict]]:
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            return None, {"status_code": 401, "detail": "No API key provided. Please set DEEPSEEK_API_KEY in your configuration."}

        return OpenAIModel(
            'deepseek-chat',
            provider=OpenAIProvider(
                base_url='https://api.deepseek.com',
                api_key=deepseek_api_key
            )
        ), None


class OllamaStrategy(ModelCreationStrategy):
    """Strategy for creating Ollama models."""
    
    def create_model(self, model_name: str, **kwargs) -> Tuple[Optional[Any], Optional[dict]]:
        # Ollama runs locally, so we don't need API keys
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OpenAIModel(
            model_name,
            provider=OpenAIProvider(base_url=base_url)
        ), None


class OpenRouterStrategy(ModelCreationStrategy):
    """Strategy for creating OpenRouter models."""
    
    def create_model(self, model_name: str, **kwargs) -> Tuple[Optional[Any], Optional[dict]]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None, {"status_code": 401, "detail": "No API key provided. Please set OPENROUTER_API_KEY in your configuration."}
        
        return OpenAIModel(
            model_name,
            provider=OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
        ), None


class GeminiStrategy(ModelCreationStrategy):
    """Strategy for creating Gemini models."""
    
    def create_model(self, model_name: str, **kwargs) -> Tuple[Optional[Any], Optional[dict]]:
        api_key = os.getenv("GOOGLE_GLA_API_KEY")
        if not api_key:
            return None, {"status_code": 401, "detail": "No API key provided. Please set GOOGLE_GLA_API_KEY in your configuration."}
        
        return GeminiModel(
            model_name,
            provider=GoogleGLAProvider(api_key=api_key)
        ), None


class AnthropicStrategy(ModelCreationStrategy):
    """Strategy for creating Anthropic models."""
    
    def create_model(self, model_name: str, **kwargs) -> Tuple[Optional[Any], Optional[dict]]:
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            return None, {"status_code": 401, "detail": "No API key provided. Please set ANTHROPIC_API_KEY in your configuration."}
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=anthropic_api_key)), None


class BedrockAnthropicStrategy(ModelCreationStrategy):
    """Strategy for creating AWS Bedrock Anthropic models."""
    
    def create_model(self, model_name: str, **kwargs) -> Tuple[Optional[Any], Optional[dict]]:
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION")

        if not aws_access_key_id or not aws_secret_access_key or not aws_region:
            return None, {"status_code": 401, "detail": "No AWS credentials provided. Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION in your configuration."}
        
        bedrock_client = AsyncAnthropicBedrock(
            aws_access_key=aws_access_key_id,
            aws_secret_key=aws_secret_access_key,
            aws_region=aws_region
        )

        return AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=bedrock_client)), None


class ModelCreationContext:
    """Context class that uses model creation strategies."""
    
    def __init__(self):
        self._strategies = {
            "openai": OpenAIStrategy(),
            "azure_openai": AzureOpenAIStrategy(),
            "deepseek": DeepseekStrategy(),
            "anthropic": AnthropicStrategy(),
            "bedrock_anthropic": BedrockAnthropicStrategy(),
            "ollama": OllamaStrategy(),
            "openrouter": OpenRouterStrategy(),
            "gemini": GeminiStrategy(),
        }
    
    def create_model(self, provider: str, model_name: str, **kwargs) -> Tuple[Optional[Any], Optional[dict]]:
        """Create a model using the appropriate strategy."""
        strategy = self._strategies.get(provider)
        if not strategy:
            return None, {"status_code": 400, "detail": f"Unsupported provider: {provider}"}
        
        return strategy.create_model(model_name, **kwargs)
    
    def register_strategy(self, provider: str, strategy: ModelCreationStrategy):
        """Register a new strategy for a provider."""
        self._strategies[provider] = strategy
    
    def get_supported_providers(self) -> list:
        """Get list of supported providers."""
        return list(self._strategies.keys())


# Global context instance
_model_context = ModelCreationContext()


@upsonic_error_handler(max_retries=1, show_error_details=True)
def get_agent_model(llm_model: str):
    """Create a model instance based on the registry entry."""
    registry_entry = get_model_registry_entry(llm_model)
    if not registry_entry:
        return None, {"status_code": 400, "detail": f"Unsupported LLM model: {llm_model}"}
    
    provider = registry_entry["provider"]
    model_name = registry_entry["model_name"]
    
    # Extract additional parameters from registry entry
    additional_params = {k: v for k, v in registry_entry.items() if k not in ["provider", "model_name"]}
    
    return _model_context.create_model(provider, model_name, **additional_params)


def register_model_strategy(provider: str, strategy: ModelCreationStrategy):
    """Register a new model creation strategy."""
    _model_context.register_strategy(provider, strategy)


def get_supported_providers() -> list:
    """Get list of supported providers."""
    return _model_context.get_supported_providers()

