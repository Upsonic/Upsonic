from __future__ import annotations
import os
from typing import Tuple, Optional, Literal, Dict, List, Any

from pydantic import Field, SecretStr
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIModel, OpenAIResponsesModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel, ThinkingConfig
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.models.openai import OpenAIResponsesModelSettings, OpenAIModelSettings
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.gemini import GeminiModelSettings

from openai import AsyncOpenAI, AsyncAzureOpenAI
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock

from upsonic.models.base import BaseModelProvider
from upsonic.exceptions import APIKeyMissingError, ConfigurationError


class BaseOpenAICompatible(BaseModelProvider):
    """Abstract base for providers using an OpenAI-compatible API."""
    model_name: str
    api_key: Optional[SecretStr] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    enable_reasoning: bool = Field(default=False)
    reasoning_effort: Literal["low", "medium", "high"] = Field(default="medium")
    reasoning_summary: str = Field(default="detailed")
    model_settings: Optional[Dict[str, Any]] = Field(default=None, description="Generic model settings like temperature, max_tokens, etc.")
    api_mode: Literal["responses", "chat"] = Field(default="chat")

    _env_key_map: str = "OVERRIDE_IN_SUBCLASS"
    _model_meta: Dict[str, Dict[str, Any]] = {}

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.model_name not in self._model_meta and self._model_meta:
             print(f"Warning: Model '{self.model_name}' is not in the predefined list for {type(self).__name__}. Metadata may be unavailable.")

    @property
    def pricing(self) -> Dict[str, float]: return self._model_meta.get(self.model_name, {}).get("pricing", {"input": 0.0, "output": 0.0})
    @property
    def capabilities(self) -> Dict[str, List[str]]: return self._model_meta.get(self.model_name, {}).get("capabilities", {})
    @property
    def required_environment_variables(self) -> List[str]: return self._model_meta.get(self.model_name, {}).get("required_environment_variables", [self._env_key_map])

    @classmethod
    def list_available_models(cls) -> List[str]:
        """Return a list of available model names for this provider."""
        if hasattr(cls, '_model_meta') and cls._model_meta:
            return sorted(list(cls._model_meta.keys()))
        return []

    def _should_validate_api_key(self) -> bool:
        """
        Determine whether API key validation should be performed.
        
        Returns False if a custom base_url is provided (indicating a local or self-hosted service
        that might not require an API key), True otherwise.
        """
        if not self._env_key_map:
            return False
            
        if not self.base_url:
            return True

        standard_endpoints = [
            "https://api.openai.com",
            "https://api.openai.com/",
            "https://api.openai.com/v1",
            "https://api.openai.com/v1/",
            
            "https://api.deepseek.com",
            "https://api.deepseek.com/",
            "https://api.deepseek.com/v1",
            "https://api.deepseek.com/v1/",
            
            "https://openrouter.ai/api/v1",
            "https://openrouter.ai/api/v1/",
            
            "http://localhost:11434/v1",
            "http://localhost:11434/v1/",
            "http://127.0.0.1:11434/v1",
            "http://127.0.0.1:11434/v1/",
        ]
        
        if self.base_url.rstrip('/') in [endpoint.rstrip('/') for endpoint in standard_endpoints]:
            return True
            
        return False

    async def _provision(self, tools: Optional[List[Any]] = None) -> Tuple[Model, Optional[ModelSettings]]:
        final_api_key = self.api_key.get_secret_value() if self.api_key else os.getenv(self._env_key_map)
        should_validate_api_key = self._should_validate_api_key()
        if should_validate_api_key and self._env_key_map and not final_api_key:
            raise APIKeyMissingError(type(self).__name__, self._env_key_map)

        if not should_validate_api_key and not final_api_key:
            final_api_key = "dummy-key-for-custom-endpoint"

        client = AsyncOpenAI(api_key=final_api_key, base_url=self.base_url)
        provider = OpenAIProvider(openai_client=client)

        if self.api_mode == "responses":
            agent_model = self._provision_responses_model(provider)
        else:
            agent_model = self._provision_chat_model(provider)

        final_settings_dict = {}
        if self.model_settings:
            final_settings_dict.update(self.model_settings)
        
        if tools and len(tools) > 0:
            final_settings_dict['parallel_tool_calls'] = False
        
        if self.enable_reasoning:
            final_settings_dict['openai_reasoning_effort'] = self.reasoning_effort
            final_settings_dict['openai_reasoning_summary'] = self.reasoning_summary
            
        if self.api_mode == "responses":
            agent_settings = OpenAIResponsesModelSettings(**final_settings_dict) if final_settings_dict else None
        else:
            agent_settings = OpenAIModelSettings(**final_settings_dict) if final_settings_dict else None
            
        return agent_model, agent_settings

    def _provision_chat_model(self, provider: OpenAIProvider) -> OpenAIModel:
        """Instantiates a standard pydantic-ai OpenAIModel."""
        return OpenAIModel(model_name=self.model_name, provider=provider)

    def _provision_responses_model(self, provider: OpenAIProvider) -> OpenAIResponsesModel:
        """Instantiates the OpenAIResponsesModel for chat/vision/tools."""
        return OpenAIResponsesModel(self.model_name, provider=provider)


class OpenAI(BaseOpenAICompatible):
    """Configuration factory for official OpenAI models."""
    _env_key_map: str = "OPENAI_API_KEY"
    _model_meta: Dict[str, Dict[str, Any]] = {
        "gpt-4o": {"pricing": {"input": 2.50, "output": 10.00}, "capabilities": {}, "required_environment_variables": ["OPENAI_API_KEY"]},
        "gpt-4.5-preview": {"pricing": {"input": 75.00, "output": 150.00}, "capabilities": {}, "required_environment_variables": ["OPENAI_API_KEY"]},
        "gpt-4.1-nano": {"pricing": {"input": 0.10, "output": 0.40}, "capabilities": {}, "required_environment_variables": ["OPENAI_API_KEY"]},
        "gpt-4.1-mini": {"pricing": {"input": 0.40, "output": 1.60}, "capabilities": {}, "required_environment_variables": ["OPENAI_API_KEY"]},
        "o3-mini": {"pricing": {"input": 1.1, "output": 4.4}, "capabilities": {}, "required_environment_variables": ["OPENAI_API_KEY"]},
        "gpt-4o-mini": {"pricing": {"input": 0.15, "output": 0.60}, "capabilities": {}, "required_environment_variables": ["OPENAI_API_KEY"]},
        "gpt-4.1": {"pricing": {"input": 2.0, "output": 8.0}, "capabilities": {}, "required_environment_variables": ["OPENAI_API_KEY"]},
        "o4-mini": {"pricing": {"input": 1.10, "output": 4.40}, "capabilities": {}, "required_environment_variables": ["OPENAI_API_KEY"]},
        "gpt-4o-audio-preview": {"pricing": {"input": 40.00, "output": 80.00}, "capabilities": {"audio": ["mp3", "wav", "webm"]}, "required_environment_variables": ["OPENAI_API_KEY"]},
        "gpt-4o-mini-audio-preview": {"pricing": {"input": 10.00, "output": 20.00}, "capabilities": {"audio": ["mp3", "wav", "webm"]}, "required_environment_variables": ["OPENAI_API_KEY"]},
    }

    def __init__(self, **data: Any):
        super().__init__(**data)

class AzureOpenAI(BaseOpenAICompatible):
    """Configuration factory for Azure OpenAI models."""
    _model_meta: Dict[str, Dict[str, Any]] = {
        "gpt-4o": {"pricing": {"input": 2.50, "output": 10.00}, "capabilities": {}, "required_environment_variables": ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_API_KEY"]},
        "gpt-4o-mini": {"pricing": {"input": 0.15, "output": 0.60}, "capabilities": {}, "required_environment_variables": ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_API_KEY"]},
    }
    azure_endpoint: Optional[str] = Field(default=None)
    api_version: Optional[str] = Field(default=None)

    async def _provision(self, tools: Optional[List[Any]] = None) -> Tuple[Model, Optional[ModelSettings]]:
        final_endpoint = self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        final_api_version = self.api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        final_api_key = self.api_key.get_secret_value() if self.api_key else os.getenv("AZURE_OPENAI_API_KEY")

        if not final_api_key:
            raise APIKeyMissingError("AzureOpenAI", "AZURE_OPENAI_API_KEY")
        if not final_endpoint:
            raise ConfigurationError("Azure Configuration Error", "AZURE_OPENAI_ENDPOINT environment variable is not set")
        if not final_api_version:
            raise ConfigurationError("Azure Configuration Error", "AZURE_OPENAI_API_VERSION environment variable is not set")

        client = AsyncAzureOpenAI(azure_endpoint=final_endpoint, api_version=final_api_version, api_key=final_api_key)
        provider = OpenAIProvider(openai_client=client)
        agent_model = OpenAIResponsesModel(self.model_name, provider=provider)
        
        agent_settings = None
        if self.enable_reasoning:
            agent_settings = OpenAIResponsesModelSettings(openai_reasoning_effort=self.reasoning_effort, openai_reasoning_summary=self.reasoning_summary)
        return agent_model, agent_settings

class Deepseek(BaseOpenAICompatible):
    """Configuration factory for Deepseek models."""
    _env_key_map: str = "DEEPSEEK_API_KEY"
    _model_meta: Dict[str, Dict[str, Any]] = {
        "deepseek-chat": {"pricing": {"input": 0.27, "output": 1.10}, "capabilities": {}, "required_environment_variables": ["DEEPSEEK_API_KEY"]},
    }
    model_name: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"

class OpenRouter(BaseOpenAICompatible):
    """Configuration factory for models accessed via OpenRouter."""
    _env_key_map: str = "OPENROUTER_API_KEY"
    _model_meta: Dict[str, Dict[str, Any]] = {}
    base_url: str = "https://openrouter.ai/api/v1"

class Ollama(BaseOpenAICompatible):
    """Configuration factory for local Ollama models."""
    _env_key_map: str = "" 
    _model_meta: Dict[str, Dict[str, Any]] = {
        "llama3.1:8b": {"pricing": {"input": 0.0, "output": 0.0}, "capabilities": {}, "required_environment_variables": []},
        "llama3.1:70b": {"pricing": {"input": 0.0, "output": 0.0}, "capabilities": {}, "required_environment_variables": []},
        "qwen3:30b": {"pricing": {"input": 0.0, "output": 0.0}, "capabilities": {}, "required_environment_variables": []},
        "gpt-oss:20b": {"pricing": {"input": 0.0, "output": 0.0}, "capabilities": {}, "required_environment_variables": []},
    }
    base_url: str = Field(default="http://localhost:11434/v1")

class BaseAnthropic(BaseModelProvider):
    """Abstract base for Anthropic-based providers."""
    model_name: str
    api_key: Optional[SecretStr] = Field(default=None)
    enable_reasoning: bool = Field(default=False)
    reasoning_budget_tokens: int = Field(default=1024)
    _model_meta: Dict[str, Dict[str, Any]] = {}

    def __init__(self, **data: Any):
        super().__init__(**data)

    @property
    def pricing(self) -> Dict[str, float]: return self._model_meta[self.model_name]['pricing']
    @property
    def capabilities(self) -> Dict[str, List[str]]: return self._model_meta[self.model_name]['capabilities']
    @property
    def required_environment_variables(self) -> List[str]: return self._model_meta[self.model_name]['required_environment_variables']

    @classmethod
    def list_available_models(cls) -> List[str]:
        """Return a list of available model names for this provider."""
        if hasattr(cls, '_model_meta') and cls._model_meta:
            return sorted(list(cls._model_meta.keys()))
        return []

class Anthropic(BaseAnthropic):
    """Configuration factory for official Anthropic Claude models."""
    _model_meta: Dict[str, Dict[str, Any]] = {
        "claude-3-5-sonnet-latest": {"pricing": {"input": 3.00, "output": 15.00}, "capabilities": {"computer_use": []}, "required_environment_variables": ["ANTHROPIC_API_KEY"]},
        "claude-3-7-sonnet-latest": {"pricing": {"input": 3.00, "output": 15.00}, "capabilities": {"computer_use": []}, "required_environment_variables": ["ANTHROPIC_API_KEY"]},
    }
    async def _provision(self, tools: Optional[List[Any]] = None) -> Tuple[Model, Optional[ModelSettings]]:
        final_api_key = self.api_key.get_secret_value() if self.api_key else os.getenv("ANTHROPIC_API_KEY")
        if not final_api_key:
            raise APIKeyMissingError("Anthropic", "ANTHROPIC_API_KEY")
        client = AsyncAnthropic(api_key=final_api_key)
        provider = AnthropicProvider(anthropic_client=client)
        agent_model = AnthropicModel(self.model_name, provider=provider)
        agent_settings = None
        if self.enable_reasoning:
            agent_settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': self.reasoning_budget_tokens}, parallel_tool_calls=False)
        return agent_model, agent_settings

class BedrockAnthropic(BaseAnthropic):
    """Configuration factory for AWS Bedrock Anthropic models."""
    _model_meta: Dict[str, Dict[str, Any]] = {
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {"pricing": {"input": 3.00, "output": 15.00}, "capabilities": {"computer_use": []}, "required_environment_variables": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"]},
    }
    async def _provision(self, tools: Optional[List[Any]] = None) -> Tuple[Model, Optional[ModelSettings]]:
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION")
        if not aws_access_key:
            raise ConfigurationError("AWS Configuration Error", "AWS_ACCESS_KEY_ID environment variable is not set")
        if not aws_secret_key:
            raise ConfigurationError("AWS Configuration Error", "AWS_SECRET_ACCESS_KEY environment variable is not set")  
        if not aws_region:
            raise ConfigurationError("AWS Configuration Error", "AWS_REGION environment variable is not set")
        client = AsyncAnthropicBedrock(aws_access_key=aws_access_key, aws_secret_key=aws_secret_key, aws_region=aws_region)
        provider = AnthropicProvider(anthropic_client=client)
        agent_model = AnthropicModel(self.model_name, provider=provider)
        agent_settings = None
        if self.enable_reasoning:
            agent_settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': self.reasoning_budget_tokens}, parallel_tool_calls=False)
        return agent_model, agent_settings

class Gemini(BaseModelProvider):
    """Configuration factory for Google's Gemini models."""
    _model_meta: Dict[str, Dict[str, Any]] = {
        "gemini-2.5-pro": {"pricing": {"input": 1.25, "output": 10.00}, "capabilities": {"image": ["png", "jpeg", "jpg", "webp", "heic", "heif"], "audio": ["wav", "mp3", "aiff", "aac", "ogg", "flac"], "video": ["mp4", "mpeg", "mpg", "mov", "avi", "flv", "webm", "wmv", "3gpp", "3gp"]}, "required_environment_variables": ["GOOGLE_GLA_API_KEY"]},
        "gemini-2.5-flash": {"pricing": {"input": 0.30, "output": 1.00}, "capabilities": {"image": ["png", "jpeg", "jpg", "webp", "heic", "heif"], "audio": ["wav", "mp3", "aiff", "aac", "ogg", "flac"], "video": ["mp4", "mpeg", "mpg", "mov", "avi", "flv", "webm", "wmv", "3gpp", "3gp"]}, "required_environment_variables": ["GOOGLE_GLA_API_KEY"]},
        "gemini-2.5-flash-lite": {"pricing": {"input": 0.10, "output": 0.40}, "capabilities": {"image": ["png", "jpeg", "jpg", "webp", "heic", "heif"], "audio": ["wav", "mp3", "aiff", "aac", "ogg", "flac"], "video": ["mp4", "mpeg", "mpg", "mov", "avi", "flv", "webm", "wmv", "3gpp", "3gp"]}, "required_environment_variables": ["GOOGLE_GLA_API_KEY"]},
        "gemini-2.0-flash": {"pricing": {"input": 0.10, "output": 0.40}, "capabilities": {"image": ["png", "jpeg", "jpg", "webp", "heic", "heif"], "audio": ["wav", "mp3", "aiff", "aac", "ogg", "flac"], "video": ["mp4", "mpeg", "mpg", "mov", "avi", "flv", "webm", "wmv", "3gpp", "3gp"]}, "required_environment_variables": ["GOOGLE_GLA_API_KEY"]},
        "gemini-1.5-pro": {"pricing": {"input": 1.25, "output": 5.00}, "capabilities": {"image": ["png", "jpeg", "jpg", "webp", "heic", "heif"], "audio": ["wav", "mp3", "aiff", "aac", "ogg", "flac"], "video": ["mp4", "mpeg", "mpg", "mov", "avi", "flv", "webm", "wmv", "3gpp", "3gp"]}, "required_environment_variables": ["GOOGLE_GLA_API_KEY"]},
        "gemini-1.5-flash": {"pricing": {"input": 0.075, "output": 0.30}, "capabilities": {"image": ["png", "jpeg", "jpg", "webp", "heic", "heif"], "audio": ["wav", "mp3", "aiff", "aac", "ogg", "flac"], "video": ["mp4", "mpeg", "mpg", "mov", "avi", "flv", "webm", "wmv", "3gpp", "3gp"]}, "required_environment_variables": ["GOOGLE_GLA_API_KEY"]},
    }
    model_name: str
    api_key: Optional[SecretStr] = Field(default=None)
    enable_reasoning: bool = Field(default=False)
    include_thoughts_in_reasoning: bool = Field(default=True)
    reasoning_budget: int = Field(default=0)
    
    def __init__(self, **data: Any):
        super().__init__(**data)

    @property
    def pricing(self) -> Dict[str, float]: return self._model_meta[self.model_name]['pricing']
    @property
    def capabilities(self) -> Dict[str, List[str]]: return self._model_meta[self.model_name]['capabilities']
    @property
    def required_environment_variables(self) -> List[str]: return self._model_meta[self.model_name]['required_environment_variables']

    @classmethod
    def list_available_models(cls) -> List[str]:
        """Return a list of available model names for this provider."""
        if hasattr(cls, '_model_meta') and cls._model_meta:
            return sorted(list(cls._model_meta.keys()))
        return []

    async def _provision(self, tools: Optional[List[Any]] = None) -> Tuple[Model, Optional[ModelSettings]]:
        final_api_key = self.api_key.get_secret_value() if self.api_key else os.getenv("GOOGLE_GLA_API_KEY")
        if not final_api_key:
            raise APIKeyMissingError("Gemini", "GOOGLE_GLA_API_KEY")
        provider = GoogleGLAProvider(api_key=final_api_key)
        agent_model = GeminiModel(self.model_name, provider=provider)
        agent_settings = None
        if self.enable_reasoning:
            agent_settings = GeminiModelSettings(gemini_thinking_config={'include_thoughts': self.include_thoughts_in_reasoning, 'thinking_budget': self.reasoning_budget})
        return agent_model, agent_settings