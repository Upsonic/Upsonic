"""
Model Factory for creating model providers from string specifications.
"""

from typing import Union, Dict, Type, Optional
from .base import BaseModelProvider
from .providers import (
    OpenAI,
    AzureOpenAI,
    Deepseek,
    OpenRouter,
    Ollama,
    Anthropic,
    BedrockAnthropic,
    Gemini
)
from .utils import ConfigurationError


class ModelFactory:
    """
    Factory class for creating model providers from string specifications.
    
    Supports formats like:
    - "openai/gpt-4o"
    - "anthropic/claude-3-5-sonnet-latest"
    - "gemini/gemini-2.5-pro"
    """
    
    PROVIDER_MAP: Dict[str, Type[BaseModelProvider]] = {
        "openai": OpenAI,
        "azure_openai": AzureOpenAI,
        "deepseek": Deepseek,
        "openrouter": OpenRouter,
        "ollama": Ollama,
        "anthropic": Anthropic,
        "bedrock_anthropic": BedrockAnthropic,
        "gemini": Gemini,
    }
    
    ALIAS_MAP: Dict[str, str] = {
        "azure": "azure_openai",
        "claude": "anthropic",
        "google": "gemini",
    }
    
    @classmethod
    def create(cls, model_spec: Union[str, BaseModelProvider], **kwargs) -> BaseModelProvider:
        """
        Create a model provider from a string specification or return the provider directly.
        
        Args:
            model_spec: Either a string like "openai/gpt-4o" or a BaseModelProvider instance
            **kwargs: Additional arguments to pass to the provider constructor
            
        Returns:
            An instance of the appropriate BaseModelProvider subclass
            
        Raises:
            ValueError: If the model specification is invalid
            ConfigurationError: If the provider or model is not supported
        """
        if isinstance(model_spec, BaseModelProvider):
            return model_spec
            
        if isinstance(model_spec, str):
            return cls._parse_model_string(model_spec, **kwargs)
            
        raise ValueError(f"Invalid model specification: {model_spec}. Must be a string or BaseModelProvider instance.")
    
    @classmethod
    def _parse_model_string(cls, model_string: str, **kwargs) -> BaseModelProvider:
        """
        Parse a model string like "openai/gpt-4o" and return the appropriate provider.
        
        Args:
            model_string: String in format "provider/model_name"
            **kwargs: Additional arguments for the provider
            
        Returns:
            An instance of the appropriate BaseModelProvider subclass
        """
        if "/" not in model_string:
            raise ValueError(
                f"Invalid model string format: {model_string}. "
                f"Expected format: 'provider/model_name' (e.g., 'openai/gpt-4o')"
            )
        
        provider_name, model_name = model_string.split("/", 1)
        
        provider_name = provider_name.lower().strip()
        
        if provider_name in cls.ALIAS_MAP:
            provider_name = cls.ALIAS_MAP[provider_name]
        
        if provider_name not in cls.PROVIDER_MAP:
            available_providers = ", ".join(sorted(cls.PROVIDER_MAP.keys()))
            raise ConfigurationError(
                f"Unknown provider '{provider_name}'. "
                f"Available providers: {available_providers}"
            )
        
        provider_class = cls.PROVIDER_MAP[provider_name]
        
        try:
            return provider_class(model_name=model_name, **kwargs)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create {provider_name} provider with model '{model_name}': {str(e)}"
            )
    
    @classmethod
    def list_supported_providers(cls) -> list[str]:
        """Return a list of all supported provider names."""
        return sorted(cls.PROVIDER_MAP.keys())
    
    @classmethod
    def list_supported_models(cls, provider_name: str) -> list[str]:
        """Return a list of supported models for a given provider."""
        provider_name = provider_name.lower().strip()
        
        if provider_name in cls.ALIAS_MAP:
            provider_name = cls.ALIAS_MAP[provider_name]
            
        if provider_name not in cls.PROVIDER_MAP:
            available_providers = ", ".join(sorted(cls.PROVIDER_MAP.keys()))
            raise ConfigurationError(
                f"Unknown provider '{provider_name}'. "
                f"Available providers: {available_providers}"
            )
        
        provider_class = cls.PROVIDER_MAP[provider_name]
        from .utils import list_available_models
        return list_available_models(provider_class)
