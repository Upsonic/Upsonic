"""
Factory for creating model providers.
"""

from typing import Dict, Type, Any, Optional
from .base import ModelProvider
from .providers.openai import OpenAIProvider


class ModelFactory:
    """
    Factory class for creating model providers.
    """
    
    _providers: Dict[str, Type[ModelProvider]] = {
        "openai": OpenAIProvider,
    }
    
    @classmethod
    def create_provider(
        self, 
        provider_name: str, 
        model_name: str, 
        **kwargs: Any
    ) -> ModelProvider:
        """
        Create a model provider instance.
        
        Args:
            provider_name: Name of the provider (e.g., 'openai')
            model_name: Name of the model to use
            **kwargs: Additional arguments for the provider
            
        Returns:
            ModelProvider instance
            
        Raises:
            ValueError: If provider is not supported
        """
        if provider_name not in self._providers:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        provider_class = self._providers[provider_name]
        return provider_class(model_name=model_name, **kwargs)
    
    @classmethod
    def register_provider(
        self, 
        name: str, 
        provider_class: Type[ModelProvider]
    ) -> None:
        """
        Register a new provider.
        
        Args:
            name: Name of the provider
            provider_class: Provider class to register
        """
        self._providers[name] = provider_class
    
    @classmethod
    def get_supported_providers(self) -> list[str]:
        """
        Get list of supported provider names.
        
        Returns:
            List of provider names
        """
        return list(self._providers.keys())
