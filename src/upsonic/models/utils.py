import os
from typing import List, Dict, Type
from decimal import Decimal

from upsonic.models.base import BaseModelProvider
from upsonic.models.providers import (
    OpenAI, AzureOpenAI, Ollama, Anthropic, Gemini 
)

from upsonic.utils.package.exception import ConfigurationError


ALL_PROVIDER_CLASSES = [OpenAI, AzureOpenAI, Ollama, Anthropic, Gemini]


def list_available_models(provider_class: Type[BaseModelProvider]) -> List[str]:
    """
    Lists the officially supported model names for a given provider class.

    Args:
        provider_class: The provider class itself (e.g., OpenAI, not an instance).

    Returns:
        A sorted list of supported model name strings.
    """
    model_meta = getattr(provider_class, '_model_meta', None)
    if not model_meta or not hasattr(model_meta, 'default'):
        return []
    return sorted(list(model_meta.default.keys()))


def get_all_supported_models() -> Dict[str, List[str]]:
    """
    Returns a complete dictionary of all supported models, grouped by provider.

    Returns:
        A dictionary where keys are provider class names and values are
        lists of their supported model names.
    """
    return {
        provider.__name__: list_available_models(provider)
        for provider in ALL_PROVIDER_CLASSES
    }


def check_provider_environment(model_provider: BaseModelProvider) -> None:
    """
    Performs a pre-flight check to ensure all required environment variables
    for a given provider instance are set.

    Args:
        model_provider: An instantiated model provider object.

    Raises:
        ConfigurationError: If any required environment variables are not set.
    """
    missing_vars = []
    for var_name in model_provider.required_environment_variables:
        if not os.getenv(var_name):
            missing_vars.append(var_name)
    
    if missing_vars:
        raise ConfigurationError(
            f"Missing required environment variables for provider '{type(model_provider).__name__}': "
            f"{', '.join(missing_vars)}. Please set them in your environment."
        )


def get_estimated_cost(
    input_tokens: int, 
    output_tokens: int, 
    model_provider: BaseModelProvider
) -> str:
    """
    Calculates the estimated cost for token usage using a configured model
    provider instance.

    This function is instance-aware and accesses pricing information directly
    from the provider object.

    Args:
        input_tokens: The number of input tokens used.
        output_tokens: The number of output tokens generated.
        model_provider: The instantiated BaseModelProvider object (e.g., OpenAI, Anthropic).

    Returns:
        A formatted string representing the estimated cost (e.g., "~$0.0005").
    """
    pricing = model_provider.pricing
    
    if not pricing or "input" not in pricing or "output" not in pricing:
        return "Cost Unknown"

    input_tokens_millions = Decimal(str(input_tokens)) / Decimal('1000000')
    output_tokens_millions = Decimal(str(output_tokens)) / Decimal('1000000')
    
    input_cost = Decimal(str(pricing["input"])) * input_tokens_millions
    output_cost = Decimal(str(pricing["output"])) * output_tokens_millions
    total_cost = input_cost + output_cost

    return f"~${float(round(total_cost, 4))}"

