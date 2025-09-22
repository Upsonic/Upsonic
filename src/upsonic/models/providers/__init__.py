"""
Model providers for the Upsonic framework.
"""

from .openai import OpenAIProvider, OpenAIModelSettings

# For backward compatibility, create an alias
OpenAI = OpenAIProvider

__all__ = [
    "OpenAIProvider",
    "OpenAIModelSettings", 
    "OpenAI",  # Alias for backward compatibility
]
