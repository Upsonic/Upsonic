"""Built-in tools for AI models.

These tools are passed directly to the model provider's API and are not
executed by the Upsonic framework. They represent native capabilities
provided by the model providers themselves.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Literal, Optional, TypedDict


__all__ = (
    'AbstractBuiltinTool',
    'WebSearchTool',
    'WebSearchUserLocation',
    'CodeExecutionTool',
    'UrlContextTool'
)


@dataclass(kw_only=True)
class AbstractBuiltinTool(ABC):
    """Abstract base class for built-in tools provided by model providers."""
    
    kind: str = 'unknown_builtin_tool'
    """Built-in tool identifier, used as a discriminator."""


@dataclass(kw_only=True)
class WebSearchTool(AbstractBuiltinTool):
    """A built-in tool that allows models to search the web for information.
    
    The exact parameters supported depend on the model provider:
    
    Supported by:
    - Anthropic
    - OpenAI Responses
    - Groq
    - Google
    """
    
    search_context_size: Literal['low', 'medium', 'high'] = 'medium'
    """Controls how much context is retrieved from web searches.
    
    Supported by:
    - OpenAI Responses
    """
    
    user_location: Optional['WebSearchUserLocation'] = None
    """Localizes search results based on user location.
    
    Supported by:
    - Anthropic
    - OpenAI Responses
    """
    
    blocked_domains: Optional[list[str]] = None
    """Domains to exclude from search results.
    
    Note: With Anthropic, you can only use one of blocked_domains or allowed_domains.
    
    Supported by:
    - Anthropic
    - Groq
    """
    
    allowed_domains: Optional[list[str]] = None
    """If provided, only these domains will be included in results.
    
    Note: With Anthropic, you can only use one of blocked_domains or allowed_domains.
    
    Supported by:
    - Anthropic
    - Groq
    """
    
    max_uses: Optional[int] = None
    """Maximum number of web searches allowed.
    
    Supported by:
    - Anthropic
    """
    
    kind: str = 'web_search'
    """The kind of tool."""


class WebSearchUserLocation(TypedDict, total=False):
    """User location information for localizing web search results.
    
    Supported by:
    - Anthropic
    - OpenAI Responses
    """
    
    city: str
    """The city where the user is located."""
    
    country: str
    """The country where the user is located.
    For OpenAI, this must be a 2-letter country code (e.g., 'US', 'GB').
    """
    
    region: str
    """The region or state where the user is located."""
    
    timezone: str
    """The timezone of the user's location."""


@dataclass(kw_only=True)
class CodeExecutionTool(AbstractBuiltinTool):
    """A built-in tool that allows models to execute code.
    
    Supported by:
    - Anthropic
    - OpenAI Responses
    - Google
    """
    
    kind: str = 'code_execution'
    """The kind of tool."""


@dataclass(kw_only=True)
class UrlContextTool(AbstractBuiltinTool):
    """Allows models to access contents from URLs.
    
    Supported by:
    - Google
    """
    
    kind: str = 'url_context'
    """The kind of tool."""
