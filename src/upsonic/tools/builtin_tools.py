"""Built-in tools for AI models.

These tools are passed directly to the model provider's API and are not
executed by the Upsonic framework. They represent native capabilities
provided by the model providers themselves.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Literal, Optional, TypedDict
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    _REQUESTS_AVAILABLE = False


try:
    from bs4 import BeautifulSoup
    _BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    _BEAUTIFULSOUP_AVAILABLE = False


try:
    try:
        from ddgs import DDGS
    except ImportError:  # Fallback for older versions of ddgs
        from duckduckgo_search import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    DDGS = None
    _DDGS_AVAILABLE = False


__all__ = (
    'AbstractBuiltinTool',
    'WebSearchTool',
    'WebSearchUserLocation',
    'CodeExecutionTool',
    'UrlContextTool',
    'WebSearch',
    'WebRead'
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


def WebSearch(query: str, max_results: int = 10) -> str:
    """
    Search the web for the given query and return formatted results.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 10)

    Returns:
        Formatted string containing search results
    """
    if not _DDGS_AVAILABLE:
        from upsonic.utils.printing import import_error
        import_error(
            package_name="duckduckgo-search",
            install_command='pip install "upsonic[tools]"',
            feature_name="DuckDuckGo search tool"
        )

    with DDGS() as ddgs:
        try:
            results = list(ddgs.text(query, max_results=max_results))
            
            formatted_results = f"Web search results for: {query}\n\n"
            for i, result in enumerate(results, 1):
                formatted_results += f"{i}. {result.get('title', 'No title')}\n"
                formatted_results += f"   URL: {result.get('href', 'No URL')}\n"
                formatted_results += f"   Description: {result.get('body', 'No description')}\n\n"
            
            return formatted_results
        except Exception as e:
            return f"Error performing web search: {str(e)}"


def WebRead(url: str) -> str:
    """
    Read and extract text content from a web page.

    Args:
        url: The URL to read from

    Returns:
        Extracted text content from the webpage
    """
    if not _REQUESTS_AVAILABLE:
        from upsonic.utils.printing import import_error
        import_error(
            package_name="requests",
            install_command='pip install "upsonic[loaders]"',
            feature_name="WebRead tool"
        )

    if not _BEAUTIFULSOUP_AVAILABLE:
        from upsonic.utils.printing import import_error
        import_error(
            package_name="beautifulsoup4",
            install_command='pip install "upsonic[loaders]"',
            feature_name="WebRead tool"
        )

    session = requests.Session()
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        if len(text) > 5000:
            text = text[:5000] + "... [Content truncated]"
        
        return f"Content from {url}:\n\n{text}"
    except requests.exceptions.RequestException as e:
        return f"Error reading from {url}: {str(e)}"
    except Exception as e:
        return f"Error processing content from {url}: {str(e)}"
    finally:
        session.close()