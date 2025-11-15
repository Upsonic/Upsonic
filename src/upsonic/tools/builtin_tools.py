"""Built-in tools for AI models.

These tools are passed directly to the model provider's API and are not
executed by the Upsonic framework. They represent native capabilities
provided by the model providers themselves.
"""

from __future__ import annotations as _annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

import pydantic
from pydantic_core import core_schema
from typing_extensions import TypedDict

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
    'WebRead',
    'ImageGenerationTool',
    'MemoryTool',
    'MCPServerTool',
)


_BUILTIN_TOOL_TYPES: dict[str, type[AbstractBuiltinTool]] = {}


@dataclass(kw_only=True)
class AbstractBuiltinTool(ABC):
    """A builtin tool that can be used by an agent.

    This class is abstract and cannot be instantiated directly.

    The builtin tools are passed to the model as part of the `ModelRequestParameters`.
    """

    kind: str = 'unknown_builtin_tool'
    """Built-in tool identifier, this should be available on all built-in tools as a discriminator."""

    @property
    def unique_id(self) -> str:
        """A unique identifier for the builtin tool.

        If multiple instances of the same builtin tool can be passed to the model, subclasses should override this property to allow them to be distinguished.
        """
        return self.kind

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _BUILTIN_TOOL_TYPES[cls.kind] = cls

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        if cls is not AbstractBuiltinTool:
            return handler(cls)

        tools = _BUILTIN_TOOL_TYPES.values()
        if len(tools) == 1:  # pragma: no cover
            tools_type = next(iter(tools))
        else:
            tools_annotated = [Annotated[tool, pydantic.Tag(tool.kind)] for tool in tools]
            tools_type = Annotated[Union[tuple(tools_annotated)], pydantic.Discriminator(_tool_discriminator)]  # noqa: UP007

        return handler(tools_type)


@dataclass(kw_only=True)
class WebSearchTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to search the web for information.

    The parameters that PydanticAI passes depend on the model, as some parameters may not be supported by certain models.

    Supported by:

    * Anthropic
    * OpenAI Responses
    * Groq
    * Google
    """

    search_context_size: Literal['low', 'medium', 'high'] = 'medium'
    """The `search_context_size` parameter controls how much context is retrieved from the web to help the tool formulate a response.

    Supported by:

    * OpenAI Responses
    """

    user_location: WebSearchUserLocation | None = None
    """The `user_location` parameter allows you to localize search results based on a user's location.

    Supported by:

    * Anthropic
    * OpenAI Responses
    """

    blocked_domains: list[str] | None = None
    """If provided, these domains will never appear in results.

    With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

    Supported by:

    * Anthropic, see <https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool#domain-filtering>
    * Groq, see <https://console.groq.com/docs/agentic-tooling#search-settings>
    """

    allowed_domains: list[str] | None = None
    """If provided, only these domains will be included in results.

    With Anthropic, you can only use one of `blocked_domains` or `allowed_domains`, not both.

    Supported by:

    * Anthropic, see <https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool#domain-filtering>
    * Groq, see <https://console.groq.com/docs/agentic-tooling#search-settings>
    """

    max_uses: int | None = None
    """If provided, the tool will stop searching the web after the given number of uses.

    Supported by:

    * Anthropic
    """

    kind: str = 'web_search'
    """The kind of tool."""


class WebSearchUserLocation(TypedDict, total=False):
    """Allows you to localize search results based on a user's location.

    Supported by:

    * Anthropic
    * OpenAI Responses
    """

    city: str
    """The city where the user is located."""

    country: str
    """The country where the user is located. For OpenAI, this must be a 2-letter country code (e.g., 'US', 'GB')."""

    region: str
    """The region or state where the user is located."""

    timezone: str
    """The timezone of the user's location."""


@dataclass(kw_only=True)
class CodeExecutionTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to execute code.

    Supported by:

    * Anthropic
    * OpenAI Responses
    * Google
    """

    kind: str = 'code_execution'
    """The kind of tool."""


@dataclass(kw_only=True)
class UrlContextTool(AbstractBuiltinTool):
    """Allows your agent to access contents from URLs.

    Supported by:

    * Google
    """

    kind: str = 'url_context'
    """The kind of tool."""

@dataclass(kw_only=True)
class ImageGenerationTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to generate images.

    Supported by:

    * OpenAI Responses
    * Google
    """

    background: Literal['transparent', 'opaque', 'auto'] = 'auto'
    """Background type for the generated image.

    Supported by:

    * OpenAI Responses. 'transparent' is only supported for 'png' and 'webp' output formats.
    """

    input_fidelity: Literal['high', 'low'] | None = None
    """
    Control how much effort the model will exert to match the style and features,
    especially facial features, of input images.

    Supported by:

    * OpenAI Responses. Default: 'low'.
    """

    moderation: Literal['auto', 'low'] = 'auto'
    """Moderation level for the generated image.

    Supported by:

    * OpenAI Responses
    """

    output_compression: int = 100
    """Compression level for the output image.

    Supported by:

    * OpenAI Responses. Only supported for 'png' and 'webp' output formats.
    """

    output_format: Literal['png', 'webp', 'jpeg'] | None = None
    """The output format of the generated image.

    Supported by:

    * OpenAI Responses. Default: 'png'.
    """

    partial_images: int = 0
    """
    Number of partial images to generate in streaming mode.

    Supported by:

    * OpenAI Responses. Supports 0 to 3.
    """

    quality: Literal['low', 'medium', 'high', 'auto'] = 'auto'
    """The quality of the generated image.

    Supported by:

    * OpenAI Responses
    """

    size: Literal['1024x1024', '1024x1536', '1536x1024', 'auto'] = 'auto'
    """The size of the generated image.

    Supported by:

    * OpenAI Responses
    """

    kind: str = 'image_generation'
    """The kind of tool."""

@dataclass(kw_only=True)
class MemoryTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to use memory.

    Supported by:

    * Anthropic
    """

    kind: str = 'memory'
    """The kind of tool."""


@dataclass(kw_only=True)
class MCPServerTool(AbstractBuiltinTool):
    """A builtin tool that allows your agent to use MCP servers.

    Supported by:

    * OpenAI Responses
    * Anthropic
    """

    id: str
    """A unique identifier for the MCP server."""

    url: str
    """The URL of the MCP server to use.

    For OpenAI Responses, it is possible to use `connector_id` by providing it as `x-openai-connector:<connector_id>`.
    """

    authorization_token: str | None = None
    """Authorization header to use when making requests to the MCP server.

    Supported by:

    * OpenAI Responses
    * Anthropic
    """

    description: str | None = None
    """A description of the MCP server.

    Supported by:

    * OpenAI Responses
    """

    allowed_tools: list[str] | None = None
    """A list of tools that the MCP server can use.

    Supported by:

    * OpenAI Responses
    * Anthropic
    """

    headers: dict[str, str] | None = None
    """Optional HTTP headers to send to the MCP server.

    Use for authentication or other purposes.

    Supported by:

    * OpenAI Responses
    """

    kind: str = 'mcp_server'

    @property
    def unique_id(self) -> str:
        return ':'.join([self.kind, self.id])


def _tool_discriminator(tool_data: dict[str, Any] | AbstractBuiltinTool) -> str:
    if isinstance(tool_data, dict):
        return tool_data.get('kind', AbstractBuiltinTool.kind)
    else:
        return tool_data.kind


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