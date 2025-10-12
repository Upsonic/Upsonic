from dataclasses import dataclass
from typing import Literal

from pydantic import TypeAdapter
from typing_extensions import TypedDict

from upsonic.tools import tool

try:
    from tavily import AsyncTavilyClient
    _TAVILY_AVAILABLE = True
except ImportError:
    AsyncTavilyClient = None
    _TAVILY_AVAILABLE = False


__all__ = ('tavily_search_tool',)


class TavilySearchResult(TypedDict):
    """A Tavily search result.

    See [Tavily Search Endpoint documentation](https://docs.tavily.com/api-reference/endpoint/search)
    for more information.
    """

    title: str
    """The title of the search result."""
    url: str
    """The URL of the search result.."""
    content: str
    """A short description of the search result."""
    score: float
    """The relevance score of the search result."""


tavily_search_ta = TypeAdapter(list[TavilySearchResult])


@dataclass
class TavilySearchTool:
    """The Tavily search tool."""

    client: AsyncTavilyClient
    """The Tavily search client."""

    async def __call__(
        self,
        query: str,
        search_deep: Literal['basic', 'advanced'] = 'basic',
        topic: Literal['general', 'news'] = 'general',
        time_range: Literal['day', 'week', 'month', 'year', 'd', 'w', 'm', 'y'] | None = None,
    ):
        """Searches Tavily for the given query and returns the results.

        Args:
            query: The search query to execute with Tavily.
            search_deep: The depth of the search.
            topic: The category of the search.
            time_range: The time range back from the current date to filter results.

        Returns:
            The search results.
        """
        results = await self.client.search(query, search_depth=search_deep, topic=topic, time_range=time_range)  # type: ignore[reportUnknownMemberType]
        return tavily_search_ta.validate_python(results['results'])


def tavily_search_tool(api_key: str):
    """Creates a Tavily search tool.

    Args:
        api_key: The Tavily API key.

            You can get one by signing up at [https://app.tavily.com/home](https://app.tavily.com/home).
    """
    if not _TAVILY_AVAILABLE:
        from upsonic.utils.printing import import_error
        import_error(
            package_name="tavily-python",
            install_command='pip install "upsonic[tools]"',
            feature_name="Tavily search tool"
        )

    # Create the tool instance
    tavily_tool = TavilySearchTool(client=AsyncTavilyClient(api_key))
    
    # Create a wrapper function instead of decorating the bound method directly
    @tool
    async def tavily_search(
        query: str,
        search_deep: Literal['basic', 'advanced'] = 'basic',
        topic: Literal['general', 'news'] = 'general',
        time_range: Literal['day', 'week', 'month', 'year', 'd', 'w', 'm', 'y'] | None = None,
    ) -> list[TavilySearchResult]:
        """Searches Tavily for the given query and returns the results.

        Args:
            query: The search query to execute with Tavily.
            search_deep: The depth of the search.
            topic: The category of the search.
            time_range: The time range back from the current date to filter results.

        Returns:
            The search results.
        """
        return await tavily_tool(query, search_deep, topic, time_range)
    
    return tavily_search