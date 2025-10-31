from __future__ import annotations
import functools
from dataclasses import KW_ONLY, dataclass
import warnings

import anyio
import anyio.to_thread
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from upsonic.tools import tool

try:
    try:
        from ddgs import DDGS
        _DDGS_PROVIDER = "ddgs"
    except ImportError:  # Fallback for older versions / alt package name
        from duckduckgo_search import DDGS
        _DDGS_PROVIDER = "duckduckgo_search"
    _DDGS_AVAILABLE = True
except ImportError:
    DDGS = None
    _DDGS_AVAILABLE = False
    _DDGS_PROVIDER = None


__all__ = ('duckduckgo_search_tool',)


class DuckDuckGoResult(TypedDict):
    """A DuckDuckGo search result."""

    title: str
    """The title of the search result."""
    href: str
    """The URL of the search result."""
    body: str
    """The body of the search result."""


duckduckgo_ta = TypeAdapter(list[DuckDuckGoResult])


@dataclass
class DuckDuckGoSearchTool:
    """The DuckDuckGo search tool."""

    client: DDGS
    """The DuckDuckGo search client."""

    _: KW_ONLY

    max_results: int | None
    """The maximum number of results. If None, returns results only from the first response."""

    async def __call__(self, query: str) -> list[DuckDuckGoResult]:
        """Searches DuckDuckGo for the given query and returns the results.

        Args:
            query: The query to search for.

        Returns:
            The search results.
        """
        search = functools.partial(self.client.text, max_results=self.max_results)
        results = await anyio.to_thread.run_sync(search, query)
        return duckduckgo_ta.validate_python(results)


def duckduckgo_search_tool(duckduckgo_client: DDGS | None = None, max_results: int | None = None):
    """Creates a DuckDuckGo search tool.

    Args:
        duckduckgo_client: The DuckDuckGo search client.
        max_results: The maximum number of results. If None, returns results only from the first response.
    """
    if not _DDGS_AVAILABLE:
        from upsonic.utils.printing import import_error
        import_error(
            package_name="duckduckgo-search",
            install_command='pip install "upsonic[tools]"',
            feature_name="DuckDuckGo search tool"
        )

    # Prepare client and create the tool instance, suppressing rename warnings if needed
    if duckduckgo_client is None:
        if _DDGS_PROVIDER == "duckduckgo_search":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    module=r"duckduckgo_search(\..*)?$",
                )
                client = DDGS()
        else:
            client = DDGS()
    else:
        client = duckduckgo_client

    ddg_tool = DuckDuckGoSearchTool(client=client, max_results=max_results)
    
    # Create a wrapper function instead of decorating the bound method directly
    @tool
    async def duckduckgo_search(query: str) -> list[DuckDuckGoResult]:
        """Searches DuckDuckGo for the given query and returns the results.

        Args:
            query: The query to search for.

        Returns:
            The search results.
        """
        return await ddg_tool(query)
    
    return duckduckgo_search