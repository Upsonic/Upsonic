from typing import Dict, List

try:
    from duckduckgo_search import DDGS
except ImportError:
    print(
        "duckduckgo-search is not installed. Please install it using 'pip install duckduckgo-search'"
    )
    exit(1)

from upsonic.tools.decorators.tool_decorator import tool
from upsonic.tools.base import Toolkit


class DuckDuckGoSearchTool(Toolkit):
    def __init__(self):
        """Initialize DuckDuckGo search tool."""
        super().__init__(name="DuckDuckGoSearchTool")

    def get_description(self) -> str:
        """Return the description of the tool."""
        return "Search DuckDuckGo for the given query and return text results."

    @tool(description="Search DuckDuckGo for the given query and return text results.")
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo for the given query and return text results.

        Args:
            query: The search query
            max_results: Maximum number of results to return (default: 10)

        Returns:
            A list of dictionaries containing search results with keys: title, body, href
        """
        try:
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "body": r.get("body", ""),
                            "href": r.get("href", ""),
                        }
                    )
                return results
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]
