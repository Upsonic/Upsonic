import requests
import json
from typing import Dict, Any, Literal, Optional, List, Union

import os

try:
    from tavily import TavilyClient
except ImportError:
    print(
        "tavily-python is not installed. Please install it using 'pip install tavily-python'"
    )
    exit(1)

from upsonic.tools.decorators.tool_decorator import tool
from upsonic.tools.base import Toolkit


class TavilySearchTool(Toolkit):
    """Tool for searching the web with Tavily Search API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_mode: Literal["detailed", "context"] = "detailed",
        max_tokens: int = 6000,
        include_answer: bool = True,
        search_depth: Literal["basic", "advanced"] = "advanced",
        format: Literal["json", "markdown"] = "markdown",
        max_results: int = 5,
    ):
        """
        Initialize the Tavily search tool.

        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY environment variable)
            search_mode: Use "detailed" for structured results or "context" for raw search context
            max_tokens: Maximum tokens in response
            include_answer: Whether to include AI-generated answer summary
            search_depth: Search depth ("basic" or "advanced")
            format: Response format ("json" or "markdown")
            max_results: Maximum number of results to return in detailed mode
        """
        super().__init__(name="tavily_search")
        self._search_mode = search_mode

        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.search_depth: Literal["basic", "advanced"] = search_depth
        self.max_tokens = max_tokens
        self.include_answer = include_answer
        self.format = format
        self.max_results = max_results
        self.client: Optional[TavilyClient] = None

    def __control__(self) -> bool:
        """
        Validate that the toolkit is ready to use.

        Returns:
            True if API key is available and client can be initialized
        """
        if not self.api_key:
            print(
                "Warning: TAVILY_API_KEY not found. Please set it as environment variable or pass it during initialization."
            )
            return False

        try:
            if not self.client:
                self.client = TavilyClient(api_key=self.api_key)
            return True
        except Exception as e:
            print(f"Warning: Failed to initialize Tavily client: {e}")
            return False

    def get_description(self) -> str:
        return "Tool for searching the web with Tavily Search API"

    def _ensure_client(self) -> TavilyClient:
        """Ensure client is initialized before use"""
        if not self.client:
            if not self.api_key:
                raise ValueError("TAVILY_API_KEY not provided")
            self.client = TavilyClient(api_key=self.api_key)
        return self.client

    @tool(description="Search the web with detailed, structured results")
    def search_detailed(self, query: str, max_results: Optional[int] = None) -> str:
        """
        Search the web with detailed, structured results.

        Args:
            query: The search query
            max_results: Maximum number of results (overrides instance default if provided)

        Returns:
            Formatted search results as JSON or markdown
        """
        client = self._ensure_client()
        max_results = max_results or self.max_results

        response = client.search(
            query=query,
            search_depth=self.search_depth,
            include_answer=self.include_answer,
            max_results=max_results,
        )

        return self._format_detailed_results(query, response)

    @tool(description="Search the web and return raw context information")
    def search_context(self, query: str) -> str:
        """
        Search the web and return raw context information.

        Args:
            query: The search query

        Returns:
            Raw search context as a string
        """
        client = self._ensure_client()
        return client.get_search_context(
            query=query,
            search_depth=self.search_depth,
            max_tokens=self.max_tokens,
        )

    def _format_detailed_results(self, query: str, response: Dict[str, Any]) -> str:
        """
        Format the detailed search results based on the configured format.

        Args:
            query: The original search query
            response: The raw response from Tavily API

        Returns:
            Formatted results as JSON or markdown
        """
        clean_response: Dict[str, Any] = {"query": query}
        if "answer" in response and self.include_answer:
            clean_response["answer"] = response["answer"]

        clean_results = self._process_results_with_token_limit(
            response.get("results", [])
        )
        clean_response["results"] = clean_results

        if self.format == "json":
            return json.dumps(clean_response) if clean_response else "No results found."
        elif self.format == "markdown":
            return self._convert_to_markdown(query, clean_response)
        else:
            return json.dumps(clean_response)

    def _process_results_with_token_limit(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process results while respecting token limit.

        Args:
            results: List of raw result items

        Returns:
            List of processed result items within token limit
        """
        clean_results = []
        current_token_count = 0

        for result in results:
            clean_result = {
                "title": result["title"],
                "url": result["url"],
                "content": result["content"],
                "score": result.get("score", 0),
            }

            result_tokens = len(json.dumps(clean_result))
            if current_token_count + result_tokens > self.max_tokens:
                break

            current_token_count += result_tokens
            clean_results.append(clean_result)

        return clean_results

    def _convert_to_markdown(self, query: str, data: Dict[str, Any]) -> str:
        """
        Convert structured data to markdown format.

        Args:
            query: The original search query
            data: Structured search results

        Returns:
            Markdown formatted string
        """
        markdown = f"# Search Results: {query}\n\n"

        if "answer" in data:
            markdown += "## Summary\n"
            markdown += f"{data['answer']}\n\n"

        if data["results"]:
            markdown += "## Sources\n\n"
            for idx, result in enumerate(data["results"], 1):
                markdown += f"### {idx}. [{result['title']}]({result['url']})\n"
                markdown += f"{result['content']}\n\n"
        else:
            markdown += "No results found."

        return markdown
