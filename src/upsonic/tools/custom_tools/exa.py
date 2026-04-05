"""
Exa Web Search & Content Retrieval Toolkit for Upsonic Framework.

This module provides comprehensive Exa API integration with support for:
- Neural/keyword/auto web search with inline content retrieval
- Fetching clean, parsed content from URLs
- Finding semantically similar pages to a given URL
- Getting LLM-generated answers with citations
- Domain, date, and text filtering for precise results
- Highlights extraction for token-efficient LLM consumption

Required Environment Variables:
-----------------------------
- EXA_API_KEY: Exa API key from https://dashboard.exa.ai

Example Usage:
    ```python
    from upsonic.tools.custom_tools.exa import ExaTools

    tools = ExaTools(api_key="your-exa-api-key")
    result = tools.search("latest AI research papers")
    ```
"""

import json
from os import getenv
from typing import Any, Dict, List, Optional

from upsonic.tools.base import ToolKit
from upsonic.tools.config import tool
from upsonic.utils.integrations.exa import serialize_exa_response
from upsonic.utils.printing import error_log

try:
    from exa_py import Exa
    _EXA_AVAILABLE = True
except ImportError:
    Exa = None
    _EXA_AVAILABLE = False


class ExaTools(ToolKit):
    """Exa web search, content retrieval, and answer toolkit."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_num_results: int = 10,
        default_search_type: str = "auto",
        default_text: bool = True,
        default_highlights: bool = False,
        default_summary: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the ExaTools toolkit.

        Args:
            api_key: Exa API key. Falls back to EXA_API_KEY env var.
            default_num_results: Default number of results for search operations.
            default_search_type: Default search type (auto, neural, keyword).
            default_text: Whether to include text content by default.
            default_highlights: Whether to include highlights by default.
            default_summary: Whether to include summaries by default.
            **kwargs: ToolKit params (include_tools, exclude_tools, timeout, etc.).
        """
        super().__init__(**kwargs)

        if not _EXA_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="exa-py",
                install_command="pip install exa-py",
                feature_name="Exa tools",
            )

        self.api_key: str = api_key or getenv("EXA_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Exa API key is required. Set EXA_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.default_num_results: int = default_num_results
        self.default_search_type: str = default_search_type
        self.default_text: bool = default_text
        self.default_highlights: bool = default_highlights
        self.default_summary: bool = default_summary

        self.client: Any = Exa(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Helper to build ContentsOptions dict
    # ------------------------------------------------------------------

    def _build_contents_options(
        self,
        text: Optional[bool] = None,
        highlights: Optional[bool] = None,
        summary: Optional[bool] = None,
        max_characters: Optional[int] = None,
        max_age_hours: Optional[int] = None,
        subpages: Optional[int] = None,
        subpage_target: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a ContentsOptions dict for search/find_similar calls.

        Returns None if no content options are requested (contents=False behavior).
        """
        opts: Dict[str, Any] = {}

        use_text = text if text is not None else self.default_text
        if use_text:
            if max_characters is not None:
                opts["text"] = {"max_characters": max_characters}
            else:
                opts["text"] = True

        use_highlights = highlights if highlights is not None else self.default_highlights
        if use_highlights:
            opts["highlights"] = True

        use_summary = summary if summary is not None else self.default_summary
        if use_summary:
            opts["summary"] = True

        if max_age_hours is not None:
            opts["max_age_hours"] = max_age_hours
        if subpages is not None:
            opts["subpages"] = subpages
        if subpage_target is not None:
            opts["subpage_target"] = subpage_target

        if not opts:
            return None
        return opts

    # ------------------------------------------------------------------
    # Async implementations
    # ------------------------------------------------------------------

    async def asearch(
        self,
        query: str,
        num_results: Optional[int] = None,
        search_type: Optional[str] = None,
        category: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        text: Optional[bool] = None,
        highlights: Optional[bool] = None,
        summary: Optional[bool] = None,
        max_characters: Optional[int] = None,
        max_age_hours: Optional[int] = None,
    ) -> str:
        try:
            import asyncio
            import functools
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                functools.partial(
                    self._search_impl,
                    query=query,
                    num_results=num_results,
                    search_type=search_type,
                    category=category,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    start_published_date=start_published_date,
                    end_published_date=end_published_date,
                    include_text=include_text,
                    exclude_text=exclude_text,
                    text=text,
                    highlights=highlights,
                    summary=summary,
                    max_characters=max_characters,
                    max_age_hours=max_age_hours,
                ),
            )
            return result
        except Exception as e:
            error_log(f"Exa async search error: {e}")
            return json.dumps({"error": str(e)})

    async def aget_contents(
        self,
        urls: List[str],
        text: Optional[bool] = None,
        highlights: Optional[bool] = None,
        summary: Optional[bool] = None,
        max_characters: Optional[int] = None,
        max_age_hours: Optional[int] = None,
        subpages: Optional[int] = None,
        subpage_target: Optional[str] = None,
    ) -> str:
        try:
            import asyncio
            import functools
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                functools.partial(
                    self._get_contents_impl,
                    urls=urls,
                    text=text,
                    highlights=highlights,
                    summary=summary,
                    max_characters=max_characters,
                    max_age_hours=max_age_hours,
                    subpages=subpages,
                    subpage_target=subpage_target,
                ),
            )
            return result
        except Exception as e:
            error_log(f"Exa async get_contents error: {e}")
            return json.dumps({"error": str(e)})

    async def afind_similar(
        self,
        url: str,
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        text: Optional[bool] = None,
        highlights: Optional[bool] = None,
        summary: Optional[bool] = None,
        max_characters: Optional[int] = None,
    ) -> str:
        try:
            import asyncio
            import functools
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                functools.partial(
                    self._find_similar_impl,
                    url=url,
                    num_results=num_results,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    start_published_date=start_published_date,
                    end_published_date=end_published_date,
                    text=text,
                    highlights=highlights,
                    summary=summary,
                    max_characters=max_characters,
                ),
            )
            return result
        except Exception as e:
            error_log(f"Exa async find_similar error: {e}")
            return json.dumps({"error": str(e)})

    async def aanswer(
        self,
        query: str,
        text: Optional[bool] = None,
    ) -> str:
        try:
            import asyncio
            import functools
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                functools.partial(
                    self._answer_impl,
                    query=query,
                    text=text,
                ),
            )
            return result
        except Exception as e:
            error_log(f"Exa async answer error: {e}")
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Internal implementations (shared by sync tools and async wrappers)
    # ------------------------------------------------------------------

    def _search_impl(
        self,
        query: str,
        num_results: Optional[int] = None,
        search_type: Optional[str] = None,
        category: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        text: Optional[bool] = None,
        highlights: Optional[bool] = None,
        summary: Optional[bool] = None,
        max_characters: Optional[int] = None,
        max_age_hours: Optional[int] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "query": query,
            "num_results": num_results or self.default_num_results,
            "type": search_type or self.default_search_type,
        }

        if category is not None:
            kwargs["category"] = category
        if include_domains is not None:
            kwargs["include_domains"] = include_domains
        if exclude_domains is not None:
            kwargs["exclude_domains"] = exclude_domains
        if start_published_date is not None:
            kwargs["start_published_date"] = start_published_date
        if end_published_date is not None:
            kwargs["end_published_date"] = end_published_date
        if include_text is not None:
            kwargs["include_text"] = include_text
        if exclude_text is not None:
            kwargs["exclude_text"] = exclude_text

        contents = self._build_contents_options(
            text=text, highlights=highlights, summary=summary,
            max_characters=max_characters, max_age_hours=max_age_hours,
        )
        if contents is not None:
            kwargs["contents"] = contents
        else:
            kwargs["contents"] = False

        result = self.client.search(**kwargs)
        return serialize_exa_response(result)

    def _get_contents_impl(
        self,
        urls: List[str],
        text: Optional[bool] = None,
        highlights: Optional[bool] = None,
        summary: Optional[bool] = None,
        max_characters: Optional[int] = None,
        max_age_hours: Optional[int] = None,
        subpages: Optional[int] = None,
        subpage_target: Optional[str] = None,
    ) -> str:
        # get_contents accepts content options as direct **kwargs.
        # The SDK defaults to text={max_characters: 10000} when no text/summary/extras
        # are provided. We must explicitly include at least one content option to
        # prevent the SDK default when the user wants no text.
        kwargs: Dict[str, Any] = {}

        use_text = text if text is not None else self.default_text
        if use_text:
            if max_characters is not None:
                kwargs["text"] = {"max_characters": max_characters}
            else:
                kwargs["text"] = True

        use_highlights = highlights if highlights is not None else self.default_highlights
        if use_highlights:
            kwargs["highlights"] = True

        use_summary = summary if summary is not None else self.default_summary
        if use_summary:
            kwargs["summary"] = True

        # If no content type was requested, pass summary=True as a minimal
        # content option to prevent the SDK from injecting its default text.
        if not use_text and not use_highlights and not use_summary:
            kwargs["summary"] = True

        if max_age_hours is not None:
            kwargs["max_age_hours"] = max_age_hours
        if subpages is not None:
            kwargs["subpages"] = subpages
        if subpage_target is not None:
            kwargs["subpage_target"] = subpage_target

        result = self.client.get_contents(urls, **kwargs)
        return serialize_exa_response(result)

    def _find_similar_impl(
        self,
        url: str,
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        text: Optional[bool] = None,
        highlights: Optional[bool] = None,
        summary: Optional[bool] = None,
        max_characters: Optional[int] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "url": url,
            "num_results": num_results or self.default_num_results,
        }

        if include_domains is not None:
            kwargs["include_domains"] = include_domains
        if exclude_domains is not None:
            kwargs["exclude_domains"] = exclude_domains
        if start_published_date is not None:
            kwargs["start_published_date"] = start_published_date
        if end_published_date is not None:
            kwargs["end_published_date"] = end_published_date

        contents = self._build_contents_options(
            text=text, highlights=highlights, summary=summary,
            max_characters=max_characters,
        )
        if contents is not None:
            kwargs["contents"] = contents
        else:
            kwargs["contents"] = False

        result = self.client.find_similar(**kwargs)
        return serialize_exa_response(result)

    def _answer_impl(
        self,
        query: str,
        text: Optional[bool] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {"query": query}
        if text is not None:
            kwargs["text"] = text

        result = self.client.answer(**kwargs)
        return serialize_exa_response(result)

    # ------------------------------------------------------------------
    # Tool methods (sync, exposed to LLM)
    # ------------------------------------------------------------------

    @tool
    def search(
        self,
        query: str,
        num_results: Optional[int] = None,
        search_type: Optional[str] = None,
        category: Optional[str] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_text: Optional[List[str]] = None,
        exclude_text: Optional[List[str]] = None,
        text: Optional[bool] = None,
        highlights: Optional[bool] = None,
        summary: Optional[bool] = None,
        max_characters: Optional[int] = None,
        max_age_hours: Optional[int] = None,
    ) -> str:
        """Search the web using Exa's neural or keyword search engine.

        Args:
            query: The search query string.
            num_results: Number of results to return (max 100).
            search_type: Search type - 'auto', 'neural', or 'keyword'.
            category: Focus on a data category - 'company', 'research paper', 'news', 'personal site', 'financial report', or 'people'.
            include_domains: Restrict results to these domains.
            exclude_domains: Exclude results from these domains.
            start_published_date: Minimum publication date in ISO 8601 format (e.g. '2024-01-01T00:00:00.000Z').
            end_published_date: Maximum publication date in ISO 8601 format.
            include_text: Strings that must be present in results.
            exclude_text: Strings that must not be present in results.
            text: Include full text content in results.
            highlights: Include relevant highlight snippets in results (token-efficient).
            summary: Include LLM-generated summary for each result.
            max_characters: Maximum characters for text content.
            max_age_hours: Maximum age of cached content in hours. 0 means always fetch fresh, -1 means cache only.

        Returns:
            JSON string containing the search results with URLs, titles, and optionally text/highlights/summaries.
        """
        try:
            return self._search_impl(
                query=query,
                num_results=num_results,
                search_type=search_type,
                category=category,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                include_text=include_text,
                exclude_text=exclude_text,
                text=text,
                highlights=highlights,
                summary=summary,
                max_characters=max_characters,
                max_age_hours=max_age_hours,
            )
        except Exception as e:
            error_log(f"Exa search error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def get_contents(
        self,
        urls: List[str],
        text: Optional[bool] = None,
        highlights: Optional[bool] = None,
        summary: Optional[bool] = None,
        max_characters: Optional[int] = None,
        max_age_hours: Optional[int] = None,
        subpages: Optional[int] = None,
        subpage_target: Optional[str] = None,
    ) -> str:
        """Retrieve clean, parsed content from a list of URLs.

        Args:
            urls: List of URLs to fetch content from.
            text: Include full text content.
            highlights: Include relevant highlight snippets (token-efficient).
            summary: Include LLM-generated summary.
            max_characters: Maximum characters for text content.
            max_age_hours: Maximum age of cached content in hours. 0 means always fetch fresh, -1 means cache only.
            subpages: Number of subpages to crawl from each URL.
            subpage_target: Terms to target when finding subpages.

        Returns:
            JSON string containing the parsed content for each URL.
        """
        try:
            return self._get_contents_impl(
                urls=urls,
                text=text,
                highlights=highlights,
                summary=summary,
                max_characters=max_characters,
                max_age_hours=max_age_hours,
                subpages=subpages,
                subpage_target=subpage_target,
            )
        except Exception as e:
            error_log(f"Exa get_contents error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def find_similar(
        self,
        url: str,
        num_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        text: Optional[bool] = None,
        highlights: Optional[bool] = None,
        summary: Optional[bool] = None,
        max_characters: Optional[int] = None,
    ) -> str:
        """Find webpages semantically similar to a given URL.

        Args:
            url: The URL to find similar pages for.
            num_results: Number of similar results to return.
            include_domains: Restrict results to these domains.
            exclude_domains: Exclude results from these domains.
            start_published_date: Minimum publication date in ISO 8601 format.
            end_published_date: Maximum publication date in ISO 8601 format.
            text: Include full text content in results.
            highlights: Include relevant highlight snippets (token-efficient).
            summary: Include LLM-generated summary for each result.
            max_characters: Maximum characters for text content.

        Returns:
            JSON string containing similar pages with URLs, titles, and optionally content.
        """
        try:
            return self._find_similar_impl(
                url=url,
                num_results=num_results,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                start_published_date=start_published_date,
                end_published_date=end_published_date,
                text=text,
                highlights=highlights,
                summary=summary,
                max_characters=max_characters,
            )
        except Exception as e:
            error_log(f"Exa find_similar error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def answer(
        self,
        query: str,
        text: Optional[bool] = None,
    ) -> str:
        """Get an LLM-generated answer to a question with cited sources from the web.

        Args:
            query: The question or query to answer.
            text: Include full text of cited sources in the response.

        Returns:
            JSON string containing the answer and citations with source URLs.
        """
        try:
            return self._answer_impl(
                query=query,
                text=text,
            )
        except Exception as e:
            error_log(f"Exa answer error: {e}")
            return json.dumps({"error": str(e)})
