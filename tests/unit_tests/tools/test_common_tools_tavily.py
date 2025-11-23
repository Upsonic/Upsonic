"""Unit tests for Tavily search tool."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from upsonic.tools.common_tools.tavily import (
    tavily_search_tool,
    TavilySearchTool,
    TavilySearchResult,
)


class TestTavilySearch:
    """Test suite for Tavily search tool."""

    @pytest.fixture
    def mock_tavily_client(self):
        """Create a mock Tavily client."""
        client = AsyncMock()
        client.search = AsyncMock(
            return_value={
                "results": [
                    {
                        "title": "Test Result",
                        "url": "http://example.com",
                        "content": "Test content",
                        "score": 0.95,
                    }
                ]
            }
        )
        return client

    @pytest.mark.asyncio
    async def test_tavily_search(self, mock_tavily_client):
        """Test Tavily search tool."""
        tool_instance = TavilySearchTool(client=mock_tavily_client)

        result = await tool_instance(
            query="test query", search_deep="basic", topic="general"
        )

        assert isinstance(result, list)
        assert len(result) > 0
        mock_tavily_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_tavily_search_error_handling(self, mock_tavily_client):
        """Test error handling."""
        mock_tavily_client.search = AsyncMock(side_effect=Exception("Tavily error"))

        tool_instance = TavilySearchTool(client=mock_tavily_client)

        with pytest.raises(Exception, match="Tavily error"):
            await tool_instance("test query")

    @patch("upsonic.tools.common_tools.tavily._TAVILY_AVAILABLE", True)
    @patch("upsonic.tools.common_tools.tavily.AsyncTavilyClient")
    def test_tavily_search_tool_creation(self, mock_client_class):
        """Test creating Tavily search tool."""
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        tool = tavily_search_tool(api_key="test_key")

        assert tool is not None
        assert callable(tool)
        mock_client_class.assert_called_once_with("test_key")

    @patch("upsonic.tools.common_tools.tavily._TAVILY_AVAILABLE", False)
    def test_tavily_search_tool_missing_dependency(self):
        """Test tool creation with missing dependency."""
        with patch(
            "upsonic.utils.printing.import_error", return_value=None
        ) as mock_error:
            with patch("upsonic.tools.common_tools.tavily.AsyncTavilyClient", None):
                try:
                    tavily_search_tool(api_key="test")
                    mock_error.assert_called_once()
                except (TypeError, AttributeError):
                    # Expected when AsyncTavilyClient is None
                    mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_tavily_search_with_time_range(self, mock_tavily_client):
        """Test search with time range."""
        tool_instance = TavilySearchTool(client=mock_tavily_client)

        result = await tool_instance(
            query="test", search_deep="advanced", topic="news", time_range="week"
        )

        assert isinstance(result, list)
        call_args = mock_tavily_client.search.call_args
        assert call_args[1]["time_range"] == "week"

    @pytest.mark.asyncio
    async def test_tavily_search_result_format(self, mock_tavily_client):
        """Test search result format."""
        mock_tavily_client.search.return_value = {
            "results": [
                {
                    "title": "Result 1",
                    "url": "http://example.com/1",
                    "content": "Content 1",
                    "score": 0.9,
                },
                {
                    "title": "Result 2",
                    "url": "http://example.com/2",
                    "content": "Content 2",
                    "score": 0.8,
                },
            ]
        }

        tool_instance = TavilySearchTool(client=mock_tavily_client)
        result = await tool_instance("test")

        assert len(result) == 2
        assert all(isinstance(r, dict) for r in result)
        assert all(
            "title" in r and "url" in r and "content" in r and "score" in r
            for r in result
        )
