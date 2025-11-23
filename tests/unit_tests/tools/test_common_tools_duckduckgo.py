"""Unit tests for DuckDuckGo search tool."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from upsonic.tools.common_tools.duckduckgo import (
    duckduckgo_search_tool,
    DuckDuckGoSearchTool,
    DuckDuckGoResult,
)


class TestDuckDuckGoSearch:
    """Test suite for DuckDuckGo search tool."""

    @pytest.fixture
    def mock_ddgs_client(self):
        """Create a mock DDGS client."""
        client = Mock()
        client.text = Mock(
            return_value=[
                {
                    "title": "Test Result",
                    "href": "http://example.com",
                    "body": "Test description",
                }
            ]
        )
        return client

    @pytest.mark.asyncio
    async def test_duckduckgo_search(self, mock_ddgs_client):
        """Test DuckDuckGo search tool."""
        tool_instance = DuckDuckGoSearchTool(client=mock_ddgs_client, max_results=5)

        result = await tool_instance("test query")

        assert isinstance(result, list)
        assert len(result) > 0
        mock_ddgs_client.text.assert_called_once_with("test query", max_results=5)

    @pytest.mark.asyncio
    async def test_duckduckgo_search_error_handling(self, mock_ddgs_client):
        """Test error handling."""
        mock_ddgs_client.text = Mock(side_effect=Exception("Search error"))

        tool_instance = DuckDuckGoSearchTool(client=mock_ddgs_client, max_results=5)

        with pytest.raises(Exception, match="Search error"):
            await tool_instance("test query")

    @patch("upsonic.tools.common_tools.duckduckgo._DDGS_AVAILABLE", True)
    @patch("upsonic.tools.common_tools.duckduckgo.DDGS")
    def test_duckduckgo_search_tool_creation(self, mock_ddgs_class):
        """Test creating DuckDuckGo search tool."""
        mock_client = Mock()
        mock_ddgs_class.return_value = mock_client

        tool = duckduckgo_search_tool(max_results=10)

        assert tool is not None
        assert callable(tool)

    @patch("upsonic.tools.common_tools.duckduckgo._DDGS_AVAILABLE", False)
    def test_duckduckgo_search_tool_missing_dependency(self):
        """Test tool creation with missing dependency."""
        with patch("upsonic.utils.printing.import_error") as mock_error:
            duckduckgo_search_tool()
            mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_duckduckgo_search_with_none_max_results(self, mock_ddgs_client):
        """Test search with None max_results."""
        tool_instance = DuckDuckGoSearchTool(client=mock_ddgs_client, max_results=None)

        result = await tool_instance("test query")

        assert isinstance(result, list)
        mock_ddgs_client.text.assert_called_once_with("test query", max_results=None)

    @pytest.mark.asyncio
    async def test_duckduckgo_search_result_format(self, mock_ddgs_client):
        """Test search result format."""
        mock_ddgs_client.text.return_value = [
            {
                "title": "Result 1",
                "href": "http://example.com/1",
                "body": "Description 1",
            },
            {
                "title": "Result 2",
                "href": "http://example.com/2",
                "body": "Description 2",
            },
        ]

        tool_instance = DuckDuckGoSearchTool(client=mock_ddgs_client, max_results=2)
        result = await tool_instance("test")

        assert len(result) == 2
        assert all(isinstance(r, dict) for r in result)
        assert all("title" in r and "href" in r and "body" in r for r in result)
