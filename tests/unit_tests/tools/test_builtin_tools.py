"""Unit tests for builtin tools."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from upsonic.tools.builtin_tools import (
    AbstractBuiltinTool,
    WebSearchTool,
    CodeExecutionTool,
    UrlContextTool,
    WebSearch,
    WebRead,
)


class TestBuiltinTools:
    """Test suite for builtin tools."""

    def test_builtin_tools_list(self):
        """Test listing builtin tools."""
        tools = [WebSearchTool(), CodeExecutionTool(), UrlContextTool()]

        assert len(tools) == 3
        assert all(isinstance(tool, AbstractBuiltinTool) for tool in tools)
        assert tools[0].kind == "web_search"
        assert tools[1].kind == "code_execution"
        assert tools[2].kind == "url_context"

    def test_builtin_tools_execution(self):
        """Test builtin tool execution."""
        # Builtin tools are passed to model providers, not executed directly
        # This test verifies they can be instantiated and configured
        web_search = WebSearchTool(search_context_size="high", max_uses=5)

        assert web_search.kind == "web_search"
        assert web_search.search_context_size == "high"
        assert web_search.max_uses == 5

    def test_web_search_tool_configuration(self):
        """Test WebSearchTool configuration."""
        from upsonic.tools.builtin_tools import WebSearchUserLocation

        location = WebSearchUserLocation(
            city="San Francisco", country="US", region="CA"
        )

        tool = WebSearchTool(user_location=location, blocked_domains=["example.com"])

        assert tool.user_location == location
        assert tool.blocked_domains == ["example.com"]

    def test_code_execution_tool(self):
        """Test CodeExecutionTool."""
        tool = CodeExecutionTool()

        assert tool.kind == "code_execution"
        assert isinstance(tool, AbstractBuiltinTool)

    def test_url_context_tool(self):
        """Test UrlContextTool."""
        tool = UrlContextTool()

        assert tool.kind == "url_context"
        assert isinstance(tool, AbstractBuiltinTool)

    @patch("upsonic.tools.builtin_tools._DDGS_AVAILABLE", True)
    @patch("upsonic.tools.builtin_tools.DDGS")
    def test_web_search_function(self, mock_ddgs):
        """Test WebSearch function."""
        mock_client = MagicMock()
        mock_client.text.return_value = [
            {"title": "Result 1", "href": "http://example.com", "body": "Description"}
        ]
        mock_ddgs.return_value.__enter__.return_value = mock_client
        mock_ddgs.return_value.__exit__ = Mock()

        result = WebSearch("test query", max_results=1)

        assert "test query" in result
        assert "Result 1" in result
        mock_client.text.assert_called_once_with("test query", max_results=1)

    @patch("upsonic.tools.builtin_tools._DDGS_AVAILABLE", False)
    def test_web_search_function_missing_dependency(self):
        """Test WebSearch with missing dependency."""
        with patch("upsonic.utils.printing.import_error") as mock_error:
            WebSearch("test")
            mock_error.assert_called_once()

    @patch("upsonic.tools.builtin_tools._REQUESTS_AVAILABLE", True)
    @patch("upsonic.tools.builtin_tools._BEAUTIFULSOUP_AVAILABLE", True)
    @patch("upsonic.tools.builtin_tools.requests")
    @patch("upsonic.tools.builtin_tools.BeautifulSoup")
    def test_web_read_function(self, mock_soup, mock_requests):
        """Test WebRead function."""
        import requests.exceptions

        mock_response = Mock()
        mock_response.content = b"<html><body>Test content</body></html>"
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.close = Mock()
        mock_requests.Session.return_value = mock_session
        mock_requests.exceptions = requests.exceptions

        # Create a callable mock for BeautifulSoup instance
        mock_soup_instance = MagicMock()
        mock_soup_instance.get_text.return_value = "Test content"
        # Make it callable to support soup(["script", "style"]) - return empty list
        mock_soup_instance.return_value = []
        mock_soup.return_value = mock_soup_instance

        result = WebRead("http://example.com")

        assert "example.com" in result
        assert "Test content" in result
        mock_session.get.assert_called_once()

    @patch("upsonic.tools.builtin_tools._REQUESTS_AVAILABLE", False)
    def test_web_read_function_missing_dependency(self):
        """Test WebRead with missing dependency."""
        with patch("upsonic.utils.printing.import_error") as mock_error:
            WebRead("http://example.com")
            mock_error.assert_called_once()
