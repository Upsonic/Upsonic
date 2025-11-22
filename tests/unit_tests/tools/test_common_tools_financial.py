"""Unit tests for financial tools."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from upsonic.tools.common_tools.financial_tools import YFinanceTools


class TestFinancialTools:
    """Test suite for financial tools."""

    @pytest.fixture
    def mock_yfinance(self):
        """Create a mock yfinance module."""
        with patch("upsonic.tools.common_tools.financial_tools.yf") as mock_yf:
            mock_ticker = Mock()
            mock_ticker.info = {
                "regularMarketPrice": 150.50,
                "currentPrice": 150.50,
                "longName": "Test Company",
                "sector": "Technology",
            }
            mock_ticker.recommendations = None
            mock_ticker.news = []
            mock_ticker.financials = None
            mock_ticker.history = Mock(return_value=None)

            mock_yf.Ticker.return_value = mock_ticker
            yield mock_yf

    @patch("upsonic.tools.common_tools.financial_tools._PANDAS_AVAILABLE", True)
    @patch("upsonic.tools.common_tools.financial_tools._YFINANCE_AVAILABLE", True)
    def test_financial_tools_list(self, mock_yfinance):
        """Test financial tools list."""
        tools = YFinanceTools(stock_price=True, company_info=True)

        functions = tools.functions()
        assert len(functions) == 2
        assert tools.get_current_stock_price in functions
        assert tools.get_company_info in functions

    @patch("upsonic.tools.common_tools.financial_tools._PANDAS_AVAILABLE", True)
    @patch("upsonic.tools.common_tools.financial_tools._YFINANCE_AVAILABLE", True)
    def test_financial_tool_execution(self, mock_yfinance):
        """Test execution."""
        tools = YFinanceTools(stock_price=True)

        result = tools.get_current_stock_price("AAPL")

        assert "150.50" in result
        mock_yfinance.Ticker.assert_called_once_with("AAPL")

    @patch("upsonic.tools.common_tools.financial_tools._PANDAS_AVAILABLE", True)
    @patch("upsonic.tools.common_tools.financial_tools._YFINANCE_AVAILABLE", True)
    def test_get_company_info(self, mock_yfinance):
        """Test get company info."""
        tools = YFinanceTools(company_info=True)

        result = tools.get_company_info("AAPL")

        assert "Test Company" in result
        assert "Technology" in result

    @patch("upsonic.tools.common_tools.financial_tools._PANDAS_AVAILABLE", True)
    @patch("upsonic.tools.common_tools.financial_tools._YFINANCE_AVAILABLE", True)
    def test_get_analyst_recommendations(self, mock_yfinance):
        """Test analyst recommendations."""
        tools = YFinanceTools(analyst_recommendations=True)

        result = tools.get_analyst_recommendations("AAPL")

        assert "No recommendations" in result or "AAPL" in result

    @patch("upsonic.tools.common_tools.financial_tools._PANDAS_AVAILABLE", True)
    @patch("upsonic.tools.common_tools.financial_tools._YFINANCE_AVAILABLE", True)
    def test_get_company_news(self, mock_yfinance):
        """Test company news."""
        tools = YFinanceTools(company_news=True)

        result = tools.get_company_news("AAPL", num_stories=3)

        assert "AAPL" in result or "No news" in result or result == "[]"

    @patch("upsonic.tools.common_tools.financial_tools._PANDAS_AVAILABLE", True)
    @patch("upsonic.tools.common_tools.financial_tools._YFINANCE_AVAILABLE", True)
    def test_enable_all_tools(self, mock_yfinance):
        """Test enabling all tools."""
        tools = YFinanceTools(enable_all=True)

        functions = tools.functions()
        assert len(functions) >= 4

    @patch("upsonic.tools.common_tools.financial_tools._PANDAS_AVAILABLE", True)
    @patch("upsonic.tools.common_tools.financial_tools._YFINANCE_AVAILABLE", True)
    def test_error_handling(self, mock_yfinance):
        """Test error handling."""
        mock_yfinance.Ticker.side_effect = Exception("API Error")

        tools = YFinanceTools(stock_price=True)
        result = tools.get_current_stock_price("INVALID")

        assert "Error" in result

    @patch("upsonic.tools.common_tools.financial_tools._PANDAS_AVAILABLE", False)
    def test_missing_pandas_dependency(self):
        """Test with missing pandas dependency."""
        with patch("upsonic.utils.printing.import_error") as mock_error:
            YFinanceTools()
            mock_error.assert_called()

    @patch("upsonic.tools.common_tools.financial_tools._YFINANCE_AVAILABLE", False)
    def test_missing_yfinance_dependency(self):
        """Test with missing yfinance dependency."""
        with patch("upsonic.utils.printing.import_error") as mock_error:
            YFinanceTools()
            mock_error.assert_called()
