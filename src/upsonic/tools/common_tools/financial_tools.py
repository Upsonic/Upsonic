"""Financial data tools using YFinance."""

from __future__ import annotations

import json
from typing import List

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    _PANDAS_AVAILABLE = False


try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    _YFINANCE_AVAILABLE = False



class YFinanceTools:
    """Comprehensive financial data toolkit using Yahoo Finance."""

    def __init__(
        self,
        stock_price: bool = True,
        company_info: bool = False,
        analyst_recommendations: bool = False,
        company_news: bool = False,
        enable_all: bool = False,
    ):
        """Initialize YFinance tools with selective functionality.

        Args:
            stock_price: Enable stock price retrieval
            company_info: Enable company information retrieval
            analyst_recommendations: Enable analyst recommendations
            company_news: Enable company news retrieval
            enable_all: Enable all available tools
        """
        if not _PANDAS_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="pandas",
                install_command='pip install "upsonic[tools]"',
                feature_name="financial tools"
            )

        if not _YFINANCE_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="yfinance",
                install_command='pip install "upsonic[tools]"',
                feature_name="financial tools"
            )

        self._tools = []
        if stock_price or enable_all:
            self._tools.append(self.get_current_stock_price)
        if company_info or enable_all:
            self._tools.append(self.get_company_info)
        if analyst_recommendations or enable_all:
            self._tools.append(self.get_analyst_recommendations)
        if company_news or enable_all:
            self._tools.append(self.get_company_news)

    def get_current_stock_price(self, symbol: str) -> str:
        """Get the current stock price for a given symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')

        Returns:
            Current stock price as formatted string
        """
        try:
            stock = yf.Ticker(symbol)
            price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
            return f"{price:.4f}" if price else f"Could not fetch current price for {symbol}"
        except Exception as e:
            return f"Error fetching current price for {symbol}: {e}"

    def get_company_info(self, symbol: str) -> str:
        """Get comprehensive company information.

        Args:
            symbol: Stock symbol

        Returns:
            JSON-formatted company information
        """
        try:
            info = yf.Ticker(symbol).info
            if not info:
                return f"Could not fetch company info for {symbol}"
            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error fetching company info for {symbol}: {e}"

    def get_analyst_recommendations(self, symbol: str) -> str:
        """Get analyst recommendations for a stock.

        Args:
            symbol: Stock symbol

        Returns:
            JSON-formatted analyst recommendations
        """
        try:
            recs = yf.Ticker(symbol).recommendations
            if recs is not None and isinstance(recs, (pd.DataFrame, pd.Series)):
                result = recs.to_json(orient="index")
                return result if result is not None else f"No recommendations for {symbol}"
            elif recs is not None:
                return json.dumps(recs, indent=2)
            else:
                return f"No recommendations for {symbol}"
        except Exception as e:
            return f"Error fetching analyst recommendations for {symbol}: {e}"

    def get_company_news(self, symbol: str, num_stories: int = 3) -> str:
        """Get recent company news.

        Args:
            symbol: Stock symbol
            num_stories: Number of news stories to retrieve

        Returns:
            JSON-formatted news stories
        """
        try:
            news = yf.Ticker(symbol).news
            if news is not None:
                return json.dumps(news[:num_stories], indent=2)
            else:
                return f"No news for {symbol}"
        except Exception as e:
            return f"Error fetching company news for {symbol}: {e}"

    def get_stock_fundamentals(self, symbol: str) -> str:
        """Get key financial fundamentals for a stock.

        Args:
            symbol: Stock symbol

        Returns:
            JSON-formatted fundamental data
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            fundamentals = {
                "symbol": symbol,
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("forwardPE", "N/A"),
                "pb_ratio": info.get("priceToBook", "N/A"),
                "dividend_yield": info.get("dividendYield", "N/A"),
                "eps": info.get("trailingEps", "N/A"),
                "beta": info.get("beta", "N/A"),
                "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            }
            return json.dumps(fundamentals, indent=2)
        except Exception as e:
            return f"Error getting fundamentals for {symbol}: {e}"

    def get_income_statements(self, symbol: str) -> str:
        """Get income statement data.

        Args:
            symbol: Stock symbol

        Returns:
            JSON-formatted income statement data
        """
        try:
            stock = yf.Ticker(symbol)
            financials = stock.financials
            if isinstance(financials, (pd.DataFrame, pd.Series)):
                result = financials.to_json(orient="index")
                return result if result is not None else f"No income statements for {symbol}"
            elif financials is not None:
                return json.dumps(financials, indent=2)
            else:
                return f"No income statements for {symbol}"
        except Exception as e:
            return f"Error fetching income statements for {symbol}: {e}"

    def get_key_financial_ratios(self, symbol: str) -> str:
        """Get key financial ratios.

        Args:
            symbol: Stock symbol

        Returns:
            JSON-formatted financial ratios
        """
        try:
            stock = yf.Ticker(symbol)
            key_ratios = stock.info
            return json.dumps(key_ratios, indent=2)
        except Exception as e:
            return f"Error fetching key financial ratios for {symbol}: {e}"

    def get_historical_stock_prices(self, symbol: str, period: str = "1mo", interval: str = "1d") -> str:
        """Get historical stock price data.

        Args:
            symbol: Stock symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        Returns:
            JSON-formatted historical price data
        """
        try:
            stock = yf.Ticker(symbol)
            historical_price = stock.history(period=period, interval=interval)
            if isinstance(historical_price, (pd.DataFrame, pd.Series)):
                result = historical_price.to_json(orient="index")
                return result if result is not None else f"No historical prices for {symbol}"
            elif historical_price is not None:
                return json.dumps(historical_price, indent=2)
            else:
                return f"No historical prices for {symbol}"
        except Exception as e:
            return f"Error fetching historical prices for {symbol}: {e}"

    def get_technical_indicators(self, symbol: str, period: str = "3mo") -> str:
        """Get technical indicator data.

        Args:
            symbol: Stock symbol
            period: Time period for analysis

        Returns:
            JSON-formatted technical indicator data
        """
        try:
            indicators = yf.Ticker(symbol).history(period=period)
            if isinstance(indicators, (pd.DataFrame, pd.Series)):
                result = indicators.to_json(orient="index")
                return result if result is not None else f"No technical indicators for {symbol}"
            elif indicators is not None:
                return json.dumps(indicators, indent=2)
            else:
                return f"No technical indicators for {symbol}"
        except Exception as e:
            return f"Error fetching technical indicators for {symbol}: {e}"

    def functions(self) -> List:
        """Return the list of tool functions to be used by the agent."""
        return self._tools

    def enable_all_tools(self) -> None:
        """Enable all available financial tools."""
        self._tools = [
            self.get_current_stock_price,
            self.get_company_info,
            self.get_analyst_recommendations,
            self.get_company_news,
            self.get_stock_fundamentals,
            self.get_income_statements,
            self.get_key_financial_ratios,
            self.get_historical_stock_prices,
            self.get_technical_indicators,
        ]