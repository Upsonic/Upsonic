from typing import Dict, List


import json
import pandas as pd

from upsonic.tools.decorators.tool_decorator import tool
from upsonic.tools.base import Toolkit

try:
    import yfinance as yf
except ImportError:
    print("yfinance is not installed. Please install it using 'pip install yfinance'")
    exit(1)


class YFinanceTools(Toolkit):
    def __init__(self):
        """Initialize YFinance tools with @tool decorator support."""
        super().__init__(name="YFinanceTools")

    def get_description(self) -> str:
        return "Tools for fetching financial data and stock information using Yahoo Finance"

    @tool(description="Get the current stock price for a given symbol")
    def get_current_stock_price(self, symbol: str) -> str:
        try:
            stock = yf.Ticker(symbol)
            price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
            return (
                f"{price:.4f}"
                if price
                else f"Could not fetch current price for {symbol}"
            )
        except Exception as e:
            return f"Error fetching current price for {symbol}: {e}"

    @tool(description="Get the company info for a given symbol")
    def get_company_info(self, symbol: str) -> str:
        try:
            info = yf.Ticker(symbol).info
            if not info:
                return f"Could not fetch company info for {symbol}"
            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error fetching company info for {symbol}: {e}"

    @tool(description="Get the analyst recommendations for a given symbol")
    def get_analyst_recommendations(self, symbol: str) -> str:
        try:
            recs = yf.Ticker(symbol).recommendations
            if recs is not None and isinstance(recs, (pd.DataFrame, pd.Series)):
                result = recs.to_json(orient="index")
                return (
                    result if result is not None else f"No recommendations for {symbol}"
                )
            elif recs is not None:
                return json.dumps(recs, indent=2)
            else:
                return f"No recommendations for {symbol}"
        except Exception as e:
            return f"Error fetching analyst recommendations for {symbol}: {e}"

    @tool(description="Get the company news for a given symbol")
    def get_company_news(self, symbol: str, num_stories: int = 3) -> str:
        try:
            news = yf.Ticker(symbol).news
            if news is not None:
                return json.dumps(news[:num_stories], indent=2)
            else:
                return f"No news for {symbol}"
        except Exception as e:
            return f"Error fetching company news for {symbol}: {e}"

    @tool(description="Get the stock fundamentals for a given symbol")
    def get_stock_fundamentals(self, symbol: str) -> str:
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

    @tool(description="Get the income statements for a given symbol")
    def get_income_statements(self, symbol: str) -> str:
        try:
            stock = yf.Ticker(symbol)
            financials = stock.financials
            if isinstance(financials, (pd.DataFrame, pd.Series)):
                result = financials.to_json(orient="index")
                return (
                    result
                    if result is not None
                    else f"No income statements for {symbol}"
                )
            elif financials is not None:
                return json.dumps(financials, indent=2)
            else:
                return f"No income statements for {symbol}"
        except Exception as e:
            return f"Error fetching income statements for {symbol}: {e}"

    @tool(description="Get the key financial ratios for a given symbol")
    def get_key_financial_ratios(self, symbol: str) -> str:
        try:
            stock = yf.Ticker(symbol)
            key_ratios = stock.info
            return json.dumps(key_ratios, indent=2)
        except Exception as e:
            return f"Error fetching key financial ratios for {symbol}: {e}"

    @tool(description="Get the historical stock prices for a given symbol")
    def get_historical_stock_prices(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> str:
        try:
            stock = yf.Ticker(symbol)
            historical_price = stock.history(period=period, interval=interval)
            if isinstance(historical_price, (pd.DataFrame, pd.Series)):
                result = historical_price.to_json(orient="index")
                return (
                    result
                    if result is not None
                    else f"No historical prices for {symbol}"
                )
            elif historical_price is not None:
                return json.dumps(historical_price, indent=2)
            else:
                return f"No historical prices for {symbol}"
        except Exception as e:
            return f"Error fetching historical prices for {symbol}: {e}"

    @tool(description="Get the technical indicators for a given symbol")
    def get_technical_indicators(self, symbol: str, period: str = "3mo") -> str:
        try:
            indicators = yf.Ticker(symbol).history(period=period)
            if isinstance(indicators, (pd.DataFrame, pd.Series)):
                result = indicators.to_json(orient="index")
                return (
                    result
                    if result is not None
                    else f"No technical indicators for {symbol}"
                )
            elif indicators is not None:
                return json.dumps(indicators, indent=2)
            else:
                return f"No technical indicators for {symbol}"
        except Exception as e:
            return f"Error fetching technical indicators for {symbol}: {e}"
