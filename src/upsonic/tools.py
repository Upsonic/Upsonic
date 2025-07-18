from typing import Dict, List
from duckduckgo_search import DDGS

import yfinance as yf
import json
import pandas as pd

class YFinanceTools:
    def __init__(
        self,
        stock_price=True,
        company_info=False,
        analyst_recommendations=False,
        company_news=False,
        enable_all=False,
    ):
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
        try:
            stock = yf.Ticker(symbol)
            price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
            return f"{price:.4f}" if price else f"Could not fetch current price for {symbol}"
        except Exception as e:
            return f"Error fetching current price for {symbol}: {e}"

    def get_company_info(self, symbol: str) -> str:
        try:
            info = yf.Ticker(symbol).info
            if not info:
                return f"Could not fetch company info for {symbol}"
            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error fetching company info for {symbol}: {e}"

    def get_analyst_recommendations(self, symbol: str) -> str:
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
        try:
            news = yf.Ticker(symbol).news
            if news is not None:
                return json.dumps(news[:num_stories], indent=2)
            else:
                return f"No news for {symbol}"
        except Exception as e:
            return f"Error fetching company news for {symbol}: {e}"

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

    def get_income_statements(self, symbol: str) -> str:
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
        try:
            stock = yf.Ticker(symbol)
            key_ratios = stock.info
            return json.dumps(key_ratios, indent=2)
        except Exception as e:
            return f"Error fetching key financial ratios for {symbol}: {e}"

    def get_historical_stock_prices(self, symbol: str, period: str = "1mo", interval: str = "1d") -> str:
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

    def functions(self):
        """Return the list of tool functions to be used by the agent."""
        return self._tools

    def enable_all_tools(self):
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


def Search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo for the given query and return text results.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 10)
        
    Returns:
        List of dictionaries containing search results with keys: title, href, body
    """
    
    
    ddgs = DDGS()
    results = list(ddgs.text(query, max_results=max_results))
    
    return results