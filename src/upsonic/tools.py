"""
Upsonic Tools Module
This module contains the tool implementations that can be used with Upsonic.
"""
from typing import Any, List, Dict, Optional, Type, Union, Callable

class Search:
    pass



class ComputerUse:
    pass

class BrowserUse:
    pass



class Wikipedia:
    def __control__() -> bool:
        # Check the import wikipedia
        try:
            import wikipedia
        except ImportError:
            raise ImportError("wikipedia is not installed. Please install it with 'pip install wikipedia'")
        
        return True
        
    def search(query: str) -> str:
        import wikipedia
        return wikipedia.search(query)
    
    def summary(query: str) -> str:
        import wikipedia
        return wikipedia.summary(query)


class DuckDuckGo:
    def __control__() -> bool:
        # Check the import duckduckgo_search
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError("duckduckgo_search is not installed. Please install it with 'pip install duckduckgo-search'")
        
        return True
    
    def search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo for the given query and return text results.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return (default: 10)
            
        Returns:
            List of dictionaries containing search results with keys: title, href, body
        """
        from duckduckgo_search import DDGS
        
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        return results
    


# Export all tool classes
__all__ = ["Search", "ComputerUse", "BrowserUse", "Wikipedia", "DuckDuckGo"] 