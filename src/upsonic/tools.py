from typing import Dict, List
from duckduckgo_search import DDGS

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