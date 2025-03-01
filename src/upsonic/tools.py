"""
Upsonic Tools Module
This module contains the tool implementations that can be used with Upsonic.
"""
from typing import Any, List, Dict, Optional, Type, Union, Callable
import os
import json
import requests
import logging
import pathlib

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
    

class SerperDev:
    @staticmethod
    def _load_api_key_from_env_file() -> Optional[str]:
        """
        Try to load the SERPER_API_KEY from a .env file using python-dotenv.
        
        Returns:
            The API key if found in .env file, None otherwise
        """
        try:
            # Try to import dotenv
            from dotenv import load_dotenv
        except ImportError:
            raise ImportError("python-dotenv is not installed. Please install it with 'pip install python-dotenv'")
        
        # Check for .env file in current directory and parent directories
        current_dir = pathlib.Path.cwd()
        
        # Look in current directory and up to 3 parent directories
        for _ in range(4):
            env_path = current_dir / '.env'
            if env_path.exists():
                # Load the .env file
                load_dotenv(dotenv_path=env_path)
                
                # Check if SERPER_API_KEY is now in environment
                if "SERPER_API_KEY" in os.environ:
                    return os.environ["SERPER_API_KEY"]
            
            # Move to parent directory
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # Reached root directory
                break
            current_dir = parent_dir
        
        return None
    
    def __control__(self) -> bool:
        # Check if requests is installed
        try:
            import requests
        except ImportError:
            raise ImportError("requests is not installed. Please install it with 'pip install requests'")
        
        # Check if python-dotenv is installed
        try:
            from dotenv import load_dotenv
        except ImportError:
            raise ImportError("python-dotenv is not installed. Please install it with 'pip install python-dotenv'")
        
        # Try to load from .env file first
        if "SERPER_API_KEY" not in os.environ:
            try:
                SerperDev._load_api_key_from_env_file()
            except ImportError:
                pass  # If dotenv is not installed, we'll check environment variables directly
        
        # Check if SERPER_API_KEY is set in environment variables
        if "SERPER_API_KEY" not in os.environ:
            raise EnvironmentError("SERPER_API_KEY environment variable is not set and could not be found in .env file")
        
        return True
    
    def __init__(self, base_url: str = "https://google.serper.dev", search_type: str = "search", n_results: int = 10, country: str = "us", 
                 location: str = None, locale: str = "en", api_key: Optional[str] = None):
        """
        Initialize the SerperDev search tool.
        
        Args:
            search_type: Type of search to perform ('search' or 'news')
            n_results: Maximum number of results to return
            country: Country code for search results
            location: Location for search results
            locale: Locale for search results
            api_key: Serper API key (optional, will try to load from environment if not provided)
        """
        self.base_url = base_url
        self.search_type = search_type
        self.n_results = n_results
        self.country = country
        self.location = location
        self.locale = locale
        
        # Set API key
        self.api_key = api_key
        
        # If API key not provided, try to load from environment or .env file
        if self.api_key is None:
            # First check environment variables
            if "SERPER_API_KEY" in os.environ:
                self.api_key = os.environ["SERPER_API_KEY"]
            else:
                # Try to load from .env file
                try:
                    api_key = self._load_api_key_from_env_file()
                    if api_key:
                        self.api_key = api_key
                    else:
                        raise EnvironmentError("SERPER_API_KEY environment variable is not set and could not be found in .env file")
                except ImportError:
                    # If dotenv is not installed and no API key in environment
                    if "SERPER_API_KEY" not in os.environ:
                        raise EnvironmentError("SERPER_API_KEY environment variable is not set and python-dotenv is not installed")
                    self.api_key = os.environ["SERPER_API_KEY"]

    def _get_search_url(self) -> str:
        """Get the appropriate endpoint URL based on search type."""
        search_type = self.search_type.lower()
        allowed_search_types = ["search", "news"]
        if search_type not in allowed_search_types:
            raise ValueError(
                f"Invalid search type: {search_type}. Must be one of: {', '.join(allowed_search_types)}"
            )
        return f"{self.base_url}/{search_type}"

    def _process_knowledge_graph(self, kg: dict) -> dict:
        """Process knowledge graph data from search results."""
        return {
            "title": kg.get("title", ""),
            "type": kg.get("type", ""),
            "website": kg.get("website", ""),
            "imageUrl": kg.get("imageUrl", ""),
            "description": kg.get("description", ""),
            "descriptionSource": kg.get("descriptionSource", ""),
            "descriptionLink": kg.get("descriptionLink", ""),
            "attributes": kg.get("attributes", {}),
        }

    def _process_organic_results(self, organic_results: list) -> list:
        """Process organic search results."""
        processed_results = []
        for result in organic_results[:self.n_results]:
            try:
                result_data = {
                    "title": result["title"],
                    "link": result["link"],
                    "snippet": result.get("snippet", ""),
                    "position": result.get("position"),
                }

                if "sitelinks" in result:
                    result_data["sitelinks"] = [
                        {
                            "title": sitelink.get("title", ""),
                            "link": sitelink.get("link", ""),
                        }
                        for sitelink in result["sitelinks"]
                    ]

                processed_results.append(result_data)
            except KeyError:
                continue
        return processed_results

    def _process_people_also_ask(self, paa_results: list) -> list:
        """Process 'People Also Ask' results."""
        processed_results = []
        for result in paa_results[:self.n_results]:
            try:
                result_data = {
                    "question": result["question"],
                    "snippet": result.get("snippet", ""),
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                }
                processed_results.append(result_data)
            except KeyError:
                continue
        return processed_results

    def _process_related_searches(self, related_results: list) -> list:
        """Process related search results."""
        processed_results = []
        for result in related_results[:self.n_results]:
            try:
                processed_results.append({"query": result["query"]})
            except KeyError:
                continue
        return processed_results

    def _process_news_results(self, news_results: list) -> list:
        """Process news search results."""
        processed_results = []
        for result in news_results[:self.n_results]:
            try:
                result_data = {
                    "title": result["title"],
                    "link": result["link"],
                    "snippet": result.get("snippet", ""),
                    "date": result.get("date", ""),
                    "source": result.get("source", ""),
                    "imageUrl": result.get("imageUrl", ""),
                }
                processed_results.append(result_data)
            except KeyError:
                continue
        return processed_results

    def _process_search_results(self, results: dict) -> dict:
        """Process search results based on search type."""
        formatted_results = {}

        if self.search_type == "search":
            if "knowledgeGraph" in results:
                formatted_results["knowledgeGraph"] = self._process_knowledge_graph(
                    results["knowledgeGraph"]
                )

            if "organic" in results:
                formatted_results["organic"] = self._process_organic_results(
                    results["organic"]
                )

            if "peopleAlsoAsk" in results:
                formatted_results["peopleAlsoAsk"] = self._process_people_also_ask(
                    results["peopleAlsoAsk"]
                )

            if "relatedSearches" in results:
                formatted_results["relatedSearches"] = self._process_related_searches(
                    results["relatedSearches"]
                )

        elif self.search_type == "news":
            if "news" in results:
                formatted_results["news"] = self._process_news_results(results["news"])

        return formatted_results

    def search(self, query: str) -> Dict[str, Any]:
        print("*************I am here")
        print(self.api_key)

        print(self.base_url)
        print(query)
        """
        Search the web using Serper API.
        
        Args:
            query: The search query
            
        Returns:
            Dictionary containing processed search results
        """
        search_url = self._get_search_url()
        payload = json.dumps({"q": query, "num": self.n_results})
        
        if self.country:
            payload = json.dumps(json.loads(payload) | {"gl": self.country})
        
        if self.location:
            payload = json.dumps(json.loads(payload) | {"location": self.location})
            
        if self.locale:
            payload = json.dumps(json.loads(payload) | {"hl": self.locale})
            
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                search_url, headers=headers, json=json.loads(payload), timeout=10
            )
            response.raise_for_status()
            results = response.json()
            
            if not results:
                raise ValueError("Empty response from Serper API")
                
            formatted_results = {
                "searchParameters": {
                    "q": query,
                    "type": self.search_type,
                    **results.get("searchParameters", {}),
                }
            }

            formatted_results.update(self._process_search_results(results))
            formatted_results["credits"] = results.get("credits", 1)
            
            return formatted_results
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error making request to Serper API: {e}"
            raise RuntimeError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON response: {e}"
            raise RuntimeError(error_msg)



# Export all tool classes
__all__ = ["Search", "ComputerUse", "BrowserUse", "Wikipedia", "DuckDuckGo", "SerperDev"] 