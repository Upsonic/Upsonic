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
from datetime import datetime
import time
import re
from .client.printing import missing_dependencies, missing_api_key

class Search:
    pass



class ComputerUse:
    pass

class BrowserUse:
    pass



class Wikipedia:
    @staticmethod
    def analyze_dependencies() -> Dict[str, bool]:
        """
        Analyze the dependencies required for Wikipedia and return their status.
        
        Returns:
            Dictionary with dependency names as keys and their availability status as values
        """
        dependencies = {
            "wikipedia": False
        }
        
        # Check each dependency
        try:
            import wikipedia
            dependencies["wikipedia"] = True
        except ImportError:
            pass
        
        return dependencies
        
    @staticmethod
    def __control__() -> bool:
        # Check the import wikipedia
        try:
            import wikipedia
            return True
        except ImportError:
            # Use the missing_dependencies function to display the error
            missing_dependencies("Wikipedia", ["wikipedia"])
            raise ImportError("Missing dependency: wikipedia. Please install it with: pip install wikipedia")
        
    def search(query: str) -> str:
        import wikipedia
        return wikipedia.search(query)
    
    def summary(query: str) -> str:
        import wikipedia
        return wikipedia.summary(query)


class DuckDuckGo:
    @staticmethod
    def analyze_dependencies() -> Dict[str, bool]:
        """
        Analyze the dependencies required for DuckDuckGo and return their status.
        
        Returns:
            Dictionary with dependency names as keys and their availability status as values
        """
        dependencies = {
            "duckduckgo_search": False
        }
        
        # Check each dependency
        try:
            import duckduckgo_search
            dependencies["duckduckgo_search"] = True
        except ImportError:
            pass
        
        return dependencies
        
    @staticmethod
    def __control__() -> bool:
        # Check the import duckduckgo_search
        try:
            import duckduckgo_search
            return True
        except ImportError:
            # Use the missing_dependencies function to display the error
            missing_dependencies("DuckDuckGo", ["duckduckgo_search"])
            raise ImportError("Missing dependency: duckduckgo_search. Please install it with: pip install duckduckgo-search")
    
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
        
        # Check if SERPER_API_KEY is set in environment variables
        if "SERPER_API_KEY" not in os.environ:
            try:
                # Try to load API key from .env file
                api_key = SerperDev._load_api_key_from_env_file()
                if api_key is None:
                    # API key not found in .env file
                    missing_api_key("SerperDev", "SERPER_API_KEY")
                    raise EnvironmentError("SERPER_API_KEY environment variable is not set and could not be found in .env file")
            except ImportError:
                # If dotenv is not installed, we can't load from .env file
                missing_api_key("SerperDev", "SERPER_API_KEY", dotenv_support=False)
                raise EnvironmentError("SERPER_API_KEY environment variable is not set and python-dotenv is not installed")
        
        return True
    
    def __init__(self, base_url: str = "https://google.serper.dev", search_type: str = "search", n_results: int = 10, country: str = "us", 
                 location: str = None, locale: str = "en", api_key: Optional[str] = None):
        """
        Initialize the SerperDev search tool.
        
        Args:
            base_url: Base URL for the Serper API (default: "https://google.serper.dev")
            search_type: Type of search to perform (default: "search")
            n_results: Number of results to return (default: 10)
            country: Country code for search (default: "us")
            location: Location for search (default: None)
            locale: Locale for search (default: "en")
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
                        # Print missing API key message
                        missing_api_key("SerperDev", "SERPER_API_KEY")
                        raise EnvironmentError("SERPER_API_KEY environment variable is not set and could not be found in .env file")
                except ImportError:
                    # If dotenv is not installed and no API key in environment
                    if "SERPER_API_KEY" not in os.environ:
                        # Print missing API key message without dotenv support
                        missing_api_key("SerperDev", "SERPER_API_KEY", dotenv_support=False)
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


class FirecrawlSearchTool:
    @staticmethod
    def analyze_dependencies() -> Dict[str, bool]:
        """
        Analyze the dependencies required for FirecrawlSearchTool and return their status.
        
        Returns:
            Dictionary with dependency names as keys and their availability status as values
        """
        dependencies = {
            "requests": False,
            "firecrawl": False,
            "python-dotenv": False
        }
        
        # Check each dependency
        try:
            import requests
            dependencies["requests"] = True
        except ImportError:
            pass
        
        try:
            import firecrawl
            dependencies["firecrawl"] = True
        except ImportError:
            pass
        
        try:
            from dotenv import load_dotenv
            dependencies["python-dotenv"] = True
        except ImportError:
            pass
        
        return dependencies
    
    @staticmethod
    def _load_api_key_from_env_file() -> Optional[str]:
        """
        Try to load the FIRECRAWL_API_KEY from a .env file using python-dotenv.
        
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
                
                # Check if FIRECRAWL_API_KEY is now in environment
                if "FIRECRAWL_API_KEY" in os.environ:
                    return os.environ["FIRECRAWL_API_KEY"]
            
            # Move to parent directory
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # Reached root directory
                break
            current_dir = parent_dir
        
        return None
    
    def __control__(self) -> bool:
        """
        Check if the required dependencies are installed and API key is available.
        
        Returns:
            True if all requirements are met
        
        Raises:
            ImportError: If required packages are not installed
            EnvironmentError: If API key is not available
        """
        # Analyze dependencies
        dependencies = self.analyze_dependencies()
        missing = [dep for dep, installed in dependencies.items() if not installed]
        
        # Print missing dependencies
        if missing:
            # Use the new printing function
            missing_dependencies("FirecrawlSearchTool", missing)
            
            # Raise ImportError with combined message for all missing dependencies
            install_cmd = "pip install " + " ".join(missing)
            raise ImportError(f"Missing dependencies: {', '.join(missing)}. Please install them with: {install_cmd}")
        
        # Check if FIRECRAWL_API_KEY is set in environment variables
        if "FIRECRAWL_API_KEY" not in os.environ:
            try:
                # Try to load API key from .env file
                api_key = FirecrawlSearchTool._load_api_key_from_env_file()
                if api_key is None:
                    # Print missing API key message
                    missing_api_key("FirecrawlSearchTool", "FIRECRAWL_API_KEY")
                    raise EnvironmentError("FIRECRAWL_API_KEY environment variable is not set and could not be found in .env file")
            except ImportError:
                # If dotenv is not installed, we can't load from .env file
                # Print missing API key message without dotenv support
                missing_api_key("FirecrawlSearchTool", "FIRECRAWL_API_KEY", dotenv_support=False)
                raise EnvironmentError("FIRECRAWL_API_KEY environment variable is not set and python-dotenv is not installed")
        
        return True
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FirecrawlSearchTool.
        
        Args:
            api_key: Firecrawl API key (optional, will try to load from environment if not provided)
        """
        # Set API key
        self.api_key = api_key
        
        # If API key not provided, try to load from environment or .env file
        if self.api_key is None:
            # First check environment variables
            if "FIRECRAWL_API_KEY" in os.environ:
                self.api_key = os.environ["FIRECRAWL_API_KEY"]
            else:
                # Try to load from .env file
                try:
                    api_key = self._load_api_key_from_env_file()
                    if api_key:
                        self.api_key = api_key
                    else:
                        # Print missing API key message
                        missing_api_key("FirecrawlSearchTool", "FIRECRAWL_API_KEY")
                        raise EnvironmentError("FIRECRAWL_API_KEY environment variable is not set and could not be found in .env file")
                except ImportError:
                    # If dotenv is not installed and no API key in environment
                    if "FIRECRAWL_API_KEY" not in os.environ:
                        # Print missing API key message without dotenv support
                        missing_api_key("FirecrawlSearchTool", "FIRECRAWL_API_KEY", dotenv_support=False)
                        raise EnvironmentError("FIRECRAWL_API_KEY environment variable is not set and python-dotenv is not installed")
                    self.api_key = os.environ["FIRECRAWL_API_KEY"]
        
        # Initialize FirecrawlApp
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            raise ImportError("firecrawl-py is not installed. Please install it with 'pip install firecrawl-py'")
    
    def search(self, query: str, limit: int = 5, tbs: Optional[str] = None, 
               lang: str = "en", country: str = "us", location: Optional[str] = None,
               timeout: int = 60000, scrape_options: Optional[Dict[str, Any]] = None) -> Any:
        """
        Search the web using Firecrawl API.
        
        Args:
            query: The search query
            limit: Maximum number of results to return (default: 5)
            tbs: Time-based search parameter
            lang: Language code for search results (default: 'en')
            country: Country code for search results (default: 'us')
            location: Location parameter for search results
            timeout: Timeout in milliseconds (default: 60000)
            scrape_options: Options for scraping search results
            
        Returns:
            Search results from Firecrawl
        """

        
        options = {
            
            "limit": limit,
            "tbs": tbs,
            "lang": lang,
            "country": country,
            "location": location,
            "timeout": timeout,
            "scrapeOptions": scrape_options or {},
        }
        from firecrawl import FirecrawlApp
        _firecrawl = FirecrawlApp(api_key=self.api_key)

        return _firecrawl.search(query=query, params=options)


class FirecrawlScrapeWebsiteTool:
    @staticmethod
    def analyze_dependencies() -> Dict[str, bool]:
        """
        Analyze the dependencies required for FirecrawlScrapeWebsiteTool and return their status.
        
        Returns:
            Dictionary with dependency names as keys and their availability status as values
        """
        dependencies = {
            "requests": False,
            "firecrawl": False,
            "python-dotenv": False
        }
        
        # Check each dependency
        try:
            import requests
            dependencies["requests"] = True
        except ImportError:
            pass
        
        try:
            import firecrawl
            dependencies["firecrawl"] = True
        except ImportError:
            pass
        
        try:
            from dotenv import load_dotenv
            dependencies["python-dotenv"] = True
        except ImportError:
            pass
        
        return dependencies
    
    def __control__(self) -> bool:
        """
        Check if the required dependencies are installed and API key is available.
        
        Returns:
            True if all requirements are met
        
        Raises:
            ImportError: If required packages are not installed
            EnvironmentError: If API key is not available
        """
        # Analyze dependencies
        dependencies = self.analyze_dependencies()
        missing = [dep for dep, installed in dependencies.items() if not installed]
        
        # Print missing dependencies
        if missing:
            # Use the new printing function
            missing_dependencies("FirecrawlScrapeWebsiteTool", missing)
            
            # Raise ImportError with combined message for all missing dependencies
            install_cmd = "pip install " + " ".join(missing)
            raise ImportError(f"Missing dependencies: {', '.join(missing)}. Please install them with: {install_cmd}")
        
        # Check if FIRECRAWL_API_KEY is set in environment variables
        if "FIRECRAWL_API_KEY" not in os.environ:
            try:
                # Try to load API key from .env file
                api_key = FirecrawlScrapeWebsiteTool._load_api_key_from_env_file()
                if api_key is None:
                    # API key not found in .env file
                    missing_api_key("FirecrawlScrapeWebsiteTool", "FIRECRAWL_API_KEY")
                    raise EnvironmentError("FIRECRAWL_API_KEY environment variable is not set and could not be found in .env file")
            except ImportError:
                # If dotenv is not installed, we can't load from .env file
                missing_api_key("FirecrawlScrapeWebsiteTool", "FIRECRAWL_API_KEY", dotenv_support=False)
                raise EnvironmentError("FIRECRAWL_API_KEY environment variable is not set and python-dotenv is not installed")
        
        return True
    
    @staticmethod
    def _load_api_key_from_env_file() -> Optional[str]:
        """
        Try to load the FIRECRAWL_API_KEY from a .env file using python-dotenv.
        
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
                
                # Check if FIRECRAWL_API_KEY is now in environment
                if "FIRECRAWL_API_KEY" in os.environ:
                    return os.environ["FIRECRAWL_API_KEY"]
            
            # Move to parent directory
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # Reached root directory
                break
            current_dir = parent_dir
        
        return None
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FirecrawlScrapeWebsiteTool.
        
        Args:
            api_key: Firecrawl API key (optional, will try to load from environment if not provided)
        """
        # Set API key
        self.api_key = api_key
        
        # If API key not provided, try to load from environment or .env file
        if self.api_key is None:
            # First check environment variables
            if "FIRECRAWL_API_KEY" in os.environ:
                self.api_key = os.environ["FIRECRAWL_API_KEY"]
            else:
                # Try to load from .env file
                try:
                    api_key = self._load_api_key_from_env_file()
                    if api_key:
                        self.api_key = api_key
                    else:
                        # Print missing API key message
                        missing_api_key("FirecrawlScrapeWebsiteTool", "FIRECRAWL_API_KEY")
                        raise EnvironmentError("FIRECRAWL_API_KEY environment variable is not set and could not be found in .env file")
                except ImportError:
                    # If dotenv is not installed and no API key in environment
                    if "FIRECRAWL_API_KEY" not in os.environ:
                        # Print missing API key message without dotenv support
                        missing_api_key("FirecrawlScrapeWebsiteTool", "FIRECRAWL_API_KEY", dotenv_support=False)
                        raise EnvironmentError("FIRECRAWL_API_KEY environment variable is not set and python-dotenv is not installed")
                    self.api_key = os.environ["FIRECRAWL_API_KEY"]
        
        # Initialize FirecrawlApp
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            raise ImportError("firecrawl-py is not installed. Please install it with 'pip install firecrawl-py'")
    
    def scrape_website(self, url: str, timeout: int = 30000, only_main_content: bool = True, 
                      formats: List[str] = None, include_tags: List[str] = None, 
                      exclude_tags: List[str] = None, headers: Dict[str, str] = None, 
                      wait_for: int = 0) -> Any:
        """
        Scrape a website using Firecrawl API.
        
        Args:
            url: Website URL to scrape
            timeout: Timeout in milliseconds (default: 30000)
            only_main_content: Whether to extract only the main content (default: True)
            formats: Output formats (default: ["markdown"])
            include_tags: HTML tags to include in the extraction
            exclude_tags: HTML tags to exclude from the extraction
            headers: Custom HTTP headers to use for the request
            wait_for: Time to wait for JavaScript execution in milliseconds
            
        Returns:
            Scraped content from the website
        """
        # Set default values
        if formats is None:
            formats = ["markdown"]
        if include_tags is None:
            include_tags = []
        if exclude_tags is None:
            exclude_tags = []
        if headers is None:
            headers = {}
        
        # Prepare scrape options
        options = {
            "formats": formats,
            "onlyMainContent": only_main_content,
            "includeTags": include_tags,
            "excludeTags": exclude_tags,
            "headers": headers,
            "waitFor": wait_for,
            "timeout": timeout,
        }
        
        # Initialize FirecrawlApp and scrape the URL
        from firecrawl import FirecrawlApp
        _firecrawl = FirecrawlApp(api_key=self.api_key)
        
        return _firecrawl.scrape_url(url, options)


class FirecrawlCrawlWebsiteTool:
    @staticmethod
    def analyze_dependencies() -> Dict[str, bool]:
        """
        Analyze the dependencies required for FirecrawlCrawlWebsiteTool and return their status.
        
        Returns:
            Dictionary with dependency names as keys and their availability status as values
        """
        dependencies = {
            "requests": False,
            "firecrawl": False,
            "python-dotenv": False
        }
        
        # Check each dependency
        try:
            import requests
            dependencies["requests"] = True
        except ImportError:
            pass
        
        try:
            import firecrawl
            dependencies["firecrawl"] = True
        except ImportError:
            pass
        
        try:
            from dotenv import load_dotenv
            dependencies["python-dotenv"] = True
        except ImportError:
            pass
        
        return dependencies
    
    def __control__(self) -> bool:
        """
        Check if the required dependencies are installed and API key is available.
        
        Returns:
            True if all requirements are met
        
        Raises:
            ImportError: If required packages are not installed
            EnvironmentError: If API key is not available
        """
        # Analyze dependencies
        dependencies = self.analyze_dependencies()
        missing = [dep for dep, installed in dependencies.items() if not installed]
        
        # Print missing dependencies
        if missing:
            # Use the new printing function
            missing_dependencies("FirecrawlCrawlWebsiteTool", missing)
            
            # Raise ImportError with combined message for all missing dependencies
            install_cmd = "pip install " + " ".join(missing)
            raise ImportError(f"Missing dependencies: {', '.join(missing)}. Please install them with: {install_cmd}")
        
        # Check if FIRECRAWL_API_KEY is set in environment variables
        if "FIRECRAWL_API_KEY" not in os.environ:
            try:
                # Try to load API key from .env file
                api_key = FirecrawlCrawlWebsiteTool._load_api_key_from_env_file()
                if api_key is None:
                    # API key not found in .env file
                    missing_api_key("FirecrawlCrawlWebsiteTool", "FIRECRAWL_API_KEY")
                    raise EnvironmentError("FIRECRAWL_API_KEY environment variable is not set and could not be found in .env file")
            except ImportError:
                # If dotenv is not installed, we can't load from .env file
                missing_api_key("FirecrawlCrawlWebsiteTool", "FIRECRAWL_API_KEY", dotenv_support=False)
                raise EnvironmentError("FIRECRAWL_API_KEY environment variable is not set and python-dotenv is not installed")
        
        return True
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FirecrawlCrawlWebsiteTool.
        
        Args:
            api_key: Firecrawl API key (optional, will try to load from environment if not provided)
        """
        # Set API key
        self.api_key = api_key
        
        # If API key not provided, try to load from environment or .env file
        if self.api_key is None:
            # First check environment variables
            if "FIRECRAWL_API_KEY" in os.environ:
                self.api_key = os.environ["FIRECRAWL_API_KEY"]
            else:
                # Try to load from .env file
                try:
                    api_key = self._load_api_key_from_env_file()
                    if api_key:
                        self.api_key = api_key
                    else:
                        # Print missing API key message
                        missing_api_key("FirecrawlCrawlWebsiteTool", "FIRECRAWL_API_KEY")
                        raise EnvironmentError("FIRECRAWL_API_KEY environment variable is not set and could not be found in .env file")
                except ImportError:
                    # If dotenv is not installed and no API key in environment
                    if "FIRECRAWL_API_KEY" not in os.environ:
                        # Print missing API key message without dotenv support
                        missing_api_key("FirecrawlCrawlWebsiteTool", "FIRECRAWL_API_KEY", dotenv_support=False)
                        raise EnvironmentError("FIRECRAWL_API_KEY environment variable is not set and python-dotenv is not installed")
                    self.api_key = os.environ["FIRECRAWL_API_KEY"]
        
        # Initialize FirecrawlApp
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            raise ImportError("firecrawl-py is not installed. Please install it with 'pip install firecrawl-py'")
    
    def crawl_website(self, url: str, crawler_options: Dict[str, Any] = None, timeout: int = 30000) -> Any:
        """
        Crawl a website using Firecrawl API.
        
        Args:
            url: Website URL to crawl
            crawler_options: Options for crawling (default: {})
            timeout: Timeout in milliseconds (default: 30000)
            
        Returns:
            Crawled content from the website
        """
        # Set default values
        if crawler_options is None:
            crawler_options = {}
        
        # Prepare crawl options
        options = {
            "crawlerOptions": crawler_options,
            "timeout": timeout,
        }
        
        # Initialize FirecrawlApp and crawl the URL
        from firecrawl import FirecrawlApp
        _firecrawl = FirecrawlApp(api_key=self.api_key)
        
        return _firecrawl.crawl_url(url, options)

    @staticmethod
    def _load_api_key_from_env_file() -> Optional[str]:
        """
        Try to load the FIRECRAWL_API_KEY from a .env file using python-dotenv.
        
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
                
                # Check if FIRECRAWL_API_KEY is now in environment
                if "FIRECRAWL_API_KEY" in os.environ:
                    return os.environ["FIRECRAWL_API_KEY"]
            
            # Move to parent directory
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # Reached root directory
                break
            current_dir = parent_dir
        
        return None


class YFinanceTool:
    @staticmethod
    def analyze_dependencies() -> Dict[str, bool]:
        """
        Analyze the dependencies required for YFinanceTool and return their status.
        
        Returns:
            Dictionary with dependency names as keys and their availability status as values
        """
        dependencies = {
            "yfinance": False,
            "pandas": False
        }
        
        # Check each dependency
        try:
            import yfinance
            dependencies["yfinance"] = True
        except ImportError:
            pass
        
        try:
            import pandas
            dependencies["pandas"] = True
        except ImportError:
            pass
        
        return dependencies
    
    def __control__(self) -> bool:
        """
        Check if the required dependencies are installed and print missing ones.
        
        Returns:
            True if all requirements are met
        
        Raises:
            ImportError: If required packages are not installed
        """
        # Analyze dependencies
        dependencies = self.analyze_dependencies()
        missing = [dep for dep, installed in dependencies.items() if not installed]
        
        # Print missing dependencies
        if missing:
            # Use the new printing function
            missing_dependencies("YFinanceTool", missing)
            
            # Raise ImportError with combined message for all missing dependencies
            install_cmd = "pip install " + " ".join(missing)
            raise ImportError(f"Missing dependencies: {', '.join(missing)}. Please install them with: {install_cmd}")
        
        return True
    
    def __init__(self):
        """
        Initialize the YFinanceTool.
        """
        # Check if dependencies are installed
        self.__control__()
    
    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic information about a ticker.
        
        Args:
            ticker: The ticker symbol (e.g., 'AAPL' for Apple)
            
        Returns:
            Dictionary containing basic information about the ticker
        """
        import yfinance as yf
        
        # Get ticker object
        ticker_obj = yf.Ticker(ticker)
        
        # Get basic info
        info = ticker_obj.info
        
        return info
    
    def get_historical_data(self, ticker: str, period: str = "1mo", interval: str = "1d") -> Dict[str, Any]:
        """
        Get historical market data for a ticker.
        
        Args:
            ticker: The ticker symbol (e.g., 'AAPL' for Apple)
            period: The period to fetch data for (default: '1mo')
                Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: The interval between data points (default: '1d')
                Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
            
        Returns:
            Dictionary containing historical data
        """
        import yfinance as yf
        import pandas as pd
        
        # Get ticker object
        ticker_obj = yf.Ticker(ticker)
        
        # Get historical data
        hist = ticker_obj.history(period=period, interval=interval)
        
        # Convert DataFrame to dictionary
        hist_dict = hist.reset_index().to_dict(orient='records')
        
        return {
            "data": hist_dict,
            "period": period,
            "interval": interval,
            "ticker": ticker
        }
    
    def get_financials(self, ticker: str) -> Dict[str, Any]:
        """
        Get financial statements for a ticker.
        
        Args:
            ticker: The ticker symbol (e.g., 'AAPL' for Apple)
            
        Returns:
            Dictionary containing financial statements
        """
        import yfinance as yf
        import pandas as pd
        
        # Get ticker object
        ticker_obj = yf.Ticker(ticker)
        
        # Get financial statements
        income_stmt = ticker_obj.income_stmt
        balance_sheet = ticker_obj.balance_sheet
        cash_flow = ticker_obj.cashflow
        
        # Convert DataFrames to dictionaries
        income_stmt_dict = income_stmt.reset_index().to_dict(orient='records') if not income_stmt.empty else []
        balance_sheet_dict = balance_sheet.reset_index().to_dict(orient='records') if not balance_sheet.empty else []
        cash_flow_dict = cash_flow.reset_index().to_dict(orient='records') if not cash_flow.empty else []
        
        return {
            "income_statement": income_stmt_dict,
            "balance_sheet": balance_sheet_dict,
            "cash_flow": cash_flow_dict,
            "ticker": ticker
        }
    
    def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Search for ticker symbols based on a query.
        
        Args:
            query: The search query (e.g., 'Apple')
            limit: Maximum number of results to return (default: 10)
            
        Returns:
            List of dictionaries containing ticker symbols and company names
        """
        import yfinance as yf
        
        try:
            # Use yfinance's search functionality
            tickers = yf.Tickers(query)
            
            # Get the tickers that were found
            found_tickers = list(tickers.tickers.keys())
            
            # Limit the number of results
            found_tickers = found_tickers[:limit]
            
            # Get info for each ticker
            result = []
            for ticker_symbol in found_tickers:
                try:
                    ticker_obj = yf.Ticker(ticker_symbol)
                    info = ticker_obj.info
                    result.append({
                        "symbol": ticker_symbol,
                        "name": info.get("shortName", "Unknown"),
                        "exchange": info.get("exchange", "Unknown"),
                        "industry": info.get("industry", "Unknown")
                    })
                except Exception as e:
                    # Skip tickers that cause errors
                    continue
            
            return result
        except Exception as e:
            # If the search fails, return an empty list
            return []



# Export all tool classes
__all__ = ["Search", "ComputerUse", "BrowserUse", "Wikipedia", "DuckDuckGo", "SerperDev", "FirecrawlSearchTool", "FirecrawlScrapeWebsiteTool", "FirecrawlCrawlWebsiteTool", "YFinanceTool"] 