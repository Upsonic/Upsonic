from pydantic import BaseModel
from typing import Dict, Any, Any
import httpx
import time
import asyncio
import concurrent.futures
import threading



from .level_two.agent import Agent
from .tasks.tasks import Task
from .agent_configuration.agent_configuration import AgentConfiguration
from .storage.storage import Storage, ClientConfig
from .tools.tools import Tools
from .markdown.markdown import Markdown
from .others.others import Others
from ..exception import ServerStatusException, TimeoutException

from .printing import connected_to_server


from .latest_upsonic_client import latest_upsonic_client


# Helper function to run a coroutine in a new thread with a new event loop
def run_coroutine_in_new_thread(coro):
    """
    Run a coroutine in a new thread with a new event loop.
    This is useful when we're in an async context but need a synchronous result.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
    """
    def run_coro_in_thread(coro):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(run_coro_in_thread, coro).result()


# Create a base class with url
class UpsonicClient(Storage, Tools, Agent, Markdown, Others):

    def __init__(self, url: str, debug: bool = False, **kwargs):
        """Initialize the Upsonic client.
        
        Args:
            url: The server URL to connect to
            debug: Whether to enable debug mode
            **kwargs: Configuration options that match ClientConfig fields
        """
        start_time = time.time()
        self.debug = debug

        # Set server type and URL first
        if "0.0.0.0" in url:
            self.server_type = "Local(Docker)"
        elif "localhost" in url:
            self.server_type = "Local(Docker)"
        elif "upsonic.ai" in url:
            self.server_type = "Cloud(Upsonic)"
        elif "devserver" in url or "localserver" in url:
            self.server_type = "Local(LocalServer)"
        else:
            self.server_type = "Cloud(Unknown)"

        # Handle local server setup
        if url == "devserver" or url == "localserver":
            url = "http://localhost:7541"
            from ..server import run_dev_server, stop_dev_server, is_tools_server_running, is_main_server_running
            if debug:
                run_dev_server(redirect_output=False)
            else:
                run_dev_server(redirect_output=True)

            import atexit
            def exit_handler():
                if is_tools_server_running() or is_main_server_running():
                    stop_dev_server()
            atexit.register(exit_handler)

        # Set URL and default model
        self.url = url
        self.default_llm_model = "openai/gpt-4o"

        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            in_async_context = True
        except RuntimeError:
            in_async_context = False

        # Check server status before proceeding
        if in_async_context:
            # We're in an async context, but __init__ can't be async
            # We need to run the async method in a new thread
            status_ok = run_coroutine_in_new_thread(self.status_async())
        else:
            # We're not in an async context, use asyncio.run
            status_ok = asyncio.run(self.status_async())
            
        if not status_ok:
            total_time = time.time() - start_time
            connected_to_server(self.server_type, "Failed", total_time)
            raise ServerStatusException("Failed to connect to the server at initialization.")
        
        # Handle configuration through ClientConfig model
        config = ClientConfig(**(kwargs or {}))
        
        # Create a dictionary of non-None values
        config_dict = {
            key: str(value) for key, value in config.model_dump().items() 
            if value is not None
        }
        
        # Bulk set the configurations if there are any
        if config_dict:
            if in_async_context:
                # We're in an async context, but __init__ can't be async
                # We need to run the async method in a new thread
                run_coroutine_in_new_thread(self.bulk_set_config_async(config_dict))
            else:
                # We're not in an async context, use asyncio.run
                asyncio.run(self.bulk_set_config_async(config_dict))

        global latest_upsonic_client
        latest_upsonic_client = self
        total_time = time.time() - start_time
        connected_to_server(self.server_type, "Established", total_time)

    def status(self) -> bool:
        """Check the server status."""
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # We're in an async context, but this is a sync method
                    # We need to run the async method in a new thread
                    return run_coroutine_in_new_thread(self.status_async())
            except RuntimeError:
                # No event loop is running, use asyncio.run
                return asyncio.run(self.status_async())
        except httpx.RequestError:
            return False

    async def status_async(self) -> bool:
        """Check the server status asynchronously."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.url + "/status")
                return response.status_code == 200
        except httpx.RequestError:
            return False

    def send_request(self, endpoint: str, data: Dict[str, Any], files: Dict[str, Any] = None, method: str = "POST", return_raw: bool = False) -> Any:
        """
        General method to send an API request.

        Args:
            endpoint: The API endpoint to send the request to.
            data: The data to send in the request.
            files: Optional files to upload.
            method: HTTP method to use (GET or POST)
            return_raw: Whether to return the raw response content instead of JSON

        Returns:
            The response from the API, either as JSON or raw content.
        """
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # We're in an async context, but this is a sync method
                    # We need to run the async method in a new thread
                    return run_coroutine_in_new_thread(
                        self.send_request_async(endpoint, data, files, method, return_raw)
                    )
            except RuntimeError:
                # No event loop is running, use asyncio.run
                return asyncio.run(self.send_request_async(endpoint, data, files, method, return_raw))
        except httpx.RequestError as e:
            raise e

    async def send_request_async(self, endpoint: str, data: Dict[str, Any], files: Dict[str, Any] = None, method: str = "POST", return_raw: bool = False) -> Any:
        """
        Asynchronous version of send_request.
        General method to send an API request asynchronously.

        Args:
            endpoint: The API endpoint to send the request to.
            data: The data to send in the request.
            files: Optional files to upload.
            method: HTTP method to use (GET or POST)
            return_raw: Whether to return the raw response content instead of JSON

        Returns:
            The response from the API, either as JSON or raw content.
        """
        async with httpx.AsyncClient() as client:
            if method.upper() == "GET":
                response = await client.get(self.url + endpoint, params=data, timeout=600.0)
            else:
                if files:
                    response = await client.post(self.url + endpoint, data=data, files=files, timeout=600.0)
                else:
                    response = await client.post(self.url + endpoint, json=data, timeout=600.0)
                
            if response.status_code == 408:
                raise TimeoutException("Request timed out")
            response.raise_for_status()
            
            return response.content if return_raw else response.json()

    def run(self, *args, **kwargs):
        """
        Run method that delegates to the appropriate async implementation.
        """
        try:
            # Try to get the current event loop
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # We're in an async context, but this is a sync method
                    # We need to run the async method in a new thread
                    return run_coroutine_in_new_thread(
                        self.run_async(*args, **kwargs)
                    )
            except RuntimeError:
                # No event loop is running, use asyncio.run
                return asyncio.run(self.run_async(*args, **kwargs))
        except Exception as e:
            raise e

    async def run_async(self, *args, **kwargs):
        """
        Asynchronous version of the run method.
        """
        llm_model = kwargs.get("llm_model", None)

        # If there is an two positional arguments we will run it in self.agent_async(first argument, second argument)
        if len(args) == 2:
            
            if isinstance(args[0], AgentConfiguration) and isinstance(args[1], Task):
                return await self.agent_async(args[0], args[1])
            elif isinstance(args[0], list):
                return await self.multi_agent_async(args[0], args[1])
        

        if len(args) == 1:
            return await self.call_async(args[0], llm_model=llm_model)