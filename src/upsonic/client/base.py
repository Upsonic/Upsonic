from pydantic import BaseModel
from typing import Dict, Any, Any
import httpx
import time


from .level_one.call import Call
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


# Create a base class with url
class UpsonicClient(Call, Storage, Tools, Agent, Markdown, Others):

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

        # Check server status before proceeding
        
        if not self.status():
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
            self.bulk_set_config(config_dict)

        global latest_upsonic_client
        latest_upsonic_client = self
        total_time = time.time() - start_time
        connected_to_server(self.server_type, "Established", total_time)

    def status(self) -> bool:
        """Check the server status."""
        try:
            with httpx.Client() as client:
                response = client.get(self.url + "/status")
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
        with httpx.Client() as client:
            if method.upper() == "GET":
                response = client.get(self.url + endpoint, params=data, timeout=600.0)
            else:
                if files:
                    response = client.post(self.url + endpoint, data=data, files=files, timeout=600.0)
                else:
                    response = client.post(self.url + endpoint, json=data, timeout=600.0)
                
            if response.status_code == 408:
                raise TimeoutException("Request timed out")
            response.raise_for_status()
            
            return response.content if return_raw else response.json()
        

    def run(self, *args, **kwargs):

        llm_model = kwargs.get("llm_model", None)

        # If there is an two positional arguments we will run it in self.agent(first argument, second argument)
        if len(args) == 2:
            
            if isinstance(args[0], AgentConfiguration) and isinstance(args[1], Task):
                return self.agent(args[0], args[1])
            elif isinstance(args[0], list):
                return self.multi_agent(args[0], args[1])
        

        if len(args) == 1:
            return self.call(args[0], llm_model=llm_model)