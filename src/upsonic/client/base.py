from pydantic import BaseModel
from typing import Dict, Any, Any
import httpx


from .level_one.call import Call
from .level_two.agent import Agent
from .storage.storage import Storage
from .tools.tools import Tools
from .markdown.markdown import Markdown
from .others.others import Others

from .printing import connected_to_server

class ServerStatusException(Exception):
    """Custom exception for server status check failures."""
    pass

class TimeoutException(Exception):
    """Custom exception for request timeout."""
    pass

# Create a base class with url
class UpsonicClient(Call, Storage, Tools, Agent, Markdown, Others):


    def __init__(self, url: str, debug: bool = False):

        self.debug = debug

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




        self.url = url
        self.default_llm_model = "gpt-4o"
        self.url = url
        self.default_llm_model = "gpt-4o"
        if not self.status():
            connected_to_server(self.server_type, "Failed")
            raise ServerStatusException("Failed to connect to the server at initialization.")
    
        connected_to_server(self.server_type, "Established")


    def set_default_llm_model(self, llm_model: str):
        self.default_llm_model = llm_model


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