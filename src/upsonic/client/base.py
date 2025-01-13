from pydantic import BaseModel
from typing import Dict, Any, Any
import httpx


from .level_one.call import Call
from .level_two.agent import Agent
from .storage.storage import Storage
from .tools.tools import Tools


from .printing import connected_to_server

class ServerStatusException(Exception):
    """Custom exception for server status check failures."""
    pass

class TimeoutException(Exception):
    """Custom exception for request timeout."""
    pass

# Create a base class with url
class UpsonicClient(Call, Storage, Tools, Agent):


    def __init__(self, url: str):



        if "0.0.0.0" in url:
            self.server_type = "Local(Docker)"
        elif "localhost" in url:
            self.server_type = "Local(Docker)"

        elif "upsonic.ai" in url:
            self.server_type = "Cloud(Upsonic)"
        elif "devserver" in url:
            self.server_type = "Local(DevServer)"
        else:
            self.server_type = "Cloud(Unknown)"

        if url == "devserver":
            
            url = "http://0.0.0.0:7541"
            from ..server import run_dev_server, stop_dev_server, is_tools_server_running, is_main_server_running
            run_dev_server()

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

    def send_request(self, endpoint: str, data: Dict[str, Any]) -> Any:
        """
        General method to send an API request.

        Args:
            endpoint: The API endpoint to send the request to.
            data: The data to send in the request.

        Returns:
            The response from the API.
        """
        with httpx.Client() as client:

            response = client.post(self.url + endpoint, json=data, timeout=600.0)
            if response.status_code == 408:
                raise TimeoutException("Request timed out")
            response.raise_for_status()
            return response.json()