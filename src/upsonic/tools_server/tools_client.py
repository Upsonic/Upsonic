import base64
import httpx
from typing import Dict, List, Any, Callable, Optional


class ToolManager:
    """Client for interacting with the Upsonic Functions API."""

    def __init__(self):
        """Initialize the Upsonic Function client."""
        self.base_url = "http://localhost:8086"


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def close(self):
        """Close the client session."""
        pass



    def install_library(self, library: str) -> Dict[str, Any]:
        """
        Call a specific tool with the given arguments.

        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments to pass to the tool

        Returns:
            Tool execution results
        """
        with httpx.Client(timeout=600.0) as session:
            response = session.post(
                f"{self.base_url}/tools/install_library",
                json={"library": library},
            )
            response.raise_for_status()
            return response.json()
        
    def uninstall_library(self, library: str) -> Dict[str, Any]:
        """
        Uninstall a library.
        """
        with httpx.Client(timeout=600.0) as session:
            response = session.post(
                f"{self.base_url}/tools/uninstall_library",
                json={"library": library},
            )
            response.raise_for_status()
            return response.json()




    def add_tool(self, function) -> Dict[str, Any]:
        """
        Add a tool.
        """
        with httpx.Client(timeout=600.0) as session:
            response = session.post(
                f"{self.base_url}/tools/add_tool",
                json={"function": function},
            )
            response.raise_for_status()
            return response.json()


    def add_mcp_tool(self, name: str, command: str, args: List[str], env: Dict[str, str]) -> Dict[str, Any]:
        """
        Add a tool.
        """
        with httpx.Client(timeout=600.0) as session:
            response = session.post(
                f"{self.base_url}/tools/add_mcp_tool",
                json={"name": name, "command": command, "args": args, "env": env},
            )
            response.raise_for_status()
            return response.json()
        
    def add_sse_mcp(self, name: str, url: str) -> Dict[str, Any]:
        """
        Add a tool.
        """
        with httpx.Client(timeout=600.0) as session:
            response = session.post(
                f"{self.base_url}/tools/add_sse_mcp",
                json={"name": name, "url": url},
            )
            response.raise_for_status()
            return response.json()

