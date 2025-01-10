from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import traceback
from ..api import app, timeout
from ...tools_server.tools_client import ToolManager
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cloudpickle
import base64


prefix = "/tools"


class InstallLibraryRequest(BaseModel):
    library: str


class CustomToolRequest(BaseModel):
    function: str


@app.post(f"{prefix}/install_library")
async def install_library(request: InstallLibraryRequest):
    """
    Endpoint to install a library.

    Args:
        library: The library to install

    Returns:
        A success message
    """
    with ToolManager() as tool_client:
        tool_client.install_library(request.library)
    return {"message": "Library installed successfully"}



@app.post(f"{prefix}/uninstall_library")
async def uninstall_library(request: InstallLibraryRequest):
    """
    Endpoint to uninstall a library.
    """
    with ToolManager() as tool_client:
        tool_client.uninstall_library(request.library)
    return {"message": "Library uninstalled successfully"}


class AddToolRequest(BaseModel):
    function: Any

@app.post(f"{prefix}/add_tool")
async def add_tool(request: AddToolRequest):
    """
    Endpoint to add a tool.
    """
    with ToolManager() as tool_client:
        tool_client.add_tool(request.function)
    return {"message": "Tool added successfully"}


class AddMCPToolRequest(BaseModel):
    command: str
    args: List[str]
    env: Dict[str, str]


@app.post(f"{prefix}/add_mcp_tool")
async def add_mcp_tool(request: AddMCPToolRequest):
    """
    Endpoint to add a tool.
    """
    with ToolManager() as tool_client:
        tool_client.add_mcp_tool(request.command, request.args, request.env)
    return {"message": "Tool added successfully"}