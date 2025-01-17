import traceback
from fastapi import HTTPException
from pydantic import BaseModel
import inspect
from typing import Any, Dict, List, Type, Callable
from functools import wraps

from .api import app, timeout

prefix = "/functions"

# Registry to store decorated functions
registered_functions: Dict[str, Dict[str, Any]] = {}


def _get_json_type(python_type: Type) -> str:
    """Convert Python type to JSON schema type."""
    type_mapping = {
        str: "string",
        int: "integer",
        bool: "boolean",
        float: "number",
        list: "array",
        dict: "object",
    }
    return type_mapping.get(python_type, "string")


def tool(description: str = "", custom_properties: Dict[str, Any] = None, custom_required: List[str] = None):
    """
    Decorator to register a function as a tool.

    Args:
        description: Optional description of the tool. If not provided, function's docstring will be used.
    """

    def decorator(func: Callable):
        sig = inspect.signature(func)


        # Get parameter info
        properties = {}
        required = []

        # Extract description from docstring if not provided
        tool_description = description
        if not tool_description and func.__doc__:
            # Get the first line of the docstring as description
            tool_description = func.__doc__.strip().split('\n')[0].strip()


        
        for param_name, param in sig.parameters.items():
            param_type = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )
            param_default = (
                None if param.default == inspect.Parameter.empty else param.default
            )

            properties[param_name] = {
                "type": _get_json_type(param_type),
                "description": f"Parameter {param_name}",
            }

            if param_default is not None:
                properties[param_name]["default"] = param_default
            
            # If parameter has no default value, it's required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)


        if custom_properties is not None:
            properties = custom_properties

        if custom_required is not None:
            required = custom_required

        # Register the function with the extracted description
        registered_functions[func.__name__] = {
            "function": func,
            "description": tool_description,
            "properties": properties,
            "required": required,
        }

        # Check if the function is async
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict


@app.post(f"{prefix}/tools")
@timeout(30.0)
async def list_tools():


    tools = []
    for name, info in registered_functions.items():

        tools.append(
            {
                "name": name,
                "description": info["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": info["properties"],
                    "required": info["required"],
                },
            }
        )

    return {"available_tools": {"tools": tools}}


@app.post(f"{prefix}/call_tool")
@timeout(30.0)
async def call_tool(request: ToolRequest):


    if request.tool_name not in registered_functions:
        raise HTTPException(
            status_code=404, detail=f"Tool {request.tool_name} not found"
        )

    try:
        func = registered_functions[request.tool_name]["function"]
        # Check if the function is async
        is_async = inspect.iscoroutinefunction(func)
        
        if is_async:
            result = await func(**request.arguments)
        else:
            result = func(**request.arguments)
            

        return {"result": result}
    except Exception as e:

        return {"status_code": 500, "detail": f"Failed to call tool: {str(e)}"}


# Example decorated functions
@tool()
async def add_numbers(a: int, b: int, c: int=0) -> int:
    "Add two numbers together"
    return a + b + c


@tool()
def concat_strings(str1: str, str2: str) -> str:
    "Concatenate two strings"
    return str1 + str2


@tool()
def Search__google(query: str, max_number: int = 20) -> list:
    """
    Search the query on Google and return the results.
    """
    try:
        from googlesearch import search as gsearch

        return list(gsearch(query, stop=max_number))
    except:
        return "An exception occurred"
    

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
@tool()
def Search__read_website(url: str, max_content_length: int = 5000) -> dict:
    """
    Read the content of a website and return the title, meta data, content, and sub-links.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        html = response.text
    except requests.RequestException as e:
        return {"error": f"Failed to retrieve the website content: {e}"}

    soup = BeautifulSoup(html, "html.parser")

    meta_properties = [
        "og:description",
        "og:site_name",
        "og:title",
        "og:type",
        "og:url",
        "description",
        "keywords",
        "author",
    ]
    meta = {}
    for property_name in meta_properties:
        tag = soup.find("meta", property=property_name) or soup.find(
            "meta", attrs={"name": property_name}
        )
        if tag:
            meta[property_name] = tag.get("content", "")

    for ignore_tag in soup(["script", "style"]):
        ignore_tag.decompose()

    title = soup.title.string.strip() if soup.title else ""
    content = soup.body.get_text(separator="\n") if soup.body else ""

    links = []
    for a in soup.find_all("a", href=True):
        link_url = urljoin(url, a["href"])
        links.append({"title": a.text.strip(), "link": link_url})

    content = re.sub(r"[\n\r\t]+", "\n", content)
    content = re.sub(r" +", " ", content)
    content = re.sub(r"[\n ]{3,}", "\n\n", content)
    content = content.strip()

    if len(content) > max_content_length:
        content = content[:max_content_length].rsplit(" ", 1)[0] + "..."

    return {"meta": meta, "title": title, "content": content, "sub_links": links}