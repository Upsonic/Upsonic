import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
import sys
import os
import asyncio
from fastapi import FastAPI
from httpx import AsyncClient

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from upsonic.tools_server.server.function_tools import app, _get_json_type

@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

def test_get_json_type():
    assert _get_json_type(str) == "string"
    assert _get_json_type(int) == "integer"
    assert _get_json_type(bool) == "boolean"
    assert _get_json_type(float) == "number"
    assert _get_json_type(list) == "array"
    assert _get_json_type(dict) == "object"
    assert _get_json_type(object) == "string"  # Default case

@pytest.mark.asyncio
async def test_list_tools(async_client):
    response = await async_client.post("/functions/tools")
    assert response.status_code == 200

    data = response.json()
    assert "available_tools" in data
    tools = data["available_tools"]["tools"]

    # Should have add_numbers and concat_strings
    assert len(tools) == 4

    # Verify add_numbers
    add_numbers = next(t for t in tools if t["name"] == "add_numbers")
    assert add_numbers["description"] == "Add two numbers together"
    assert add_numbers["inputSchema"]["required"] == ["a", "b"]

    # Verify concat_strings
    concat_strings = next(t for t in tools if t["name"] == "concat_strings")
    assert concat_strings["description"] == "Concatenate two strings"
    assert concat_strings["inputSchema"]["required"] == ["str1", "str2"]

@pytest.mark.asyncio
async def test_example_functions(async_client):
    try:
        # Test add_numbers with timeout using asyncio.timeout
        async with asyncio.timeout(5):  # 5 second timeout
            response = await async_client.post(
                "/functions/call_tool",
                json={"tool_name": "add_numbers", "arguments": {"a": 10, "b": 20}},
            )
            assert response.status_code == 200
            assert response.json() == {"result": 30}

        # Test concat_strings
        async with asyncio.timeout(5):
            response = await async_client.post(
                "/functions/call_tool",
                json={"tool_name": "concat_strings", "arguments": {"str1": "Hello, ", "str2": "World!"}},
            )
            assert response.status_code == 200
            assert response.json() == {"result": "Hello, World!"}
    except asyncio.TimeoutError:
        pytest.fail("Test timed out")

@pytest.mark.asyncio
async def test_error_handling(async_client):
    try:
        # Test non-existent tool
        async with asyncio.timeout(5):
            response = await async_client.post(
                "/functions/call_tool", json={"tool_name": "non_existent", "arguments": {}}
            )
            assert response.status_code == 404
            error_response = response.json()
            assert "detail" in error_response
            assert error_response["detail"] == "Tool non_existent not found"

        # Test invalid arguments
        async with asyncio.timeout(5):
            response = await async_client.post(
                "/functions/call_tool",
                json={"tool_name": "add_numbers", "arguments": {"a": "not_a_number", "b": 20}},
            )
            # The API returns a 200 status code with an error message in the response
            assert response.status_code == 200
            error_response = response.json()
            assert "error" in error_response or "detail" in error_response
            # The error message should indicate a failure to call the tool
            error_msg = error_response.get("error", error_response.get("detail", ""))
            assert "failed to call tool" in error_msg.lower() or "can only concatenate" in error_msg.lower()
    except asyncio.TimeoutError:
        pytest.fail("Test timed out")
