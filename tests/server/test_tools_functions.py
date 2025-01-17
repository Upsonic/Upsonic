import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from upsonic.tools_server.server.function_tools import app, _get_json_type

client = TestClient(app)


def test_get_json_type():
    assert _get_json_type(str) == "string"
    assert _get_json_type(int) == "integer"
    assert _get_json_type(bool) == "boolean"
    assert _get_json_type(float) == "number"
    assert _get_json_type(list) == "array"
    assert _get_json_type(dict) == "object"
    assert _get_json_type(object) == "string"  # Default case


@pytest.mark.asyncio
async def test_list_tools():
    response = client.post("/functions/tools")
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
async def test_example_functions():
    # Test add_numbers
    response = client.post(
        "/functions/call_tool",
        json={"tool_name": "add_numbers", "arguments": {"a": 10, "b": 20}},
    )
    assert response.status_code == 200
    assert response.json()["result"] == 30

    # Test concat_strings
    response = client.post(
        "/functions/call_tool",
        json={
            "tool_name": "concat_strings",
            "arguments": {"str1": "Hello, ", "str2": "World!"},
        },
    )
    assert response.status_code == 200
    assert response.json()["result"] == "Hello, World!"


@pytest.mark.asyncio
async def test_error_handling():
    # Test non-existent tool
    response = client.post(
        "/functions/call_tool", json={"tool_name": "non_existent", "arguments": {}}
    )
    assert response.status_code == 404

    # Test invalid arguments
    response = client.post(
        "/functions/call_tool",
        json={"tool_name": "add_numbers", "arguments": {"a": "not_a_number", "b": 3}},
    )
    assert response.json()["status_code"] == 500
