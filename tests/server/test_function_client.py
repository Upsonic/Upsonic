from upsonic.tools_server.function_client import FunctionToolManager


def test_function_client_basic():
    # Initialize the client
    client = FunctionToolManager()

    # Get list of tools
    tool_list = client.tools()

    # Check if we got any tools
    assert len(tool_list) > 0

    # Test the first tool (add_numbers)
    first_tool = tool_list[0]

    # Check tool attributes
    assert hasattr(first_tool, "__name__")
    assert hasattr(first_tool, "__doc__")
    assert hasattr(first_tool, "__annotations__")

    # Test tool execution
    result = first_tool(1, 2)
    assert isinstance(result, dict)
    assert "result" in result


def test_get_tools_by_name():
    client = FunctionToolManager()

    # Get specific tools by name
    tools = client.get_tools_by_name(["add_numbers"])

    # Check if we got the tool
    assert len(tools) == 1
    assert tools[0].__name__ == "add_numbers"

    # Test the tool
    result = tools[0](5, 3)
    assert isinstance(result, dict)
    assert "result" in result
    assert result["result"] == 8


def test_direct_tool_call():
    client = FunctionToolManager()

    # Test direct tool call
    result = client.call_tool("add_numbers", {"a": 10, "b": 20})
    assert isinstance(result, dict)
    assert "result" in result
    assert result["result"] == 30
