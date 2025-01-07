from upsonic.server.tools.mcp_client import MCPToolManager
import time


def test_get_tools_by_name():
    client = MCPToolManager(command="uvx", args=["mcp-server-fetch"])

    # Get specific tools by name
    tools = client.get_tools_by_name(["fetch"])

    # Check if we got the tool
    assert len(tools) == 1
    assert tools[0].__name__ == "fetch"
    time.sleep(5)

    # Test the tool
    result = tools[0](url="http://localhost:8086/status")
    assert isinstance(result, dict)
    assert "result" in result
    assert not result["result"].get("isError", True)
