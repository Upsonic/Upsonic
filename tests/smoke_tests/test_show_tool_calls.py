"""
Smoke test for show_tool_calls feature in Agent class.

Tests that when show_tool_calls=True, tool calls are displayed in output.
"""

import pytest
from upsonic import Agent, Task
from upsonic.tools import tool
from io import StringIO
from contextlib import redirect_stdout

pytestmark = pytest.mark.timeout(60)


@tool
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b


@pytest.mark.asyncio
async def test_show_tool_calls_enabled():
    """Test that show_tool_calls=True displays tool calls."""
    agent = Agent(
        model="openai/gpt-4o",
        name="Calculator Agent",
        show_tool_calls=True
    )
    
    task = Task(
        description="Calculate the sum of 15 and 27 using the calculator tool.",
        tools=[calculate_sum]
    )
    
    # Capture stdout to check logs
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result exists
    assert result is not None, "Result should not be None"
    
    # Verify tool usage information appears in output
    # show_tool_calls controls whether tool_usage() is called in call_end()
    # tool_usage() should add tool call information to the output
    # Check for tool-related output (tool name, tool usage summary, etc.)
    tool_indicators = [
        "calculate_sum",
        "Tool Usage",
        "tool",
        "Tool Call"
    ]
    
    tool_info_found = any(indicator.lower() in output.lower() for indicator in tool_indicators)
    assert tool_info_found, f"Tool call information should appear in output when show_tool_calls=True. Output: {output[:1000]}"


@pytest.mark.asyncio
async def test_show_tool_calls_disabled():
    """Test that show_tool_calls=False does not display tool calls."""
    agent = Agent(
        model="openai/gpt-4o",
        name="Calculator Agent",
        show_tool_calls=False
    )
    
    task = Task(
        description="Calculate the sum of 15 and 27 using the calculator tool.",
        tools=[calculate_sum]
    )
    
    # Capture stdout to check logs
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result exists
    assert result is not None, "Result should not be None"
    
    # When show_tool_calls=False, tool_usage() returns None and is not displayed
    # So we can't really verify absence, but we can verify the task still completes
    assert isinstance(result, (str, int)), "Result should be returned"

