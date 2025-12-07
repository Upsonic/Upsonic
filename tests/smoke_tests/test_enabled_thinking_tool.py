"""
Smoke test for enabled_thinking_tool feature.

Tests that when enabled_thinking_tool=True, we see specific logs:
- "Orchestrator Activated:"
- "Executing Tool Step"
- "Plan complete. Preparing for final synthesis."
"""

import pytest
from upsonic import Agent, Task
from upsonic.tools import tool
from io import StringIO
from contextlib import redirect_stdout

pytestmark = pytest.mark.timeout(120)


@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"Weather in {city}: Sunny, 25°C"


@tool
def get_population(city: str) -> str:
    """Get population information for a city."""
    return f"Population of {city}: 2.1 million"


@pytest.mark.asyncio
async def test_enabled_thinking_tool_logs():
    """Test that enabled_thinking_tool produces expected logs."""
    agent = Agent(
        model="openai/gpt-4o",
        name="Planning Agent",
        enable_thinking_tool=True
    )
    
    task = Task(
        description="Get the weather and population for Paris, then provide a summary. You need to use multiple tools to complete this task.",
        tools=[get_weather, get_population]
    )
    
    # Capture stdout to check logs
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result exists
    assert result is not None, "Result should not be None"
    
    # Verify expected logs appear
    assert "Orchestrator Activated" in output, f"'Orchestrator Activated' should appear in logs. Output: {output[:1000]}"
    assert "Executing Tool Step" in output, f"'Executing Tool Step' should appear in logs. Output: {output[:1000]}"
    assert "Plan complete. Preparing for final synthesis" in output, f"'Plan complete. Preparing for final synthesis' should appear in logs. Output: {output[:1000]}"

