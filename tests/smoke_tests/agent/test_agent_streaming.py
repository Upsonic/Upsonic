"""
Test 27: Agent streaming testing
Success criteria: Agent streaming works without any error!
"""
import pytest
from io import StringIO
from contextlib import redirect_stdout

from upsonic import Agent, Task

pytestmark = pytest.mark.timeout(120)


@pytest.mark.asyncio
async def test_agent_stream_async():
    """Test Agent streaming with astream method."""
    agent = Agent(model="openai/gpt-4o", name="Streaming Agent", debug=True)
    
    task = Task(description="Write a short story about a robot learning to paint. Make it exactly 3 sentences.")
    
    output_buffer = StringIO()
    accumulated_text = ""
    
    try:
        with redirect_stdout(output_buffer):
            async for text_chunk in agent.astream(task, events=False):
                accumulated_text += text_chunk
                assert isinstance(text_chunk, str), "Stream chunks should be strings"
        
        output = output_buffer.getvalue()
        
        # Verify streaming worked
        assert accumulated_text is not None, "Should have accumulated text"
        assert len(accumulated_text) > 0, "Should have received text chunks"
        assert "robot" in accumulated_text.lower() or "paint" in accumulated_text.lower(), \
            "Streamed text should contain story content"
        
        # Verify final output
        run_output = agent.get_run_output()
        assert run_output is not None, "Run output should not be None"
        final_output = run_output.content or run_output._accumulated_text
        assert final_output is not None, "Final output should not be None"
        assert isinstance(final_output, str), "Final output should be a string"
        assert len(final_output) > 0, "Final output should not be empty"
        
        # Verify run output status
        assert run_output.is_complete, "Run should be complete"
        
    finally:
        pass  # Agent cleanup handled automatically


@pytest.mark.asyncio
async def test_agent_stream_sync():
    """Test Agent streaming with stream method (synchronous wrapper)."""
    agent = Agent(model="openai/gpt-4o", name="Streaming Agent", debug=True)
    
    task = Task(description="Count from 1 to 5, one number per line.")
    
    output_buffer = StringIO()
    accumulated_text = ""
    
    try:
        with redirect_stdout(output_buffer):
            for text_chunk in agent.stream(task, events=False):
                accumulated_text += text_chunk
                assert isinstance(text_chunk, str), "Stream chunks should be strings"
        
        output = output_buffer.getvalue()
        
        # Verify streaming worked
        assert accumulated_text is not None, "Should have accumulated text"
        assert len(accumulated_text) > 0, "Should have received text chunks"
        
        # Verify final output
        run_output = agent.get_run_output()
        assert run_output is not None, "Run output should not be None"
        final_output = run_output.content or run_output._accumulated_text
        assert final_output is not None, "Final output should not be None"
        assert isinstance(final_output, str), "Final output should be a string"
        
    finally:
        pass  # Agent cleanup handled automatically


@pytest.mark.asyncio
async def test_agent_stream_events():
    """Test Agent streaming events."""
    agent = Agent(model="openai/gpt-4o", name="Streaming Agent", debug=True)
    
    task = Task(description="What is 2 + 2?")
    
    output_buffer = StringIO()
    events_received = []
    
    try:
        with redirect_stdout(output_buffer):
            async for event in agent.astream(task, events=True):
                events_received.append(event)
                assert event is not None, "Events should not be None"
        
        # Verify events were received
        assert len(events_received) > 0, "Should have received streaming events"
        
        # Verify final output still works
        run_output = agent.get_run_output()
        assert run_output is not None, "Run output should not be None"
        final_output = run_output.content or run_output._accumulated_text
        assert final_output is not None, "Final output should not be None"
        
    finally:
        pass  # Agent cleanup handled automatically


@pytest.mark.asyncio
async def test_agent_stream_with_tools():
    """Test Agent streaming with tools."""
    from upsonic.tools import tool
    
    @tool
    def add_numbers(a: int, b: int) -> int:
        """Adds two numbers."""
        return a + b
    
    agent = Agent(
        model="openai/gpt-4o",
        name="Streaming Agent",
        tools=[add_numbers],
        debug=True
    )
    
    task = Task(description="Use the add_numbers tool to calculate 15 + 27")
    
    output_buffer = StringIO()
    accumulated_text = ""
    
    try:
        with redirect_stdout(output_buffer):
            async for text_chunk in agent.astream(task, events=False):
                accumulated_text += text_chunk
        
        output = output_buffer.getvalue()
        
        # Verify streaming worked
        assert accumulated_text is not None, "Should have accumulated text"
        assert len(accumulated_text) > 0, "Should have received text chunks"
        
        # Verify tool was called (check logs or run output)
        run_output = agent.get_run_output()
        tool_called = False
        if run_output and run_output.tools:
            tool_called = any(t.tool_name == "add_numbers" for t in run_output.tools)
        
        assert "add_numbers" in output.lower() or "42" in accumulated_text or tool_called, \
            "Tool should have been called or result mentioned"
        
        # Verify final output
        final_output = run_output.content or run_output._accumulated_text if run_output else None
        assert final_output is not None, "Final output should not be None"
        
    finally:
        pass  # Agent cleanup handled automatically

