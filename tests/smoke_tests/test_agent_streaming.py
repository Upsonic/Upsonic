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
    """Test Agent streaming with stream_async method."""
    agent = Agent(model="openai/gpt-4o", name="Streaming Agent", debug=True)
    
    task = Task(description="Write a short story about a robot learning to paint. Make it exactly 3 sentences.")
    
    output_buffer = StringIO()
    accumulated_text = ""
    
    try:
        with redirect_stdout(output_buffer):
            stream_result = await agent.stream_async(task)
            
            async with stream_result:
                async for text_chunk in stream_result.stream_output():
                    accumulated_text += text_chunk
                    assert isinstance(text_chunk, str), "Stream chunks should be strings"
        
        output = output_buffer.getvalue()
        
        # Verify streaming worked
        assert accumulated_text is not None, "Should have accumulated text"
        assert len(accumulated_text) > 0, "Should have received text chunks"
        assert "robot" in accumulated_text.lower() or "paint" in accumulated_text.lower(), \
            "Streamed text should contain story content"
        
        # Verify final output
        final_output = stream_result.get_final_output()
        assert final_output is not None, "Final output should not be None"
        assert isinstance(final_output, str), "Final output should be a string"
        assert len(final_output) > 0, "Final output should not be empty"
        
        # Verify stream result attributes
        assert hasattr(stream_result, '_is_complete'), "Stream result should track completion"
        assert stream_result._is_complete is True, "Stream should be complete"
        
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
            stream_result = agent.stream(task)
            
            async with stream_result:
                async for text_chunk in stream_result.stream_output():
                    accumulated_text += text_chunk
                    assert isinstance(text_chunk, str), "Stream chunks should be strings"
        
        output = output_buffer.getvalue()
        
        # Verify streaming worked
        assert accumulated_text is not None, "Should have accumulated text"
        assert len(accumulated_text) > 0, "Should have received text chunks"
        
        # Verify final output
        final_output = stream_result.get_final_output()
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
            stream_result = await agent.stream_async(task)
            
            async with stream_result:
                async for event in stream_result.stream_events():
                    events_received.append(event)
                    assert event is not None, "Events should not be None"
        
        # Verify events were received
        assert len(events_received) > 0, "Should have received streaming events"
        
        # Verify final output still works
        final_output = stream_result.get_final_output()
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
            stream_result = await agent.stream_async(task)
            
            async with stream_result:
                async for text_chunk in stream_result.stream_output():
                    accumulated_text += text_chunk
        
        output = output_buffer.getvalue()
        
        # Verify streaming worked
        assert accumulated_text is not None, "Should have accumulated text"
        assert len(accumulated_text) > 0, "Should have received text chunks"
        
        # Verify tool was called (check logs)
        assert "add_numbers" in output.lower() or "42" in accumulated_text, \
            "Tool should have been called or result mentioned"
        
        # Verify final output
        final_output = stream_result.get_final_output()
        assert final_output is not None, "Final output should not be None"
        
    finally:
        pass  # Agent cleanup handled automatically

