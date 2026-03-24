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
    agent = Agent(model="anthropic/claude-sonnet-4-6", name="Streaming Agent", debug=True)
    
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
        final_output = run_output.output or run_output.accumulated_text
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
    agent = Agent(model="anthropic/claude-sonnet-4-6", name="Streaming Agent", debug=True)
    
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
        final_output = run_output.output or run_output.accumulated_text
        assert final_output is not None, "Final output should not be None"
        assert isinstance(final_output, str), "Final output should be a string"
        
    finally:
        pass  # Agent cleanup handled automatically


@pytest.mark.asyncio
async def test_agent_stream_events():
    """Test Agent streaming events."""
    agent = Agent(model="anthropic/claude-sonnet-4-6", name="Streaming Agent", debug=True)
    
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
        final_output = run_output.output or run_output.accumulated_text
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
        model="anthropic/claude-sonnet-4-6",
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
        final_output = run_output.output or run_output.accumulated_text if run_output else None
        assert final_output is not None, "Final output should not be None"

    finally:
        pass  # Agent cleanup handled automatically


@pytest.mark.asyncio
async def test_streaming_price_id_isolated_per_run():
    """Each streaming run should get its own price_id for cost tracking isolation."""
    agent = Agent(model="anthropic/claude-sonnet-4-6", name="Cost Tracking Agent")

    task1 = Task(description="Say hello")
    task2 = Task(description="Say goodbye")

    async for _ in agent.astream(task1):
        pass
    price_id_1 = task1.price_id_

    async for _ in agent.astream(task2):
        pass
    price_id_2 = task2.price_id_

    assert price_id_1 is not None, "Task 1 should have a price_id"
    assert price_id_2 is not None, "Task 2 should have a price_id"
    assert price_id_1 != price_id_2, "Each streaming run must have its own price_id"


@pytest.mark.asyncio
async def test_streaming_tool_count_reset_between_runs():
    """Agent _tool_call_count must reset between streaming runs."""
    from upsonic.tools import tool

    @tool
    def greet(name: str) -> str:
        """Greet a person by name."""
        return f"Hello, {name}!"

    agent = Agent(
        model="anthropic/claude-sonnet-4-6",
        name="Tool Reset Agent",
        tools=[greet],
    )

    task1 = Task(description="Use the greet tool to greet Alice")
    async for _ in agent.astream(task1):
        pass

    count_after_first = agent._tool_call_count

    task2 = Task(description="Use the greet tool to greet Bob")
    async for _ in agent.astream(task2):
        pass

    count_after_second = agent._tool_call_count

    assert count_after_first > 0, "First run should have made tool calls"
    assert count_after_second > 0, "Second run should have made tool calls"
    assert count_after_second <= count_after_first + 5, \
        "Tool count should not accumulate across runs indefinitely"


@pytest.mark.asyncio
async def test_streaming_task_response_available_after_stream():
    """task.response should be set after streaming completes."""
    agent = Agent(model="anthropic/claude-sonnet-4-6", name="Response Agent")

    task = Task(description="What is 2 + 2? Answer with just the number.")

    accumulated = ""
    async for chunk in agent.astream(task):
        accumulated += chunk

    assert task.response is not None, "task.response should be set after streaming"
    assert len(str(task.response)) > 0, "task.response should not be empty"
    assert accumulated is not None and len(accumulated) > 0, "Should have streamed text"


@pytest.mark.asyncio
async def test_streaming_cost_tracking_works():
    """task.total_cost should be available after streaming (CallManagementStep runs)."""
    agent = Agent(model="anthropic/claude-sonnet-4-6", name="Cost Agent")

    task = Task(description="Say hi")

    async for _ in agent.astream(task):
        pass

    assert task.price_id_ is not None, "price_id should be set"
    assert task.total_input_token is not None, "total_input_token should be tracked"
    assert task.total_output_token is not None, "total_output_token should be tracked"
    assert task.total_input_token > 0, "Should have used input tokens"
    assert task.total_output_token > 0, "Should have used output tokens"


@pytest.mark.asyncio
async def test_streaming_run_output_complete():
    """AgentRunOutput should be complete with proper status after streaming."""
    agent = Agent(model="anthropic/claude-sonnet-4-6", name="Status Agent")

    task = Task(description="Say one word")

    async for _ in agent.astream(task):
        pass

    run_output = agent.get_run_output()
    assert run_output is not None, "Run output should exist"
    assert run_output.is_complete, "Run should be marked complete"
    assert run_output.output is not None, "Output should be set"
    assert run_output.run_id is not None, "Run ID should be set"
    assert task.run_id is not None, "Task should have run_id"
    assert task.status is not None, "Task status should be set"
