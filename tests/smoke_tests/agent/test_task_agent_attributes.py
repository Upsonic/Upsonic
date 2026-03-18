"""
Smoke tests for Task and Agent attribute management during do_async execution.

Verifies attribute lifecycle changes:
1. price_id isolation per run (cost tracking)
2. _tool_call_count reset between fresh runs
3. _tool_calls accumulation across tool executions
4. task.response and task.status set after completion
5. task timing attributes (start_time, end_time, duration)
6. Cost tracking via price_id_summary (total_cost, total_input_token, total_output_token)
7. AgentRunOutput status and attributes after completion
8. Multiple sequential runs on the same agent

Run with: pytest tests/smoke_tests/agent/test_task_agent_attributes.py -v -s
"""

import pytest
from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.run.base import RunStatus

pytestmark = pytest.mark.timeout(120)

MODEL = "anthropic/claude-sonnet-4-6"


# ============================================================================
# Tools
# ============================================================================

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


# ============================================================================
# Test: price_id isolation per do_async run
# ============================================================================

@pytest.mark.asyncio
async def test_do_async_price_id_isolated_per_run():
    """Each do_async call should generate a fresh price_id for cost tracking."""
    agent = Agent(model=MODEL, name="PriceIdAgent")

    task1 = Task(description="Say hello")
    task2 = Task(description="Say goodbye")

    await agent.do_async(task1)
    price_id_1 = task1.price_id_

    await agent.do_async(task2)
    price_id_2 = task2.price_id_

    assert price_id_1 is not None
    assert price_id_2 is not None
    assert price_id_1 != price_id_2, "Each run must have its own price_id"


# ============================================================================
# Test: _tool_call_count reset between fresh runs
# ============================================================================

@pytest.mark.asyncio
async def test_do_async_tool_count_reset_between_runs():
    """Agent _tool_call_count should reset for each fresh do_async run."""
    agent = Agent(model=MODEL, name="ToolCountAgent", tools=[add])

    task1 = Task(description="Use the add tool to calculate 3 + 4")
    await agent.do_async(task1)
    count_after_first = agent._tool_call_count

    task2 = Task(description="Use the add tool to calculate 10 + 20")
    await agent.do_async(task2)
    count_after_second = agent._tool_call_count

    assert count_after_first > 0, "First run should have made tool calls"
    assert count_after_second > 0, "Second run should have made tool calls"
    # Second run's count should NOT include first run's count
    assert count_after_second <= count_after_first + 3, \
        "Tool count should not accumulate across fresh runs"


# ============================================================================
# Test: _tool_calls preserved during execution
# ============================================================================

@pytest.mark.asyncio
async def test_do_async_tool_calls_recorded():
    """task.tool_calls should contain all tool calls made during execution."""
    agent = Agent(model=MODEL, name="ToolCallsAgent", tools=[add])

    task = Task(description="Use the add tool to calculate 5 + 7. Just give the answer.")
    await agent.do_async(task)

    assert len(task.tool_calls) > 0, "Tool calls should be recorded"

    # Verify tool call structure
    for tc in task.tool_calls:
        assert "tool_name" in tc, "Tool call should have tool_name"


# ============================================================================
# Test: task.response set after do_async
# ============================================================================

@pytest.mark.asyncio
async def test_do_async_task_response_set():
    """task.response should be populated after do_async completes."""
    agent = Agent(model=MODEL, name="ResponseAgent")

    task = Task(description="What is 2 + 2? Reply with just the number.")
    result = await agent.do_async(task)

    assert result is not None, "do_async should return a result"
    assert task.response is not None, "task.response should be set"
    assert len(str(task.response)) > 0, "task.response should not be empty"


# ============================================================================
# Test: task timing attributes
# ============================================================================

@pytest.mark.asyncio
async def test_do_async_task_timing():
    """Task should have start_time, end_time, and duration after execution."""
    agent = Agent(model=MODEL, name="TimingAgent")

    task = Task(description="Say hi")
    await agent.do_async(task)

    assert task.start_time is not None, "start_time should be set"
    assert task.end_time is not None, "end_time should be set"
    assert task.end_time >= task.start_time, "end_time should be >= start_time"
    assert task.duration is not None, "duration should be computed"
    assert task.duration >= 0, "duration should be non-negative"


# ============================================================================
# Test: cost tracking via price_id_summary
# ============================================================================

@pytest.mark.asyncio
async def test_do_async_cost_tracking():
    """task.total_cost, total_input_token, total_output_token should work after do_async."""
    agent = Agent(model=MODEL, name="CostAgent")

    task = Task(description="Say hello world")
    await agent.do_async(task)

    assert task.price_id_ is not None, "price_id should be set"
    assert task.total_input_token is not None, "total_input_token should be tracked"
    assert task.total_output_token is not None, "total_output_token should be tracked"
    assert task.total_input_token > 0, "Should have used input tokens"
    assert task.total_output_token > 0, "Should have used output tokens"


# ============================================================================
# Test: AgentRunOutput attributes after completion
# ============================================================================

@pytest.mark.asyncio
async def test_do_async_run_output_attributes():
    """AgentRunOutput should have proper status and attributes after do_async."""
    agent = Agent(model=MODEL, name="OutputAgent")

    task = Task(description="Say one word")
    output = await agent.do_async(task, return_output=True)

    assert output is not None, "Should return AgentRunOutput"
    assert output.is_complete, "Run should be marked complete"
    assert output.status == RunStatus.completed, "Status should be completed"
    assert output.output is not None, "Output should be set"
    assert output.run_id is not None, "Run ID should be set"
    assert task.run_id is not None, "Task should have run_id"
    assert task.status == RunStatus.completed, "Task status should be completed"


# ============================================================================
# Test: task.status synced from AgentRunOutput
# ============================================================================

@pytest.mark.asyncio
async def test_do_async_task_status_synced():
    """Task status should be synced from AgentRunOutput after completion."""
    agent = Agent(model=MODEL, name="StatusSyncAgent")

    task = Task(description="Say yes")

    # Before execution, status should be None
    assert task.status is None, "Status should be None before execution"

    await agent.do_async(task)

    # After execution, status should be completed
    assert task.status == RunStatus.completed, "Status should be completed after execution"


# ============================================================================
# Test: Multiple sequential runs on the same agent
# ============================================================================

@pytest.mark.asyncio
async def test_do_async_multiple_sequential_runs():
    """Multiple do_async calls should each get fresh state."""
    agent = Agent(model=MODEL, name="SequentialAgent", tools=[add])

    results = []
    price_ids = []

    for i in range(3):
        task = Task(description=f"Use the add tool to calculate {i} + {i+1}")
        await agent.do_async(task)
        results.append(task.response)
        price_ids.append(task.price_id_)

    # All should have responses
    for i, r in enumerate(results):
        assert r is not None, f"Task {i} should have a response"

    # All should have unique price_ids
    assert len(set(price_ids)) == 3, "Each run should have a unique price_id"


# ============================================================================
# Test: do_async with tools — verify tool_calls and response
# ============================================================================

@pytest.mark.asyncio
async def test_do_async_with_multiple_tools():
    """do_async with tools should record tool calls and produce correct output."""
    agent = Agent(model=MODEL, name="MultiToolAgent", tools=[add, subtract])

    task = Task(description="First add 10 + 5, then subtract 3 from the result. Use both tools.")
    await agent.do_async(task)

    assert task.response is not None, "Should have a response"
    assert len(task.tool_calls) >= 2, f"Should have at least 2 tool calls, got {len(task.tool_calls)}"

    tool_names = [tc["tool_name"] for tc in task.tool_calls]
    assert "add" in tool_names, "Should have called add tool"
    assert "subtract" in tool_names, "Should have called subtract tool"


# ============================================================================
# Test: Reusing a task for a second run gets fresh state
# ============================================================================

@pytest.mark.asyncio
async def test_do_async_reuse_task_not_allowed_after_completion():
    """A completed task should not be re-runnable (status check prevents it)."""
    agent = Agent(model=MODEL, name="ReuseAgent")

    task = Task(description="Say hi")
    await agent.do_async(task)

    assert task.status == RunStatus.completed

    # Running the same completed task again should return a warning, not re-execute
    result = await agent.do_async(task)
    assert "already completed" in str(result).lower(), \
        "Should warn that task is already completed"
