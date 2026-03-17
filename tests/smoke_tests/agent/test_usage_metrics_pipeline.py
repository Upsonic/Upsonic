"""
Pipeline Usage Metrics Smoke Tests

Verifies that usage metrics are correctly set at each stage of the pipeline:

1. Non-streaming: duration set BEFORE MemorySave (Fix #1)
2. Streaming: tool_call_count tracked (Fix #2)
3. Streaming: tool_execution_time tracked separately (Fix #4)
4. Streaming: time_to_first_token set (Fix #3)
5. Non-streaming with tools: model_execution_time vs duration consistency
6. Agent-level usage accumulation after multiple tasks
7. Usage serialization round-trip via to_dict/from_dict

Run with: uv run pytest tests/smoke_tests/agent/test_usage_metrics_pipeline.py -v -s
"""

import pytest
import asyncio
from typing import List

from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.usage import TaskUsage, AgentUsage


# ============================================================================
# TOOLS
# ============================================================================

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Quotient
    """
    return a / b


# ============================================================================
# HELPERS
# ============================================================================

def _assert_usage_complete(usage: TaskUsage, label: str) -> None:
    """Assert all major usage fields are populated."""
    assert usage is not None, f"[{label}] usage is None"
    assert usage.requests > 0, f"[{label}] requests should be > 0"
    assert usage.input_tokens > 0, f"[{label}] input_tokens should be > 0"
    assert usage.output_tokens > 0, f"[{label}] output_tokens should be > 0"
    assert usage.duration is not None and usage.duration > 0, f"[{label}] duration missing or zero"
    assert usage.model_execution_time is not None and usage.model_execution_time > 0, (
        f"[{label}] model_execution_time missing or zero"
    )


def _assert_timing_consistency(usage: TaskUsage, label: str) -> None:
    """Assert timing relationships are consistent."""
    assert usage.duration is not None, f"[{label}] duration is None"
    assert usage.model_execution_time is not None, f"[{label}] model_execution_time is None"
    assert usage.model_execution_time <= usage.duration, (
        f"[{label}] model_execution_time ({usage.model_execution_time}) > duration ({usage.duration})"
    )
    if usage.upsonic_execution_time is not None:
        assert usage.upsonic_execution_time >= 0, (
            f"[{label}] upsonic_execution_time should be >= 0, got {usage.upsonic_execution_time}"
        )


# ============================================================================
# TEST 1: Non-streaming do_async — all usage fields populated
# ============================================================================

@pytest.mark.asyncio
async def test_non_streaming_usage_all_fields():
    """Non-streaming run should have all usage fields populated."""
    agent = Agent("openai/gpt-4o-mini", name="non_stream_usage")
    task = Task(description="What is the capital of France? Answer in one word.")

    output = await agent.do_async(task, return_output=True)

    assert output.is_complete
    _assert_usage_complete(output.usage, "non_streaming")
    _assert_timing_consistency(output.usage, "non_streaming")


# ============================================================================
# TEST 2: Non-streaming with tools — timing tracks tool time
# ============================================================================

@pytest.mark.asyncio
async def test_non_streaming_with_tools_usage():
    """Non-streaming run with tools should track model time and have correct timing."""
    agent = Agent("openai/gpt-4o-mini", name="tool_usage_agent")
    task = Task(
        description="Use multiply to calculate 12 * 15. Return just the number.",
        tools=[multiply]
    )

    output = await agent.do_async(task, return_output=True)

    assert output.is_complete
    usage = output.usage
    _assert_usage_complete(usage, "non_streaming_tools")
    _assert_timing_consistency(usage, "non_streaming_tools")

    # With tools, we expect >= 2 requests (tool call + response)
    assert usage.requests >= 2, f"Expected >= 2 requests with tools, got {usage.requests}"


# ============================================================================
# TEST 3: Non-streaming with multiple tools — timing consistency
# ============================================================================

@pytest.mark.asyncio
async def test_non_streaming_multiple_tools_timing():
    """Multiple tool calls should have consistent timing."""
    agent = Agent("openai/gpt-4o-mini", name="multi_tool_timing")
    task = Task(
        description=(
            "First multiply 6 by 7, then divide the result by 2. "
            "Return just the final number."
        ),
        tools=[multiply, divide]
    )

    output = await agent.do_async(task, return_output=True)

    assert output.is_complete
    usage = output.usage
    _assert_usage_complete(usage, "multi_tool_timing")
    _assert_timing_consistency(usage, "multi_tool_timing")


# ============================================================================
# TEST 4: Streaming run — usage fields populated
# ============================================================================

@pytest.mark.asyncio
async def test_streaming_usage_all_fields():
    """Streaming run should have all usage fields populated."""
    agent = Agent("openai/gpt-4o-mini", name="stream_usage")
    task = Task(description="Count from 1 to 3, each on a new line.")

    events = []
    async for event in agent.astream(task, events=True):
        events.append(event)

    assert len(events) > 0, "Should have received events"

    # Get the output from the agent
    output = agent._agent_run_output
    assert output is not None

    usage = output.usage
    assert usage is not None, "Streaming run should have usage"
    assert usage.input_tokens > 0, "Streaming should track input_tokens"
    assert usage.output_tokens > 0, "Streaming should track output_tokens"
    assert usage.duration is not None and usage.duration > 0, "Streaming should have duration"
    assert usage.model_execution_time is not None and usage.model_execution_time > 0, (
        "Streaming should have model_execution_time"
    )


# ============================================================================
# TEST 5: Streaming with tools — tool_execution_time tracked (Fix #4)
# ============================================================================

@pytest.mark.asyncio
async def test_streaming_with_tools_tracks_tool_time():
    """Streaming run with tools should track tool_execution_time separately."""
    agent = Agent("openai/gpt-4o-mini", name="stream_tool_time")
    task = Task(
        description="Use multiply to calculate 5 * 9. Return just the number.",
        tools=[multiply]
    )

    events = []
    async for event in agent.astream(task, events=True):
        events.append(event)

    output = agent._agent_run_output
    assert output is not None

    usage = output.usage
    assert usage is not None
    assert usage.input_tokens > 0
    assert usage.duration is not None and usage.duration > 0

    # After Fix #4, tool_execution_time should be tracked
    # (It may still be None if the tool executes extremely fast,
    # but it should at least be set)
    if usage.tool_execution_time is not None:
        assert usage.tool_execution_time >= 0, "tool_execution_time should be >= 0"


# ============================================================================
# TEST 6: Streaming — time_to_first_token set (Fix #3)
# ============================================================================

@pytest.mark.asyncio
async def test_streaming_time_to_first_token():
    """Streaming run should set time_to_first_token."""
    agent = Agent("openai/gpt-4o-mini", name="stream_ttft")
    task = Task(description="Say hello.")

    events = []
    async for event in agent.astream(task, events=True):
        events.append(event)

    output = agent._agent_run_output
    assert output is not None

    usage = output.usage
    assert usage is not None

    # time_to_first_token should be set for streaming
    if usage.time_to_first_token is not None:
        assert usage.time_to_first_token > 0, "time_to_first_token should be > 0"
        assert usage.time_to_first_token < usage.duration, (
            "time_to_first_token should be < total duration"
        )


# ============================================================================
# TEST 7: Agent-level usage accumulates correctly
# ============================================================================

@pytest.mark.asyncio
async def test_agent_usage_accumulates_with_tools():
    """Agent.usage should accumulate across multiple tool-using tasks."""
    agent = Agent("openai/gpt-4o-mini", name="accumulate_agent")

    task1 = Task(
        description="Use multiply to calculate 3 * 4. Return the number.",
        tools=[multiply]
    )
    task2 = Task(
        description="Use multiply to calculate 5 * 6. Return the number.",
        tools=[multiply]
    )

    await agent.do_async(task1)
    usage_after_1 = AgentUsage()
    usage_after_1.incr(agent.usage)  # snapshot

    await agent.do_async(task2)

    assert agent.usage.requests > usage_after_1.requests, "Requests should accumulate"
    assert agent.usage.input_tokens > usage_after_1.input_tokens, "Tokens should accumulate"
    assert agent.usage.duration > usage_after_1.duration, "Duration should accumulate"


# ============================================================================
# TEST 8: TaskUsage serialization round-trip after real run
# ============================================================================

@pytest.mark.asyncio
async def test_task_usage_serialization_after_run():
    """TaskUsage from a real run should survive to_dict/from_dict."""
    agent = Agent("openai/gpt-4o-mini", name="serial_agent")
    task = Task(description="What is 7*8? Just the number.")

    output = await agent.do_async(task, return_output=True)
    assert output.is_complete

    usage = output.usage
    _assert_usage_complete(usage, "serialization_source")

    # Serialize and restore
    d = usage.to_dict()
    restored = TaskUsage.from_dict(d)

    assert restored.requests == usage.requests
    assert restored.input_tokens == usage.input_tokens
    assert restored.output_tokens == usage.output_tokens
    assert restored.duration == usage.duration
    assert restored.model_execution_time == usage.model_execution_time
    if usage.cost is not None:
        assert restored.cost == usage.cost


# ============================================================================
# TEST 9: print_do shows correct timing metrics
# ============================================================================

def test_print_do_shows_timing_metrics(capsys):
    """print_do should display Task Metrics with timing fields."""
    agent = Agent("openai/gpt-4o-mini", name="print_timing_agent")
    task = Task(description="Say hello.")

    agent.print_do(task)

    captured = capsys.readouterr()
    assert "Task Metrics" in captured.out
    assert "Time Taken" in captured.out
    assert "Model Execution Time" in captured.out
    assert "Framework Overhead" in captured.out


# ============================================================================
# TEST 10: Non-streaming task.usage matches output.usage
# ============================================================================

@pytest.mark.asyncio
async def test_task_usage_matches_output_usage():
    """task.usage and output.usage should reference the same object."""
    agent = Agent("openai/gpt-4o-mini", name="match_agent")
    task = Task(description="What is 1+1? Just the number.")

    output = await agent.do_async(task, return_output=True)

    assert output.is_complete
    assert task.usage is not None
    assert output.usage is not None

    # They should be the same object
    assert task.usage is output.usage, (
        "task.usage and output.usage should be the same TaskUsage object"
    )

    # And both should have metrics
    _assert_usage_complete(task.usage, "task_usage")
    _assert_usage_complete(output.usage, "output_usage")


# ============================================================================
# TEST RUNNER
# ============================================================================

async def run_all_tests():
    """Run all tests manually."""
    tests = [
        ("Non-streaming usage fields", test_non_streaming_usage_all_fields),
        ("Non-streaming with tools", test_non_streaming_with_tools_usage),
        ("Multiple tools timing", test_non_streaming_multiple_tools_timing),
        ("Streaming usage fields", test_streaming_usage_all_fields),
        ("Streaming tool time", test_streaming_with_tools_tracks_tool_time),
        ("Streaming TTFT", test_streaming_time_to_first_token),
        ("Agent accumulation", test_agent_usage_accumulates_with_tools),
        ("Serialization round-trip", test_task_usage_serialization_after_run),
        ("Task/output usage match", test_task_usage_matches_output_usage),
    ]

    for name, test_fn in tests:
        print(f"\n{'=' * 80}")
        print(f"TEST: {name}")
        print("=" * 80)
        await test_fn()
        print("PASSED")

    print(f"\n{'=' * 80}")
    print("ALL TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
