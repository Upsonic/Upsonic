"""
Streaming Pipeline Verification — Reflection + CallManagement in action.

Test 1: Pure text streaming with reflection enabled
Test 2: Event streaming with tools

Run with:
  uv run tests/smoke_tests/agent/test_streaming_pipeline_verification.py
"""

import asyncio
import time
import pytest
from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.reflection import ReflectionConfig
from upsonic.run.events.events import (
    PipelineStartEvent,
    PipelineEndEvent,
    StepStartEvent,
    StepEndEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    ReflectionEvent,
    ReliabilityEvent,
    ExecutionCompleteEvent,
)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@pytest.mark.asyncio
async def test_text_streaming_with_reflection():
    print("=" * 80)
    print("  TEST 1: TEXT STREAMING — REFLECTION ENABLED")
    print("=" * 80)

    agent = Agent(
        model="anthropic/claude-sonnet-4-6",
        name="ReflectiveStreamAgent",
        tools=[multiply],
        reflection=True,
        reflection_config=ReflectionConfig(max_iterations=1, acceptance_threshold=0.95),
        print=True,
    )

    task = Task(
        description="Use the multiply tool to calculate 9 * 7 and explain the result clearly.",
        tools=[multiply],
    )

    print("\nStreaming...\n")
    t_stream_start: float = time.time()
    async for chunk in agent.astream(task):
        print(chunk, end="", flush=True)
    t_stream_end: float = time.time()
    stream_elapsed_s: float = t_stream_end - t_stream_start
    print("\n")
    print("--- Manual timing (time.time): text streaming ---")
    print(f"  start={t_stream_start:.6f}  end={t_stream_end:.6f}  elapsed={stream_elapsed_s:.3f}s")

    print("=" * 80)
    print("  TEST 1 COMPLETE")
    print("=" * 80)


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_event_streaming_with_tools():
    print("\n" + "=" * 80)
    print("  TEST 2: EVENT STREAMING — ALL STEPS")
    print("=" * 80)

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="EventStreamAgent",
        tools=[multiply],
        reflection=True,
        reflection_config=ReflectionConfig(max_iterations=1, acceptance_threshold=0.5),
        print=True,
    )

    task = Task(
        description="Use the multiply tool to calculate 12 * 5. Then tell me the result in one sentence.",
        tools=[multiply],
    )

    print("\nStreaming events...\n")
    t_stream_start: float = time.time()
    async for event in agent.astream(task, events=True):
        if isinstance(event, PipelineStartEvent):
            print(f"  [PipelineStart] steps={event.total_steps} streaming={event.is_streaming}", flush=True)
        elif isinstance(event, StepStartEvent):
            print(f"    [{event.step_index:>2}] START: {event.step_name}", flush=True)
        elif isinstance(event, StepEndEvent):
            print(f"    [{event.step_index:>2}] END:   {event.step_name} → {event.status} ({event.execution_time:.3f}s)", flush=True)
        elif isinstance(event, ToolCallEvent):
            print(f"  [ToolCall] {event.tool_name}({event.tool_args})", flush=True)
        elif isinstance(event, ToolResultEvent):
            print(f"  [ToolResult] {event.tool_name} → {event.result_preview}", flush=True)
        elif isinstance(event, TextDeltaEvent):
            print(event.content, end="", flush=True)
        elif isinstance(event, ReflectionEvent):
            print(f"  [Reflection] applied={event.reflection_applied}", flush=True)
        elif isinstance(event, ReliabilityEvent):
            print(f"  [Reliability] applied={event.reliability_applied}", flush=True)
        elif isinstance(event, ExecutionCompleteEvent):
            print(f"  [ExecutionComplete] type={event.output_type} tools={event.total_tool_calls}", flush=True)
        elif isinstance(event, PipelineEndEvent):
            print(f"  [PipelineEnd] status={event.status} duration={event.total_duration:.2f}s steps={event.executed_steps}/{event.total_steps}", flush=True)

    t_stream_end: float = time.time()
    stream_elapsed_s: float = t_stream_end - t_stream_start
    print("\n--- Manual timing (time.time): event streaming ---")
    print(f"  start={t_stream_start:.6f}  end={t_stream_end:.6f}  elapsed={stream_elapsed_s:.3f}s")

    print("\n" + "=" * 80)
    print("  TEST 2 COMPLETE")
    print("=" * 80)


async def main():
    await test_text_streaming_with_reflection()
    await test_event_streaming_with_tools()


if __name__ == "__main__":
    asyncio.run(main())
