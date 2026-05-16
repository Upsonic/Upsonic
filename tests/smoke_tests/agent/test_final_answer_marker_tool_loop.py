"""Tool-loop behavior smoke tests for ``stream_final_answer``.

Covers how the marker behaves across complex tool-execution patterns:

  - Deep recursion (multiple sequential tool calls before sentinel)
  - Marker at-most-once when both sentinel and ``final_result`` output-tool
    paths are reachable in the same run (structured output + tools)
  - Anthropic extended thinking (``ThinkingDeltaEvent``) interleaved with
    text and tool calls
  - ``tool_call_count`` exemption holds across multi-turn recursion
"""

from __future__ import annotations

import pytest

from _final_answer_marker_helpers import (
    AnswerSchema,
    FINAL_ANSWER_MARKER_TOOL_NAME,
    Task,
    add,
    build_agent_default,
    collect_events,
    has_anthropic,
    skip_if_no_provider,
)

pytestmark = skip_if_no_provider


# ---------------------------------------------------------------------------
# Deep recursion — 3+ sequential add calls before the sentinel turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_turn_three_tools_then_marker():
    """Force a deep recursion: the agent must call multiple sequential add
    tools before reaching the final answer. The marker still fires exactly
    once, AFTER all real tool results and BEFORE the final answer text.
    """
    from upsonic.run.events.events import (
        FinalAnswerStartEvent,
        FinalOutputEvent,
        ToolCallEvent,
        ToolResultEvent,
    )

    agent = build_agent_default(name="DeepRecursionAgent", tools=[add])
    task = Task(
        description=(
            "Step by step using ONLY the add tool: "
            "1. compute 5+5, "
            "2. then add 10 to that result, "
            "3. then add 20 to that result. "
            "Report each intermediate result and the final value in one sentence."
        )
    )

    events = await collect_events(agent, task, stream_final_answer=True)

    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 1, (
        f"Multi-turn recursion must produce exactly one marker; "
        f"got {len(fa_starts)}. Timeline: {[type(e).__name__ for e in events]}"
    )

    real_tool_calls = [
        e for e in events
        if isinstance(e, ToolCallEvent) and e.tool_name == "add"
    ]
    assert len(real_tool_calls) >= 2, (
        f"Expected ≥2 add tool calls to validate deep recursion; "
        f"got {len(real_tool_calls)}"
    )

    real_tool_results = [
        i for i, e in enumerate(events)
        if isinstance(e, ToolResultEvent) and e.tool_name == "add"
    ]
    marker_idx = events.index(fa_starts[0])
    assert all(i < marker_idx for i in real_tool_results), (
        f"Marker (idx {marker_idx}) must come AFTER all real ToolResultEvents; "
        f"got results at indices {real_tool_results}"
    )

    sentinel_leaks = [
        e for e in events
        if isinstance(e, ToolCallEvent) and e.tool_name == FINAL_ANSWER_MARKER_TOOL_NAME
    ]
    assert len(sentinel_leaks) == 0

    final_outputs = [e for e in events if isinstance(e, FinalOutputEvent)]
    assert len(final_outputs) == 1

    # Sentinel exempt from tool_call_count even across multiple recursions.
    assert agent._tool_call_count == len(real_tool_calls), (
        f"agent._tool_call_count ({agent._tool_call_count}) must equal real "
        f"tool calls ({len(real_tool_calls)}); sentinel should be exempt "
        f"regardless of recursion depth"
    )


# ---------------------------------------------------------------------------
# At-most-once: structured output + tools — only one trigger path wins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_marker_at_most_once_when_both_paths_reachable():
    """When BOTH structured output (final_result tool path) AND a tool-using
    workflow are configured, the marker must still fire exactly once per
    run — never twice across the different trigger paths.
    """
    from upsonic.run.events.events import FinalAnswerStartEvent, FinalOutputEvent

    agent = build_agent_default(name="AtMostOnceAgent", tools=[add])
    task = Task(
        description="Use add to compute 11+11.",
        response_format=AnswerSchema,
    )

    events = await collect_events(agent, task, stream_final_answer=True)

    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 1, (
        f"Marker must fire AT MOST ONCE per run regardless of which paths "
        f"are reachable; got {len(fa_starts)}. Triggers: "
        f"{[e.triggered_by for e in fa_starts]}"
    )

    final_outputs = [e for e in events if isinstance(e, FinalOutputEvent)]
    assert len(final_outputs) == 1
    assert final_outputs[0].output_type in ("structured", "text")


# ---------------------------------------------------------------------------
# Anthropic extended thinking — marker holds across thinking deltas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not has_anthropic(), reason="Extended thinking is Anthropic-specific")
async def test_extended_thinking_does_not_break_marker():
    """With extended thinking enabled on Anthropic, ThinkingDeltaEvent fires
    interleaved with text/tool events. The marker must still fire exactly
    once and precede the final-answer TextDeltaEvents.
    """
    from upsonic.agent.agent import Agent
    from upsonic.run.events.events import (
        FinalAnswerStartEvent,
        FinalOutputEvent,
        TextDeltaEvent,
    )
    from upsonic.models.anthropic import AnthropicModelSettings

    settings = AnthropicModelSettings(
        anthropic_thinking={"type": "enabled", "budget_tokens": 2048},
        max_tokens=4096,
    )
    agent = Agent(
        model="anthropic/claude-sonnet-4-6",
        name="ThinkingAgent",
        tools=[add],
        settings=settings,
    )
    task = Task(description="Use add tool to compute 21+21 and explain briefly.")

    events = []
    async for event in agent.astream(task, events=True, stream_final_answer=True):
        events.append(event)

    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 1, (
        f"Marker must fire exactly once even with extended thinking; "
        f"got {len(fa_starts)}"
    )

    marker_idx = events.index(fa_starts[0])
    text_after_marker = [
        e for i, e in enumerate(events)
        if i > marker_idx and isinstance(e, TextDeltaEvent)
    ]
    assert text_after_marker, (
        "Final-answer text deltas must appear after the marker even with "
        "thinking enabled"
    )

    final_outputs = [e for e in events if isinstance(e, FinalOutputEvent)]
    assert len(final_outputs) == 1
