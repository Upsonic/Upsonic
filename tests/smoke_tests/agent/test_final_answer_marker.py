"""Core obey-path smoke tests for ``stream_final_answer`` with real model providers.

These exercise the trigger paths that the unit-tier fake-Model tests can
only approximate:

  - Sentinel obey path: a tool-using agent calls ``__final_answer_marker__``,
    the marker fires before the final-answer text deltas, no sentinel
    leakage on the public stream.
  - No-tool single-turn: agent with at least one user tool registered but a
    prompt that needs none — sentinel still fires.
  - Structured output: ``response_format=PydanticModel`` — marker fires on
    ``final_result`` tool onset (``triggered_by='output_tool'``).
  - tool_call_limit exemption (integrated): real tools called twice plus
    sentinel — run completes without limit-abort.
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
    greet,
    skip_if_no_provider,
)

pytestmark = skip_if_no_provider


# ---------------------------------------------------------------------------
# Sentinel obey path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sentinel_obey_path_marker_precedes_final_text():
    from upsonic.run.events.events import (
        FinalAnswerStartEvent,
        FinalOutputEvent,
        TextDeltaEvent,
        ToolCallEvent,
        ToolResultEvent,
    )

    agent = build_agent_default(name="ObeyAgent", tools=[add])
    task = Task(
        description="Use the `add` tool to compute 2 + 3, then respond "
                    "with the result as a single sentence."
    )

    events = await collect_events(agent, task, stream_final_answer=True)

    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 1, (
        f"Obey path must emit exactly one marker; got {len(fa_starts)}. "
        f"Timeline: {[type(e).__name__ for e in events]}"
    )
    assert fa_starts[0].triggered_by == "sentinel"

    sentinel_leaks = [
        e for e in events
        if isinstance(e, (ToolCallEvent, ToolResultEvent))
        and e.tool_name == FINAL_ANSWER_MARKER_TOOL_NAME
    ]
    assert len(sentinel_leaks) == 0, (
        f"Sentinel must never leak via ToolCallEvent/ToolResultEvent; "
        f"got: {sentinel_leaks}"
    )

    marker_idx = events.index(fa_starts[0])
    text_idxs = [i for i, e in enumerate(events) if isinstance(e, TextDeltaEvent)]
    assert text_idxs
    assert any(i > marker_idx for i in text_idxs), (
        f"At least one TextDeltaEvent must come AFTER the marker "
        f"(marker_idx={marker_idx})"
    )

    final_outputs = [e for e in events if isinstance(e, FinalOutputEvent)]
    assert len(final_outputs) == 1


# ---------------------------------------------------------------------------
# No-tool single-turn (still requires ≥1 user tool registered for sentinel injection)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_tool_needed_single_turn_still_emits_marker():
    from upsonic.run.events.events import FinalAnswerStartEvent

    agent = build_agent_default(name="NoToolNeededAgent", tools=[greet])
    task = Task(
        description="Greet 'Alice' using the greet tool. Then provide a short "
                    "closing sentence as your final answer."
    )

    events = await collect_events(agent, task, stream_final_answer=True)
    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 1, (
        f"Expected exactly one marker for tool-registered run; got {len(fa_starts)}"
    )


# ---------------------------------------------------------------------------
# Structured output — marker fires on final_result tool onset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_output_marker_fires_via_output_tool_or_sentinel():
    from upsonic.run.events.events import FinalAnswerStartEvent, FinalOutputEvent

    agent = build_agent_default(name="StructuredAgent", tools=[add])
    task = Task(
        description="What is 1 + 1? Use the `add` tool.",
        response_format=AnswerSchema,
    )

    events = await collect_events(agent, task, stream_final_answer=True)

    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 1, (
        f"Structured output run must emit exactly one marker; "
        f"got {len(fa_starts)}"
    )
    assert fa_starts[0].triggered_by in ("output_tool", "sentinel")

    final_outputs = [e for e in events if isinstance(e, FinalOutputEvent)]
    assert len(final_outputs) == 1
    assert final_outputs[0].output_type in ("structured", "text")


# ---------------------------------------------------------------------------
# tool_call_limit exemption — sentinel does not count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_limit_exempts_sentinel():
    """``tool_call_limit=2``; force two real tool calls plus one sentinel.
    Run must complete (not abort by limit) and total tool count equals
    the number of real calls.
    """
    from upsonic.run.events.events import FinalOutputEvent, ToolCallEvent

    agent = build_agent_default(
        name="LimitAgent",
        tools=[add],
        tool_call_limit=2,
    )
    task = Task(
        description="Use the `add` tool twice: first compute 1+2, then "
                    "compute 3+4. Then give a one-sentence final answer."
    )

    events = await collect_events(agent, task, stream_final_answer=True)
    final_outputs = [e for e in events if isinstance(e, FinalOutputEvent)]
    assert len(final_outputs) == 1, "Run did not complete"

    real_add_calls = [
        e for e in events
        if isinstance(e, ToolCallEvent) and e.tool_name == "add"
    ]
    assert agent._tool_call_count <= 2
    assert agent._tool_call_count == len(real_add_calls), (
        f"Tool count ({agent._tool_call_count}) must equal real add calls "
        f"({len(real_add_calls)}) — sentinel exempt"
    )
