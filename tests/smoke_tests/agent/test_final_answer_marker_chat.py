"""Chat-layer integration smoke tests for ``stream_final_answer``.

Verifies that ``Chat.invoke(stream=True, events=True, stream_final_answer=True)``
and ``Chat.stream(events=True, stream_final_answer=True)`` correctly:

  - Forward the flag to the underlying ``Agent.astream``
  - Emit ``FinalAnswerStartEvent`` exactly once on the obey path
  - Suppress sentinel ``ToolCallEvent`` / ``ToolResultEvent`` leakage
  - Pass kwargs through Chat method signatures
"""

from __future__ import annotations

import inspect

import pytest

from _final_answer_marker_helpers import (
    Chat,
    FINAL_ANSWER_MARKER_TOOL_NAME,
    add,
    build_agent_default,
    greet,
    skip_if_no_provider,
)

pytestmark = skip_if_no_provider


# ---------------------------------------------------------------------------
# Chat.invoke obey path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_invoke_emits_marker_exactly_once():
    from upsonic.run.events.events import (
        FinalAnswerStartEvent,
        FinalOutputEvent,
        ToolCallEvent,
        ToolResultEvent,
    )

    agent = build_agent_default(name="ChatInvokeAgent", tools=[add])
    chat = Chat(session_id="test-chat-invoke", user_id="test-user", agent=agent)
    try:
        events = []
        result = await chat.invoke(
            "Use the add tool to compute 7+3, then state the result in one sentence.",
            stream=True,
            events=True,
            stream_final_answer=True,
        )
        async for event in result:
            events.append(event)
    finally:
        await chat.close()

    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 1, (
        f"Chat.invoke MUST emit exactly one FinalAnswerStartEvent on obey path; "
        f"got {len(fa_starts)}. Timeline: {[type(e).__name__ for e in events]}"
    )
    assert fa_starts[0].triggered_by == "sentinel"

    sentinel_leaks = [
        e for e in events
        if isinstance(e, (ToolCallEvent, ToolResultEvent))
        and e.tool_name == FINAL_ANSWER_MARKER_TOOL_NAME
    ]
    assert len(sentinel_leaks) == 0, (
        f"Sentinel must never leak via ToolCallEvent/ToolResultEvent. "
        f"Leaks: {sentinel_leaks}"
    )

    final_outputs = [e for e in events if isinstance(e, FinalOutputEvent)]
    assert len(final_outputs) == 1


# ---------------------------------------------------------------------------
# Chat.stream obey path + position invariant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_stream_marker_precedes_text_deltas():
    from upsonic.run.events.events import (
        FinalAnswerStartEvent,
        TextDeltaEvent,
    )

    agent = build_agent_default(name="ChatStreamAgent", tools=[greet])
    chat = Chat(session_id="test-chat-stream", user_id="test-user", agent=agent)
    try:
        events = []
        stream = chat.stream(
            "Greet 'Bob' using the greet tool, then say goodbye in one short sentence.",
            events=True,
            stream_final_answer=True,
        )
        async for event in stream:
            events.append(event)
    finally:
        await chat.close()

    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 1, (
        f"Chat.stream MUST emit exactly one FinalAnswerStartEvent; "
        f"got {len(fa_starts)}"
    )
    assert fa_starts[0].triggered_by == "sentinel"

    marker_idx = events.index(fa_starts[0])
    text_idxs = [i for i, e in enumerate(events) if isinstance(e, TextDeltaEvent)]
    assert text_idxs, "Final-turn must produce text deltas"
    assert any(i > marker_idx for i in text_idxs), (
        f"At least one TextDeltaEvent must come AFTER FinalAnswerStartEvent "
        f"(marker_idx={marker_idx}, first 5 text idxs={text_idxs[:5]})"
    )


# ---------------------------------------------------------------------------
# Forward-compat: Chat signatures accept the flag via **kwargs
# ---------------------------------------------------------------------------


def test_chat_invoke_signature_passes_kwargs_through():
    """Chat.invoke must accept stream_final_answer via **kwargs (the
    reliability gate uses kwargs.get('stream_final_answer'))."""
    sig = inspect.signature(Chat.invoke)
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    assert has_kwargs, (
        "Chat.invoke must accept **kwargs so the flag can pass through "
        "to the underlying Agent.astream"
    )


def test_chat_stream_signature_passes_kwargs_through():
    sig = inspect.signature(Chat.stream)
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    assert has_kwargs, (
        "Chat.stream must accept **kwargs so the flag can pass through "
        "to the underlying Agent.astream"
    )
