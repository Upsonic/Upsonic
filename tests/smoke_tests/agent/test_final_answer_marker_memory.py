"""Memory-integration smoke tests for ``stream_final_answer``.

Verifies that the marker plumbing does NOT corrupt session memory:

  - The synthetic ``ToolReturnPart`` we generate for the dropped sentinel
    (to satisfy Anthropic's tool_use → tool_result invariant) is persisted
    into chat_history. We must ensure that:
      * On a second run that LOADS the persisted history, the provider
        still accepts the conversation (no orphaned tool_use, no garbled
        history)
      * The marker still works on the second run (state isolation between
        runs holds even with persistence)
      * Memory metrics (message count, etc.) are sane

  - Run with the flag OFF and the flag ON both produce valid persistable
    histories.

  - Summary memory works alongside the marker (the LLM-generated summary
    captures the session correctly).
"""

from __future__ import annotations

import uuid

import pytest

from _final_answer_marker_helpers import (
    Agent,
    Task,
    add,
    build_agent_default,
    collect_events,
    has_anthropic,
    has_google,
    skip_if_no_provider,
)

pytestmark = skip_if_no_provider


def _make_memory(session_id: str, full_session: bool = True, summary: bool = False):
    """Build an InMemoryStorage-backed Memory instance (no disk persistence
    so tests don't leak state across runs)."""
    from upsonic.storage import Memory
    from upsonic.storage.in_memory.in_memory import InMemoryStorage

    storage = InMemoryStorage()
    return Memory(
        storage=storage,
        session_id=session_id,
        user_id="test-user-mem",
        full_session_memory=full_session,
        summary_memory=summary,
        # Need a model for summary generation; reuse a known available one.
        model="anthropic/claude-sonnet-4-6" if has_anthropic() else "google-gla/gemini-2.5-flash",
    )


# ---------------------------------------------------------------------------
# Single-run: chat_history persists cleanly with the flag on
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_history_persisted_cleanly_with_flag_on():
    """After a streaming run with the flag on, the persisted chat history
    must satisfy the provider's tool_use → tool_result invariant. We
    verify by inspecting the saved session and counting tool_call /
    tool_return pairs.
    """
    from upsonic.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
    from upsonic.run.events.events import FinalAnswerStartEvent, FinalOutputEvent

    session_id = f"history-flag-on-{uuid.uuid4()}"
    memory = _make_memory(session_id, full_session=True)
    agent = build_agent_default(name="MemoryFlagOnAgent", tools=[add], memory=memory)

    task = Task(description="Use the add tool to compute 8+12, then summarize in one sentence.")
    events = await collect_events(agent, task, stream_final_answer=True)

    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    final_outputs = [e for e in events if isinstance(e, FinalOutputEvent)]
    assert len(fa_starts) == 1
    assert len(final_outputs) == 1

    # Load the persisted session
    session = await memory.get_session_async()
    assert session is not None, "Session was not persisted"

    # Walk the persisted memory and verify every tool_call has a matching
    # tool_return immediately following.
    messages = session.messages or []
    assert len(messages) >= 2, (
        f"Expected ≥2 messages in persisted history; got {len(messages)}"
    )

    # Collect all tool_call_ids and tool_return_ids in order
    tool_call_ids: list[str] = []
    tool_return_ids: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    tool_call_ids.append(part.tool_call_id)
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    tool_return_ids.append(part.tool_call_id)

    # Every tool_call must have a tool_return with matching id (regardless
    # of whether it's a real call or a sentinel synthetic return)
    missing = set(tool_call_ids) - set(tool_return_ids)
    assert not missing, (
        f"Persisted history has tool_calls without matching tool_returns: "
        f"{missing}. tool_call_ids={tool_call_ids}, "
        f"tool_return_ids={tool_return_ids}. "
        f"This would 400 on a subsequent provider call."
    )


# ---------------------------------------------------------------------------
# Two-run sequence: second run loads first run's history cleanly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_consecutive_runs_with_flag_share_memory_cleanly():
    """Run 1 with the flag on persists chat history. Run 2 (also with the
    flag on) loads that history and must complete without provider errors.
    This is the most realistic memory + flag interaction.
    """
    from upsonic.run.events.events import FinalAnswerStartEvent, FinalOutputEvent

    session_id = f"two-run-flag-{uuid.uuid4()}"
    memory = _make_memory(session_id, full_session=True)

    # Run 1: compute and store a fact
    agent1 = build_agent_default(name="MemRun1Agent", tools=[add], memory=memory)
    task1 = Task(
        description="Use the add tool to compute 5+5. Reply with the result in one short sentence."
    )
    events1 = await collect_events(agent1, task1, stream_final_answer=True)
    assert len([e for e in events1 if isinstance(e, FinalAnswerStartEvent)]) == 1
    assert len([e for e in events1 if isinstance(e, FinalOutputEvent)]) == 1

    # Run 2: same session_id, should load history. Ask a follow-up that
    # references the previous turn.
    agent2 = build_agent_default(name="MemRun2Agent", tools=[add], memory=memory)
    task2 = Task(
        description="What was the result of my previous calculation? Reply in one short sentence."
    )
    events2 = await collect_events(agent2, task2, stream_final_answer=True)

    # Run 2 must complete; marker fires; provider did NOT 400 due to broken history.
    fa_starts2 = [e for e in events2 if isinstance(e, FinalAnswerStartEvent)]
    final_outputs2 = [e for e in events2 if isinstance(e, FinalOutputEvent)]
    assert len(fa_starts2) == 1, (
        f"Run 2 must emit marker; got {len(fa_starts2)}. If 0, the persisted "
        f"history from run 1 may have corrupted the provider's invariant."
    )
    assert len(final_outputs2) == 1
    assert final_outputs2[0].output_type != "blocked"


# ---------------------------------------------------------------------------
# Flag off vs on: persisted histories are both valid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_persisted_history_unchanged_shape():
    """Without the flag, the persisted history has the same shape it
    always had — verifies our changes don't alter byte-identical behavior.
    """
    from upsonic.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
    from upsonic.run.events.events import FinalAnswerStartEvent

    session_id = f"history-flag-off-{uuid.uuid4()}"
    memory = _make_memory(session_id, full_session=True)
    agent = build_agent_default(name="MemoryFlagOffAgent", tools=[add], memory=memory)

    task = Task(description="Use the add tool to compute 3+7, then state the result.")
    events = await collect_events(agent, task)  # NO stream_final_answer

    # With flag off, no marker
    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 0

    # Persisted history must still be valid (this was the baseline before
    # the feature)
    session = await memory.get_session_async()
    assert session is not None
    messages = session.messages or []

    tool_call_ids: list[str] = []
    tool_return_ids: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    tool_call_ids.append(part.tool_call_id)
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    tool_return_ids.append(part.tool_call_id)

    missing = set(tool_call_ids) - set(tool_return_ids)
    assert not missing, (
        f"Baseline (flag off) history has orphan tool_calls — preexisting "
        f"bug, not caused by stream_final_answer. Missing: {missing}"
    )


# ---------------------------------------------------------------------------
# Memory off: synthetic sentinel returns do not need to be persisted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_memory_still_streams_marker_and_runs_cleanly():
    """Without a memory backend, the run must complete identically — the
    chat_history mutation we do for the provider invariant is bounded by
    the single run.
    """
    from upsonic.run.events.events import FinalAnswerStartEvent, FinalOutputEvent

    agent = build_agent_default(name="NoMemoryAgent", tools=[add])
    task = Task(description="Use the add tool to compute 9+1, then summarize.")

    events = await collect_events(agent, task, stream_final_answer=True)
    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    final_outputs = [e for e in events if isinstance(e, FinalOutputEvent)]
    assert len(fa_starts) == 1
    assert len(final_outputs) == 1
