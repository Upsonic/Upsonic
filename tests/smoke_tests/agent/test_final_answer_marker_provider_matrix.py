"""Provider-parity smoke tests for ``stream_final_answer``.

The primary smoke target is Anthropic Sonnet (covered in
``test_final_answer_marker.py``). This module verifies that Google Gemini
produces the same observable contract: marker emitted exactly once, no
sentinel leakage, marker precedes final-answer text deltas.

A small autouse fixture clears Upsonic's module-level
``_cached_async_http_client`` between each test. Without it, Google's
``BaseApiClient._async_request_once`` reuses an ``httpx.AsyncClient`` bound
to a previously closed pytest-asyncio event loop and crashes with
``RuntimeError: Event loop is closed``.
"""

from __future__ import annotations

import pytest

from _final_answer_marker_helpers import (
    AnswerSchema,
    FINAL_ANSWER_MARKER_TOOL_NAME,
    Task,
    add,
    build_agent_google,
    clear_cached_http_clients,
    collect_events,
    has_google,
    skip_if_no_provider,
)

pytestmark = skip_if_no_provider


# ---------------------------------------------------------------------------
# Per-test HTTP-client cache reset (Google-specific fix)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_http_client_per_test():
    """Force a fresh httpx.AsyncClient per test to keep Google's SDK happy
    across pytest-asyncio's function-scoped event loops."""
    clear_cached_http_clients()
    yield
    clear_cached_http_clients()


# ---------------------------------------------------------------------------
# Gemini obey path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not has_google(), reason="Gemini parity test needs Google key + SDK")
async def test_gemini_obey_path_emits_marker():
    from upsonic.run.events.events import (
        FinalAnswerStartEvent,
        ToolCallEvent,
        ToolResultEvent,
    )

    agent = build_agent_google(name="GeminiObeyAgent", tools=[add])
    task = Task(
        description="Use the `add` tool to compute 6+9, then state the result in one sentence."
    )

    events = await collect_events(agent, task, stream_final_answer=True)

    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 1, (
        f"Gemini must emit exactly one marker (parity with Anthropic); "
        f"got {len(fa_starts)}. Timeline: {[type(e).__name__ for e in events]}"
    )
    assert fa_starts[0].triggered_by == "sentinel"

    sentinel_leaks = [
        e for e in events
        if isinstance(e, (ToolCallEvent, ToolResultEvent))
        and e.tool_name == FINAL_ANSWER_MARKER_TOOL_NAME
    ]
    assert len(sentinel_leaks) == 0

    real_add_calls = [
        e for e in events
        if isinstance(e, ToolCallEvent) and e.tool_name == "add"
    ]
    assert len(real_add_calls) >= 1


# ---------------------------------------------------------------------------
# Gemini structured output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not has_google(), reason="Gemini parity test needs Google key + SDK")
async def test_gemini_structured_output_marker():
    """Gemini structured output path emits marker on final_result tool onset."""
    from upsonic.run.events.events import FinalAnswerStartEvent, FinalOutputEvent

    agent = build_agent_google(name="GeminiStructuredAgent", tools=[add])
    task = Task(
        description="Use the `add` tool to compute 4+4.",
        response_format=AnswerSchema,
    )

    events = await collect_events(agent, task, stream_final_answer=True)

    fa_starts = [e for e in events if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 1, (
        f"Gemini structured-output path must emit exactly one marker; "
        f"got {len(fa_starts)}"
    )
    assert fa_starts[0].triggered_by in ("sentinel", "output_tool")

    final_outputs = [e for e in events if isinstance(e, FinalOutputEvent)]
    assert len(final_outputs) == 1
