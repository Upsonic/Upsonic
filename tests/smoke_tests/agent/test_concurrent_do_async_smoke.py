"""End-to-end concurrency isolation for do_async on one Agent (QA bug F2).

ContextVar isolation of per-run state (`_agent_run_output` / `_tool_call_count` /
`run_id`) is proven deterministically by the unit suite
(`tests/unit_tests/agent/test_concurrent_do_async.py`). This smoke is
end-to-end defense-in-depth on the *real* pipeline — two concurrent runs with
genuine tool execution — for the highest-blast-radius fix.

Pinned to gpt-4o (deterministic tool calling) via `without_model_override`,
since the smoke-suite global gpt-5 override would make tool calls flaky and F2
is model-agnostic. The sync tool's `time.sleep` runs in an executor thread
(`loop.run_in_executor`), so the event loop stays free and the two coroutines
genuinely interleave at the tool await + the model round-trips — exactly where
a run-state bleed would corrupt one call with the other's state.

Run with: uv run pytest tests/smoke_tests/agent/test_concurrent_do_async_smoke.py -v -s
"""
import asyncio
import time

import pytest

from upsonic import Agent
from upsonic.tools import tool
from tests.smoke_tests._model_selection import without_model_override


pytestmark = pytest.mark.timeout(180)


@tool
def tag(value: str) -> str:
    """Echo the given value back after a short delay.

    Args:
        value: The string to return unchanged.

    Returns:
        The same string.
    """
    time.sleep(0.2)
    return value


@pytest.mark.asyncio
async def test_concurrent_do_async_no_state_bleed_e2e():
    # Pin to a deterministic tool-calling model; F2 is about run-state
    # isolation, not model behavior.
    with without_model_override():
        agent = Agent("openai/gpt-4o", tools=[tag])

    prompt_a = "Call the tag tool with 'ALPHA' and report exactly what it returns."
    prompt_b = "Call the tag tool with 'BETA' and report exactly what it returns."

    out_a, out_b = await asyncio.gather(
        agent.do_async(prompt_a, return_output=True),
        agent.do_async(prompt_b, return_output=True),
    )

    # Neither call lost its output to the other.
    assert out_a is not None and out_b is not None

    # Each result carries ITS OWN tool value — no cross-attribution of
    # self._agent_run_output across the concurrent calls. Containment, never
    # equality (the model may wrap the value in text).
    text_a = str(out_a.output).upper()
    text_b = str(out_b.output).upper()
    assert "ALPHA" in text_a and "BETA" not in text_a, f"call A bled: {out_a.output!r}"
    assert "BETA" in text_b and "ALPHA" not in text_b, f"call B bled: {out_b.output!r}"

    # Each run's tool-call count is its own (self._tool_call_count not clobbered).
    assert out_a.usage.tool_calls >= 1, f"A tool_calls={out_a.usage.tool_calls}"
    assert out_b.usage.tool_calls >= 1, f"B tool_calls={out_b.usage.tool_calls}"
