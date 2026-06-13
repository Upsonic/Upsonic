"""MemorySaveStep finalize (24/24) + registry tool-time wiring — end to end.

Two pre-existing framework defects surfaced by QA, both fixed here:

- **Bug A — MemorySaveStep never finalized.** It was the only non-streaming
  step lacking the ``finally: if step_result: self._finalize_step_result(...)``
  block every sibling has, so ``execution_stats`` under-counted ("Pipeline
  completed: 23/24 steps") and ``step_statuses['memory_save']`` was never
  recorded.
- **Bug B — tool wall time never reached the usage registry.** The agent folded
  per-tool elapsed only into the per-run snapshot; since ``Agent.usage`` is now
  derived purely from the registry, ``Agent.usage.tool_execution_time`` stayed
  ``0`` despite real tool calls.

The existing pipeline metrics smoke test asserts only on ``output.usage`` (the
snapshot, which always worked), so it never caught Bug B — these assert on the
registry-derived ``Agent.usage`` and on ``execution_stats`` directly.

Pinned to gpt-4o for deterministic tool calling (the smoke-suite global gpt-5
override would make tool-calling flaky and both fixes are model-agnostic).

Run with: uv run pytest tests/smoke_tests/agent/test_step_finalize_and_tool_time_smoke.py -v -s
"""
import pytest

from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.usage_registry import get_default_registry
from tests.smoke_tests._model_selection import without_model_override


pytestmark = pytest.mark.timeout(180)


@tool
def adder(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The sum of a and b.
    """
    return a + b


@pytest.mark.asyncio
async def test_memory_save_finalizes_and_tool_time_recorded():
    with without_model_override():
        agent = Agent("openai/gpt-4o", tools=[adder])

    task = Task("Use the adder tool to add 21 and 21, then state the result.")
    output = await agent.do_async(task, return_output=True)

    # ---- Bug A: every step finalized (24/24); memory_save recorded ----
    stats = output.execution_stats
    assert stats is not None, "execution_stats should be populated after a run"
    assert "memory_save" in stats.step_statuses, (
        f"memory_save missing from step_statuses {list(stats.step_statuses)} "
        "— MemorySaveStep did not finalize"
    )
    assert stats.executed_steps == stats.total_steps, (
        f"executed {stats.executed_steps}/{stats.total_steps} steps — "
        "a step did not finalize (the 23/24 regression)"
    )

    # ---- Bug B: registry-derived tool time reflects the real tool call ----
    assert agent.usage.tool_execution_time > 0, (
        f"agent.usage.tool_execution_time={agent.usage.tool_execution_time} — "
        "tool wall time not mirrored into the usage registry"
    )
    tool_entries = get_default_registry().entries(
        agent_usage_id=agent.agent_usage_id, kind="tool"
    )
    assert tool_entries, "expected at least one kind='tool' registry entry"
    assert all(e.tool_execution_time > 0 for e in tool_entries)
