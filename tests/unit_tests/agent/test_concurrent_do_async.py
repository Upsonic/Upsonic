"""Concurrency isolation for do_async on a single agent instance (QA bug F2).

Before the fix, run-scoped state (run_id, output, tool counters) lived directly
on the agent instance, so two concurrent ``do_async`` calls clobbered each
other — a slow call would read None / the other call's run_id after awaiting.

These tests patch ``_do_async_pipeline`` as a controllable seam that reads the
agent's run state across an ``await`` boundary, which is exactly where the bleed
used to happen. They also assert the ContextVar mechanism the streaming path
relies on is isolated per asyncio task.
"""
import asyncio

import pytest
from unittest.mock import patch

from upsonic.agent.agent import Agent
from upsonic.agent.run_state import (
    AgentRunState,
    get_run_state,
    set_run_state,
    reset_run_state,
)
from upsonic.tasks.tasks import Task


@pytest.mark.asyncio
async def test_concurrent_do_async_no_state_bleed():
    """Two concurrent do_async calls on ONE agent keep isolated run state."""
    agent = Agent("anthropic/claude-haiku-4-5")

    delays = {"ALPHA": 0.15, "BETA": 0.02}

    async def fake_pipeline(*, task, run_id, return_output=False, **kwargs):
        # Read run-scoped state, cross an await (where the other call runs its
        # whole lifecycle including its finally), then verify our state held.
        entry_run_id = agent.run_id
        await asyncio.sleep(delays[task.description])
        assert agent.run_id == entry_run_id == run_id, (
            f"run_id bled: entry={entry_run_id} now={agent.run_id} expected={run_id}"
        )
        agent._agent_run_output.output = f"RESULT_{task.description}"
        return f"RESULT_{task.description}"

    with patch.object(agent, "_do_async_pipeline", side_effect=fake_pipeline):
        r_alpha, r_beta = await asyncio.gather(
            agent.do_async(Task("ALPHA")),
            agent.do_async(Task("BETA")),
        )

    assert r_alpha == "RESULT_ALPHA"
    assert r_beta == "RESULT_BETA"


@pytest.mark.asyncio
async def test_many_concurrent_do_async_all_results_correct():
    """A fan-out of concurrent runs each returns its own result, none None."""
    agent = Agent("anthropic/claude-haiku-4-5")

    async def fake_pipeline(*, task, run_id, return_output=False, **kwargs):
        # Stagger so completion order differs from start order.
        await asyncio.sleep((hash(task.description) % 5) / 100.0)
        agent._agent_run_output.output = task.description
        return task.description

    labels = [f"task-{i}" for i in range(8)]
    with patch.object(agent, "_do_async_pipeline", side_effect=fake_pipeline):
        results = await asyncio.gather(*(agent.do_async(Task(x)) for x in labels))

    assert results == labels
    assert all(r is not None for r in results)


@pytest.mark.asyncio
async def test_run_state_contextvar_isolated_per_task():
    """The ContextVar that astream/do_async share is isolated per asyncio task."""

    async def run(tag, delay):
        token = set_run_state(AgentRunState(run_id=tag))
        try:
            await asyncio.sleep(delay)
            # After awaiting (other tasks ran), our state must be intact.
            assert get_run_state().run_id == tag
            return get_run_state().run_id
        finally:
            reset_run_state(token)

    out = await asyncio.gather(run("A", 0.1), run("B", 0.01), run("C", 0.05))
    assert out == ["A", "B", "C"]
    # No scope leaks back to the outer context.
    assert get_run_state() is None
