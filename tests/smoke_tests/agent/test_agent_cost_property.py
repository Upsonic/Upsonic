"""
Smoke tests for the ``Agent.cost`` property.

The ``cost`` property exposes ``Agent.usage`` (an :class:`AgentUsage`) as a
flat dict mirroring :meth:`Task.get_total_cost`, but accumulated across
every task the agent has executed in the session. This matters for
autonomous agents and prebuilts (e.g. :class:`AppliedScientist`) that
dispatch several tasks per run.

Run with:
    uv run pytest tests/smoke_tests/agent/test_agent_cost_property.py -v -s
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from upsonic import Agent, Task
from upsonic.agent.autonomous_agent.autonomous_agent import AutonomousAgent


EXPECTED_KEYS: List[str] = [
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "estimated_cost",
    "requests",
    "tool_calls",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
]


def _assert_cost_shape(cost: Optional[Dict[str, Any]], label: str) -> None:
    assert cost is not None, f"[{label}] cost is None"
    for key in EXPECTED_KEYS:
        assert key in cost, f"[{label}] cost missing key {key!r}; got {sorted(cost)}"
    assert cost["total_tokens"] == cost["input_tokens"] + cost["output_tokens"], (
        f"[{label}] total_tokens != input + output: {cost}"
    )


class TestCostBeforeAnyRun:

    def test_cost_is_none_on_fresh_agent(self) -> None:
        agent = Agent(model="openai/gpt-4o-mini")
        assert agent.cost is None

    def test_cost_is_none_on_fresh_autonomous_agent(self, tmp_path) -> None:
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=str(tmp_path),
            enable_filesystem=False,
            enable_shell=False,
        )
        assert agent.cost is None


class TestCostAfterSingleTask:

    def test_cost_populated_after_one_do(self) -> None:
        agent = Agent(model="openai/gpt-4o-mini")
        agent.do(Task("What is 2+2? Answer with just the number."))

        cost = agent.cost
        _assert_cost_shape(cost, "single_do")
        assert cost["input_tokens"] > 0
        assert cost["output_tokens"] > 0
        assert cost["requests"] >= 1


class TestCostAccumulatesAcrossTasks:

    def test_cost_sums_two_sequential_tasks(self) -> None:
        agent = Agent(model="openai/gpt-4o-mini")

        agent.do(Task("Say hello in French. One word."))
        first = agent.cost
        _assert_cost_shape(first, "after_task_1")
        first_input = first["input_tokens"]
        first_output = first["output_tokens"]
        first_requests = first["requests"]

        agent.do(Task("Say hello in German. One word."))
        second = agent.cost
        _assert_cost_shape(second, "after_task_2")

        assert second["input_tokens"] > first_input, (
            f"input_tokens did not grow: {first_input} -> {second['input_tokens']}"
        )
        assert second["output_tokens"] > first_output, (
            f"output_tokens did not grow: {first_output} -> {second['output_tokens']}"
        )
        assert second["requests"] > first_requests, (
            f"requests did not grow: {first_requests} -> {second['requests']}"
        )

    def test_cost_matches_per_task_token_sum(self) -> None:
        """Aggregated cost on the agent should track the sum of per-task tokens."""
        agent = Agent(model="openai/gpt-4o-mini")
        t1 = Task("What is 1+1? Answer with just the number.")
        t2 = Task("What is 2+2? Answer with just the number.")

        agent.do(t1)
        agent.do(t2)

        cost = agent.cost
        _assert_cost_shape(cost, "two_tasks_sum")

        per_task_input = (t1.usage.input_tokens or 0) + (t2.usage.input_tokens or 0)
        per_task_output = (t1.usage.output_tokens or 0) + (t2.usage.output_tokens or 0)

        # Exact match isn't guaranteed because of internal sub-agent calls
        # (memory summarisation, policies, …) that contribute to agent.usage
        # but not to the user-facing task TaskUsage. The agent total must be
        # at least the user-task sum.
        assert cost["input_tokens"] >= per_task_input - 5, (
            f"agent input ({cost['input_tokens']}) < per-task input sum ({per_task_input})"
        )
        assert cost["output_tokens"] >= per_task_output - 5, (
            f"agent output ({cost['output_tokens']}) < per-task output sum ({per_task_output})"
        )


class TestCostOnAutonomousAgent:

    def test_autonomous_cost_after_one_do(self, tmp_path) -> None:
        """``AutonomousAgent.cost`` must work the same as ``Agent.cost``."""
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=str(tmp_path),
            enable_filesystem=False,
            enable_shell=False,
        )
        agent.do(Task("Reply with the single word OK."))

        cost = agent.cost
        _assert_cost_shape(cost, "autonomous_single_do")
        assert cost["input_tokens"] > 0
        assert cost["output_tokens"] > 0

    def test_autonomous_cost_accumulates_across_runs(self, tmp_path) -> None:
        agent = AutonomousAgent(
            model="openai/gpt-4o-mini",
            workspace=str(tmp_path),
            enable_filesystem=False,
            enable_shell=False,
        )

        agent.do(Task("Reply with the single word ONE."))
        first = agent.cost
        _assert_cost_shape(first, "autonomous_after_1")

        agent.do(Task("Reply with the single word TWO."))
        second = agent.cost
        _assert_cost_shape(second, "autonomous_after_2")

        assert second["input_tokens"] > first["input_tokens"]
        assert second["output_tokens"] > first["output_tokens"]
        assert second["total_tokens"] > first["total_tokens"]
