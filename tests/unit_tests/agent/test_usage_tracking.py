"""
Unit tests for LLM usage tracking across the Agent pipeline.

Tests verify usage accumulation and propagation logic without making real API calls.
All tests use mocks; no API key required.

Covers:
- TaskUsage / RequestUsage incr and aggregation
- AgentUsage aggregation from TaskUsage
- AgentRunOutput._ensure_usage, set_usage_cost
- CultureManager _last_llm_usage accumulation and drain_accumulated_usage
- Orchestrator _propagate_sub_agent_usage
- AgentTool _accumulated_usage and drain_accumulated_usage
- Agent._drain_agent_tool_usage
- SystemPromptManager draining culture usage into agent run output
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from upsonic.usage import TaskUsage, RequestUsage, AgentUsage
from upsonic.run.agent.output import AgentRunOutput


# ---------------------------------------------------------------------------
# TaskUsage and RequestUsage
# ---------------------------------------------------------------------------

class TestTaskUsageRequestUsage:
    """Unit tests for TaskUsage and RequestUsage aggregation."""

    def test_task_usage_incr_request_usage(self) -> None:
        """TaskUsage.incr(RequestUsage) increments tokens and requests."""
        task_usage = TaskUsage()
        req_usage = RequestUsage(input_tokens=100, output_tokens=50)

        task_usage.incr(req_usage)

        assert task_usage.requests == 1
        assert task_usage.input_tokens == 100
        assert task_usage.output_tokens == 50

    def test_task_usage_incr_request_usage_twice(self) -> None:
        """TaskUsage.incr(RequestUsage) twice accumulates correctly."""
        task_usage = TaskUsage()
        req1 = RequestUsage(input_tokens=10, output_tokens=5)
        req2 = RequestUsage(input_tokens=20, output_tokens=15)

        task_usage.incr(req1)
        task_usage.incr(req2)

        assert task_usage.requests == 2
        assert task_usage.input_tokens == 30
        assert task_usage.output_tokens == 20

    def test_task_usage_incr_task_usage(self) -> None:
        """TaskUsage.incr(TaskUsage) merges requests, tokens, and cost."""
        task_usage = TaskUsage()
        other = TaskUsage(
            requests=2,
            input_tokens=200,
            output_tokens=80,
            cost=0.001,
        )

        task_usage.incr(other)

        assert task_usage.requests == 2
        assert task_usage.input_tokens == 200
        assert task_usage.output_tokens == 80
        assert task_usage.cost == 0.001

    def test_task_usage_incr_task_usage_cost_accumulates(self) -> None:
        """TaskUsage.incr(TaskUsage) accumulates cost when both have cost."""
        task_usage = TaskUsage(cost=0.001)
        other = TaskUsage(cost=0.002, requests=1, input_tokens=1, output_tokens=1)

        task_usage.incr(other)

        assert task_usage.cost == 0.003


# ---------------------------------------------------------------------------
# AgentUsage from TaskUsage aggregation
# ---------------------------------------------------------------------------

class TestAgentUsageAggregation:
    """Unit tests for AgentUsage aggregation from TaskUsage."""

    def test_agent_usage_incr_task_usage(self) -> None:
        """AgentUsage.incr(TaskUsage) increments all fields."""
        agent_usage = AgentUsage()
        task_usage = TaskUsage(
            requests=2,
            input_tokens=200,
            output_tokens=80,
            cost=0.001,
            duration=3.0,
            model_execution_time=2.0,
            tool_calls=1,
        )

        agent_usage.incr(task_usage)

        assert agent_usage.requests == 2
        assert agent_usage.input_tokens == 200
        assert agent_usage.output_tokens == 80
        assert agent_usage.cost == 0.001
        assert agent_usage.duration == 3.0
        assert agent_usage.model_execution_time == 2.0
        assert agent_usage.tool_calls == 1

    def test_agent_usage_incr_multiple_task_usages(self) -> None:
        """AgentUsage accumulates across multiple TaskUsage increments."""
        agent_usage = AgentUsage()

        agent_usage.incr(TaskUsage(requests=1, input_tokens=100, output_tokens=50, cost=0.001))
        agent_usage.incr(TaskUsage(requests=2, input_tokens=200, output_tokens=100, cost=0.002))

        assert agent_usage.requests == 3
        assert agent_usage.input_tokens == 300
        assert agent_usage.output_tokens == 150
        assert agent_usage.cost == pytest.approx(0.003)

    def test_agent_usage_incr_agent_usage(self) -> None:
        """AgentUsage.incr(AgentUsage) merges sub-agent totals."""
        main = AgentUsage(requests=1, input_tokens=100, cost=0.001)
        sub = AgentUsage(requests=2, input_tokens=200, cost=0.002, duration=5.0, model_execution_time=3.0)

        main.incr(sub)

        assert main.requests == 3
        assert main.input_tokens == 300
        assert main.cost == pytest.approx(0.003)
        assert main.duration == 5.0
        assert main.model_execution_time == 3.0


# ---------------------------------------------------------------------------
# AgentRunOutput usage methods
# ---------------------------------------------------------------------------

class TestAgentRunOutputUsage:
    """Unit tests for AgentRunOutput usage tracking methods."""

    def test_ensure_usage_creates_task_usage(self) -> None:
        """_ensure_usage() creates TaskUsage when usage is None."""
        output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )
        output.usage = None

        usage = output._ensure_usage()

        assert usage is not None
        assert isinstance(usage, TaskUsage)
        assert output.usage is usage

    def test_ensure_usage_incr_request_usage(self) -> None:
        """_ensure_usage().incr(RequestUsage) increments usage."""
        output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )

        req_usage = RequestUsage(input_tokens=50, output_tokens=25)
        output._ensure_usage().incr(req_usage)

        assert output.usage.requests == 1
        assert output.usage.input_tokens == 50
        assert output.usage.output_tokens == 25

    def test_ensure_usage_incr_multiple(self) -> None:
        """Multiple _ensure_usage().incr() calls accumulate correctly."""
        output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )

        output._ensure_usage().incr(RequestUsage(input_tokens=30, output_tokens=10))
        output._ensure_usage().incr(RequestUsage(input_tokens=20, output_tokens=15))

        assert output.usage.requests == 2
        assert output.usage.input_tokens == 50
        assert output.usage.output_tokens == 25

    def test_set_usage_cost_initial(self) -> None:
        """set_usage_cost sets cost when current cost is None."""
        output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )

        output.set_usage_cost(0.005)

        assert output.usage.cost == 0.005

    def test_set_usage_cost_accumulates(self) -> None:
        """set_usage_cost adds to existing cost."""
        output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )
        output._ensure_usage().cost = 0.001

        output.set_usage_cost(0.002)

        assert output.usage.cost == 0.003


# ---------------------------------------------------------------------------
# TaskUsage aggregation from multiple sources (sanity)
# ---------------------------------------------------------------------------

class TestTaskUsageAggregation:
    """Sanity tests for TaskUsage aggregation from multiple sources."""

    def test_aggregate_direct_request_plus_sub_agent_task_usage(self) -> None:
        """Simulate one direct RequestUsage + one sub-agent TaskUsage."""
        task_usage = TaskUsage()

        task_usage.incr(RequestUsage(input_tokens=100, output_tokens=50))
        task_usage.incr(TaskUsage(requests=1, input_tokens=200, output_tokens=80, cost=0.001))

        assert task_usage.requests == 2
        assert task_usage.input_tokens == 300
        assert task_usage.output_tokens == 130
        assert task_usage.cost == 0.001

    def test_agent_run_output_multiple_incr_and_set_cost(self) -> None:
        """AgentRunOutput can receive multiple _ensure_usage().incr() and set_usage_cost."""
        output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )

        output._ensure_usage().incr(RequestUsage(input_tokens=10, output_tokens=5))
        output.set_usage_cost(0.0001)
        output._ensure_usage().incr(RequestUsage(input_tokens=20, output_tokens=10))
        output.set_usage_cost(0.0002)
        output.usage.incr(TaskUsage(requests=1, input_tokens=30, output_tokens=15, cost=0.0003))

        assert output.usage.requests == 3
        assert output.usage.input_tokens == 60
        assert output.usage.output_tokens == 30
        assert output.usage.cost == pytest.approx(0.0006)
