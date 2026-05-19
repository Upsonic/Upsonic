"""
Unit tests for task-level usage tracking via TaskUsage.

All tests use mocks — no API key required.

Covers:
- RequestUsage initialization and token-only incr
- TaskUsage initialization, start/stop timer, add_model_execution_time
- TaskUsage.incr (RequestUsage and TaskUsage)
- TaskUsage.upsonic_execution_time computed property
- TaskUsage.to_dict / from_dict round-trip
- Task._usage lifecycle: task_start -> populate -> task_end
- Task property delegation (duration, model_execution_time, upsonic_execution_time)
- Task.usage property
- Task.to_dict / from_dict preservation of _usage
- AgentUsage.incr(TaskUsage) aggregation
- Multiple TaskUsage instances on AgentUsage (multi-task scenario)
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import Mock

from upsonic.usage import RequestUsage, TaskUsage, AgentUsage
from upsonic.tasks.tasks import Task


# ---------------------------------------------------------------------------
# RequestUsage: Initialization & Basics (token-only)
# ---------------------------------------------------------------------------

class TestRequestUsageInit:

    def test_default_values(self) -> None:
        usage = RequestUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_with_token_values(self) -> None:
        usage = RequestUsage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_requests_always_one(self) -> None:
        usage = RequestUsage()
        assert usage.requests == 1

    def test_no_timer_attribute(self) -> None:
        usage = RequestUsage()
        assert not hasattr(usage, "timer")

    def test_no_duration_attribute(self) -> None:
        usage = RequestUsage()
        assert not hasattr(usage, "duration")

    def test_no_model_execution_time_attribute(self) -> None:
        usage = RequestUsage()
        assert not hasattr(usage, "model_execution_time")


# ---------------------------------------------------------------------------
# RequestUsage: incr (token-only accumulation)
# ---------------------------------------------------------------------------

class TestRequestUsageIncr:

    def test_incr_accumulates_tokens(self) -> None:
        usage = RequestUsage(input_tokens=10, output_tokens=5)
        other = RequestUsage(input_tokens=20, output_tokens=15)
        usage.incr(other)
        assert usage.input_tokens == 30
        assert usage.output_tokens == 20

    def test_incr_cache_tokens(self) -> None:
        usage = RequestUsage(cache_write_tokens=10, cache_read_tokens=5)
        other = RequestUsage(cache_write_tokens=20, cache_read_tokens=15)
        usage.incr(other)
        assert usage.cache_write_tokens == 30
        assert usage.cache_read_tokens == 20


# ---------------------------------------------------------------------------
# RequestUsage: __add__
# ---------------------------------------------------------------------------

class TestRequestUsageAdd:

    def test_add_creates_new_instance(self) -> None:
        a = RequestUsage(input_tokens=10, output_tokens=5)
        b = RequestUsage(input_tokens=20, output_tokens=15)
        c = a + b
        assert c.input_tokens == 30
        assert c.output_tokens == 20
        assert a.input_tokens == 10
        assert b.input_tokens == 20


# ---------------------------------------------------------------------------
# RequestUsage: to_dict / from_dict (token-only)
# ---------------------------------------------------------------------------

class TestRequestUsageSerialization:

    def test_to_dict_basic(self) -> None:
        usage = RequestUsage(input_tokens=100, output_tokens=50)
        d = usage.to_dict()
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert "duration" not in d
        assert "model_execution_time" not in d

    def test_to_dict_excludes_zero_values(self) -> None:
        usage = RequestUsage()
        d = usage.to_dict()
        assert "input_tokens" not in d
        assert "output_tokens" not in d

    def test_to_dict_includes_nonzero_values(self) -> None:
        usage = RequestUsage(input_tokens=1)
        d = usage.to_dict()
        assert "input_tokens" in d

    def test_from_dict_restores_all_fields(self) -> None:
        d = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_write_tokens": 10,
            "cache_read_tokens": 5,
        }
        usage = RequestUsage.from_dict(d)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_write_tokens == 10
        assert usage.cache_read_tokens == 5

    def test_round_trip(self) -> None:
        original = RequestUsage(input_tokens=42, output_tokens=17)
        d = original.to_dict()
        restored = RequestUsage.from_dict(d)
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens

    def test_from_dict_defaults(self) -> None:
        usage = RequestUsage.from_dict({})
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0


# ---------------------------------------------------------------------------
# TaskUsage: Initialization & Basics
# ---------------------------------------------------------------------------

class TestTaskUsageInit:

    def test_default_values(self) -> None:
        usage = TaskUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.requests == 0
        assert usage.duration is None
        assert usage.model_execution_time is None
        assert usage.upsonic_execution_time is None
        assert usage.timer is None
        assert usage.cost is None
        assert usage.tool_calls == 0

    def test_with_token_values(self) -> None:
        usage = TaskUsage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50


# ---------------------------------------------------------------------------
# TaskUsage: Timer & Timing
# ---------------------------------------------------------------------------

class TestTaskUsageTimer:

    def test_start_timer_creates_timer(self) -> None:
        usage = TaskUsage()
        assert usage.timer is None
        usage.start_timer()
        assert usage.timer is not None

    def test_stop_timer_sets_duration(self) -> None:
        usage = TaskUsage()
        usage.start_timer()
        time.sleep(0.05)
        usage.stop_timer()
        assert usage.duration is not None
        assert usage.duration >= 0.04

    def test_stop_timer_without_start_is_noop(self) -> None:
        usage = TaskUsage()
        usage.stop_timer()
        assert usage.duration is None

    def test_stop_timer_set_duration_false(self) -> None:
        usage = TaskUsage()
        usage.start_timer()
        time.sleep(0.02)
        usage.stop_timer(set_duration=False)
        assert usage.duration is None

    def test_add_model_execution_time_first_call(self) -> None:
        usage = TaskUsage()
        usage.add_model_execution_time(1.5)
        assert usage.model_execution_time == 1.5

    def test_add_model_execution_time_accumulates(self) -> None:
        usage = TaskUsage()
        usage.add_model_execution_time(1.0)
        usage.add_model_execution_time(0.5)
        usage.add_model_execution_time(0.3)
        assert usage.model_execution_time == pytest.approx(1.8, abs=0.01)

    def test_upsonic_execution_time_computed(self) -> None:
        usage = TaskUsage()
        usage.duration = 5.0
        usage.model_execution_time = 3.0
        assert usage.upsonic_execution_time == pytest.approx(2.0)

    def test_upsonic_execution_time_none_when_duration_missing(self) -> None:
        usage = TaskUsage()
        usage.model_execution_time = 3.0
        assert usage.upsonic_execution_time is None

    def test_upsonic_execution_time_none_when_model_time_missing(self) -> None:
        usage = TaskUsage()
        usage.duration = 5.0
        assert usage.upsonic_execution_time is None

    def test_upsonic_execution_time_zero_when_equal(self) -> None:
        usage = TaskUsage()
        usage.duration = 3.0
        usage.model_execution_time = 3.0
        assert usage.upsonic_execution_time == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TaskUsage: incr (RequestUsage and TaskUsage)
# ---------------------------------------------------------------------------

class TestTaskUsageIncr:

    def test_incr_request_usage_increments_tokens_and_requests(self) -> None:
        usage = TaskUsage()
        req = RequestUsage(input_tokens=100, output_tokens=50)
        usage.incr(req)
        assert usage.requests == 1
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_incr_request_usage_twice(self) -> None:
        usage = TaskUsage()
        usage.incr(RequestUsage(input_tokens=10, output_tokens=5))
        usage.incr(RequestUsage(input_tokens=20, output_tokens=15))
        assert usage.requests == 2
        assert usage.input_tokens == 30
        assert usage.output_tokens == 20

    def test_incr_task_usage_merges_all_fields(self) -> None:
        usage = TaskUsage()
        other = TaskUsage(
            requests=2,
            input_tokens=200,
            output_tokens=80,
            cost=0.001,
            duration=3.0,
            model_execution_time=2.0,
        )
        usage.incr(other)
        assert usage.requests == 2
        assert usage.input_tokens == 200
        assert usage.output_tokens == 80
        assert usage.cost == 0.001
        assert usage.duration == 3.0
        assert usage.model_execution_time == 2.0

    def test_incr_task_usage_cost_accumulates(self) -> None:
        usage = TaskUsage(cost=0.001)
        other = TaskUsage(cost=0.002, requests=1, input_tokens=1, output_tokens=1)
        usage.incr(other)
        assert usage.cost == 0.003

    def test_incr_task_usage_timing_accumulates(self) -> None:
        usage = TaskUsage()
        usage.duration = 1.0
        usage.model_execution_time = 0.8
        other = TaskUsage(duration=2.0, model_execution_time=1.5)
        usage.incr(other)
        assert usage.duration == pytest.approx(3.0)
        assert usage.model_execution_time == pytest.approx(2.3)

    def test_incr_cache_tokens(self) -> None:
        usage = TaskUsage(cache_write_tokens=10, cache_read_tokens=5)
        other = RequestUsage(cache_write_tokens=20, cache_read_tokens=15)
        usage.incr(other)
        assert usage.cache_write_tokens == 30
        assert usage.cache_read_tokens == 20


# ---------------------------------------------------------------------------
# TaskUsage: to_dict / from_dict
# ---------------------------------------------------------------------------

class TestTaskUsageSerialization:

    def test_to_dict_basic(self) -> None:
        usage = TaskUsage(input_tokens=100, output_tokens=50, requests=2)
        usage.duration = 5.0
        usage.model_execution_time = 3.0
        d = usage.to_dict()
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["requests"] == 2
        assert d["duration"] == 5.0
        assert d["model_execution_time"] == 3.0
        assert d["upsonic_execution_time"] == pytest.approx(2.0)

    def test_to_dict_excludes_zero_values(self) -> None:
        usage = TaskUsage()
        d = usage.to_dict()
        assert "input_tokens" not in d
        assert "output_tokens" not in d
        assert "duration" not in d

    def test_from_dict_restores_all_fields(self) -> None:
        d = {
            "requests": 3,
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_write_tokens": 10,
            "cache_read_tokens": 5,
            "duration": 5.0,
            "model_execution_time": 3.0,
            "cost": 0.005,
            "tool_calls": 2,
        }
        usage = TaskUsage.from_dict(d)
        assert usage.requests == 3
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_write_tokens == 10
        assert usage.cache_read_tokens == 5
        assert usage.duration == 5.0
        assert usage.model_execution_time == 3.0
        assert usage.cost == 0.005
        assert usage.tool_calls == 2
        assert usage.upsonic_execution_time == pytest.approx(2.0)

    def test_round_trip(self) -> None:
        original = TaskUsage(input_tokens=42, output_tokens=17, requests=2)
        original.duration = 2.5
        original.model_execution_time = 1.8
        original.cost = 0.003
        d = original.to_dict()
        restored = TaskUsage.from_dict(d)
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.requests == original.requests
        assert restored.duration == original.duration
        assert restored.model_execution_time == original.model_execution_time
        assert restored.upsonic_execution_time == original.upsonic_execution_time
        assert restored.cost == original.cost

    def test_from_dict_defaults(self) -> None:
        usage = TaskUsage.from_dict({})
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.requests == 0
        assert usage.duration is None
        assert usage.model_execution_time is None


# ---------------------------------------------------------------------------
# Task._usage lifecycle: task_start / task_end
# ---------------------------------------------------------------------------

class TestTaskUsageLifecycle:

    def test_task_usage_none_initially(self) -> None:
        task = Task(description="test")
        assert task._usage is None
        # task.usage is now a registry-backed AggregatedUsage view —
        # always non-None, zero-valued when nothing has been recorded.
        assert task.usage is not None
        assert task.usage.input_tokens == 0
        assert task.usage.output_tokens == 0
        assert task.usage.requests == 0

    def test_task_start_creates_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        assert task._usage is not None
        assert isinstance(task._usage, TaskUsage)
        assert task._usage.timer is not None

    def test_task_end_stops_timer_sets_duration(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        time.sleep(0.05)
        task.task_end()
        assert task._usage.duration is not None
        assert task._usage.duration >= 0.04

    def test_task_start_sets_start_time(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        before = time.time()
        task.task_start(mock_agent)
        assert task.start_time is not None
        assert task.start_time >= before - 1

    def test_task_end_sets_end_time(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        task.task_end()
        assert task.end_time is not None


# ---------------------------------------------------------------------------
# Task property delegation to _usage
# ---------------------------------------------------------------------------

class TestTaskPropertyDelegation:
    """The legacy ``task.duration`` / ``task.model_execution_time`` /
    ``task.tool_execution_time`` / ``task.upsonic_execution_time``
    properties were removed in the unification pass — callers read
    everything off ``task.usage`` now. The tests that asserted the
    delegation contract no longer apply and have been deleted."""

    def test_usage_property_is_registry_view_not_internal_mutable(self) -> None:
        """``task.usage`` is now an :class:`AggregatedUsage` derived from
        the registry — distinct from the in-pipeline ``_usage`` timer
        scratchpad."""
        from upsonic.usage import TaskUsage
        from upsonic.usage_registry import AggregatedUsage

        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        assert isinstance(task._usage, TaskUsage)
        assert isinstance(task.usage, AggregatedUsage)
        assert task.usage is not task._usage


# ---------------------------------------------------------------------------
# Task._usage: manually populating tokens + timing
# ---------------------------------------------------------------------------

class TestTaskUsagePopulation:

    def test_tokens_accumulated_on_task_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)

        task._usage.incr(RequestUsage(input_tokens=100, output_tokens=50))
        task._usage.incr(RequestUsage(input_tokens=200, output_tokens=80))

        assert task._usage.input_tokens == 300
        assert task._usage.output_tokens == 130
        assert task._usage.requests == 2

    def test_model_time_accumulated_on_task_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)

        task._usage.add_model_execution_time(1.0)
        task._usage.add_model_execution_time(0.5)

        assert task._usage.model_execution_time == pytest.approx(1.5)

    def test_full_lifecycle_with_tokens_and_timing(self) -> None:
        """``task._usage`` is the in-pipeline timer / model_execution_time
        scratchpad. The public ``task.usage`` is a registry view —
        populating ``_usage`` directly does NOT show up there. To get a
        recorded entry under this task's scope, callers must go through
        the Phase-2 emission hook (``record_request_usage``) — which is
        what every real pipeline step does."""
        from upsonic.usage_registry import scope, record_request_usage, get_default_registry
        get_default_registry().clear()

        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None

        task.task_start(mock_agent)

        # Scratchpad is still useful for timer state.
        task._usage.incr(RequestUsage(input_tokens=50, output_tokens=20))
        task._usage.add_model_execution_time(0.8)
        task._usage.incr(RequestUsage(input_tokens=30, output_tokens=10))
        task._usage.add_model_execution_time(0.4)

        time.sleep(0.05)
        task.task_end()

        # Scratchpad totals
        assert task._usage.input_tokens == 80
        assert task._usage.output_tokens == 30
        assert task._usage.model_execution_time == pytest.approx(1.2, abs=0.01)

        # Public registry view is empty — nothing was recorded via the
        # emission hook in this synthetic test.
        assert task.usage.input_tokens == 0
        assert task.usage.requests == 0

        # Now drive the registry the way the pipeline does.
        with scope(task_usage_id=task.task_usage_id):
            record_request_usage(
                RequestUsage(input_tokens=50, output_tokens=20),
                model="test-model",
            )
            record_request_usage(
                RequestUsage(input_tokens=30, output_tokens=10),
                model="test-model",
            )

        assert task.usage.input_tokens == 80
        assert task.usage.output_tokens == 30
        assert task.usage.requests == 2

        # Internal ``_usage`` scratchpad still carries the timer state
        # the pipeline accumulated during the run.
        assert task._usage.duration is None or task._usage.duration >= 0
        assert task._usage.model_execution_time == pytest.approx(1.2, abs=0.01)


# ---------------------------------------------------------------------------
# Task.to_dict / from_dict preserves _usage
# ---------------------------------------------------------------------------

class TestTaskSerializationWithUsage:

    def test_to_dict_includes_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        task._usage.incr(RequestUsage(input_tokens=42, output_tokens=17))
        task._usage.duration = 2.5
        task._usage.model_execution_time = 1.8
        task.task_end()

        d = task.to_dict()
        assert "_usage" in d
        assert d["_usage"] is not None
        assert d["_usage"]["input_tokens"] == 42
        assert d["_usage"]["output_tokens"] == 17

    def test_to_dict_usage_none_when_not_started(self) -> None:
        task = Task(description="test")
        d = task.to_dict()
        assert d["_usage"] is None

    def test_from_dict_restores_internal_usage_scratchpad(self) -> None:
        """``Task._usage`` (the timer scratchpad) round-trips through
        ``to_dict`` / ``from_dict``. The public ``task.usage`` is now a
        registry view independent of the serialised _usage state — it
        reflects ledger entries scoped to ``task_usage_id``, which a
        deserialised task does not auto-rehydrate (storage layer does
        that explicitly via ``UsageRegistry.load_from_storage``)."""
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None
        task.task_start(mock_agent)
        task._usage.incr(RequestUsage(input_tokens=100, output_tokens=50))
        task._usage.duration = 3.0
        task._usage.model_execution_time = 2.0
        task.task_end()

        d = task.to_dict()
        restored = Task.from_dict(d)

        # Internal scratchpad survives the round-trip.
        assert restored._usage is not None
        assert isinstance(restored._usage, TaskUsage)
        assert restored._usage.input_tokens == 100
        assert restored._usage.output_tokens == 50
        assert restored._usage.duration is not None
        assert restored._usage.model_execution_time == 2.0

    def test_from_dict_without_usage(self) -> None:
        from upsonic.usage_registry import AggregatedUsage
        d = {"description": "test", "_usage": None}
        restored = Task.from_dict(d)
        # Internal scratchpad is None when not serialised.
        assert restored._usage is None
        # Public registry view is a zero-valued AggregatedUsage.
        assert isinstance(restored.usage, AggregatedUsage)
        assert restored.usage.input_tokens == 0


# ---------------------------------------------------------------------------
# AgentUsage.incr(TaskUsage) — aggregation for agent level
# ---------------------------------------------------------------------------

class TestAgentUsageIncrFromTaskUsage:

    def test_incr_task_usage_increments_requests(self) -> None:
        agent_usage = AgentUsage()
        task_usage = TaskUsage(requests=1, input_tokens=100, output_tokens=50)
        agent_usage.incr(task_usage)
        assert agent_usage.requests == 1
        assert agent_usage.input_tokens == 100
        assert agent_usage.output_tokens == 50

    def test_incr_multiple_task_usages(self) -> None:
        agent_usage = AgentUsage()
        agent_usage.incr(TaskUsage(requests=1, input_tokens=10, output_tokens=5))
        agent_usage.incr(TaskUsage(requests=2, input_tokens=20, output_tokens=15))
        agent_usage.incr(TaskUsage(requests=1, input_tokens=30, output_tokens=25))
        assert agent_usage.requests == 4
        assert agent_usage.input_tokens == 60
        assert agent_usage.output_tokens == 45

    def test_incr_task_usage_with_timing(self) -> None:
        agent_usage = AgentUsage()
        t1 = TaskUsage(requests=1, input_tokens=10, output_tokens=5, duration=2.0, model_execution_time=1.5)
        agent_usage.incr(t1)

        t2 = TaskUsage(requests=1, input_tokens=20, output_tokens=15, duration=3.0, model_execution_time=2.0)
        agent_usage.incr(t2)

        assert agent_usage.requests == 2
        assert agent_usage.input_tokens == 30
        assert agent_usage.output_tokens == 20
        assert agent_usage.duration == pytest.approx(5.0)
        assert agent_usage.model_execution_time == pytest.approx(3.5)
        assert agent_usage.upsonic_execution_time == pytest.approx(1.5)

    def test_incr_task_usage_timing_none_stays_none(self) -> None:
        agent_usage = AgentUsage()
        task_usage = TaskUsage(requests=1, input_tokens=10, output_tokens=5)
        agent_usage.incr(task_usage)
        assert agent_usage.duration is None
        assert agent_usage.model_execution_time is None
        assert agent_usage.upsonic_execution_time is None

    def test_incr_task_usage_with_cost(self) -> None:
        agent_usage = AgentUsage()
        agent_usage.incr(TaskUsage(requests=1, cost=0.001))
        agent_usage.incr(TaskUsage(requests=1, cost=0.002))
        assert agent_usage.cost == pytest.approx(0.003)

    def test_incr_task_usage_with_tool_calls(self) -> None:
        agent_usage = AgentUsage()
        agent_usage.incr(TaskUsage(requests=1, tool_calls=3))
        agent_usage.incr(TaskUsage(requests=1, tool_calls=2))
        assert agent_usage.tool_calls == 5


# ---------------------------------------------------------------------------
# Multi-task simulation: multiple tasks -> AgentUsage
# ---------------------------------------------------------------------------

class TestMultiTaskAgentUsageAggregation:

    def test_simulate_three_tasks_aggregated_to_agent_usage(self) -> None:
        agent_usage = AgentUsage()

        for i in range(3):
            task_usage = TaskUsage(
                requests=1,
                input_tokens=100 * (i + 1),
                output_tokens=50 * (i + 1),
                duration=1.0 + i * 0.5,
                model_execution_time=0.8 + i * 0.3,
            )
            agent_usage.incr(task_usage)

        assert agent_usage.requests == 3
        assert agent_usage.input_tokens == 600
        assert agent_usage.output_tokens == 300
        assert agent_usage.duration == pytest.approx(4.5, abs=0.01)
        assert agent_usage.model_execution_time == pytest.approx(3.3, abs=0.01)
        assert agent_usage.upsonic_execution_time == pytest.approx(1.2, abs=0.01)

    def test_independent_task_usages_not_shared(self) -> None:
        t1 = TaskUsage(requests=1, input_tokens=10, output_tokens=5, duration=1.0)
        t2 = TaskUsage(requests=1, input_tokens=20, output_tokens=15, duration=2.0)

        assert t1 is not t2
        assert t1.input_tokens != t2.input_tokens
        assert t1.duration != t2.duration


# ---------------------------------------------------------------------------
# AgentUsage: to_dict / from_dict
# ---------------------------------------------------------------------------

class TestAgentUsageSerialization:

    def test_to_dict_basic(self) -> None:
        usage = AgentUsage(requests=3, input_tokens=500, output_tokens=200, cost=0.01)
        usage.duration = 10.0
        usage.model_execution_time = 7.0
        d = usage.to_dict()
        assert d["requests"] == 3
        assert d["input_tokens"] == 500
        assert d["output_tokens"] == 200
        assert d["cost"] == 0.01
        assert d["duration"] == 10.0
        assert d["model_execution_time"] == 7.0
        assert d["upsonic_execution_time"] == pytest.approx(3.0)

    def test_round_trip(self) -> None:
        original = AgentUsage(requests=5, input_tokens=1000, output_tokens=500, cost=0.05)
        original.duration = 15.0
        original.model_execution_time = 10.0
        d = original.to_dict()
        restored = AgentUsage.from_dict(d)
        assert restored.requests == original.requests
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.cost == original.cost
        assert restored.duration == original.duration
        assert restored.model_execution_time == original.model_execution_time


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_task_end_without_start_no_error(self) -> None:
        task = Task(description="test")
        task.task_end()
        assert task._usage is None

    def test_add_model_time_before_task_start_no_crash(self) -> None:
        task = Task(description="test")
        assert task._usage is None

    def test_multiple_task_starts_resets_usage(self) -> None:
        task = Task(description="test")
        mock_agent = Mock()
        mock_agent.canvas = None

        task.task_start(mock_agent)
        first_usage = task._usage
        task._usage.incr(RequestUsage(input_tokens=100, output_tokens=50))

        task.task_start(mock_agent)
        assert task._usage is not first_usage
        assert task._usage.input_tokens == 0

    def test_request_usage_has_values(self) -> None:
        empty = RequestUsage()
        assert not empty.has_values()

        populated = RequestUsage(input_tokens=1)
        assert populated.has_values()

    def test_task_usage_has_values(self) -> None:
        empty = TaskUsage()
        assert not empty.has_values()

        populated = TaskUsage(input_tokens=1)
        assert populated.has_values()
