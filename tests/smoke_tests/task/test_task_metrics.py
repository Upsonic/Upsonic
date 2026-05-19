"""
Smoke test for task metrics via agent.do().
Requires OPENAI_API_KEY to be set.

Metrics are now read from ``task.usage`` (a :class:`TaskUsage`) — the old
``task.total_cost`` / ``task.total_input_token`` / ``task.total_output_token``
and the ``price_id`` machinery were removed by the usage-registry refactor.
"""
import pytest
from upsonic import Task, Agent
from upsonic.usage import TaskUsage


@pytest.fixture
def agent() -> Agent:
    return Agent(name="MetricsTestAgent", model="openai/gpt-4o-mini")


def _assert_usage_positive(task: Task) -> None:
    assert task.usage is not None
    assert isinstance(task.usage, TaskUsage)
    assert task.usage.input_tokens > 0
    assert task.usage.output_tokens > 0
    assert task.usage.cost is None or task.usage.cost >= 0


class TestTaskMetricsViaDo:

    @pytest.mark.asyncio
    async def test_all_task_metrics_after_do_async(self, agent: Agent) -> None:
        task = Task("What is 2+2? Answer with just the number.")
        await agent.do_async(task)

        assert task.response is not None and isinstance(task.response, str) and task.response
        assert task.start_time is not None
        assert task.end_time is not None
        assert task.usage.duration is not None and task.usage.duration >= 0

        _assert_usage_positive(task)
        assert isinstance(task.tool_calls, list)

    def test_all_task_metrics_after_do(self, agent: Agent) -> None:
        task = Task("What is 3+3? Answer with just the number.")
        agent.do(task)

        assert task.response is not None
        assert task.start_time is not None
        assert task.end_time is not None
        assert task.usage.duration is not None and task.usage.duration >= 0
        _assert_usage_positive(task)

    @pytest.mark.asyncio
    async def test_all_task_metrics_after_print_do_async(self, agent: Agent) -> None:
        task = Task("What is 4+4? Answer with just the number.")
        await agent.print_do_async(task)

        assert task.response is not None
        assert task.usage.duration is not None and task.usage.duration >= 0
        _assert_usage_positive(task)

    def test_all_task_metrics_after_print_do(self, agent: Agent) -> None:
        task = Task("What is 5+5? Answer with just the number.")
        agent.print_do(task)

        assert task.response is not None
        assert task.usage.duration is not None and task.usage.duration >= 0
        _assert_usage_positive(task)

    @pytest.mark.asyncio
    async def test_metrics_independent_across_tasks(self, agent: Agent) -> None:
        """Two separate tasks should each have their own independent metrics."""
        task_a = Task("Say hello.")
        task_b = Task("Say goodbye.")

        await agent.do_async(task_a)
        await agent.do_async(task_b)

        # task_usage_id replaces the old price_id as the per-task scope tag.
        assert task_a.task_usage_id != task_b.task_usage_id
        _assert_usage_positive(task_a)
        _assert_usage_positive(task_b)
        assert task_a.usage is not task_b.usage
