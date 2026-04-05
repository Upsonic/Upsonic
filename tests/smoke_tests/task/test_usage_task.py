"""
Smoke tests for task-level usage tracking via TaskUsage.

Verifies that task._usage (TaskUsage) — the single source of truth for
task-level metrics — properly tracks:
  - duration (total wall-clock time)
  - model_execution_time (time in LLM API calls)
  - upsonic_execution_time (framework overhead)
  - input_tokens / output_tokens (accumulated from model responses)
  - to_dict / from_dict round-trip
  - independence across multiple tasks on the same agent
  - consistency between task.usage and task.duration / task.model_execution_time / task.upsonic_execution_time
  - print_do and print_do_async produce output without errors

Requires OPENAI_API_KEY to be set.
Run with: uv run pytest tests/smoke_tests/task/test_usage_task.py -v -s
"""

import pytest
from typing import Optional

from upsonic import Agent, Task
from upsonic.usage import TaskUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(tools: list = None) -> Agent:
    return Agent(
        model="openai/gpt-4o-mini",
        name="TaskUsageTestAgent",
        tools=tools or [],
    )


def _assert_task_usage_complete(task: Task, label: str) -> None:
    """Assert that task._usage has all three timing fields and tokens populated."""
    usage: Optional[TaskUsage] = task.usage
    assert usage is not None, f"[{label}] task.usage is None"

    assert usage.duration is not None and usage.duration > 0, (
        f"[{label}] duration missing or zero: {usage.duration}"
    )
    assert usage.model_execution_time is not None and usage.model_execution_time > 0, (
        f"[{label}] model_execution_time missing or zero: {usage.model_execution_time}"
    )
    assert usage.upsonic_execution_time is not None and usage.upsonic_execution_time >= 0, (
        f"[{label}] upsonic_execution_time missing or negative: {usage.upsonic_execution_time}"
    )
    assert usage.input_tokens > 0, f"[{label}] input_tokens should be > 0, got {usage.input_tokens}"
    assert usage.output_tokens > 0, f"[{label}] output_tokens should be > 0, got {usage.output_tokens}"

    assert usage.model_execution_time <= usage.duration, (
        f"[{label}] model_execution_time ({usage.model_execution_time}) > duration ({usage.duration})"
    )


# ---------------------------------------------------------------------------
# 1. Basic do — simple text task
# ---------------------------------------------------------------------------

class TestTaskUsageBasic:

    def test_task_usage_after_do(self) -> None:
        agent = _make_agent()
        task = Task("What is 2+2? Answer with just the number.")
        agent.do(task)

        _assert_task_usage_complete(task, "do_sync")

    @pytest.mark.asyncio
    async def test_task_usage_after_do_async(self) -> None:
        agent = _make_agent()
        task = Task("What is 3+3? Answer with just the number.")
        await agent.do_async(task)

        _assert_task_usage_complete(task, "do_async")


# ---------------------------------------------------------------------------
# 2. print_do — ensure timing is printed and populated
# ---------------------------------------------------------------------------

class TestTaskUsagePrintDo:

    def test_task_usage_after_print_do(self, capsys) -> None:
        agent = _make_agent()
        task = Task("What is 5+5? Answer with just the number.")
        agent.print_do(task)

        _assert_task_usage_complete(task, "print_do")

        captured = capsys.readouterr()
        assert "Task Metrics" in captured.out
        assert "Time Taken" in captured.out
        assert "Model Execution Time" in captured.out
        assert "Framework Overhead" in captured.out

    @pytest.mark.asyncio
    async def test_task_usage_after_print_do_async(self, capsys) -> None:
        agent = _make_agent()
        task = Task("What is 7+7? Answer with just the number.")
        await agent.print_do_async(task)

        _assert_task_usage_complete(task, "print_do_async")

        captured = capsys.readouterr()
        assert "Task Metrics" in captured.out


# ---------------------------------------------------------------------------
# 3. Task property delegation — task.duration, task.model_execution_time, etc.
# ---------------------------------------------------------------------------

class TestTaskPropertyDelegation:

    def test_task_properties_delegate_to_usage(self) -> None:
        agent = _make_agent()
        task = Task("Say hello.")
        agent.do(task)

        usage = task.usage
        assert usage is not None
        assert isinstance(usage, TaskUsage)

        assert task.duration == usage.duration
        assert task.model_execution_time == usage.model_execution_time
        assert task.upsonic_execution_time == usage.upsonic_execution_time


# ---------------------------------------------------------------------------
# 4. Tool call task — model_execution_time accumulates across LLM calls
# ---------------------------------------------------------------------------

class TestTaskUsageWithTools:

    def test_task_usage_with_tool_call(self) -> None:
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers together.

            Args:
                a: First number.
                b: Second number.
            """
            return a * b

        agent = _make_agent(tools=[multiply])
        task = Task("Multiply 6 by 7 using the multiply tool.")
        agent.do(task)

        _assert_task_usage_complete(task, "tool_call")
        assert task.usage.input_tokens > 50, "Tool-calling task should have substantial tokens"


# ---------------------------------------------------------------------------
# 5. Multiple tasks on ONE agent — each task has independent usage
# ---------------------------------------------------------------------------

class TestMultipleTasksSameAgent:

    def test_independent_task_usage_across_runs(self) -> None:
        agent = _make_agent()

        task1 = Task("Say hello in French.")
        task2 = Task("Say hello in German.")
        task3 = Task("Say hello in Japanese.")

        agent.do(task1)
        agent.do(task2)
        agent.do(task3)

        for label, t in [("task1", task1), ("task2", task2), ("task3", task3)]:
            _assert_task_usage_complete(t, label)

        assert task1.usage.duration != task2.usage.duration or task1.usage.input_tokens != task2.usage.input_tokens, (
            "Tasks should have independent usage, not identical"
        )

        assert task1.usage is not task2.usage, "Each task should have its own TaskUsage instance"
        assert task2.usage is not task3.usage, "Each task should have its own TaskUsage instance"

    @pytest.mark.asyncio
    async def test_independent_task_usage_print_do_multiple(self, capsys) -> None:
        agent = _make_agent()

        task1 = Task("Count to 3.")
        task2 = Task("Count to 5.")

        agent.print_do(task1)
        agent.print_do(task2)

        _assert_task_usage_complete(task1, "multi_print_task1")
        _assert_task_usage_complete(task2, "multi_print_task2")

        captured = capsys.readouterr()
        assert captured.out.count("Task Metrics") == 2, "Should print Task Metrics for each task"


# ---------------------------------------------------------------------------
# 6. to_dict / from_dict round-trip preserves usage
# ---------------------------------------------------------------------------

class TestTaskUsageSerialization:

    def test_to_dict_from_dict_preserves_usage(self) -> None:
        agent = _make_agent()
        task = Task("What is the capital of France?")
        agent.do(task)

        _assert_task_usage_complete(task, "before_serialization")

        d = task.to_dict()
        assert "_usage" in d
        assert d["_usage"] is not None
        assert "duration" in d["_usage"]
        assert "model_execution_time" in d["_usage"]
        assert "input_tokens" in d["_usage"]
        assert "output_tokens" in d["_usage"]

        restored = Task.from_dict(d)
        assert restored.usage is not None
        assert isinstance(restored.usage, TaskUsage)
        assert restored.usage.duration == task.usage.duration
        assert restored.usage.model_execution_time == task.usage.model_execution_time
        assert restored.usage.input_tokens == task.usage.input_tokens
        assert restored.usage.output_tokens == task.usage.output_tokens
        assert restored.duration == task.duration


# ---------------------------------------------------------------------------
# 7. Structured output — usage tracked correctly
# ---------------------------------------------------------------------------

class TestTaskUsageStructuredOutput:

    def test_structured_output_task_usage(self) -> None:
        from pydantic import BaseModel, Field

        class MathAnswer(BaseModel):
            answer: int = Field(description="The numeric answer")

        agent = _make_agent()
        task = Task("What is 10 * 5?", response_format=MathAnswer)
        agent.do(task)

        _assert_task_usage_complete(task, "structured_output")
        assert isinstance(task.response, MathAnswer)
        assert task.response.answer == 50


# ---------------------------------------------------------------------------
# 8. Timing sanity — model_execution_time < duration always
# ---------------------------------------------------------------------------

class TestTaskTimingSanity:

    def test_model_time_always_less_than_duration(self) -> None:
        agent = _make_agent()
        tasks = [
            Task("Say yes."),
            Task("Say no."),
            Task("Say maybe."),
        ]

        for t in tasks:
            agent.do(t)

        for i, t in enumerate(tasks):
            u = t.usage
            assert u is not None, f"Task {i} has no usage"
            assert u.model_execution_time <= u.duration, (
                f"Task {i}: model_time ({u.model_execution_time:.3f}) > duration ({u.duration:.3f})"
            )
            assert u.upsonic_execution_time >= 0, (
                f"Task {i}: upsonic_time negative ({u.upsonic_execution_time:.3f})"
            )


# ---------------------------------------------------------------------------
# 9. Duration formula — duration == model + tool + pause + overhead
# ---------------------------------------------------------------------------

TOLERANCE = 0.01  # seconds


def _assert_duration_formula(usage: TaskUsage, label: str) -> None:
    """Assert duration == model_execution_time + tool_execution_time + pause_time + upsonic_execution_time."""
    model = usage.model_execution_time or 0.0
    tool = usage.tool_execution_time or 0.0
    pause = usage.pause_time or 0.0
    overhead = usage.upsonic_execution_time or 0.0
    expected = model + tool + pause + overhead
    diff = abs(usage.duration - expected)
    assert diff < TOLERANCE, (
        f"[{label}] duration formula mismatch: "
        f"duration={usage.duration:.4f} != model({model:.4f}) + tool({tool:.4f}) "
        f"+ pause({pause:.4f}) + overhead({overhead:.4f}) = {expected:.4f}  (diff={diff:.4f})"
    )


class TestTaskDurationFormula:

    def test_simple_task_duration_formula(self) -> None:
        """duration == model + tool + pause + overhead for a simple text task."""
        agent = _make_agent()
        task = Task("What is 2+2? Answer with just the number.")
        agent.do(task)
        _assert_duration_formula(task.usage, "simple_formula")

    def test_sync_task_duration_formula(self) -> None:
        """duration formula holds for sync do runs."""
        agent = _make_agent()
        task = Task("What is 5+5? Answer with just the number.")
        agent.do(task)
        _assert_duration_formula(task.usage, "sync_formula")

    def test_tool_task_duration_formula(self) -> None:
        """duration formula holds when tools are used (tool_execution_time > 0)."""
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers together.

            Args:
                a: First number.
                b: Second number.
            """
            return a * b

        agent = _make_agent(tools=[multiply])
        task = Task("Multiply 6 by 7 using the multiply tool.")
        agent.do(task)

        usage = task.usage
        _assert_duration_formula(usage, "tool_formula")
        assert usage.tool_execution_time is not None and usage.tool_execution_time >= 0, (
            f"tool_execution_time should be set, got {usage.tool_execution_time}"
        )

    def test_multiple_tasks_each_satisfies_formula(self) -> None:
        """Each task's duration formula holds independently when running multiple tasks."""
        agent = _make_agent()
        tasks = [
            Task("Say hello in French."),
            Task("Say hello in German."),
            Task("Say hello in Japanese."),
        ]
        for t in tasks:
            agent.do(t)
        for i, t in enumerate(tasks):
            _assert_duration_formula(t.usage, f"multi_task_{i}")


# ---------------------------------------------------------------------------
# 10. Agent accumulation — agent.usage == sum(task.usage) for all fields
# ---------------------------------------------------------------------------

class TestAgentAccumulatesTaskMetrics:

    def test_agent_duration_equals_sum_of_task_durations(self) -> None:
        """Agent duration == sum of individual task durations."""
        agent = _make_agent()
        t1 = Task("Say yes.")
        t2 = Task("Say no.")
        t3 = Task("Say maybe.")

        agent.do(t1)
        agent.do(t2)
        agent.do(t3)

        sum_dur = t1.usage.duration + t2.usage.duration + t3.usage.duration
        diff = abs(agent.usage.duration - sum_dur)
        assert diff < TOLERANCE, (
            f"Agent duration ({agent.usage.duration:.4f}) != "
            f"sum of task durations ({sum_dur:.4f}), diff={diff:.4f}"
        )

    def test_agent_model_time_equals_sum_of_task_model_times(self) -> None:
        """Agent model_execution_time == sum of individual task model_execution_times."""
        agent = _make_agent()
        t1 = Task("Say red.")
        t2 = Task("Say blue.")

        agent.do(t1)
        agent.do(t2)

        sum_model = t1.usage.model_execution_time + t2.usage.model_execution_time
        diff = abs(agent.usage.model_execution_time - sum_model)
        assert diff < TOLERANCE, (
            f"Agent model_execution_time ({agent.usage.model_execution_time:.4f}) != "
            f"sum ({sum_model:.4f}), diff={diff:.4f}"
        )

    def test_agent_tokens_equal_sum_of_task_tokens(self) -> None:
        """Agent tokens == exact sum of individual task tokens."""
        agent = _make_agent()
        t1 = Task("Say one.")
        t2 = Task("Say two.")
        t3 = Task("Say three.")

        agent.do(t1)
        agent.do(t2)
        agent.do(t3)

        sum_in = t1.usage.input_tokens + t2.usage.input_tokens + t3.usage.input_tokens
        sum_out = t1.usage.output_tokens + t2.usage.output_tokens + t3.usage.output_tokens

        assert agent.usage.input_tokens == sum_in, (
            f"Agent input_tokens ({agent.usage.input_tokens}) != sum ({sum_in})"
        )
        assert agent.usage.output_tokens == sum_out, (
            f"Agent output_tokens ({agent.usage.output_tokens}) != sum ({sum_out})"
        )

    def test_agent_requests_equal_sum_of_task_requests(self) -> None:
        """Agent requests == exact sum of individual task requests."""
        agent = _make_agent()
        t1 = Task("Say A.")
        t2 = Task("Say B.")

        o1 = agent.do(t1, return_output=True)
        o2 = agent.do(t2, return_output=True)

        sum_reqs = o1.usage.requests + o2.usage.requests
        assert agent.usage.requests == sum_reqs, (
            f"Agent requests ({agent.usage.requests}) != sum ({sum_reqs})"
        )

    def test_agent_tool_calls_equal_sum_of_task_tool_calls(self) -> None:
        """Agent tool_calls == sum of individual task tool_calls."""
        def add(a: int, b: int) -> int:
            """Add two numbers.

            Args:
                a: First number.
                b: Second number.
            """
            return a + b

        agent = _make_agent(tools=[add])
        t1 = Task("Use the add tool to add 1 and 2.")
        t2 = Task("Use the add tool to add 3 and 4.")

        o1 = agent.do(t1, return_output=True)
        o2 = agent.do(t2, return_output=True)

        sum_calls = o1.usage.tool_calls + o2.usage.tool_calls
        assert agent.usage.tool_calls == sum_calls, (
            f"Agent tool_calls ({agent.usage.tool_calls}) != sum ({sum_calls})"
        )

    def test_agent_tool_time_equals_sum_of_task_tool_times(self) -> None:
        """Agent tool_execution_time == sum of individual task tool_execution_times."""
        def slow_op(x: int) -> int:
            """Do a slow operation.

            Args:
                x: Input number.
            """
            import time
            time.sleep(0.3)
            return x * 2

        agent = _make_agent(tools=[slow_op])
        t1 = Task("Use the slow_op tool with input 5.")
        t2 = Task("Use the slow_op tool with input 10.")

        agent.do(t1)
        agent.do(t2)

        sum_tool = (t1.usage.tool_execution_time or 0.0) + (t2.usage.tool_execution_time or 0.0)
        agent_tool = agent.usage.tool_execution_time or 0.0
        diff = abs(agent_tool - sum_tool)
        assert diff < TOLERANCE, (
            f"Agent tool_execution_time ({agent_tool:.4f}) != "
            f"sum ({sum_tool:.4f}), diff={diff:.4f}"
        )
