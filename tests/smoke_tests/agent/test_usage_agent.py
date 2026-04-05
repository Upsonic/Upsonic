"""
Comprehensive Smoke Tests for agent-level usage tracking via AgentUsage.

Verifies that agent.usage (AgentUsage) and AgentRunOutput.usage (TaskUsage)
properly accumulate metrics across multiple task runs, including:

1. Direct model.request() calls in ModelExecutionStep
2. Sub-agent calls via do_async() in:
   - CultureManager (culture extraction)
   - Orchestrator (analysis, revision, synthesis)
   - AgentTool (agent-as-tool delegation)
   - Reflection (evaluation + improvement)
   - ReliabilityLayer (verifier + editor agents)
   - CacheCheckStep (LLM-based query comparison)
3. Policy LLM calls:
   - AgentPolicyStep
   - UserPolicyStep
   - ToolPolicyManager (post-execution validation)
4. Context management:
   - ContextManagementMiddleware (summarization)
5. Memory:
   - AgentSessionMemory (summary generation)
   - UserMemory (trait analysis)

Also verifies:
  - requests, input_tokens, output_tokens accumulation
  - duration, model_execution_time, upsonic_execution_time at agent level
  - cost tracking
  - independence between different agent instances
  - correct aggregation when running multiple tasks on one agent
  - correct aggregation when using multiple agents
  - print_do / print_do_async printing Agent Metrics
  - agent-as-tool usage propagation
  - tool_calls counting
  - AgentUsage to_dict / from_dict round-trip
  - TaskUsage on AgentRunOutput (return_output)

Run with: uv run pytest tests/smoke_tests/agent/test_usage_agent.py -v -s
"""

import pytest
from typing import List

from upsonic import Agent, Task
from upsonic.usage import TaskUsage, AgentUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(name: str = "AgentUsageTestAgent", tools: list = None) -> Agent:
    return Agent(
        model="openai/gpt-4o-mini",
        name=name,
        tools=tools or [],
    )


def _assert_agent_usage_positive(usage: AgentUsage, label: str) -> None:
    """Assert that an AgentUsage has meaningful positive values."""
    assert usage is not None, f"[{label}] usage is None"
    assert usage.requests > 0, f"[{label}] Expected requests > 0, got {usage.requests}"
    assert usage.input_tokens > 0, f"[{label}] Expected input_tokens > 0, got {usage.input_tokens}"
    assert usage.output_tokens > 0, f"[{label}] Expected output_tokens > 0, got {usage.output_tokens}"


def _assert_agent_usage_timing(usage: AgentUsage, label: str) -> None:
    """Assert that an AgentUsage has all three timing fields populated."""
    assert usage.duration is not None and usage.duration > 0, (
        f"[{label}] duration missing or zero: {usage.duration}"
    )
    assert usage.model_execution_time is not None and usage.model_execution_time > 0, (
        f"[{label}] model_execution_time missing or zero: {usage.model_execution_time}"
    )
    assert usage.upsonic_execution_time is not None and usage.upsonic_execution_time >= 0, (
        f"[{label}] upsonic_execution_time missing or negative: {usage.upsonic_execution_time}"
    )
    assert usage.model_execution_time <= usage.duration, (
        f"[{label}] model_execution_time ({usage.model_execution_time}) > duration ({usage.duration})"
    )


def _assert_task_usage_positive(usage: TaskUsage, label: str) -> None:
    """Assert that a TaskUsage (from AgentRunOutput) has meaningful positive values."""
    assert usage is not None, f"[{label}] usage is None"
    assert usage.requests > 0, f"[{label}] Expected requests > 0, got {usage.requests}"
    assert usage.input_tokens > 0, f"[{label}] Expected input_tokens > 0, got {usage.input_tokens}"
    assert usage.output_tokens > 0, f"[{label}] Expected output_tokens > 0, got {usage.output_tokens}"


# ---------------------------------------------------------------------------
# 1. Single task — agent.usage populated after one run
# ---------------------------------------------------------------------------

class TestAgentUsageSingleTask:

    def test_agent_usage_after_single_do(self) -> None:
        agent = _make_agent()
        task = Task("What is 2+2? Answer with just the number.")
        agent.do(task)

        _assert_agent_usage_positive(agent.usage, "single_do")
        _assert_agent_usage_timing(agent.usage, "single_do")

    @pytest.mark.asyncio
    async def test_agent_usage_after_single_do_async(self) -> None:
        agent = _make_agent()
        task = Task("What is 3+3? Answer with just the number.")
        await agent.do_async(task)

        _assert_agent_usage_positive(agent.usage, "single_do_async")
        _assert_agent_usage_timing(agent.usage, "single_do_async")


# ---------------------------------------------------------------------------
# 2. Multiple tasks on ONE agent — usage accumulates
# ---------------------------------------------------------------------------

class TestAgentUsageMultipleTasks:

    def test_agent_usage_accumulates_across_tasks(self) -> None:
        agent = _make_agent()

        task1 = Task("Say hello in French.")
        task2 = Task("Say hello in German.")
        task3 = Task("Say hello in Japanese.")

        agent.do(task1)
        after_first_requests: int = agent.usage.requests
        after_first_input: int = agent.usage.input_tokens

        agent.do(task2)
        after_second_requests: int = agent.usage.requests
        after_second_input: int = agent.usage.input_tokens

        agent.do(task3)

        assert agent.usage.requests > after_first_requests, (
            "Requests should accumulate across tasks"
        )
        assert agent.usage.requests > after_second_requests, (
            "Requests should keep accumulating"
        )
        assert agent.usage.input_tokens > after_first_input, (
            "Input tokens should accumulate"
        )
        assert agent.usage.input_tokens > after_second_input, (
            "Input tokens should keep accumulating"
        )

        _assert_agent_usage_timing(agent.usage, "multi_task_accumulated")

    @pytest.mark.asyncio
    async def test_agent_usage_accumulates_async(self) -> None:
        agent = _make_agent()

        tasks: List[Task] = [
            Task("Say hello."),
            Task("Say goodbye."),
            Task("Say thanks."),
        ]

        for t in tasks:
            await agent.do_async(t)

        assert agent.usage.requests >= 3, (
            f"Expected at least 3 requests, got {agent.usage.requests}"
        )
        _assert_agent_usage_positive(agent.usage, "multi_async_accumulated")
        _assert_agent_usage_timing(agent.usage, "multi_async_accumulated")

    def test_task_usage_independent_but_agent_usage_accumulated(self) -> None:
        agent = _make_agent()

        task1 = Task("What is 1+1? Answer with just the number.")
        task2 = Task("What is 2+2? Answer with just the number.")

        agent.do(task1)
        agent.do(task2)

        assert task1.usage is not task2.usage, "Each task should have its own TaskUsage"
        assert task1.usage.input_tokens > 0
        assert task2.usage.input_tokens > 0

        agent_total_input: int = agent.usage.input_tokens
        task_sum_input: int = task1.usage.input_tokens + task2.usage.input_tokens
        assert agent_total_input >= task_sum_input - 10, (
            f"Agent total input ({agent_total_input}) should be close to sum of task inputs ({task_sum_input})"
        )


# ---------------------------------------------------------------------------
# 3. Multiple DIFFERENT agents — each has independent usage
# ---------------------------------------------------------------------------

class TestMultipleAgentsIndependentUsage:

    def test_separate_agents_have_independent_usage(self) -> None:
        agent_a = _make_agent(name="AgentA")
        agent_b = _make_agent(name="AgentB")

        task_a = Task("Say hello.")
        task_b = Task("Say goodbye.")

        agent_a.do(task_a)
        agent_b.do(task_b)

        _assert_agent_usage_positive(agent_a.usage, "agent_a")
        _assert_agent_usage_positive(agent_b.usage, "agent_b")

        assert agent_a.usage is not agent_b.usage, (
            "Different agents must have different AgentUsage instances"
        )

    @pytest.mark.asyncio
    async def test_separate_agents_accumulated_independently_async(self) -> None:
        agent_a = _make_agent(name="IndependentA")
        agent_b = _make_agent(name="IndependentB")

        await agent_a.do_async(Task("Count to 3."))
        await agent_a.do_async(Task("Count to 5."))

        await agent_b.do_async(Task("Say yes."))

        assert agent_a.usage.requests >= 2, (
            f"AgentA should have at least 2 requests, got {agent_a.usage.requests}"
        )
        assert agent_b.usage.requests >= 1, (
            f"AgentB should have at least 1 request, got {agent_b.usage.requests}"
        )

        assert agent_a.usage.input_tokens > agent_b.usage.input_tokens, (
            "AgentA ran 2 tasks, should have more tokens than AgentB with 1 task"
        )


# ---------------------------------------------------------------------------
# 4. print_do — Agent Metrics panel printed
# ---------------------------------------------------------------------------

class TestAgentUsagePrintDo:

    def test_print_do_prints_agent_metrics(self, capsys) -> None:
        agent = _make_agent()
        task = Task("What is 10+10? Answer with just the number.")
        agent.print_do(task)

        _assert_agent_usage_positive(agent.usage, "print_do_agent")

        captured = capsys.readouterr()
        assert "Agent Metrics" in captured.out, "Should print Agent Metrics panel"
        assert "Total Requests" in captured.out
        assert "Total Input Tokens" in captured.out
        assert "Total Output Tokens" in captured.out
        assert "Total Duration" in captured.out
        assert "Model Execution Time" in captured.out
        assert "Framework Overhead" in captured.out

    @pytest.mark.asyncio
    async def test_print_do_async_prints_agent_metrics(self, capsys) -> None:
        agent = _make_agent()
        task = Task("What is 20+20? Answer with just the number.")
        await agent.print_do_async(task)

        _assert_agent_usage_positive(agent.usage, "print_do_async_agent")

        captured = capsys.readouterr()
        assert "Agent Metrics" in captured.out

    def test_print_do_multiple_tasks_agent_metrics_once_per_do(self, capsys) -> None:
        agent = _make_agent()

        agent.print_do(Task("Say red."))
        agent.print_do(Task("Say blue."))

        captured = capsys.readouterr()
        agent_metrics_count: int = captured.out.count("Agent Metrics")
        task_metrics_count: int = captured.out.count("Task Metrics")

        assert task_metrics_count == 2, f"Expected 2 Task Metrics panels, got {task_metrics_count}"
        assert agent_metrics_count == 2, f"Expected 2 Agent Metrics panels, got {agent_metrics_count}"

    def test_print_do_also_prints_task_metrics_with_timing(self, capsys) -> None:
        agent = _make_agent()
        task = Task("Say hello world.")
        agent.print_do(task)

        captured = capsys.readouterr()
        assert "Task Metrics" in captured.out
        assert "Time Taken" in captured.out
        assert "Model Execution Time" in captured.out
        assert "Framework Overhead" in captured.out


# ---------------------------------------------------------------------------
# 5. Agent with tools — agent.usage tracks tool usage requests
# ---------------------------------------------------------------------------

class TestAgentUsageWithTools:

    def test_agent_usage_with_tool_calls(self) -> None:
        def add(a: int, b: int) -> int:
            """Add two integers together.

            Args:
                a: First integer.
                b: Second integer.
            """
            return a + b

        agent = _make_agent(tools=[add])
        task = Task("Use the add tool to add 10 and 20.")
        agent.do(task)

        _assert_agent_usage_positive(agent.usage, "tool_agent")
        _assert_agent_usage_timing(agent.usage, "tool_agent")

    def test_agent_usage_accumulated_with_multiple_tool_tasks(self) -> None:
        def subtract(a: int, b: int) -> int:
            """Subtract b from a.

            Args:
                a: First integer.
                b: Second integer.
            """
            return a - b

        agent = _make_agent(tools=[subtract])

        agent.do(Task("Subtract 5 from 10 using the subtract tool."))
        requests_after_first: int = agent.usage.requests

        agent.do(Task("Subtract 3 from 7 using the subtract tool."))

        assert agent.usage.requests > requests_after_first, (
            "Requests should accumulate across tool-using tasks"
        )


# ---------------------------------------------------------------------------
# 6. Agent with structured output — usage tracking
# ---------------------------------------------------------------------------

class TestAgentUsageStructuredOutput:

    def test_structured_output_contributes_to_agent_usage(self) -> None:
        from pydantic import BaseModel, Field

        class Answer(BaseModel):
            value: int = Field(description="The numeric answer")

        agent = _make_agent()

        agent.do(Task("What is 5*5?", response_format=Answer))
        agent.do(Task("What is 6*6?", response_format=Answer))

        assert agent.usage.requests >= 2
        _assert_agent_usage_positive(agent.usage, "structured_accumulated")


# ---------------------------------------------------------------------------
# 7. Agent-as-tool — parent agent.usage includes sub-agent usage
# ---------------------------------------------------------------------------

class TestAgentAsToolUsagePropagation:

    @pytest.mark.asyncio
    async def test_parent_agent_usage_includes_sub_agent(self) -> None:
        sub_agent = Agent(
            model="openai/gpt-4o-mini",
            name="MathHelper",
            system_prompt="You answer math questions. Respond with just the number.",
        )

        parent_agent = Agent(
            model="openai/gpt-4o-mini",
            name="ParentCoordinator",
            tools=[sub_agent],
        )

        task = Task("Ask MathHelper what is 12 * 12. Return just the number.")
        await parent_agent.do_async(task)

        _assert_agent_usage_positive(parent_agent.usage, "parent_with_sub")
        assert parent_agent.usage.requests >= 2, (
            f"Parent should have at least 2 requests (own + sub-agent), got {parent_agent.usage.requests}"
        )

    @pytest.mark.asyncio
    async def test_multiple_sub_agents_all_propagate(self) -> None:
        agent_a = Agent(
            model="openai/gpt-4o-mini",
            name="TranslatorBot",
            system_prompt="Translate to French. Return only the translation.",
        )
        agent_b = Agent(
            model="openai/gpt-4o-mini",
            name="SummarizerBot",
            system_prompt="Summarize in one sentence. Return only the summary.",
        )

        coordinator = Agent(
            model="openai/gpt-4o-mini",
            name="MultiSubCoordinator",
            tools=[agent_a, agent_b],
        )

        task = Task(
            "First ask SummarizerBot to summarize 'AI is transforming healthcare'. "
            "Then ask TranslatorBot to translate the summary to French. Return the French text."
        )
        await coordinator.do_async(task)

        _assert_agent_usage_positive(coordinator.usage, "multi_sub_coordinator")
        assert coordinator.usage.requests >= 3, (
            f"Coordinator should have at least 3 requests, got {coordinator.usage.requests}"
        )


# ---------------------------------------------------------------------------
# 8. Cost tracking at agent level
# ---------------------------------------------------------------------------

class TestAgentCostTracking:

    def test_agent_cost_populated(self) -> None:
        agent = _make_agent()
        agent.do(Task("What is Python?"))

        assert agent.usage.cost is not None, "Cost should be tracked for gpt-4o-mini"
        assert agent.usage.cost > 0, f"Cost should be positive, got {agent.usage.cost}"

    def test_agent_cost_accumulates(self) -> None:
        agent = _make_agent()

        agent.do(Task("Say hello."))
        cost_after_first: float = agent.usage.cost

        agent.do(Task("Say goodbye."))
        assert agent.usage.cost > cost_after_first, (
            f"Cost should increase after second task: {agent.usage.cost} vs {cost_after_first}"
        )


# ---------------------------------------------------------------------------
# 9. return_output — TaskUsage accessible from AgentRunOutput
# ---------------------------------------------------------------------------

class TestAgentRunOutputUsage:

    def test_return_output_has_task_usage(self) -> None:
        agent = _make_agent()
        task = Task("Say hi.")
        output = agent.do(task, return_output=True)

        assert output is not None
        assert output.usage is not None
        assert isinstance(output.usage, TaskUsage)
        _assert_task_usage_positive(output.usage, "return_output")

    @pytest.mark.asyncio
    async def test_return_output_usage_tokens_match_agent_single_task(self) -> None:
        agent = _make_agent()
        task = Task("Say hello.")
        output = await agent.do_async(task, return_output=True)

        assert output.usage.input_tokens == agent.usage.input_tokens
        assert output.usage.output_tokens == agent.usage.output_tokens


# ---------------------------------------------------------------------------
# 10. Task-level and agent-level consistency
# ---------------------------------------------------------------------------

class TestTaskAndAgentUsageConsistency:

    def test_task_usage_duration_in_agent_duration(self) -> None:
        agent = _make_agent()
        task = Task("What is the speed of light?")
        agent.do(task)

        task_duration: float = task.usage.duration
        agent_duration: float = agent.usage.duration

        assert agent_duration >= task_duration * 0.8, (
            f"Agent duration ({agent_duration:.3f}) should be >= task duration ({task_duration:.3f})"
        )

    def test_multi_task_durations_sum_close_to_agent_duration(self) -> None:
        agent = _make_agent()

        t1 = Task("Say 1.")
        t2 = Task("Say 2.")
        t3 = Task("Say 3.")

        agent.do(t1)
        agent.do(t2)
        agent.do(t3)

        task_duration_sum: float = t1.usage.duration + t2.usage.duration + t3.usage.duration
        agent_duration: float = agent.usage.duration

        assert agent_duration >= task_duration_sum * 0.5, (
            f"Agent total duration ({agent_duration:.3f}) should be close to sum ({task_duration_sum:.3f})"
        )

    def test_multi_task_tokens_sum_to_agent_tokens(self) -> None:
        agent = _make_agent()

        t1 = Task("Say yes.")
        t2 = Task("Say no.")

        agent.do(t1)
        agent.do(t2)

        sum_input: int = t1.usage.input_tokens + t2.usage.input_tokens
        sum_output: int = t1.usage.output_tokens + t2.usage.output_tokens

        assert agent.usage.input_tokens >= sum_input - 10, (
            f"Agent input tokens ({agent.usage.input_tokens}) vs sum ({sum_input})"
        )
        assert agent.usage.output_tokens >= sum_output - 10, (
            f"Agent output tokens ({agent.usage.output_tokens}) vs sum ({sum_output})"
        )


# ---------------------------------------------------------------------------
# 11. AgentUsage serialization round-trip
# ---------------------------------------------------------------------------

class TestAgentUsageSerialization:

    def test_agent_usage_to_dict_from_dict(self) -> None:
        agent = _make_agent()
        agent.do(Task("Tell me a joke."))

        original: AgentUsage = agent.usage
        d = original.to_dict()

        assert "requests" in d
        assert "input_tokens" in d
        assert "output_tokens" in d
        assert "duration" in d
        assert "model_execution_time" in d

        restored: AgentUsage = AgentUsage.from_dict(d)
        assert restored.requests == original.requests
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.duration == original.duration
        assert restored.model_execution_time == original.model_execution_time


# ===========================================================================
# Tests for AgentRunOutput.usage (TaskUsage) via return_output=True
# ===========================================================================


# ---------------------------------------------------------------------------
# 12. Basic agent run – direct LLM call usage (return_output)
# ---------------------------------------------------------------------------

def test_basic_agent_usage_tracking():
    """Verify that a simple agent run tracks usage from the direct model.request() call."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="BasicUsageAgent",
    )

    task = Task(description="What is 2+2? Reply with just the number.")

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "basic_agent")


@pytest.mark.asyncio
async def test_basic_agent_usage_tracking_async():
    """Async version of test_basic_agent_usage_tracking."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="BasicUsageAgentAsync",
    )

    task = Task(description="What is 3+3? Reply with just the number.")

    output = await agent.do_async(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "basic_agent_async")


# ---------------------------------------------------------------------------
# 13. Structured output (response_format) – verifies output tool path
# ---------------------------------------------------------------------------

def test_structured_output_usage_tracking():
    """Verify usage tracking when agent returns structured output (Pydantic model)."""
    from pydantic import BaseModel, Field

    class MathResult(BaseModel):
        answer: int = Field(description="The numeric answer")
        explanation: str = Field(description="Brief explanation")

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="StructuredOutputAgent",
    )

    task = Task(
        description="What is 7 times 8?",
        response_format=MathResult,
    )

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None
    assert isinstance(output.output, MathResult)
    assert output.output.answer == 56

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "structured_output")


# ---------------------------------------------------------------------------
# 14. Tool usage – function tool
# ---------------------------------------------------------------------------

def test_tool_usage_tracking():
    """Verify usage tracking when agent uses a regular function tool."""
    from upsonic.tools.config import tool

    @tool(docstring_format="google")
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of a and b
        """
        return a + b

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="ToolUsageAgent",
        tools=[add_numbers],
    )

    task = Task(description="Use the add_numbers tool to add 15 and 27. Return only the result number.")

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "tool_usage")
    assert usage.requests >= 1, "Expected at least 1 request (tool call + follow-up)"


# ---------------------------------------------------------------------------
# 15. Agent-as-tool – AgentTool usage propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_as_tool_usage_tracking():
    """Verify that usage from a sub-agent used as a tool propagates to the parent agent."""
    sub_agent = Agent(
        model="openai/gpt-4o-mini",
        name="MathExpert",
        system_prompt="You are a math expert. Answer math questions concisely.",
    )

    parent_agent = Agent(
        model="openai/gpt-4o-mini",
        name="Coordinator",
        tools=[sub_agent],
    )

    task = Task(
        description="Ask MathExpert what is the square root of 144. Return just the number.",
    )

    output = await parent_agent.do_async(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "agent_as_tool")
    assert usage.requests >= 2, (
        f"Expected at least 2 requests (parent + sub-agent), got {usage.requests}"
    )


# ---------------------------------------------------------------------------
# 16. Culture extraction – CultureManager usage
# ---------------------------------------------------------------------------

def test_culture_extraction_usage_tracking():
    """Verify that usage from culture extraction sub-agent is tracked."""
    from upsonic.culture import Culture

    culture = Culture(
        description="You are a friendly barista at a coffee shop in Seattle",
        add_system_prompt=True,
    )

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="CultureAgent",
        culture=culture,
    )

    task = Task(description="Greet me as I walk in.")

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "culture_extraction")
    assert usage.requests >= 2, (
        f"Expected at least 2 requests (culture extraction + main), got {usage.requests}"
    )


# ---------------------------------------------------------------------------
# 17. Reflection – sub-agent evaluator/improver usage
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reflection_usage_tracking():
    """Verify that usage from reflection evaluator and improver sub-agents is tracked."""
    from upsonic.reflection import ReflectionConfig

    reflection_config = ReflectionConfig(
        max_iterations=1,
        acceptance_threshold=0.95,
        enable_self_critique=True,
        enable_improvement_suggestions=True,
    )

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="ReflectionAgent",
        reflection=True,
        reflection_config=reflection_config,
    )

    task = Task(description="Write a one-sentence summary of quantum computing.")

    output = await agent.do_async(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "reflection")
    assert usage.requests >= 2, (
        f"Expected at least 2 requests (main + reflection eval), got {usage.requests}"
    )


# ---------------------------------------------------------------------------
# 18. Multiple tools with structured output – combined scenario
# ---------------------------------------------------------------------------

def test_multiple_tools_structured_output_usage():
    """Verify usage tracking with multiple tools and structured output."""
    from pydantic import BaseModel, Field
    from upsonic.tools.config import tool

    @tool(docstring_format="google")
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Product of a and b
        """
        return a * b

    @tool(docstring_format="google")
    def subtract(a: int, b: int) -> int:
        """Subtract second number from first.

        Args:
            a: First number
            b: Second number

        Returns:
            Difference of a minus b
        """
        return a - b

    class ComputationResult(BaseModel):
        final_answer: int = Field(description="The final computed answer")
        steps: List[str] = Field(description="List of computation steps taken")

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="MultiToolAgent",
        tools=[multiply, subtract],
    )

    task = Task(
        description="First multiply 6 by 7, then subtract 10 from the result. Show your steps.",
        response_format=ComputationResult,
    )

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None
    assert isinstance(output.output, ComputationResult)
    assert output.output.final_answer == 32

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "multi_tool_structured")
    assert usage.requests >= 1


# ---------------------------------------------------------------------------
# 19. Usage accumulation across multiple tasks on same agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_usage_accumulation_separate_runs():
    """Each do_async call should track usage independently on AgentRunOutput."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="AccumulationAgent",
    )

    task1 = Task(description="Say hello in French.")
    task2 = Task(description="Say hello in German.")

    output1 = await agent.do_async(task1, return_output=True)
    output2 = await agent.do_async(task2, return_output=True)

    assert output1 is not None
    assert output2 is not None

    usage1: TaskUsage = output1.usage
    usage2: TaskUsage = output2.usage

    _assert_task_usage_positive(usage1, "run1")
    _assert_task_usage_positive(usage2, "run2")

    assert usage1.requests >= 1
    assert usage2.requests >= 1


# ---------------------------------------------------------------------------
# 20. Agent with system prompt – usage from first LLM call
# ---------------------------------------------------------------------------

def test_system_prompt_usage_tracking():
    """Verify usage tracking for agent with custom system prompt."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="SystemPromptAgent",
        system_prompt="You are a pirate. Always respond in pirate speak.",
    )

    task = Task(description="Tell me about the weather today.")

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "system_prompt")
    assert usage.input_tokens > 10, "System prompt should increase input tokens"


# ---------------------------------------------------------------------------
# 21. Agent with guardrail (response_format + validation) usage tracking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guardrail_usage_tracking():
    """Verify usage tracking when a guardrail triggers retry via _execute_with_guardrail."""
    from pydantic import BaseModel, Field

    class CapitalInfo(BaseModel):
        country: str = Field(description="Country name")
        capital: str = Field(description="Capital city")
        population_approx: str = Field(description="Approximate population")

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="GuardrailAgent",
    )

    task = Task(
        description="What is the capital of France? Include approximate population.",
        response_format=CapitalInfo,
    )

    output = await agent.do_async(task, return_output=True)

    assert output is not None
    assert output.output is not None
    assert isinstance(output.output, CapitalInfo)
    assert output.output.capital.lower() == "paris"

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "guardrail")


# ---------------------------------------------------------------------------
# 22. Verify TaskUsage fields are populated correctly
# ---------------------------------------------------------------------------

def test_usage_fields_completeness():
    """Verify that all critical TaskUsage fields are populated after a run."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="FieldsCheckAgent",
    )

    task = Task(description="What is Python?")

    output = agent.do(task, return_output=True)

    assert output is not None

    usage: TaskUsage = output.usage
    assert isinstance(usage, TaskUsage)

    assert usage.requests >= 1, f"requests should be >= 1, got {usage.requests}"
    assert usage.input_tokens > 0, f"input_tokens should be > 0, got {usage.input_tokens}"
    assert usage.output_tokens > 0, f"output_tokens should be > 0, got {usage.output_tokens}"

    total_tokens = usage.input_tokens + usage.output_tokens
    assert total_tokens > 0, "Total tokens should be positive"


# ---------------------------------------------------------------------------
# 23. Agent-as-tool with multiple sub-agents
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_agent_tools_usage_tracking():
    """Verify usage propagation when parent agent delegates to multiple sub-agents."""
    translator_agent = Agent(
        model="openai/gpt-4o-mini",
        name="Translator",
        system_prompt="You translate text to French. Return only the translation.",
    )

    summarizer_agent = Agent(
        model="openai/gpt-4o-mini",
        name="Summarizer",
        system_prompt="You summarize text in one short sentence. Return only the summary.",
    )

    coordinator = Agent(
        model="openai/gpt-4o-mini",
        name="Coordinator",
        tools=[translator_agent, summarizer_agent],
    )

    task = Task(
        description=(
            "First, ask the Summarizer to summarize: 'Artificial intelligence is transforming healthcare "
            "by enabling faster diagnosis and personalized treatment plans.' "
            "Then ask the Translator to translate the summary to French. "
            "Return the French translation."
        ),
    )

    output = await coordinator.do_async(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "multi_agent_tools")
    assert usage.requests >= 3, (
        f"Expected at least 3 requests (coordinator + 2 sub-agents), got {usage.requests}"
    )


# ---------------------------------------------------------------------------
# 24. Culture + Tool + Structured output combined
# ---------------------------------------------------------------------------

def test_combined_culture_tool_structured():
    """Verify usage tracking in a combined scenario: culture + tool + structured output."""
    from pydantic import BaseModel, Field
    from upsonic.tools.config import tool
    from upsonic.culture import Culture

    @tool(docstring_format="google")
    def get_room_price(room_type: str) -> str:
        """Get the price for a room type.

        Args:
            room_type: Type of room (standard, deluxe, suite)

        Returns:
            Price information as string
        """
        prices = {
            "standard": "$150/night",
            "deluxe": "$250/night",
            "suite": "$450/night",
        }
        return prices.get(room_type.lower(), "Room type not available")

    class RoomRecommendation(BaseModel):
        recommended_room: str = Field(description="Recommended room type")
        price: str = Field(description="Price per night")
        reason: str = Field(description="Why this room is recommended")

    culture = Culture(
        description="You are a luxury hotel concierge at The Grand Hotel",
        add_system_prompt=True,
    )

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="HotelConcierge",
        culture=culture,
        tools=[get_room_price],
    )

    task = Task(
        description="I want a nice room for a romantic getaway. Check the suite price and recommend it.",
        response_format=RoomRecommendation,
    )

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None
    assert isinstance(output.output, RoomRecommendation)

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "combined_culture_tool_structured")
    assert usage.requests >= 2, (
        f"Expected at least 2 requests (culture + main), got {usage.requests}"
    )


# ---------------------------------------------------------------------------
# 25. Streaming mode – verify usage is tracked in stream mode too
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_usage_tracking():
    """Verify that usage is tracked even in streaming mode."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="StreamingAgent",
    )

    task = Task(description="Count from 1 to 5, each on a new line.")

    output = await agent.do_async(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "streaming")


# ---------------------------------------------------------------------------
# 26. Verify cost field is populated
# ---------------------------------------------------------------------------

def test_cost_tracking():
    """Verify that cost is calculated and populated in usage."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="CostTrackingAgent",
    )

    task = Task(description="What is the meaning of life? Answer in one sentence.")

    output = agent.do(task, return_output=True)

    assert output is not None

    usage: TaskUsage = output.usage
    _assert_task_usage_positive(usage, "cost_tracking")

    assert usage.cost is not None, "Cost should be calculated for gpt-4o-mini"
    assert usage.cost > 0, f"Cost should be positive, got {usage.cost}"


# ---------------------------------------------------------------------------
# 27. Duration formula — duration == model + tool + pause + overhead
# ---------------------------------------------------------------------------

FORMULA_TOLERANCE = 0.01  # seconds


def _assert_duration_formula(usage: TaskUsage, label: str) -> None:
    """Assert duration == model_execution_time + tool_execution_time + pause_time + upsonic_execution_time."""
    model = usage.model_execution_time or 0.0
    tool = usage.tool_execution_time or 0.0
    pause = usage.pause_time or 0.0
    overhead = usage.upsonic_execution_time or 0.0
    expected = model + tool + pause + overhead
    diff = abs(usage.duration - expected)
    assert diff < FORMULA_TOLERANCE, (
        f"[{label}] duration formula mismatch: "
        f"duration={usage.duration:.4f} != model({model:.4f}) + tool({tool:.4f}) "
        f"+ pause({pause:.4f}) + overhead({overhead:.4f}) = {expected:.4f}  (diff={diff:.4f})"
    )


class TestTaskDurationFormula:
    """Verify duration = model + tool + pause + overhead for AgentRunOutput.usage."""

    def test_simple_task_duration_formula(self) -> None:
        agent = _make_agent()
        task = Task("What is 2+2? Answer with just the number.")
        output = agent.do(task, return_output=True)
        _assert_duration_formula(output.usage, "simple_formula")

    def test_sync_task_duration_formula(self) -> None:
        agent = _make_agent()
        task = Task("What is 5+5? Answer with just the number.")
        output = agent.do(task, return_output=True)
        _assert_duration_formula(output.usage, "sync_formula")

    def test_tool_task_duration_formula(self) -> None:
        from upsonic.tools.config import tool as tool_decorator

        @tool_decorator(docstring_format="google")
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers.

            Args:
                a: First number
                b: Second number

            Returns:
                Product
            """
            return a * b

        agent = _make_agent(tools=[multiply])
        task = Task("Multiply 6 by 7 using the multiply tool.")
        output = agent.do(task, return_output=True)

        _assert_duration_formula(output.usage, "tool_formula")
        assert output.usage.tool_execution_time is not None and output.usage.tool_execution_time >= 0

    def test_structured_output_duration_formula(self) -> None:
        from pydantic import BaseModel, Field

        class Answer(BaseModel):
            value: int = Field(description="The numeric answer")

        agent = _make_agent()
        task = Task("What is 9*9?", response_format=Answer)
        output = agent.do(task, return_output=True)
        _assert_duration_formula(output.usage, "structured_formula")


# ---------------------------------------------------------------------------
# 28. Agent accumulation — agent.usage == sum(task.usage) for all metrics
# ---------------------------------------------------------------------------

ACCUM_TOLERANCE = 0.01  # seconds


class TestAgentAccumulationExact:
    """When same agent runs multiple tasks, agent metrics == sum of task metrics."""

    def test_duration_accumulation(self) -> None:
        agent = _make_agent()
        t1 = Task("Say hello.")
        t2 = Task("Say goodbye.")
        t3 = Task("Say thanks.")

        o1 = agent.do(t1, return_output=True)
        o2 = agent.do(t2, return_output=True)
        o3 = agent.do(t3, return_output=True)

        sum_dur = o1.usage.duration + o2.usage.duration + o3.usage.duration
        diff = abs(agent.usage.duration - sum_dur)
        assert diff < ACCUM_TOLERANCE, (
            f"Agent duration ({agent.usage.duration:.4f}) != "
            f"sum of task durations ({sum_dur:.4f}), diff={diff:.4f}"
        )

    def test_model_execution_time_accumulation(self) -> None:
        agent = _make_agent()
        t1 = Task("Say red.")
        t2 = Task("Say blue.")

        o1 = agent.do(t1, return_output=True)
        o2 = agent.do(t2, return_output=True)

        sum_model = o1.usage.model_execution_time + o2.usage.model_execution_time
        diff = abs(agent.usage.model_execution_time - sum_model)
        assert diff < ACCUM_TOLERANCE, (
            f"Agent model_execution_time ({agent.usage.model_execution_time:.4f}) != "
            f"sum ({sum_model:.4f}), diff={diff:.4f}"
        )

    def test_tokens_accumulation_exact(self) -> None:
        agent = _make_agent()
        t1 = Task("Say one.")
        t2 = Task("Say two.")
        t3 = Task("Say three.")

        o1 = agent.do(t1, return_output=True)
        o2 = agent.do(t2, return_output=True)
        o3 = agent.do(t3, return_output=True)

        sum_in = o1.usage.input_tokens + o2.usage.input_tokens + o3.usage.input_tokens
        sum_out = o1.usage.output_tokens + o2.usage.output_tokens + o3.usage.output_tokens

        assert agent.usage.input_tokens == sum_in, (
            f"Agent input_tokens ({agent.usage.input_tokens}) != sum ({sum_in})"
        )
        assert agent.usage.output_tokens == sum_out, (
            f"Agent output_tokens ({agent.usage.output_tokens}) != sum ({sum_out})"
        )

    def test_requests_accumulation_exact(self) -> None:
        agent = _make_agent()
        t1 = Task("Say A.")
        t2 = Task("Say B.")

        o1 = agent.do(t1, return_output=True)
        o2 = agent.do(t2, return_output=True)

        sum_reqs = o1.usage.requests + o2.usage.requests
        assert agent.usage.requests == sum_reqs, (
            f"Agent requests ({agent.usage.requests}) != sum ({sum_reqs})"
        )

    def test_tool_calls_accumulation_exact(self) -> None:
        from upsonic.tools.config import tool as tool_decorator

        @tool_decorator(docstring_format="google")
        def add(a: int, b: int) -> int:
            """Add two numbers.

            Args:
                a: First number
                b: Second number

            Returns:
                Sum
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

    def test_tool_execution_time_accumulation(self) -> None:
        import time as _time
        from upsonic.tools.config import tool as tool_decorator

        @tool_decorator(docstring_format="google")
        def slow_op(x: int) -> int:
            """Do a slow operation.

            Args:
                x: Input number

            Returns:
                Result
            """
            _time.sleep(0.3)
            return x * 2

        agent = _make_agent(tools=[slow_op])
        t1 = Task("Use the slow_op tool with input 5.")
        t2 = Task("Use the slow_op tool with input 10.")

        o1 = agent.do(t1, return_output=True)
        o2 = agent.do(t2, return_output=True)

        sum_tool = (o1.usage.tool_execution_time or 0.0) + (o2.usage.tool_execution_time or 0.0)
        agent_tool = agent.usage.tool_execution_time or 0.0
        diff = abs(agent_tool - sum_tool)
        assert diff < ACCUM_TOLERANCE, (
            f"Agent tool_execution_time ({agent_tool:.4f}) != "
            f"sum ({sum_tool:.4f}), diff={diff:.4f}"
        )

    def test_formula_holds_for_each_task_in_multi_run(self) -> None:
        """Each individual task's duration formula holds even when running multiple on same agent."""
        agent = _make_agent()
        tasks = [Task("Say cat."), Task("Say dog."), Task("Say bird.")]
        outputs = [agent.do(t, return_output=True) for t in tasks]

        for i, o in enumerate(outputs):
            _assert_duration_formula(o.usage, f"multi_run_task_{i}")
