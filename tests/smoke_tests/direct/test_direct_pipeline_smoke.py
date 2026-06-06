"""Smoke tests for `Direct` — real Anthropic API calls.

These exercise the reduced ("direct") pipeline profile end-to-end against
``anthropic/claude-haiku-4-5``. The key is read from the environment, falling
back to the repo ``.env``; the module is skipped when no key is available.
"""
import os
import pathlib

import pytest
from pydantic import BaseModel

from upsonic import Agent, Direct, Task, Graph
from upsonic.context.sources import TaskOutputSource


def _ensure_anthropic_key() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    env = pathlib.Path(__file__).resolve().parents[3] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    os.environ["ANTHROPIC_API_KEY"] = val
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


pytestmark = pytest.mark.skipif(
    not _ensure_anthropic_key(),
    reason="ANTHROPIC_API_KEY not available for live smoke test",
)

MODEL = "anthropic/claude-haiku-4-5"


class Person(BaseModel):
    name: str
    age: int


@pytest.mark.asyncio
async def test_smoke_direct_text_output():
    d = Direct(MODEL, print=False)
    result = await d.do_async(
        Task("What is 2 + 2? Reply with only the number.")
    )
    assert isinstance(result, str)
    assert "4" in result


@pytest.mark.asyncio
async def test_smoke_direct_structured_output():
    d = Direct(MODEL, print=False)
    result = await d.do_async(
        Task("Extract the person from this text: 'Ada Lovelace, age 36.'",
             response_format=Person),
    )
    assert isinstance(result, Person)
    assert result.age == 36
    assert "ada" in result.name.lower()


@pytest.mark.asyncio
async def test_smoke_direct_plain_string_context():
    d = Direct(MODEL, print=False)
    result = await d.do_async(
        Task("What is the secret code? Reply with only the code.",
             context=["The secret code is ZEBRA42."]),
    )
    assert "ZEBRA42" in str(result).upper().replace(" ", "")


@pytest.mark.asyncio
async def test_smoke_direct_taskoutputsource_via_state():
    """Direct resolves a Graph-style TaskOutputSource from `state` and the
    upstream output reaches the live model."""

    class _State:
        def get_task_output(self, task_id):
            return "The magic number is 4242." if task_id == "prev" else None

    d = Direct(MODEL, print=False)
    result = await d.do_async(
        Task("What is the magic number? Reply with only the number.",
             context=[TaskOutputSource(task_description_or_id="prev")]),
        state=_State(),
    )
    assert "4242" in str(result)


@pytest.mark.asyncio
async def test_smoke_graph_direct_passthrough():
    """`Direct` as a Graph `default_agent`; the downstream task sees the
    upstream task's output via the graph's auto-injected TaskOutputSource."""
    direct = Direct(MODEL, print=False)
    graph = Graph(default_agent=direct, show_progress=False)

    task1 = Task("Reply with exactly this token and nothing else: KIWI777")
    task2 = Task(
        "What exact token did the previous step output? Reply with only that token."
    )
    graph.add(task1 >> task2)

    await graph.run_async(show_progress=False)

    final_output = graph.get_output()
    assert final_output is not None
    assert "KIWI777" in str(final_output)


@pytest.mark.asyncio
async def test_smoke_graph_agent_passthrough():
    """A plain Agent as Graph default_agent resolves the downstream
    TaskOutputSource — the Agent pipeline now honors Graph state (previously it
    silently ignored it)."""
    agent = Agent(MODEL, name="GraphAgent")
    graph = Graph(default_agent=agent, show_progress=False)

    task1 = Task("Reply with exactly this token and nothing else: MANGO314")
    task2 = Task(
        "What exact token did the previous step output? Reply with only that token."
    )
    graph.add(task1 >> task2)

    await graph.run_async(show_progress=False)

    final_output = graph.get_output()
    assert final_output is not None
    assert "MANGO314" in str(final_output)


@pytest.mark.asyncio
async def test_smoke_direct_usage_matches_agent_usage():
    """Direct.usage is shape-compatible with Agent.usage and both report real,
    matching token counts for the same minimal task/model."""
    task_text = "Reply with exactly this and nothing else: USAGE"

    direct = Direct(MODEL, print=False)
    await direct.do_async(Task(task_text))
    du = direct.usage

    # Same minimal configuration as Direct's internal agent, so prompts match.
    agent = Agent(MODEL, memory=None, tools=[], reflection=False)
    await agent.do_async(Task(task_text))
    au = agent.usage

    assert du.requests == au.requests == 1
    assert du.input_tokens > 0 and du.output_tokens > 0
    assert au.input_tokens > 0 and au.output_tokens > 0
    # Identical prompt + model + config ⇒ identical input token count.
    assert du.input_tokens == au.input_tokens
