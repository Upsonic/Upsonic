"""Whole-task retry must not repeat completed tools by default."""

import json
from collections.abc import Iterator
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI

from upsonic import Agent, Task
from upsonic.exceptions import AgentRetryUnsafeError
from upsonic.models.openai import OpenAIChatModel
from upsonic.providers.openai import OpenAIProvider
from upsonic.run.tools.tools import ToolExecution
from upsonic.tools.config import ToolConfig, tool
from upsonic.utils.package.exception import ModelHTTPError
from upsonic.utils.retry import retryable


def _model(responses: Iterator[tuple[int, dict]]) -> OpenAIChatModel:
    def respond(request: httpx.Request) -> httpx.Response:
        status, body = next(responses)
        return httpx.Response(status, json=body)

    client = AsyncOpenAI(
        api_key="test",
        base_url="https://model.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    return OpenAIChatModel("gpt-4o-mini", provider=OpenAIProvider(openai_client=client))


def _call(name: str, call_id: str = "call_1") -> tuple[int, dict]:
    return 200, {
        "id": "response",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps({"invoice": "INV-1"}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


ERROR = (429, {"error": {"message": "rate limit", "type": "requests"}})
DONE = (
    200,
    {
        "id": "response",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["raise", "return_false"])
@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize("task_tools", [False, True])
async def test_post_tool_model_failure_does_not_repeat_tool(
    mode: str,
    sequential: bool,
    task_tools: bool,
) -> None:
    payments: list[str] = []

    @tool(sequential=sequential)
    def pay(invoice: str) -> str:
        """Pay an invoice."""
        payments.append(invoice)
        return "receipt"

    model = _model(iter([_call("pay"), ERROR] * 3))
    agent = Agent(model, retry=3, mode=mode, tools=[] if task_tools else [pay])
    task = Task("Pay INV-1", tools=[pay] if task_tools else [])
    try:
        with pytest.raises(AgentRetryUnsafeError) as error:
            await agent.do_async(task)
        assert error.value.tool_names == ("pay",)
        assert isinstance(error.value.__cause__, ModelHTTPError)
        assert payments == ["INV-1"]
        assert len(agent._agent_run_output.tools) == 1
        assert agent._agent_run_output.tools[0].result
    finally:
        await model.client.close()


@pytest.mark.asyncio
async def test_explicit_retry_safe_tool_preserves_task_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    @tool(retry_safe=True)
    def lookup(invoice: str) -> str:
        """Look up an invoice."""
        calls.append(invoice)
        return "paid"

    model = _model(iter([_call("lookup"), ERROR, _call("lookup"), DONE]))
    agent = Agent(model, retry=3, tools=[lookup])
    monkeypatch.setattr("upsonic.utils.retry.asyncio.sleep", AsyncMock())
    try:
        assert await agent.do_async(Task("Look up INV-1")) == "Done"
        assert calls == ["INV-1", "INV-1"]
    finally:
        await model.client.close()


@pytest.mark.asyncio
async def test_failure_before_any_tool_preserves_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(iter([ERROR, DONE]))
    agent = Agent(model, retry=3)
    monkeypatch.setattr("upsonic.utils.retry.asyncio.sleep", AsyncMock())
    try:
        assert await agent.do_async(Task("Hello")) == "Done"
    finally:
        await model.client.close()


@pytest.mark.asyncio
async def test_completed_task_in_list_is_not_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payments: list[str] = []

    def pay(invoice: str) -> str:
        """Pay an invoice."""
        payments.append(invoice)
        return "receipt"

    model = _model(iter([_call("pay"), DONE, ERROR, ERROR, ERROR]))
    agent = Agent(model, retry=3, tools=[pay])
    monkeypatch.setattr("upsonic.utils.retry.asyncio.sleep", AsyncMock())
    try:
        with pytest.raises(ModelHTTPError):
            await agent.do_async(["Pay INV-1", "Hello"])
        assert payments == ["INV-1"]
    finally:
        await model.client.close()


@pytest.mark.asyncio
async def test_two_legitimate_identical_tool_calls_are_not_deduplicated() -> None:
    payments: list[str] = []

    def pay(invoice: str) -> str:
        """Pay an invoice."""
        payments.append(invoice)
        return "receipt"

    model = _model(iter([_call("pay"), _call("pay", "call_2"), DONE]))
    agent = Agent(model, retry=3, tools=[pay])
    try:
        assert await agent.do_async(Task("Pay twice")) == "Done"
        assert payments == ["INV-1", "INV-1"]
    finally:
        await model.client.close()


@pytest.mark.parametrize("retry_safe", [False, True])
def test_retry_safety_survives_tool_execution_round_trip(retry_safe: bool) -> None:
    execution = ToolExecution(tool_name="lookup", retry_safe=retry_safe)
    assert ToolExecution.from_dict(execution.to_dict()).retry_safe is retry_safe
    assert ToolExecution.from_dict({"tool_name": "old_tool"}).retry_safe is False


def test_retry_safety_requires_explicit_boolean() -> None:
    assert ToolConfig().retry_safe is False
    with pytest.raises(ValueError):
        ToolConfig(retry_safe="true")


def test_sync_retry_decorator_propagates_safety_error() -> None:
    class Runner:
        calls = 0

        @retryable(retries=3, mode="return_false", delay=0)
        def run(self) -> None:
            self.calls += 1
            raise AgentRetryUnsafeError(("pay",))

    runner = Runner()
    with pytest.raises(AgentRetryUnsafeError):
        runner.run()
    assert runner.calls == 1
