"""Chat SessionState latch + reopen recovery (QA bug batch F3 + F10).

Before the fix, a failed invoke set ERROR in the except handler but the finally
unconditionally restored IDLE, so the failure was invisible and the error-state
guard was dead code. Now ERROR latches; reopen() clears it non-destructively.
"""
import pytest
from unittest.mock import MagicMock
from typing import Any, AsyncIterator

from upsonic.chat import Chat, SessionState
from upsonic.tasks.tasks import Task


class FlakyAgent:
    """Mock agent whose do_async / astream can be toggled to raise."""

    def __init__(self) -> None:
        self.model = MagicMock()
        self.model.model_name = "test-model"
        self.memory = None
        self.name = "FlakyAgent"
        self.debug = False
        self.tools = []
        self.should_raise = False

    async def do_async(self, task: Task, **kwargs: Any) -> str:
        if self.should_raise:
            raise RuntimeError("boom")
        return f"ok: {task.description}"

    async def astream(self, task: Task, events: bool = False, **kwargs: Any) -> AsyncIterator[Any]:
        if self.should_raise:
            raise RuntimeError("stream boom")
        for word in ["Hello", " ", "World"]:
            yield word


def _make_chat() -> tuple[Chat, FlakyAgent]:
    agent = FlakyAgent()
    chat = Chat(session_id="s1", user_id="u1", agent=agent)
    return chat, agent


@pytest.mark.asyncio
async def test_successful_invoke_returns_to_idle():
    # Characterization: a normal invoke leaves the session IDLE.
    chat, _ = _make_chat()
    await chat.invoke("hello")
    assert chat.state == SessionState.IDLE


@pytest.mark.asyncio
async def test_failed_invoke_latches_error():
    chat, agent = _make_chat()
    agent.should_raise = True
    with pytest.raises(RuntimeError):
        await chat.invoke("hello")
    assert chat.state == SessionState.ERROR


@pytest.mark.asyncio
async def test_invoke_in_error_state_raises_clear_message():
    chat, agent = _make_chat()
    agent.should_raise = True
    with pytest.raises(RuntimeError):
        await chat.invoke("hello")
    # Next invoke is guarded with an actionable message naming the recovery path.
    with pytest.raises(RuntimeError, match="reopen"):
        await chat.invoke("again")


@pytest.mark.asyncio
async def test_reopen_clears_error_on_open_session():
    chat, agent = _make_chat()
    agent.should_raise = True
    with pytest.raises(RuntimeError):
        await chat.invoke("hello")
    assert chat.state == SessionState.ERROR

    # Session was never closed — just errored. reopen() must still clear it.
    chat.reopen()
    assert chat.state == SessionState.IDLE

    agent.should_raise = False
    result = await chat.invoke("recovered")
    assert "recovered" in result
    assert chat.state == SessionState.IDLE


@pytest.mark.asyncio
async def test_close_then_reopen_duration_is_cumulative():
    # Characterization: reopen continues duration, never resets to ~0.
    chat, _ = _make_chat()
    await chat.invoke("hello")
    await chat.close()
    frozen = chat.duration
    chat.reopen()
    assert chat.duration >= frozen - 1e-3  # cumulative, not reset


@pytest.mark.asyncio
async def test_closed_and_errored_reopen_clears_both():
    chat, agent = _make_chat()
    agent.should_raise = True
    with pytest.raises(RuntimeError):
        await chat.invoke("hello")
    await chat.close()  # closed AND errored
    chat.reopen()
    assert chat.state == SessionState.IDLE
    agent.should_raise = False
    assert "ok" in await chat.invoke("after")


@pytest.mark.asyncio
async def test_stream_abandonment_returns_to_idle():
    chat, _ = _make_chat()
    stream = await chat.invoke("hello", stream=True)
    async for _chunk in stream:
        break  # abandon mid-stream
    # Closing the abandoned generator throws GeneratorExit into it (this is what
    # the next invoke's stale-stream cleanup, or GC, does). Abandonment is not an
    # error, so the finally must land the session in IDLE — not stuck STREAMING.
    await stream.aclose()
    assert chat.state == SessionState.IDLE
