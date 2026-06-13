"""Closed-session invocation guard (QA bug batch F4 + F5 + F7).

Before the fix, invoking a chat after close() — or after an `async with Chat`
block exited — failed silently (can_accept_invocation ignored _is_closed), which
read to users as "history lost". Now it raises SessionClosedError pointing at
reopen(), and close -> reopen -> invoke resumes cleanly.
"""
import pytest
from unittest.mock import MagicMock
from typing import Any

from upsonic.chat import Chat, SessionState
from upsonic.exceptions import SessionClosedError
from upsonic.tasks.tasks import Task


class MockAgent:
    def __init__(self) -> None:
        self.model = MagicMock()
        self.model.model_name = "test-model"
        self.memory = None
        self.name = "MockAgent"
        self.debug = False
        self.tools = []

    async def do_async(self, task: Task, **kwargs: Any) -> str:
        return f"ok: {task.description}"


@pytest.mark.asyncio
async def test_invoke_after_close_raises_session_closed():
    chat = Chat(session_id="s1", user_id="u1", agent=MockAgent())
    await chat.invoke("hello")
    await chat.close()
    with pytest.raises(SessionClosedError, match="reopen"):
        await chat.invoke("are you there?")


@pytest.mark.asyncio
async def test_invoke_after_context_manager_exit_raises():
    async with Chat(session_id="s2", user_id="u1", agent=MockAgent()) as chat:
        await chat.invoke("inside block")
    # Block exited -> auto-closed. Invoking now must raise, not fail silently.
    with pytest.raises(SessionClosedError):
        await chat.invoke("after block")


@pytest.mark.asyncio
async def test_close_then_reopen_then_invoke_resumes():
    chat = Chat(session_id="s3", user_id="u1", agent=MockAgent())
    await chat.invoke("first")
    await chat.close()
    chat.reopen()
    result = await chat.invoke("second")
    assert "second" in result
    assert chat.state == SessionState.IDLE


@pytest.mark.asyncio
async def test_reopen_duration_is_cumulative():
    chat = Chat(session_id="s4", user_id="u1", agent=MockAgent())
    await chat.invoke("first")
    await chat.close()
    frozen = chat.duration
    chat.reopen()
    # Duration continues from the frozen value, not reset to ~0.
    assert chat.duration >= frozen - 1e-3
