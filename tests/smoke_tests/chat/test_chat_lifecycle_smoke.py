"""End-to-end chat lifecycle smoke tests for the QA bug batch (F3 + F4 + F6).

Exercises against a real model (selected by smoke_model / the LLM_MODEL_KEY
override in conftest):
  - F6: chat.usage.tool_calls reflects executed tool calls.
  - F3: a failed invoke latches SessionState.ERROR; reopen() recovers
        non-destructively with conversation history intact.
  - F4: invoking a closed session raises SessionClosedError.

The failure in the F3 test is injected deterministically into a pipeline step
(model-independent), since the global LLM_MODEL_KEY override means an invalid
model string would be replaced before it could fail.

Run with: uv run pytest tests/smoke_tests/chat/test_chat_lifecycle_smoke.py -v -s
"""
import pytest

from upsonic import Agent, Chat
from upsonic.chat import SessionState
from upsonic.exceptions import SessionClosedError
from upsonic.tools import tool
from tests.smoke_tests._model_selection import smoke_model
from tests._pipeline_injection import (
    inject_error_into_step,
    clear_error_injection,
)


pytestmark = pytest.mark.timeout(300)

MODEL = smoke_model()


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the sum."""
    return a + b


@pytest.mark.asyncio
async def test_chat_usage_tool_calls_counted():
    agent = Agent(MODEL, tools=[add_numbers])
    chat = Chat(session_id="smoke-tool-calls", user_id="u1", agent=agent)
    await chat.invoke("Use the add_numbers tool to add 15 and 27. Return only the number.")
    assert chat.usage.tool_calls >= 1


@pytest.mark.asyncio
async def test_error_latch_then_reopen_preserves_history():
    agent = Agent(MODEL)
    chat = Chat(session_id="smoke-error-reopen", user_id="u1", agent=agent)

    await chat.invoke("My name is Alice. Please remember it.")

    # Force a failure deterministically (model-independent).
    clear_error_injection()
    inject_error_into_step("response_processing", RuntimeError, "boom", trigger_count=5)
    try:
        with pytest.raises(Exception):
            await chat.invoke("Hello")
    finally:
        clear_error_injection()
    assert chat.state == SessionState.ERROR

    # The latched ERROR guards the next invoke with an actionable message.
    with pytest.raises(RuntimeError, match="reopen"):
        await chat.invoke("Are you there?")

    # Non-destructive recovery: reopen clears ERROR and history survives.
    chat.reopen()
    assert chat.state == SessionState.IDLE
    response = await chat.invoke("What is my name?")
    assert "alice" in response.lower()


@pytest.mark.asyncio
async def test_closed_session_invoke_raises():
    agent = Agent(MODEL)
    chat = Chat(session_id="smoke-closed", user_id="u1", agent=agent)
    await chat.invoke("Hello there.")
    await chat.close()
    with pytest.raises(SessionClosedError):
        await chat.invoke("Anyone home?")
