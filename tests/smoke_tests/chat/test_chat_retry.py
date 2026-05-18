"""
Chat Retry Smoke Tests

Chat used to have its own retry wrapper around ``agent.do_async``. That
wrapper re-invoked the agent on the SAME ``Task`` object, which —
because the previous attempt may have left ``task.status`` as completed
or problematic — surfaced the ``[Agent] Task is already completed.
Cannot re-run a completed task.`` warning on the retry.

Chat now delegates retries entirely to ``Agent.do_async``'s
``@retryable`` decorator (configured via ``Agent(retry=N)``). The
Agent's retry block calls ``task.reset_run_state()`` between attempts,
which clears ``task.status`` / ``task._response`` and avoids the
warning entirely for retried attempts on the same Task.

Scenarios:
  1. Successful retry: 2 attempts fail, 3rd succeeds — Chat returns
     the successful response, no "Task is already completed" warning.
  2. Chat usage metrics accumulate across retry attempts (mirrors
     ``test_retry_metrics.py``).
  3. All-fail retry: agent raises after retries exhausted; Chat
     propagates the exception, no warning text in the response.

Run with: uv run pytest tests/smoke_tests/chat/test_chat_retry.py -v -s
"""

import logging
import pytest

from upsonic import Agent, Chat
from upsonic.tools import tool
from tests._pipeline_injection import (
    inject_error_into_step,
    clear_error_injection,
)


pytestmark = pytest.mark.timeout(300)

MODEL = "openai/gpt-4o-mini"

COMPLETED_WARNING_FRAGMENT = "Task is already completed"


@tool
def simple_math(a: int, b: int) -> int:
    """Add two numbers and return the sum."""
    return a + b


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _LogCapture:
    """Capture WARNING-level log records emitted to the ``upsonic`` logger.

    The "Task is already completed" warning is produced via
    ``warning_log`` in ``src/upsonic/agent/agent.py`` which writes to the
    Python ``logging`` system. ``capsys`` does NOT see this output, so
    we install a temporary handler.
    """

    def __init__(self):
        self.records: list[logging.LogRecord] = []
        self._handler: logging.Handler | None = None
        self._logger: logging.Logger | None = None
        self._prev_level: int | None = None

    def __enter__(self):
        self._logger = logging.getLogger("upsonic")
        self._prev_level = self._logger.level
        self._logger.setLevel(logging.WARNING)

        class _ListHandler(logging.Handler):
            def __init__(self, store):
                super().__init__(level=logging.WARNING)
                self._store = store

            def emit(self, record: logging.LogRecord) -> None:
                self._store.append(record)

        self._handler = _ListHandler(self.records)
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._logger and self._handler:
            self._logger.removeHandler(self._handler)
        if self._logger and self._prev_level is not None:
            self._logger.setLevel(self._prev_level)
        return False

    def has_message_containing(self, fragment: str) -> bool:
        return any(fragment in record.getMessage() for record in self.records)


# ---------------------------------------------------------------------------
# 1. Successful retry — no completed-task warning during retried attempts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_retry_success_no_completed_warning():
    """Agent retry=3 (3 total attempts), first 2 fail post-model,
    3rd succeeds. The retried attempts must NOT trigger the
    'Task is already completed' warning."""
    clear_error_injection()
    inject_error_into_step(
        "response_processing", RuntimeError, "boom-post-model", trigger_count=2
    )

    agent = Agent(MODEL, name="ChatRetryAgent", retry=3)
    chat = Chat(
        session_id="chat_retry_success",
        user_id="u1",
        agent=agent,
    )
    try:
        with _LogCapture() as logs:
            response = await chat.invoke("What is 5 + 3? Reply with just the number.")

        assert isinstance(response, str)
        assert response, "Chat must return a non-empty response after retry"
        assert COMPLETED_WARNING_FRAGMENT not in response, (
            f"Response contains the completed-task warning text: {response!r}"
        )
        assert not logs.has_message_containing(COMPLETED_WARNING_FRAGMENT), (
            "The 'Task is already completed' warning fired during retry. "
            "Agent's retry block should reset task state before each attempt."
        )
    finally:
        clear_error_injection()
        await chat.close()


# ---------------------------------------------------------------------------
# 2. Chat usage metrics accumulate across retry attempts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_retry_usage_accumulates_across_attempts():
    """``agent.usage`` after a successful retry must include the failed
    attempts' partial token usage (captured by Agent's retry block and
    the exhaustion hook). Mirrors ``test_retry_success_metrics_accumulate_all_attempts``
    from ``test_retry_metrics.py`` but driven via Chat."""
    clear_error_injection()
    inject_error_into_step(
        "response_processing", RuntimeError, "boom", trigger_count=2
    )

    agent = Agent(MODEL, name="ChatRetryMetricsAgent", retry=3)
    chat = Chat(
        session_id="chat_retry_metrics",
        user_id="u2",
        agent=agent,
    )
    try:
        response = await chat.invoke("What is 7 + 2?")
        assert response

        # agent.usage must be populated and reflect ~3× per-attempt counters.
        assert agent.usage is not None
        assert agent.usage.requests > 0

        # Read the run output from the agent — it carries the FINAL attempt's
        # usage. Multiplying by 3 (total attempts) is the expected lower bound
        # for agent.usage when each attempt does a similar amount of work.
        per_attempt = agent._agent_run_output.usage.requests
        assert agent.usage.requests == per_attempt * 3, (
            f"agent.usage.requests={agent.usage.requests} != "
            f"per_attempt({per_attempt})×3 — failed attempts not accumulated"
        )
    finally:
        clear_error_injection()
        await chat.close()


# ---------------------------------------------------------------------------
# 3. All-fail retry — exception propagates, no warning leaked as response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_retry_all_fail_raises():
    """When all retry attempts fail at a post-model step, Chat must
    propagate the underlying exception (it does NOT swallow it into
    a warning-as-response)."""
    clear_error_injection()
    inject_error_into_step(
        "response_processing", RuntimeError, "perma-boom", trigger_count=10
    )

    agent = Agent(MODEL, name="ChatAllFailAgent", retry=3)
    chat = Chat(
        session_id="chat_all_fail",
        user_id="u4",
        agent=agent,
    )
    try:
        raised: Exception | None = None
        try:
            await chat.invoke("What is 4 + 4?")
        except Exception as e:
            raised = e

        assert raised is not None, "All-fail retry must raise after exhausting attempts"
        assert "perma-boom" in str(raised) or "INJECTED ERROR" in str(raised), (
            f"Raised exception text mismatch: {raised!r}"
        )

        # Even on full failure, agent.usage should reflect the captured
        # partial usage from all 3 attempts (retry block: 1, 2; hook: 3).
        if agent.usage is not None:
            assert agent.usage.requests > 0
    finally:
        clear_error_injection()
        await chat.close()
