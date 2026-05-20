"""
Chat Streaming Failure Smoke Tests

End-to-end regression for the streaming pipeline's mid-stream failure
path. Mirrors the autonomous-HQ chat.py screenshot symptom:

Before the fix, when ``StreamModelExecutionStep`` raised mid-stream,
``PipelineManager.execute_stream``'s ``finally`` block called
``mark_completed()`` after the ``except`` had already called
``mark_error()`` — because the failing step was never appended to
``context.step_results``. The task ended up with ``status=completed``
even though the run failed. Any subsequent call passing the SAME
``Task`` instance to ``Agent.astream`` / ``Agent.do_async`` then
tripped ``_validate_task_for_new_run``'s completed-task guard with:
  ``[Agent] Task is already completed (run_id=…). Cannot re-run a completed task.``

After the fix the task stays in ``RunStatus.error`` and the
completed-guard never fires.

This file uses a deterministic ``Model`` subclass instead of a real
provider so the streaming exception path is exercised reliably without
network flakiness. ``inject_error_into_step`` is deliberately NOT used
here: it patches ``Step.run`` (non-streaming) and HITL is not
supported in streaming yet.

Scenarios:
  1. Single streaming invoke fails → task ends up in ``RunStatus.error``,
     not ``RunStatus.completed``.
  2. Failing step's ``StepResult`` is appended to ``step_results`` so
     observability tooling (``pipeline_failed`` panel, sentry log, the
     ``get_error_step`` API) can identify the failure point.
  3. Re-using the failed ``Task`` instance MUST NOT trigger the
     completed-task guard's "Task is already completed" warning.

Run with: ``uv run pytest tests/smoke_tests/chat/test_chat_streaming_failure.py -v -s``
"""

import logging
import time
from contextlib import asynccontextmanager

import pytest

from upsonic import Agent, Chat
from upsonic.messages.messages import ModelResponse, TextPart
from upsonic.models import Model
from upsonic.run.base import RunStatus
from upsonic.usage import RequestUsage


pytestmark = pytest.mark.timeout(120)

COMPLETED_WARNING_FRAGMENT = "Task is already completed"


class _MidStreamFailModel(Model):
    """Deterministic streaming failure model.

    ``request`` succeeds (so non-streaming preflight steps like
    ``LLMManager``, ``ModelSelection``, ``MessageAssembly`` complete
    without issue) and ``request_stream`` raises inside ``__anext__``
    — landing the failure inside ``StreamModelExecutionStep``.
    """

    @property
    def model_name(self) -> str:
        return "smoke-fail-model"

    @property
    def system(self) -> str:
        return "smoke-test-provider"

    async def request(self, messages, model_settings, model_request_parameters):
        return ModelResponse(
            parts=[TextPart(content="ok")],
            model_name=self.model_name,
            timestamp=time.time(),
            usage=RequestUsage(input_tokens=1, output_tokens=1, details={}),
            provider_name="smoke-test-provider",
            provider_response_id="id",
            provider_details={},
            finish_reason="stop",
        )

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters):
        class _FailingIter:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, exc_type, exc_val, exc_tb):
                pass

            def __aiter__(self_inner):
                return self_inner

            async def __anext__(self_inner):
                # Empty-message exception mirrors the original Anthropic
                # streaming failure shape from the autonomous-HQ screenshot.
                raise Exception("")

        yield _FailingIter()


class _UpsonicLogCapture:
    """Capture WARNING-level records emitted to the ``upsonic`` logger.

    ``warning_log`` in ``src/upsonic/utils/printing.py`` writes both to
    a Rich console (captured by ``capsys``) AND to the ``upsonic.user``
    Python logger (NOT captured by ``capsys``). We install a temporary
    list-handler to catch the latter.
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
        store = self.records

        class _ListHandler(logging.Handler):
            def __init__(self):
                super().__init__(level=logging.WARNING)

            def emit(self, record: logging.LogRecord) -> None:
                store.append(record)

        self._handler = _ListHandler()
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._logger is not None and self._handler is not None:
            self._logger.removeHandler(self._handler)
        if self._logger is not None and self._prev_level is not None:
            self._logger.setLevel(self._prev_level)

    def has_message_containing(self, fragment: str) -> bool:
        return any(fragment in record.getMessage() for record in self.records)


def _build_chat(session_suffix: str) -> tuple[Chat, Agent]:
    agent = Agent(model=_MidStreamFailModel(), name="SmokeStreamFailAgent")
    chat = Chat(
        session_id=f"smoke-stream-fail-{session_suffix}",
        user_id="smoke-user",
        agent=agent,
    )
    return chat, agent


# ---------------------------------------------------------------------------
# 1. Task ends up in RunStatus.error, not RunStatus.completed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_failure_leaves_run_in_error_state():
    """After a mid-stream failure the run must be marked as ``error``.
    A run incorrectly marked as ``completed`` would cause the next
    invoke to bypass any continue-run logic and silently misbehave.
    """
    chat, agent = _build_chat("state")

    try:
        with pytest.raises(Exception):
            stream = await chat.invoke("trigger failure", stream=True, events=True)
            async for _ in stream:
                pass

        out = agent.get_run_output()
        assert out is not None, "Agent must have a run output after invoke"
        assert out.status == RunStatus.error, (
            f"Expected RunStatus.error after streaming failure, got {out.status}. "
            f"The pipeline finally block is overwriting mark_error() with "
            f"mark_completed()."
        )
    finally:
        await chat.close()


# ---------------------------------------------------------------------------
# 2. Failing step is appended to step_results (observability)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_failure_records_failing_step_in_step_results():
    """``step_results[-1]`` must reflect the failing step's ERROR status
    (not the previous successful step's COMPLETED status). This is
    what ``get_error_step()`` and the ``pipeline_failed`` panel rely on.
    """
    from upsonic.agent.pipeline.step import StepStatus

    chat, agent = _build_chat("observability")

    try:
        with pytest.raises(Exception):
            stream = await chat.invoke("trigger failure", stream=True, events=True)
            async for _ in stream:
                pass

        out = agent.get_run_output()
        assert out is not None
        assert out.step_results, (
            "step_results must not be empty after a failed streaming run"
        )
        last_step = out.step_results[-1]
        assert last_step.status in (StepStatus.ERROR, StepStatus.CANCELLED), (
            f"Last recorded step must be ERROR/CANCELLED, got {last_step.status}. "
            f"Step.run_stream is not finalizing the failing step into step_results."
        )
        assert last_step.name, (
            f"Failing StepResult must carry a name (got {last_step.name!r}). "
            f"StreamModelExecutionStep's exception branch is constructing "
            f"StepResult without name/step_number."
        )
    finally:
        await chat.close()


# ---------------------------------------------------------------------------
# 3. Re-using the failed Task instance does NOT trip the completed-guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reused_failed_task_does_not_trip_completed_guard():
    """Direct end-to-end regression for the autonomous-HQ screenshot.

    Passing the SAME ``Task`` instance back through ``chat.invoke``
    must NOT emit the ``Task is already completed`` warning. It is OK
    for the framework to refuse with the *problematic-run* warning
    (``status=ERROR``) — but the ``is_completed`` branch must never
    fire on a task that actually failed.
    """
    chat, _ = _build_chat("reuse")

    # Hook ``_normalize_input`` so we can capture the Task object Chat
    # creates on the first (string) invoke and re-feed it on the second.
    captured_tasks: list = []
    original_normalize = chat._normalize_input

    def _capture(input_data, context=None):
        t = original_normalize(input_data, context)
        captured_tasks.append(t)
        return t

    chat._normalize_input = _capture

    try:
        with _UpsonicLogCapture() as logs:
            # 1st invoke (string) — fresh Task, fails mid-stream.
            with pytest.raises(Exception):
                stream = await chat.invoke("first", stream=True, events=True)
                async for _ in stream:
                    pass

            assert captured_tasks, "Chat must materialize a Task on first invoke"
            failed_task = captured_tasks[0]

            # 2nd invoke with the SAME Task instance — the validate call
            # runs again against a task whose status is now error/completed.
            try:
                stream = await chat.invoke(failed_task, stream=True, events=True)
                async for _ in stream:
                    pass
            except Exception:
                pass

        # Hard regression assertion.
        assert not logs.has_message_containing(COMPLETED_WARNING_FRAGMENT), (
            f"Regression: re-using a failed Task fired the completed-task "
            f"guard ('Task is already completed'). Pipeline finally is "
            f"overwriting mark_error() with mark_completed().\n"
            f"Captured warnings: "
            f"{[r.getMessage() for r in logs.records]}"
        )
    finally:
        await chat.close()
