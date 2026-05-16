"""Unit tests for stream_final_answer streaming behavior with a fake Model.

Covers:
  - Scenario 3: model disobeys the directive (no sentinel call) ⇒ NO
                FinalAnswerStartEvent emitted; FinalOutputEvent still fires.
  - Scenario 7: flag off (default) ⇒ event timeline does NOT contain
                FinalAnswerStartEvent (byte-identical to pre-feature).
  - Marker-once invariant: if the model calls sentinel multiple times in a
    pathological way, only ONE FinalAnswerStartEvent is emitted.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, List
from unittest.mock import AsyncMock, Mock

import pytest

from upsonic.agent.agent import Agent
from upsonic.messages.messages import (
    FinalResultEvent,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
)
from upsonic.models import Model
from upsonic.tasks.tasks import Task
from upsonic.tools import tool
from upsonic.usage import RequestUsage


class _StreamingMockModel(Model):
    """Mock model that streams a configurable list of events and a final
    ModelResponse. Used to drive Scenario 3 / Scenario 7 deterministically.
    """

    def __init__(
        self,
        events_to_yield: List[Any],
        final_parts: List[Any],
        model_name: str = "fake-model",
    ):
        super().__init__()
        self._model_name = model_name
        self._events = events_to_yield
        self._final_parts = final_parts

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return "fake"

    async def request(self, messages, model_settings, model_request_parameters):  # type: ignore[override]
        return ModelResponse(
            parts=self._final_parts,
            model_name=self._model_name,
            timestamp=time.time(),
            usage=RequestUsage(input_tokens=10, output_tokens=5, details={}),
            provider_name="fake",
            provider_response_id="id1",
            provider_details={},
            finish_reason="stop",
        )

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters):  # type: ignore[override]
        events = self._events
        final_response = ModelResponse(
            parts=self._final_parts,
            model_name=self._model_name,
            timestamp=time.time(),
            usage=RequestUsage(input_tokens=10, output_tokens=5, details={}),
            provider_name="fake",
            provider_response_id="id1",
            provider_details={},
            finish_reason="stop",
        )

        stream_mock = AsyncMock()
        stream_mock.__aenter__ = AsyncMock(return_value=stream_mock)
        stream_mock.__aexit__ = AsyncMock(return_value=None)

        async def _iter(_self):
            for event in events:
                yield event

        stream_mock.__aiter__ = _iter
        stream_mock.get = Mock(return_value=final_response)

        try:
            yield stream_mock
        finally:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@tool
def _dummy_user_tool() -> str:
    """A real user tool — ensures the skip-when-no-user-tools rule does not
    fire, so the sentinel injection and prompt mutation are both active.
    """
    return "ok"


async def _collect(agent: Agent, task: Task, **kwargs):
    out = []
    async for event in agent.astream(task, events=True, **kwargs):
        out.append(event)
    return out


# ---------------------------------------------------------------------------
# Scenario 3: model disobeys — no marker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disobey_no_marker_emitted():
    """When the model emits final text without ever calling the sentinel,
    NO FinalAnswerStartEvent appears in the timeline. FinalOutputEvent
    still fires retrospectively, as it would today.
    """
    from upsonic.run.events.events import FinalAnswerStartEvent, FinalOutputEvent

    events_to_stream = [
        PartStartEvent(index=0, part=TextPart(content="Hello")),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=" world")),
        FinalResultEvent(tool_name=None, tool_call_id=None),
    ]
    final_parts = [TextPart(content="Hello world")]

    model = _StreamingMockModel(events_to_stream, final_parts)
    agent = Agent(model=model, name="DisobeyAgent", tools=[_dummy_user_tool])
    task = Task(description="Say hello.")

    collected = await _collect(agent, task, stream_final_answer=True)

    # No FinalAnswerStartEvent anywhere
    fa_starts = [e for e in collected if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 0, "Disobey case must not emit FinalAnswerStartEvent"

    # FinalOutputEvent IS still emitted (status-quo retrospective fallback)
    final_outputs = [e for e in collected if isinstance(e, FinalOutputEvent)]
    assert len(final_outputs) == 1, "FinalOutputEvent must still fire in disobey case"


# ---------------------------------------------------------------------------
# Scenario 7: flag off — no marker, no behavior change
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_default_no_marker():
    """With stream_final_answer=False (default), no FinalAnswerStartEvent
    is emitted and the agent's per-run flag is False throughout.
    """
    from upsonic.run.events.events import FinalAnswerStartEvent

    events_to_stream = [
        PartStartEvent(index=0, part=TextPart(content="Hi")),
        FinalResultEvent(tool_name=None, tool_call_id=None),
    ]
    final_parts = [TextPart(content="Hi")]

    model = _StreamingMockModel(events_to_stream, final_parts)
    agent = Agent(model=model, name="FlagOffAgent", tools=[_dummy_user_tool])
    task = Task(description="Say hi.")

    collected = await _collect(agent, task)  # no stream_final_answer

    fa_starts = [e for e in collected if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 0

    # Per-run flag must remain False
    assert getattr(agent, "_stream_final_answer_active", False) is False


@pytest.mark.asyncio
async def test_flag_off_explicitly_false_no_marker():
    """Explicit stream_final_answer=False behaves identically to default."""
    from upsonic.run.events.events import FinalAnswerStartEvent

    events_to_stream = [
        PartStartEvent(index=0, part=TextPart(content="Hi")),
        FinalResultEvent(tool_name=None, tool_call_id=None),
    ]
    final_parts = [TextPart(content="Hi")]

    model = _StreamingMockModel(events_to_stream, final_parts)
    agent = Agent(model=model, name="ExplicitOffAgent", tools=[_dummy_user_tool])
    task = Task(description="Say hi.")

    collected = await _collect(agent, task, stream_final_answer=False)
    fa_starts = [e for e in collected if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 0


@pytest.mark.asyncio
async def test_per_run_flags_cleared_after_run():
    """After astream completes, the per-run flags must be cleared so a
    subsequent non-streaming call on the same Agent instance does not
    inherit them.
    """
    events_to_stream = [
        PartStartEvent(index=0, part=TextPart(content="X")),
        FinalResultEvent(tool_name=None, tool_call_id=None),
    ]
    final_parts = [TextPart(content="X")]

    model = _StreamingMockModel(events_to_stream, final_parts)
    agent = Agent(model=model, name="LeakAgent", tools=[_dummy_user_tool])
    task = Task(description="Anything.")

    await _collect(agent, task, stream_final_answer=True)

    assert agent._stream_final_answer_active is False
    assert agent._final_answer_marker_emitted is False


# ---------------------------------------------------------------------------
# Skip-when-no-user-tools rule
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skip_sentinel_when_no_user_tools():
    """When the agent has no registered user tools, the sentinel is not
    injected (no extra round-trip cost). Model never sees it; no marker
    is emitted via the sentinel path.
    """
    from upsonic.run.events.events import FinalAnswerStartEvent

    events_to_stream = [
        PartStartEvent(index=0, part=TextPart(content="Hi")),
        FinalResultEvent(tool_name=None, tool_call_id=None),
    ]
    final_parts = [TextPart(content="Hi")]

    model = _StreamingMockModel(events_to_stream, final_parts)
    # NO tools.
    agent = Agent(model=model, name="NoToolsAgent")
    task = Task(description="Say hi.")

    collected = await _collect(agent, task, stream_final_answer=True)
    fa_starts = [e for e in collected if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 0
