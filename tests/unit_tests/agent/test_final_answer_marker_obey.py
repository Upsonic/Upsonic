"""Unit tests for the obey-path: model calls the sentinel correctly.

Covers:
  - Scenario 1 (unit variant): when the model calls __final_answer_marker__,
    exactly one FinalAnswerStartEvent(triggered_by='sentinel') is emitted.
  - Sentinel call is suppressed: no ToolCallEvent / ToolResultEvent for it.
  - Scenario 9a: sentinel does NOT increment tool_call_count.
  - Parallel-batch invalidation: sentinel + real tool ⇒ sentinel dropped,
    no marker, real tool executes.
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
    ToolCallPartDelta,
)
from upsonic.models import Model
from upsonic.tasks.tasks import Task
from upsonic.tools import tool
from upsonic.tools.framework_tools import FINAL_ANSWER_MARKER_TOOL_NAME
from upsonic.usage import RequestUsage


# ---------------------------------------------------------------------------
# Multi-turn streaming mock — first turn calls sentinel; second turn streams text
# ---------------------------------------------------------------------------


class _TwoTurnSentinelModel(Model):
    """Turn 1: model emits a ToolCallPart for __final_answer_marker__.
    Turn 2: model emits final-answer text deltas.
    """

    def __init__(self, final_text: str = "The final answer."):
        super().__init__()
        self._model_name = "fake-two-turn"
        self._final_text = final_text
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return "fake"

    async def request(self, messages, model_settings, model_request_parameters):
        return ModelResponse(
            parts=[TextPart(content=self._final_text)],
            model_name=self._model_name,
            timestamp=time.time(),
            usage=RequestUsage(input_tokens=10, output_tokens=5, details={}),
            provider_name="fake",
            provider_response_id="id1",
            provider_details={},
            finish_reason="stop",
        )

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters):
        self._call_count += 1
        if self._call_count == 1:
            # Turn 1: sentinel call.
            events = [
                PartStartEvent(
                    index=0,
                    part=ToolCallPart(
                        tool_name=FINAL_ANSWER_MARKER_TOOL_NAME,
                        args={},
                        tool_call_id="tc_sentinel_1",
                    ),
                ),
                FinalResultEvent(tool_name=FINAL_ANSWER_MARKER_TOOL_NAME, tool_call_id="tc_sentinel_1"),
            ]
            final_response = ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=FINAL_ANSWER_MARKER_TOOL_NAME,
                        args={},
                        tool_call_id="tc_sentinel_1",
                    ),
                ],
                model_name=self._model_name,
                timestamp=time.time(),
                usage=RequestUsage(input_tokens=10, output_tokens=2, details={}),
                provider_name="fake",
                provider_response_id="id_t1",
                provider_details={},
                finish_reason="tool_calls",
            )
        else:
            # Turn 2: final text deltas.
            events = [
                PartStartEvent(index=0, part=TextPart(content=self._final_text[:3])),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=self._final_text[3:])),
                FinalResultEvent(tool_name=None, tool_call_id=None),
            ]
            final_response = ModelResponse(
                parts=[TextPart(content=self._final_text)],
                model_name=self._model_name,
                timestamp=time.time(),
                usage=RequestUsage(input_tokens=10, output_tokens=5, details={}),
                provider_name="fake",
                provider_response_id="id_t2",
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
# Mixed-batch streaming mock — sentinel + real tool in same parallel batch
# ---------------------------------------------------------------------------


class _MixedBatchModel(Model):
    """Turn 1: model emits sentinel AND a real tool in the same batch.
    Per the parallel-batch policy, sentinel must be dropped and the real
    tool must execute. No marker fires from this batch.
    Turn 2: model emits final text.
    """

    def __init__(self):
        super().__init__()
        self._model_name = "fake-mixed-batch"
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return "fake"

    async def request(self, messages, model_settings, model_request_parameters):
        return ModelResponse(
            parts=[TextPart(content="done")],
            model_name=self._model_name,
            timestamp=time.time(),
            usage=RequestUsage(input_tokens=10, output_tokens=5, details={}),
            provider_name="fake",
            provider_response_id="id1",
            provider_details={},
            finish_reason="stop",
        )

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters):
        self._call_count += 1
        if self._call_count == 1:
            # Turn 1: mixed batch — sentinel + real tool.
            events = [
                PartStartEvent(
                    index=0,
                    part=ToolCallPart(
                        tool_name=FINAL_ANSWER_MARKER_TOOL_NAME,
                        args={},
                        tool_call_id="tc_sentinel",
                    ),
                ),
                PartStartEvent(
                    index=1,
                    part=ToolCallPart(
                        tool_name="_dummy_user_tool",
                        args={},
                        tool_call_id="tc_real",
                    ),
                ),
                FinalResultEvent(tool_name=None, tool_call_id=None),
            ]
            final_response = ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=FINAL_ANSWER_MARKER_TOOL_NAME,
                        args={},
                        tool_call_id="tc_sentinel",
                    ),
                    ToolCallPart(
                        tool_name="_dummy_user_tool",
                        args={},
                        tool_call_id="tc_real",
                    ),
                ],
                model_name=self._model_name,
                timestamp=time.time(),
                usage=RequestUsage(input_tokens=10, output_tokens=2, details={}),
                provider_name="fake",
                provider_response_id="id_t1",
                provider_details={},
                finish_reason="tool_calls",
            )
        else:
            events = [
                PartStartEvent(index=0, part=TextPart(content="done")),
                FinalResultEvent(tool_name=None, tool_call_id=None),
            ]
            final_response = ModelResponse(
                parts=[TextPart(content="done")],
                model_name=self._model_name,
                timestamp=time.time(),
                usage=RequestUsage(input_tokens=10, output_tokens=5, details={}),
                provider_name="fake",
                provider_response_id="id_t2",
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


@tool
def _dummy_user_tool() -> str:
    """User tool used to satisfy the ≥1-user-tool injection precondition."""
    return "ok"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sentinel_call_emits_marker_once():
    from upsonic.run.events.events import FinalAnswerStartEvent

    model = _TwoTurnSentinelModel()
    agent = Agent(model=model, name="SentinelAgent", tools=[_dummy_user_tool])
    task = Task(description="Anything.")

    collected = []
    async for event in agent.astream(task, events=True, stream_final_answer=True):
        collected.append(event)

    fa_starts = [e for e in collected if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 1
    assert fa_starts[0].triggered_by == "sentinel"


@pytest.mark.asyncio
async def test_sentinel_call_suppresses_tool_call_event():
    """Sentinel must NOT appear as a ToolCallEvent or ToolResultEvent in the stream."""
    from upsonic.run.events.events import ToolCallEvent, ToolResultEvent

    model = _TwoTurnSentinelModel()
    agent = Agent(model=model, name="NoSentinelEventAgent", tools=[_dummy_user_tool])
    task = Task(description="Anything.")

    collected = []
    async for event in agent.astream(task, events=True, stream_final_answer=True):
        collected.append(event)

    sentinel_tool_calls = [
        e for e in collected
        if isinstance(e, ToolCallEvent) and e.tool_name == FINAL_ANSWER_MARKER_TOOL_NAME
    ]
    sentinel_tool_results = [
        e for e in collected
        if isinstance(e, ToolResultEvent) and e.tool_name == FINAL_ANSWER_MARKER_TOOL_NAME
    ]
    assert len(sentinel_tool_calls) == 0
    assert len(sentinel_tool_results) == 0


@pytest.mark.asyncio
async def test_sentinel_call_does_not_increment_tool_call_count():
    """Scenario 9a integrated: sentinel call must NOT increment
    self._tool_call_count even when it's the only tool the model invokes.
    """
    model = _TwoTurnSentinelModel()
    agent = Agent(
        model=model,
        name="LimitAgent",
        tools=[_dummy_user_tool],
        tool_call_limit=1,
    )
    task = Task(description="Anything.")

    # Sentinel-only run: even with tool_call_limit=1, the run completes
    # because sentinel is exempt.
    async for _event in agent.astream(task, events=True, stream_final_answer=True):
        pass

    # _tool_call_count must remain 0 — no real user tool was executed.
    assert agent._tool_call_count == 0


@pytest.mark.asyncio
async def test_sentinel_marker_precedes_text_deltas():
    """In the obey case, FinalAnswerStartEvent must appear BEFORE any
    TextDeltaEvent containing the final answer.
    """
    from upsonic.run.events.events import FinalAnswerStartEvent, TextDeltaEvent

    model = _TwoTurnSentinelModel(final_text="hello")
    agent = Agent(model=model, name="OrderingAgent", tools=[_dummy_user_tool])
    task = Task(description="Anything.")

    collected = []
    async for event in agent.astream(task, events=True, stream_final_answer=True):
        collected.append(event)

    marker_index = next(
        i for i, e in enumerate(collected) if isinstance(e, FinalAnswerStartEvent)
    )
    first_text_index = next(
        (i for i, e in enumerate(collected) if isinstance(e, TextDeltaEvent)),
        None,
    )
    assert first_text_index is not None, "Final-turn text deltas must appear"
    assert marker_index < first_text_index, (
        f"Marker (index {marker_index}) must precede first TextDeltaEvent "
        f"(index {first_text_index})"
    )


@pytest.mark.asyncio
async def test_mixed_batch_invalidates_sentinel_no_marker():
    """Parallel-batch policy: sentinel + non-sentinel tool ⇒ sentinel dropped,
    NO FinalAnswerStartEvent emitted, real tool executes normally.
    """
    from upsonic.run.events.events import (
        FinalAnswerStartEvent,
        ToolCallEvent,
    )

    model = _MixedBatchModel()
    agent = Agent(model=model, name="MixedBatchAgent", tools=[_dummy_user_tool])
    task = Task(description="Anything.")

    collected = []
    async for event in agent.astream(task, events=True, stream_final_answer=True):
        collected.append(event)

    # No marker fired
    fa_starts = [e for e in collected if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 0, (
        "Mixed batch (sentinel + real tool) must NOT emit FinalAnswerStartEvent"
    )

    # Sentinel does NOT appear as a ToolCallEvent
    sentinel_calls = [
        e for e in collected
        if isinstance(e, ToolCallEvent) and e.tool_name == FINAL_ANSWER_MARKER_TOOL_NAME
    ]
    assert len(sentinel_calls) == 0

    # Real tool DID emit a ToolCallEvent
    real_calls = [
        e for e in collected
        if isinstance(e, ToolCallEvent) and e.tool_name == "_dummy_user_tool"
    ]
    assert len(real_calls) == 1
