"""Unit tests for stream_final_answer on cache-hit and policy-blocked paths.

Covers:
  - Scenario 5 (cache hit): with task._cached_result=True, the per-character
    cached text stream is preceded by FinalAnswerStartEvent(triggered_by='cache_hit').
    The marker fires before any TextDeltaEvent.
  - Scenario 6 (policy blocked): with task._policy_blocked=True, NO
    FinalAnswerStartEvent is emitted. FinalOutputEvent(output_type='blocked')
    is the sole terminal signal.

Both scenarios stub the relevant flags on the Task so no live model call
is needed. The pipeline branches in StreamModelExecutionStep.execute_stream
are exercised directly.
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
    PartStartEvent,
    TextPart,
)
from upsonic.models import Model
from upsonic.tasks.tasks import Task
from upsonic.tools import tool
from upsonic.usage import RequestUsage


@tool
def _dummy_user_tool() -> str:
    """Satisfies the ≥1-user-tool injection precondition."""
    return "ok"


class _MinimalModel(Model):
    """A model that never gets called when the task is cache-hit or policy-blocked
    — the StreamModelExecutionStep.execute_stream branches return before
    invoking request_stream(). But the agent still constructs the model
    object during pipeline setup, so it must be a valid Model subclass.
    """

    def __init__(self):
        super().__init__()

    @property
    def model_name(self) -> str:
        return "stub-model"

    @property
    def system(self) -> str:
        return "stub"

    async def request(self, messages, model_settings, model_request_parameters):
        # Should not be called in cache/policy paths.
        return ModelResponse(
            parts=[TextPart(content="should not be reached")],
            model_name=self.model_name,
            timestamp=time.time(),
            usage=RequestUsage(input_tokens=0, output_tokens=0, details={}),
            provider_name="stub",
            provider_response_id="x",
            provider_details={},
            finish_reason="stop",
        )

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters):
        stream_mock = AsyncMock()
        stream_mock.__aenter__ = AsyncMock(return_value=stream_mock)
        stream_mock.__aexit__ = AsyncMock(return_value=None)

        async def _iter(_self):
            for event in []:
                yield event

        stream_mock.__aiter__ = _iter
        stream_mock.get = Mock(return_value=ModelResponse(
            parts=[TextPart(content="")],
            model_name=self.model_name,
            timestamp=time.time(),
            usage=RequestUsage(input_tokens=0, output_tokens=0, details={}),
            provider_name="stub",
            provider_response_id="x",
            provider_details={},
            finish_reason="stop",
        ))
        try:
            yield stream_mock
        finally:
            pass


# ---------------------------------------------------------------------------
# Scenario 5: cache hit — marker fires before per-char text deltas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_marker_precedes_text_deltas():
    """When task._cached_result is True, FinalAnswerStartEvent(triggered_by=
    'cache_hit') is the FIRST event in the per-character text stream.
    """
    from upsonic.run.events.events import (
        FinalAnswerStartEvent,
        FinalOutputEvent,
        TextDeltaEvent,
    )
    from upsonic.run.agent.output import AgentRunOutput

    agent = Agent(
        model=_MinimalModel(),
        name="CacheHitAgent",
        tools=[_dummy_user_tool],
    )
    task = Task(description="Anything.")

    # Construct the streaming step and a minimal AgentRunOutput context.
    # We invoke execute_stream directly so we exercise the cache branch
    # without going through the full pipeline (which has its own cache
    # detection step that we don't need here).
    from upsonic.agent.pipeline.steps import StreamModelExecutionStep

    cached_text = "Cached final answer."
    run_output = AgentRunOutput(
        run_id="r1",
        agent_id="a1",
        is_streaming=True,
    )
    run_output.output = cached_text

    # Inject cache hit signal.
    task._cached_result = True

    # Activate the per-run flag manually (mirrors what astream does).
    agent._stream_final_answer_active = True
    agent._final_answer_marker_emitted = False

    step = StreamModelExecutionStep()
    collected = []
    async for event in step.execute_stream(run_output, task, agent, _MinimalModel()):
        collected.append(event)

    fa_starts = [e for e in collected if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 1, f"Expected exactly one FinalAnswerStartEvent, got {len(fa_starts)}"
    assert fa_starts[0].triggered_by == "cache_hit"

    # FinalAnswerStartEvent must come BEFORE any TextDeltaEvent in the stream.
    fa_idx = collected.index(fa_starts[0])
    text_indices = [i for i, e in enumerate(collected) if isinstance(e, TextDeltaEvent)]
    assert text_indices, "Expected at least one TextDeltaEvent in cache-hit stream"
    assert all(i > fa_idx for i in text_indices), (
        "Every TextDeltaEvent must come AFTER FinalAnswerStartEvent in cache-hit path"
    )

    # FinalOutputEvent still fires retrospectively with output_type='cached'.
    final_outputs = [e for e in collected if isinstance(e, FinalOutputEvent)]
    assert len(final_outputs) == 1
    assert final_outputs[0].output_type == "cached"


@pytest.mark.asyncio
async def test_cache_hit_no_marker_when_flag_off():
    """When stream_final_answer is False, cache-hit path emits NO marker."""
    from upsonic.run.events.events import FinalAnswerStartEvent, FinalOutputEvent
    from upsonic.run.agent.output import AgentRunOutput

    agent = Agent(
        model=_MinimalModel(),
        name="CacheHitNoFlagAgent",
        tools=[_dummy_user_tool],
    )
    task = Task(description="Anything.")
    cached_text = "Cached."

    run_output = AgentRunOutput(
        run_id="r1", agent_id="a1", is_streaming=True,
    )
    run_output.output = cached_text
    task._cached_result = True

    # Flag NOT set → marker must not fire.
    agent._stream_final_answer_active = False
    agent._final_answer_marker_emitted = False

    from upsonic.agent.pipeline.steps import StreamModelExecutionStep
    step = StreamModelExecutionStep()

    collected = []
    async for event in step.execute_stream(run_output, task, agent, _MinimalModel()):
        collected.append(event)

    fa_starts = [e for e in collected if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 0
    # Existing FinalOutputEvent still works as today.
    final_outputs = [e for e in collected if isinstance(e, FinalOutputEvent)]
    assert len(final_outputs) == 1
    assert final_outputs[0].output_type == "cached"


# ---------------------------------------------------------------------------
# Scenario 6: policy blocked — NO marker; FinalOutputEvent(blocked) only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_blocked_emits_no_marker():
    """When task._policy_blocked is True, FinalAnswerStartEvent must NOT
    be emitted regardless of the flag. The pipeline returns immediately
    with FinalOutputEvent(output_type='blocked').
    """
    from upsonic.run.events.events import FinalAnswerStartEvent, FinalOutputEvent
    from upsonic.run.agent.output import AgentRunOutput

    agent = Agent(
        model=_MinimalModel(),
        name="PolicyBlockedAgent",
        tools=[_dummy_user_tool],
    )
    task = Task(description="Anything.")
    run_output = AgentRunOutput(
        run_id="r1", agent_id="a1", is_streaming=True,
    )

    task._policy_blocked = True

    # Activate the flag — gate must STILL suppress the marker for
    # policy-blocked runs (correctness invariant).
    agent._stream_final_answer_active = True
    agent._final_answer_marker_emitted = False

    from upsonic.agent.pipeline.steps import StreamModelExecutionStep
    step = StreamModelExecutionStep()

    collected = []
    async for event in step.execute_stream(run_output, task, agent, _MinimalModel()):
        collected.append(event)

    fa_starts = [e for e in collected if isinstance(e, FinalAnswerStartEvent)]
    assert len(fa_starts) == 0, (
        "Policy-blocked path must NEVER emit FinalAnswerStartEvent, "
        f"even with the flag on. Got: {fa_starts}"
    )

    final_outputs = [e for e in collected if isinstance(e, FinalOutputEvent)]
    assert len(final_outputs) == 1
    assert final_outputs[0].output_type == "blocked"
    assert final_outputs[0].output is None
