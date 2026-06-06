"""Unit tests for ``Direct`` streaming (``stream`` / ``astream``).

``Direct`` streams by delegating to its internal ``Agent`` running the reduced
``"direct"`` streaming pipeline (a bare streaming LLM call — no memory, tools,
reflection, reliability, or policies). The model's ``request_stream`` is patched
to canned events so the tests are deterministic and need no network or API key.
A real model object is used (only ``request_stream`` is replaced) so the model
metadata the pipeline reads is real.
"""
import os

# A real Anthropic model object is constructed for metadata; the client never
# makes a call (we patch ``request_stream``), so a placeholder key is sufficient.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-placeholder-for-construction")

import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

from upsonic import Agent, Direct, Task
from upsonic.models import infer_model, ModelResponse, TextPart
from upsonic.messages.messages import (
    PartStartEvent, PartDeltaEvent, FinalResultEvent, TextPartDelta,
)
from upsonic.usage import RequestUsage


TEST_MODEL = "anthropic/claude-haiku-4-5"


def _patched_streaming_model(chunks=("Hello", " world", "!"),
                             input_tokens=11, output_tokens=7):
    """Return (model, captured) where ``model.request_stream`` yields the given
    text ``chunks`` as streaming events; ``captured`` records the messages the
    model was streamed with."""
    model = infer_model(TEST_MODEL)
    captured: dict = {}

    @asynccontextmanager
    async def fake_request_stream(messages=None, model_settings=None,
                                  model_request_parameters=None, **kwargs):
        captured["messages"] = messages

        events = [PartStartEvent(index=0, part=TextPart(content=chunks[0]))]
        for chunk in chunks[1:]:
            events.append(
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=chunk))
            )
        events.append(FinalResultEvent(tool_name=None, tool_call_id=None))

        stream_mock = AsyncMock()
        stream_mock.__aenter__ = AsyncMock(return_value=stream_mock)
        stream_mock.__aexit__ = AsyncMock(return_value=None)

        async def _aiter(_self):
            for event in events:
                yield event

        stream_mock.__aiter__ = _aiter
        stream_mock.get = Mock(return_value=ModelResponse(
            parts=[TextPart(content="".join(chunks))],
            model_name="claude-haiku-4-5",
            timestamp=time.time(),
            usage=RequestUsage(
                input_tokens=input_tokens, output_tokens=output_tokens, details={}
            ),
            provider_name="anthropic",
            provider_response_id="resp-stream",
            provider_details={},
            finish_reason="stop",
        ))

        yield stream_mock

    model.request_stream = fake_request_stream
    return model, captured


class _FakeState:
    """Minimal stand-in for a Graph State exposing get_task_output."""
    def __init__(self, mapping):
        self._mapping = mapping

    def get_task_output(self, task_id):
        return self._mapping.get(task_id)


# --------------------------------------------------------------------------
# astream / stream yield the streamed text, reassembled in order
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_direct_astream_yields_text():
    model, _ = _patched_streaming_model(("Red", ", Blue", ", Yellow"))
    direct = Direct(model=model, print=False)
    chunks = [c async for c in direct.astream(Task("colors"))]
    assert all(isinstance(c, str) for c in chunks)
    assert "".join(chunks) == "Red, Blue, Yellow"


def test_direct_stream_yields_text():
    model, _ = _patched_streaming_model(("Hi", " there"))
    direct = Direct(model=model, print=False)
    chunks = list(direct.stream(Task("greet")))
    assert "".join(chunks) == "Hi there"


# --------------------------------------------------------------------------
# events=True yields event objects, not plain text
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_direct_astream_events_yields_event_objects():
    model, _ = _patched_streaming_model(("a", "b"))
    direct = Direct(model=model, print=False)
    items = [e async for e in direct.astream(Task("x"), events=True)]
    assert items, "events mode must yield at least one event"
    assert any(not isinstance(e, str) for e in items), \
        "events=True must yield stream-event objects, not plain text"


# --------------------------------------------------------------------------
# Direct streams through the reduced "direct" streaming pipeline
# --------------------------------------------------------------------------
def test_direct_uses_lean_streaming_profile():
    model, _ = _patched_streaming_model()
    agent = Direct(model=model, print=False)._build_internal_agent()

    lean = [s.name for s in agent._select_streaming_pipeline_steps()]
    full = [s.name for s in agent._create_streaming_pipeline_steps()]

    assert len(lean) == 10
    assert len(full) == 22
    assert set(lean).issubset(set(full))
    assert {"stream_model_execution", "stream_finalization"} <= set(lean)

    # Every optional/expensive step is dropped — a bare streaming LLM call with
    # no memory (so no chat-history step; context_build stays for context/Graph).
    for dropped in (
        "tool_setup", "reflection", "reliability", "user_policy", "agent_policy",
        "cache_check", "cache_storage", "storage_connection", "llm_manager",
        "memory_prepare", "chat_history", "stream_memory_message_tracking",
    ):
        assert dropped not in lean, f"lean streaming must drop {dropped}"
    assert "context_build" in lean, "context_build must stay (powers context/Graph)"

    default_agent = Agent(TEST_MODEL, print=False)
    assert default_agent._pipeline_profile == "agent"
    assert len(default_agent._select_streaming_pipeline_steps()) == 22


# --------------------------------------------------------------------------
# Streaming usage delegates to the internal Agent and matches a standalone Agent
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_direct_stream_usage_matches_agent_stream_usage():
    direct = Direct(model=_patched_streaming_model()[0], print=False)
    async for _ in direct.astream(Task("hi")):
        pass
    du = direct.usage

    agent = Agent(model=_patched_streaming_model()[0], memory=None, tools=[], reflection=False)
    async for _ in agent.astream(Task("hi")):
        pass
    au = agent.usage

    assert du.input_tokens == au.input_tokens == 11
    assert du.output_tokens == au.output_tokens == 7
    assert du.requests == au.requests == 1


# --------------------------------------------------------------------------
# astream threads Graph state so TaskOutputSource resolves and reaches the model
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_direct_astream_threads_state_for_taskoutputsource():
    from upsonic.context.sources import TaskOutputSource

    model, captured = _patched_streaming_model()
    direct = Direct(model=model, print=False)
    state = _FakeState({"upstream": "UPSTREAM_RESULT_42"})
    task = Task(
        "Continue from the previous step.",
        context=[TaskOutputSource(task_description_or_id="upstream")],
    )
    async for _ in direct.astream(task, state=state):
        pass
    assert "UPSTREAM_RESULT_42" in str(captured["messages"]), \
        "astream must thread state so TaskOutputSource resolves and reaches the model"
