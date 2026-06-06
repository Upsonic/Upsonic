"""Unit tests for `Direct`'s observable behaviour.

Each test asserts what a caller observes — the returned value and that a task's
context/attachments reach the model request — independently of how ``Direct``
executes internally.

The model's ``request`` is patched to a canned response so the tests are
deterministic and need no network or API key. A real model object is used (only
``request`` is replaced) so the model metadata the pipeline reads is real.
"""
import os

# A real Anthropic model object is constructed for metadata; the client never
# makes a call (we patch `.request`), so a placeholder key is sufficient.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-placeholder-for-construction")

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from upsonic import Agent, Direct, Task
from upsonic.models import infer_model, ModelResponse, TextPart


TEST_MODEL = "anthropic/claude-haiku-4-5"


class Person(BaseModel):
    """Structured-output model used by the tests."""
    name: str
    age: int


class _Usage:
    def __init__(self, input_tokens=11, output_tokens=7):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def _patched_model(canned_text: str):
    """Return (model, captured) where model.request is replaced by a canned
    response and `captured` records the messages the model was called with."""
    model = infer_model(TEST_MODEL)
    captured: dict = {}

    async def fake_request(messages=None, model_settings=None,
                           model_request_parameters=None, **kwargs):
        captured["messages"] = messages
        captured["params"] = model_request_parameters
        return ModelResponse(
            parts=[TextPart(content=canned_text)],
            model_name="claude-haiku-4-5",
            timestamp="2026-01-01T00:00:00Z",
            usage=_Usage(),
            provider_name="anthropic",
            provider_response_id="resp-golden",
            provider_details={},
            finish_reason="stop",
        )

    model.request = fake_request
    return model, captured


# --------------------------------------------------------------------------
# Golden 1 — plain text output
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_golden_text_output():
    model, _ = _patched_model("The capital is Paris.")
    direct = Direct(model=model, print=False)
    result = await direct.do_async(Task("What is the capital of France?"))
    assert isinstance(result, str)
    assert result == "The capital is Paris."


# --------------------------------------------------------------------------
# Golden 2 — structured (Pydantic) output
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_golden_structured_output():
    model, _ = _patched_model('{"name": "Ada", "age": 36}')
    direct = Direct(model=model, print=False)
    result = await direct.do_async(
        Task("Extract the person.", response_format=Person)
    )
    assert isinstance(result, Person)
    assert result.name == "Ada"
    assert result.age == 36


# --------------------------------------------------------------------------
# Golden 3 — plain-string context reaches the model
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_golden_plain_string_context_reaches_model():
    model, captured = _patched_model("ok")
    direct = Direct(model=model, print=False)
    await direct.do_async(
        Task("Use the context.", context=["The secret word is BANANA."]),
    )
    assert "BANANA" in str(captured["messages"]), \
        "plain-string task.context must reach the model input"


# --------------------------------------------------------------------------
# Golden 4 — Graph TaskOutputSource resolves and reaches the model
# --------------------------------------------------------------------------
class _FakeState:
    """Minimal stand-in for a Graph State exposing get_task_output."""
    def __init__(self, mapping):
        self._mapping = mapping

    def get_task_output(self, task_id):
        return self._mapping.get(task_id)


@pytest.mark.asyncio
async def test_golden_taskoutputsource_reaches_model():
    from upsonic.context.sources import TaskOutputSource

    model, captured = _patched_model("ok")
    direct = Direct(model=model, print=False)
    state = _FakeState({"upstream": "UPSTREAM_RESULT_42"})
    task = Task(
        "Continue from the previous step.",
        context=[TaskOutputSource(task_description_or_id="upstream")],
    )
    await direct.do_async(task, state=state)
    assert "UPSTREAM_RESULT_42" in str(captured["messages"]), \
        "TaskOutputSource output must be resolved from state and reach the model input"


# --------------------------------------------------------------------------
# Golden 5 — attachments reach the model
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_golden_attachments_reach_model(tmp_path):
    img = tmp_path / "tiny.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nFAKEPNGDATA")
    model, captured = _patched_model("described")
    direct = Direct(model=model, print=False)
    await direct.do_async(
        Task("Describe the attachment.", attachments=[str(img)]),
    )
    # The binary content (or its container) must appear in the model messages.
    blob = str(captured["messages"])
    assert "BinaryContent" in blob or "image" in blob.lower(), \
        "attachment must reach the model input"


# --------------------------------------------------------------------------
# Golden 6 — sync do() wrapper returns the same content
# --------------------------------------------------------------------------
def test_golden_sync_do():
    model, _ = _patched_model("sync result")
    direct = Direct(model=model, print=False)
    result = direct.do(Task("hello"))
    assert result == "sync result"


# --------------------------------------------------------------------------
# Golden 7 — the "direct" profile selects the reduced pipeline, "agent" the full
# --------------------------------------------------------------------------
def test_direct_and_agent_pipeline_profiles():
    model, _ = _patched_model("x")
    direct = Direct(model=model, print=False)
    agent = direct._build_internal_agent()
    assert agent._pipeline_profile == "direct"
    assert len(agent._select_pipeline_steps()) == 13
    assert len(agent._create_direct_pipeline_steps()) == 13

    default_agent = type(agent)("anthropic/claude-haiku-4-5", print=False)
    assert default_agent._pipeline_profile == "agent"
    assert len(default_agent._create_agent_pipeline_steps()) == 24
    assert len(default_agent._select_pipeline_steps()) == 24


# --------------------------------------------------------------------------
# Direct's print behaviour is identical to Agent's (method default + override)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_direct_print_follows_agent_semantics():
    """Direct does no print resolution of its own; the internal Agent resolves
    it exactly like ``Agent.do``/``print_do``: do() off by default, print_do()
    on, and a constructor ``print=False`` overrides even print_do()."""
    direct = Direct(model=_patched_model("ok")[0], print=None)
    agent = direct._build_internal_agent()

    await direct.do_async(Task("hi"))
    assert agent._agent_run_output.print_flag is False   # do ⇒ method default False

    await direct.print_do_async(Task("hi"))
    assert agent._agent_run_output.print_flag is True     # print_do ⇒ method default True

    # Constructor print=False (forwarded to the Agent) forces it off even for print_do.
    direct_off = Direct(model=_patched_model("ok")[0], print=False)
    agent_off = direct_off._build_internal_agent()
    await direct_off.print_do_async(Task("hi"))
    assert agent_off._agent_run_output.print_flag is False


# --------------------------------------------------------------------------
# Metrics finalizer — task_end() fires exactly once per run in both profiles
# --------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["direct", "agent"])
async def test_task_end_called_exactly_once(profile):
    """end_time / duration are populated, and task_end() fires exactly once —
    so duration is never double-counted (the direct profile finalizes in
    CallManagementStep; the full profile in MemorySaveStep, guarded on end_time)."""
    model, _ = _patched_model("ok")
    task = Task("hi")

    # Task is a Pydantic model — patch task_end at the class level and count
    # invocations while still running the real finalizer.
    original_task_end = Task.task_end
    count = {"n": 0}

    def _spy(self):
        count["n"] += 1
        return original_task_end(self)

    with patch.object(Task, "task_end", _spy):
        if profile == "direct":
            await Direct(model=model, print=False).do_async(task)
        else:
            await Agent(model=model, memory=None, tools=[], reflection=False).do_async(task)

    assert count["n"] == 1, f"{profile}: task_end() called {count['n']} times"
    assert task.end_time is not None
    assert task._usage is not None and task._usage.duration is not None


# --------------------------------------------------------------------------
# Direct.usage delegates to the internal Agent and matches a standalone Agent
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_direct_usage_matches_agent_usage():
    """``Direct.usage`` returns the internal Agent's usage (same shape as
    ``Agent.usage``), and matches a standalone Agent running the same task with
    the same per-response usage."""
    direct = Direct(model=_patched_model("ok")[0], print=False)
    await direct.do_async(Task("hi"))
    du = direct.usage

    agent = Agent(model=_patched_model("ok")[0], memory=None, tools=[], reflection=False)
    await agent.do_async(Task("hi"))
    au = agent.usage

    # _patched_model reports input_tokens=11, output_tokens=7 per response.
    assert du.input_tokens == au.input_tokens == 11
    assert du.output_tokens == au.output_tokens == 7
    assert du.requests == au.requests == 1
