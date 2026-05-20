"""Unit tests for `Agent.fallback_model` (parallel-field design).

Tests verify the parallel-field wiring:
- `self.model` and `self.fallback_model` are resolved identically via `infer_model`.
- `_request_with_fallback(...)` retries on `ModelHTTPError(401/403/404)`.
- User-supplied settings/profile apply symmetrically at construction.
- Cross-provider reasoning settings are keyed per-target (Q7 refactor).
- Properties (`settings`/`profile`/`fallback_settings`/`fallback_profile`) round-trip.
- Per-call model override inherits the construction-time fallback.
- Streaming (`astream`) does NOT trigger fallback — locked gap.

Run with: uv run pytest tests/unit_tests/agent/test_agent_fallback.py -v
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import patch

import pytest

from upsonic import Agent
from upsonic.messages import ModelMessage, ModelResponse, TextPart
from upsonic.models import Model, ModelRequestParameters
from upsonic.models.settings import ModelSettings
from upsonic.utils.package.exception import ModelAPIError, ModelHTTPError


# --------------------------------------------------------------------------- #
# Test helpers                                                                #
# --------------------------------------------------------------------------- #


def _make_response(model_name: str = "stub", text: str = "ok") -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)], model_name=model_name)


def _params() -> ModelRequestParameters:
    return ModelRequestParameters()


class _StubModel(Model):
    """Minimal `Model` for Agent fallback tests."""

    def __init__(
        self,
        *,
        name: str = "stub",
        system: str = "openai",
        response: Optional[ModelResponse] = None,
        raises: Optional[BaseException] = None,
    ):
        super().__init__()
        self._name = name
        self._system = system
        self._response = response if response is not None else _make_response(name)
        self._raises = raises
        self.call_count = 0
        self.last_model_settings: Optional[ModelSettings] = None

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def system(self) -> str:
        return self._system

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.call_count += 1
        self.last_model_settings = model_settings
        if self._raises is not None:
            raise self._raises
        return self._response


# --------------------------------------------------------------------------- #
# 1–2. Construction wiring                                                    #
# --------------------------------------------------------------------------- #


def test_agent_no_fallback_param_unchanged():
    primary = _StubModel(name="primary")
    agent = Agent(model=primary)

    assert agent.model is primary
    assert agent.fallback_model is None


def test_agent_resolves_fallback_via_infer_model():
    primary_stub = _StubModel(name="resolved-primary")
    fallback_stub = _StubModel(name="resolved-fallback")

    def _fake_infer(spec):
        # Mirror real `infer_model`: pass-through if already a Model.
        if isinstance(spec, Model):
            return spec
        if spec == "openai/gpt-4o":
            return primary_stub
        if spec == "anthropic/claude-sonnet-4-6":
            return fallback_stub
        raise ValueError(f"unexpected spec: {spec}")

    with patch("upsonic.models.infer_model", side_effect=_fake_infer):
        agent = Agent(
            model="openai/gpt-4o",
            fallback_model="anthropic/claude-sonnet-4-6",
        )

    assert agent.model is primary_stub
    assert agent.fallback_model is fallback_stub


# --------------------------------------------------------------------------- #
# 3–4. Trigger matrix                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403, 404])
async def test_agent_request_returns_fallback_response_on_auth_error(status_code, caplog):
    """Parameterised over the trigger set: each status code fires fallback,
    warning_log emits with the expected message."""
    import logging

    fb_resp = _make_response("fallback", "from-fallback")
    primary = _StubModel(
        name="primary",
        raises=ModelHTTPError(status_code=status_code, model_name="primary"),
    )
    fallback = _StubModel(name="fallback", response=fb_resp)
    agent = Agent(model=primary, fallback_model=fallback)

    caplog.set_level(logging.WARNING)
    result = await agent._request_with_fallback([], None, _params())

    assert result is fb_resp
    assert primary.call_count == 1
    assert fallback.call_count == 1
    # Warning log emission
    fb_warnings = [r for r in caplog.records if "ModelFallback" in r.getMessage()]
    assert len(fb_warnings) >= 1
    msg = fb_warnings[0].getMessage()
    assert "primary" in msg
    assert f"status={status_code}" in msg
    assert "fallback" in msg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        ModelHTTPError(status_code=400, model_name="p", body={"__type": "ValidationException"}),
        ModelHTTPError(status_code=429, model_name="p"),
        ModelHTTPError(status_code=500, model_name="p"),
        ModelAPIError(model_name="p", message="dns failure"),
        ValueError("boom"),
    ],
)
async def test_agent_request_propagates_non_auth_error(exc):
    primary = _StubModel(raises=exc)
    fallback = _StubModel()
    agent = Agent(model=primary, fallback_model=fallback)

    with pytest.raises(type(exc)):
        await agent._request_with_fallback([], None, _params())

    assert fallback.call_count == 0


# --------------------------------------------------------------------------- #
# 5–6. Public-API path                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_401_propagates_without_fallback():
    """No fallback configured; agent.do(task) end-to-end propagates 401."""
    from upsonic.tasks.tasks import Task

    primary = _StubModel(
        raises=ModelHTTPError(status_code=401, model_name="primary"),
    )
    agent = Agent(model=primary)

    with pytest.raises(ModelHTTPError) as excinfo:
        await agent.do_async(Task("hi"))

    assert excinfo.value.status_code == 401


@pytest.mark.asyncio
async def test_agent_per_call_override_inherits_fallback():
    """`agent.do(task, model=...)` swaps the primary; the construction-time
    fallback is read at call time so the per-call primary still gets retry.
    """
    from upsonic.tasks.tasks import Task

    fb_resp = _make_response("fallback", "served-by-fallback")
    primary = _StubModel(name="primary")
    fallback = _StubModel(name="fallback", response=fb_resp)
    agent = Agent(model=primary, fallback_model=fallback)

    per_call_primary = _StubModel(
        name="per-call",
        raises=ModelHTTPError(status_code=401, model_name="per-call"),
    )

    def _fake_infer(spec):
        if isinstance(spec, Model):
            return spec
        if spec == "per-call/x":
            return per_call_primary
        raise ValueError(f"unexpected spec: {spec}")

    with patch("upsonic.models.infer_model", side_effect=_fake_infer):
        result = await agent.do_async(Task("hi"), model="per-call/x")

    assert "served-by-fallback" in str(result)
    assert per_call_primary.call_count == 1
    assert fallback.call_count == 1


# --------------------------------------------------------------------------- #
# 7. Settings/profile ownership: primary only at construction;                #
#    reasoning settings: agent-level intent applied per-target                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "primary_system,fallback_system",
    [("openai", "anthropic"), ("anthropic", "openai")],
)
def test_agent_settings_apply_to_primary_only_at_construction(primary_system, fallback_system):
    """The constructor's ``settings`` / ``profile`` params describe the
    PRIMARY model only — they do NOT leak to ``self.fallback_model``. Users
    configure the fallback's settings post-construction via the
    ``fallback_settings`` / ``fallback_profile`` properties.

    Reasoning settings (``reasoning_effort``, ``thinking_enabled``, ...) are
    agent-level intent: each model receives its own provider-shape reasoning
    dict via ``_get_model_specific_reasoning_settings_for(target)``.
    """
    p = _StubModel(name="p", system=primary_system)
    f = _StubModel(name="f", system=fallback_system)
    agent = Agent(
        model=p,
        settings={"temperature": 0.5},
        fallback_model=f,
        reasoning_effort="high",
        thinking_enabled=True,
        thinking_budget=1024,
    )

    # User-supplied `settings` (`temperature`) lands ONLY on primary.
    assert p._settings.get("temperature") == 0.5
    assert f._settings.get("temperature") is None  # fallback does NOT inherit primary's settings

    # Reasoning settings are agent-level, applied per-target with provider-shape keys:
    if primary_system == "openai":
        assert "openai_reasoning_effort" in p._settings
        assert "openai_reasoning_effort" not in f._settings
        assert "anthropic_thinking" in f._settings
        assert "anthropic_thinking" not in p._settings
    else:  # primary is anthropic, fallback is openai
        assert "anthropic_thinking" in p._settings
        assert "anthropic_thinking" not in f._settings
        assert "openai_reasoning_effort" in f._settings
        assert "openai_reasoning_effort" not in p._settings


def test_agent_fallback_settings_property_configures_fallback_post_construction():
    """The recommended user flow for configuring fallback's settings: pass
    ``settings=...`` to the constructor (lands on primary), then use the
    ``fallback_settings`` property setter for the fallback.
    """
    p = _StubModel(name="p")
    f = _StubModel(name="f")
    agent = Agent(model=p, settings={"temperature": 0.5}, fallback_model=f)

    # After construction: primary has user's settings, fallback does not.
    assert agent.settings.get("temperature") == 0.5
    assert agent.fallback_settings.get("temperature") is None

    # User explicitly configures fallback via the property:
    agent.fallback_settings = {"temperature": 0.7}
    assert agent.fallback_settings == {"temperature": 0.7}
    assert agent.settings.get("temperature") == 0.5  # primary unchanged


# --------------------------------------------------------------------------- #
# 8. Properties round-trip + AttributeError on no-fallback                    #
# --------------------------------------------------------------------------- #


def test_agent_properties_round_trip():
    p = _StubModel(name="p")
    f = _StubModel(name="f")
    agent = Agent(model=p, fallback_model=f)

    # Round-trip: settings
    agent.settings = {"temperature": 0.8}
    assert agent.settings == {"temperature": 0.8}
    assert p._settings == {"temperature": 0.8}

    # Round-trip: fallback_settings, independent of settings
    agent.fallback_settings = {"temperature": 0.9}
    assert agent.fallback_settings == {"temperature": 0.9}
    assert f._settings == {"temperature": 0.9}
    assert agent.settings == {"temperature": 0.8}  # primary unchanged

    # Round-trip: profile + fallback_profile
    sentinel_p = object()
    sentinel_f = object()
    agent.profile = sentinel_p
    agent.fallback_profile = sentinel_f
    assert agent.profile is sentinel_p
    assert agent.fallback_profile is sentinel_f
    assert p._profile is sentinel_p
    assert f._profile is sentinel_f

    # No-fallback Agent raises on fallback_* setters
    agent_no_fb = Agent(model=_StubModel(name="solo"))
    with pytest.raises(AttributeError, match="no fallback_model configured"):
        agent_no_fb.fallback_settings = {"temperature": 0.1}
    with pytest.raises(AttributeError, match="no fallback_model configured"):
        agent_no_fb.fallback_profile = object()


# --------------------------------------------------------------------------- #
# 9. Streaming-fallback gap LOCKED                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_astream_does_not_fall_back_on_auth_error():
    """`astream()` goes through `model.request_stream(...)`, which is NOT
    routed through `_request_with_fallback`. This locks the documented
    streaming-fallback gap (see fallback_model docstring + agent.md).

    A future contributor adding streaming fallback must update this test
    consciously.
    """
    from contextlib import asynccontextmanager

    from upsonic.tasks.tasks import Task

    class _StreamErrorStub(_StubModel):
        @asynccontextmanager
        async def request_stream(self, messages, model_settings, model_request_parameters):
            raise ModelHTTPError(status_code=401, model_name=self._name)
            yield  # unreachable — keeps the type checker happy

    class _StreamOkStub(_StubModel):
        @asynccontextmanager
        async def request_stream(self, messages, model_settings, model_request_parameters):
            yield _make_response(self._name, "streamed-ok")

    primary = _StreamErrorStub(name="primary")
    fallback = _StreamOkStub(name="fallback")
    agent = Agent(model=primary, fallback_model=fallback)

    with pytest.raises(ModelHTTPError) as excinfo:
        async for _ in agent.astream(Task("hi")):
            pass

    assert excinfo.value.status_code == 401
    # Fallback NEVER touched — gap is locked.
    assert fallback.call_count == 0
