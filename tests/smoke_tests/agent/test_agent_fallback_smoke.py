"""Smoke tests for `Agent.fallback_model` end-to-end against REAL providers.

These tests deliberately corrupt the OpenAI API key (or pass an invalid model
identifier) so the primary call returns HTTP 401/404, then verify that the
configured Anthropic fallback actually serves the request. Uses minimal
prompts to minimise token cost.

For each test we triangulate three evidence channels:
- Logs: `caplog` records for the `ModelFallback` warning.
- Outputs: actual response text from Anthropic + `response.model_name`.
- Introspection: `agent.model._settings`, `agent.fallback_model._settings`,
  property round-trip values.

Requires:
- Valid ``ANTHROPIC_API_KEY`` in env / `.env`.
- The ``models`` extra installed: ``uv sync --extra models``.

Run with: uv run pytest tests/smoke_tests/agent/test_agent_fallback_smoke.py -v -s
"""

from __future__ import annotations

import logging
import os

import pytest

from upsonic import Agent, Task
from upsonic.models.instrumented import InstrumentedModel
from upsonic.utils.package.exception import ModelHTTPError


_INVALID_OPENAI_KEY = "sk-INVALID-FORCED-FOR-FALLBACK-SMOKE-TEST-do-not-use"


def _require_anthropic_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — smoke test requires a real Anthropic key")


def _require_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — control test requires a real OpenAI key")


def _fallback_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "ModelFallback" in r.getMessage()]


# --------------------------------------------------------------------------- #
# 1. Basic auth fallback (401 from invalid OpenAI key)                        #
# --------------------------------------------------------------------------- #


def test_smoke_agent_falls_back_to_anthropic_on_broken_openai(monkeypatch, caplog):
    _require_anthropic_key()
    monkeypatch.setenv("OPENAI_API_KEY", _INVALID_OPENAI_KEY)
    caplog.set_level(logging.WARNING)

    agent = Agent(
        model="openai/gpt-4o-mini",
        fallback_model="anthropic/claude-haiku-4-5",
    )

    # Parallel-field wiring (no wrapper class).
    assert agent.fallback_model is not None
    assert agent.model.model_name.startswith("gpt-")
    assert agent.fallback_model.model_name.startswith("claude-")

    task = Task("Reply with exactly the single word: ok")
    result = agent.do(task)

    # Output: real text from Anthropic.
    assert isinstance(result, str)
    assert "ok" in result.lower()

    # Logs: ModelFallback warning with all expected substrings.
    warnings = _fallback_warnings(caplog)
    assert len(warnings) >= 1, f"no ModelFallback warning. All warnings: {[r.getMessage() for r in caplog.records]}"
    msg = warnings[0].getMessage()
    assert "gpt-4o-mini" in msg
    assert "status=401" in msg
    assert "claude-haiku-4-5" in msg


# --------------------------------------------------------------------------- #
# 2. Wrap stack under `instrument=True`                                       #
# --------------------------------------------------------------------------- #


def test_smoke_agent_fallback_with_instrument_true(monkeypatch, caplog):
    """`instrument=True` wraps `self.model` with `InstrumentedModel`. The
    parallel-field design routes through `_request_with_fallback` which reads
    `self.model` (the instrumented wrapper) and `self.fallback_model` (raw
    Anthropic) — fallback still fires on 401.
    """
    _require_anthropic_key()
    monkeypatch.setenv("OPENAI_API_KEY", _INVALID_OPENAI_KEY)
    caplog.set_level(logging.WARNING)

    agent = Agent(
        model="openai/gpt-4o-mini",
        fallback_model="anthropic/claude-haiku-4-5",
        instrument=True,
    )

    # Primary is wrapped by InstrumentedModel; fallback is the raw Anthropic.
    assert isinstance(agent.model, InstrumentedModel)
    assert agent.fallback_model is not None
    # The fallback itself should NOT be instrumented (only primary is wrapped).
    assert not isinstance(agent.fallback_model, InstrumentedModel)

    task = Task("Reply with exactly the single word: ok")
    result = agent.do(task)

    assert isinstance(result, str)
    assert "ok" in result.lower()

    warnings = _fallback_warnings(caplog)
    assert len(warnings) >= 1


# --------------------------------------------------------------------------- #
# 3. Negative control: valid keys, primary serves, no fallback warning        #
# --------------------------------------------------------------------------- #


def test_smoke_agent_primary_serves_when_keys_valid(caplog):
    _require_anthropic_key()
    _require_openai_key()
    caplog.set_level(logging.WARNING)

    agent = Agent(
        model="openai/gpt-4o-mini",
        fallback_model="anthropic/claude-haiku-4-5",
    )

    task = Task("Reply with exactly the single word: ok")
    result = agent.do(task)

    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # Primary served → no ModelFallback warning fired.
    assert _fallback_warnings(caplog) == []


# --------------------------------------------------------------------------- #
# 4. Per-call model override inherits the construction-time fallback          #
# --------------------------------------------------------------------------- #


def test_smoke_agent_per_call_override_inherits_fallback(monkeypatch, caplog):
    """User passes `agent.do(task, model="...")` per-call. The override
    swaps `self.model`; `self.fallback_model` is read at call time from
    the construction-time field, so the override still gets fallback-protected.
    """
    _require_anthropic_key()
    monkeypatch.setenv("OPENAI_API_KEY", _INVALID_OPENAI_KEY)
    caplog.set_level(logging.WARNING)

    agent = Agent(
        model="openai/gpt-4o",
        fallback_model="anthropic/claude-haiku-4-5",
    )

    task = Task("Reply with exactly the single word: ok")
    # Per-call override to a different OpenAI model — also fails with the broken key.
    result = agent.do(task, model="openai/gpt-4o-mini")

    assert isinstance(result, str)
    assert "ok" in result.lower()

    # The warning should mention the per-call override's model name (gpt-4o-mini),
    # NOT the constructor model (gpt-4o).
    warnings = _fallback_warnings(caplog)
    assert len(warnings) >= 1
    msg = warnings[0].getMessage()
    assert "gpt-4o-mini" in msg, f"warning should reference per-call override; got: {msg}"


# --------------------------------------------------------------------------- #
# 5. Settings ownership: constructor `settings` → primary only                #
# --------------------------------------------------------------------------- #


def test_smoke_agent_settings_only_on_primary_not_fallback(monkeypatch, caplog):
    """`Agent(settings=X, fallback_model=...)` writes `X` to `self.model._settings`
    only. `self.fallback_model._settings` does NOT inherit user's settings.
    Verified via introspection + the real fallback call still succeeds.
    """
    _require_anthropic_key()
    monkeypatch.setenv("OPENAI_API_KEY", _INVALID_OPENAI_KEY)
    caplog.set_level(logging.WARNING)

    agent = Agent(
        model="openai/gpt-4o-mini",
        settings={"temperature": 0.0},
        fallback_model="anthropic/claude-haiku-4-5",
    )

    # Introspection: ownership boundary
    assert agent.model._settings.get("temperature") == 0.0
    assert agent.fallback_model._settings.get("temperature") is None

    # Via properties:
    assert agent.settings.get("temperature") == 0.0
    assert agent.fallback_settings.get("temperature") is None

    # Real call: fallback still serves correctly (its own defaults).
    task = Task("Reply with exactly the single word: ok")
    result = agent.do(task)
    assert "ok" in result.lower()
    assert len(_fallback_warnings(caplog)) >= 1


# --------------------------------------------------------------------------- #
# 6. `fallback_settings` property setter — post-construction configuration    #
# --------------------------------------------------------------------------- #


def test_smoke_agent_fallback_settings_via_property_post_construction(monkeypatch, caplog):
    """User configures the fallback's settings via the `fallback_settings`
    property setter after construction. Verified via introspection +
    real call to the fallback.
    """
    _require_anthropic_key()
    monkeypatch.setenv("OPENAI_API_KEY", _INVALID_OPENAI_KEY)
    caplog.set_level(logging.WARNING)

    agent = Agent(
        model="openai/gpt-4o-mini",
        fallback_model="anthropic/claude-haiku-4-5",
    )

    # Initially, fallback's _settings has no user temperature.
    assert agent.fallback_settings.get("temperature") is None

    # Configure via property setter.
    agent.fallback_settings = {"temperature": 0.0}

    # Introspection: property reads through; primary untouched.
    assert agent.fallback_settings.get("temperature") == 0.0
    assert agent.fallback_model._settings.get("temperature") == 0.0
    assert agent.settings.get("temperature") is None  # primary unchanged

    # Real call: fallback serves with the property-set settings.
    task = Task("Reply with exactly the single word: ok")
    result = agent.do(task)
    assert "ok" in result.lower()
    assert len(_fallback_warnings(caplog)) >= 1


# --------------------------------------------------------------------------- #
# 7. `fallback_settings` setter raises when no fallback configured            #
# --------------------------------------------------------------------------- #


def test_smoke_agent_fallback_settings_setter_raises_without_fallback():
    """No real API call — pure construction + property semantics. Smoke-tier
    because we want the AttributeError path under realistic env (real env
    vars loaded by uv from .env)."""
    _require_openai_key()

    agent = Agent(model="openai/gpt-4o-mini")  # no fallback_model

    assert agent.fallback_model is None
    assert agent.fallback_settings is None

    with pytest.raises(AttributeError, match="no fallback_model configured"):
        agent.fallback_settings = {"temperature": 0.0}
    with pytest.raises(AttributeError, match="no fallback_model configured"):
        agent.fallback_profile = object()


# --------------------------------------------------------------------------- #
# 8. Reasoning settings: agent-level intent, per-target provider keys         #
# --------------------------------------------------------------------------- #


def test_smoke_agent_reasoning_per_target_provider_keys(monkeypatch, caplog):
    """`reasoning_effort` (an OpenAI-shape param) maps to
    `openai_reasoning_effort` on the OpenAI primary; the same agent-level
    param does NOT pollute the Anthropic fallback's `_settings` (anthropic
    uses different keys for thinking — which we don't enable here).
    """
    _require_anthropic_key()
    monkeypatch.setenv("OPENAI_API_KEY", _INVALID_OPENAI_KEY)
    caplog.set_level(logging.WARNING)

    agent = Agent(
        model="openai/gpt-4o-mini",
        fallback_model="anthropic/claude-haiku-4-5",
        reasoning_effort="low",
    )

    # Introspection: per-target reasoning keys
    assert "openai_reasoning_effort" in agent.model._settings
    assert agent.model._settings["openai_reasoning_effort"] == "low"
    # Fallback (anthropic) does NOT get openai_reasoning_effort
    assert "openai_reasoning_effort" not in agent.fallback_model._settings

    # Real call: fallback serves.
    task = Task("Reply with exactly the single word: ok")
    result = agent.do(task)
    assert "ok" in result.lower()
    assert len(_fallback_warnings(caplog)) >= 1


# --------------------------------------------------------------------------- #
# 9. No fallback configured → primary 401 propagates                          #
# --------------------------------------------------------------------------- #


def test_smoke_agent_propagates_when_no_fallback_configured(monkeypatch, caplog):
    """No `fallback_model` → broken OpenAI key raises ModelHTTPError(401)
    instead of being silently masked."""
    monkeypatch.setenv("OPENAI_API_KEY", _INVALID_OPENAI_KEY)
    caplog.set_level(logging.WARNING)

    agent = Agent(model="openai/gpt-4o-mini")  # NO fallback

    task = Task("hi")
    with pytest.raises(ModelHTTPError) as excinfo:
        agent.do(task)

    assert excinfo.value.status_code == 401
    # No ModelFallback warning should fire (no fallback to switch to).
    assert _fallback_warnings(caplog) == []
