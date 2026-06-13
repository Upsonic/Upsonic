"""Unit tests for the @retryable decorator semantics (QA bug batch F1).

Covers three contracts:
  F1a — ExecutionTimeoutError is retried (it is a transient failure).
  F1b — GuardrailValidationError is NOT retried, but emits an actionable warning.
  F1c — an explicitly-passed call param (retry=) overrides the instance attribute.

The decorator is exercised through a minimal stub that mirrors how Agent wires it
(``self.retry`` set in ``__init__``; ``retries_from_param="retry"``), so the tests
stay free of model/network dependencies.
"""
import pytest

import upsonic.utils.printing as printing
from upsonic.utils.retry import retryable
from upsonic.exceptions import ExecutionTimeoutError
from upsonic.utils.package.exception import GuardrailValidationError


class _Stub:
    """Mirrors the Agent retry wiring: instance ``retry`` + per-call ``retry`` param."""

    def __init__(self, retry=1):
        self.retry = retry
        self.calls = 0

    @retryable(retries_from_param="retry")
    async def run(self, fail_times=0, exc=RuntimeError, retry=None):
        self.calls += 1
        if self.calls <= fail_times:
            raise exc("boom")
        return "ok"


@pytest.mark.asyncio
async def test_per_call_retry_overrides_instance_default():
    # F1c: instance default is 1, but the call passes retry=3 → 3 attempts.
    s = _Stub(retry=1)
    assert await s.run(fail_times=2, retry=3) == "ok"
    assert s.calls == 3


@pytest.mark.asyncio
async def test_instance_retry_used_when_no_call_param():
    # F1c: no call param → fall back to instance attribute (3).
    s = _Stub(retry=3)
    assert await s.run(fail_times=2) == "ok"
    assert s.calls == 3


@pytest.mark.asyncio
async def test_explicit_none_falls_back_to_instance():
    # retry=None is the "not specified" sentinel → use instance attribute.
    s = _Stub(retry=2)
    assert await s.run(fail_times=1, retry=None) == "ok"
    assert s.calls == 2


@pytest.mark.asyncio
async def test_timeout_is_retried():
    # F1a: a timeout on the first attempt is retried and the second attempt succeeds.
    s = _Stub(retry=1)
    assert await s.run(fail_times=1, exc=ExecutionTimeoutError, retry=2) == "ok"
    assert s.calls == 2


@pytest.mark.asyncio
async def test_guardrail_not_retried_but_warned(monkeypatch):
    # F1b: guardrail error is not retried (1 attempt) and a warning is emitted
    # before re-raise pointing to Task(guardrail_retries=N).
    warnings = []
    monkeypatch.setattr(
        printing, "warning_log", lambda msg, ctx="Upsonic": warnings.append(msg)
    )

    s = _Stub(retry=1)
    with pytest.raises(GuardrailValidationError):
        await s.run(fail_times=5, exc=GuardrailValidationError, retry=3)

    assert s.calls == 1
    assert any("guardrail_retries" in w for w in warnings), warnings
