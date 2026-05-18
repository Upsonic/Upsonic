"""Test-only helper for injecting errors into pipeline steps.

Replaces the in-tree ``_ERROR_INJECTIONS`` global that used to live in
``src/upsonic/agent/pipeline/step.py``. The public API (``inject_error_into_step``,
``clear_error_injection``) is unchanged so existing tests only need to switch
their import path.

Mechanism: on first ``inject_error_into_step`` call we monkey-patch
``Step.run`` to consult an in-process registry; ``clear_error_injection``
restores the original ``Step.run`` once the registry is empty so other tests
are unaffected. Production code carries no test-specific hooks.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from upsonic.agent.pipeline.step import Step, StepResult, StepStatus

_INJECTIONS: Dict[str, Dict[str, Any]] = {}
_ORIGINAL_RUN = Step.run


async def _patched_run(
    self,
    context,
    task,
    agent,
    model,
    step_number,
    pipeline_manager=None,
):
    injection = _INJECTIONS.get(self.name)
    if injection is not None and injection["triggered"] < injection["trigger_count"]:
        injection["triggered"] += 1
        msg = injection["message"]
        if "INJECTED ERROR" not in msg:
            msg = f"INJECTED ERROR: {msg}"
        exc = injection["exception_type"](msg)
        # Mirror the old in-tree behaviour: finalize an ERROR ``StepResult``
        # on the context before raising, so ``get_problematic_step()`` /
        # ``continue_run_async`` can find a resume point downstream.
        err_result = StepResult(
            name=self.name,
            step_number=step_number,
            status=StepStatus.ERROR,
            message=str(exc),
            execution_time=0.0,
        )
        self._finalize_step_result(err_result, context)
        raise exc
    return await _ORIGINAL_RUN(
        self,
        context,
        task,
        agent,
        model,
        step_number,
        pipeline_manager=pipeline_manager,
    )

def inject_error_into_step(
    step_name: str,
    exception_type: type = RuntimeError,
    message: str = "Injected error",
    trigger_count: int = 1,
) -> None:
    """Arm a per-step error trap.

    The next ``trigger_count`` invocations of ``Step.run`` whose ``self.name``
    matches ``step_name`` will raise ``exception_type(message)`` (prefixed with
    ``"INJECTED ERROR: "``) before the step body runs. Subsequent invocations
    pass through to the real implementation.
    """
    _INJECTIONS[step_name] = {
        "exception_type": exception_type,
        "message": message,
        "trigger_count": trigger_count,
        "triggered": 0,
    }
    if Step.run is _ORIGINAL_RUN:
        Step.run = _patched_run  # type: ignore[method-assign]


def clear_error_injection(step_name: Optional[str] = None) -> None:
    """Disarm one or all step traps and restore ``Step.run`` when empty."""
    if step_name is not None:
        _INJECTIONS.pop(step_name, None)
    else:
        _INJECTIONS.clear()
    if not _INJECTIONS:
        Step.run = _ORIGINAL_RUN  # type: ignore[method-assign]
