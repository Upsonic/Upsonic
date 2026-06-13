"""Heartbeat console-output opt-in (QA bug batch F4 / heartbeat).

execute_heartbeat used to force the print flags to False unconditionally, so a
heartbeat could never render console panels. The new print= parameter opts in
without changing the silent default.
"""
import pytest

from upsonic.agent.autonomous_agent.autonomous_agent import AutonomousAgent


class _HeartbeatStub:
    """Minimal stand-in exposing only what aexecute_heartbeat touches."""

    def __init__(self) -> None:
        self.heartbeat = True
        self.heartbeat_message = "ping"
        self._print_param = None
        self.print = True  # agent is configured to print normally
        self.captured_print = None
        self.captured_kwargs = None

    async def do_async(self, task, **kwargs):
        # Capture the print state and kwargs at the moment of the call.
        self.captured_print = self.print
        self.captured_print_param = self._print_param
        self.captured_kwargs = kwargs

    def get_run_output(self):
        return None  # short-circuit; we only care about how do_async was invoked


@pytest.mark.asyncio
async def test_heartbeat_default_is_silent():
    stub = _HeartbeatStub()
    await AutonomousAgent.aexecute_heartbeat(stub)
    # Default suppresses: print forced False during the run, _print_method_default=False.
    assert stub.captured_print is False
    assert stub.captured_print_param is False
    assert stub.captured_kwargs.get("_print_method_default") is False
    # Flags restored afterwards.
    assert stub.print is True
    assert stub._print_param is None


@pytest.mark.asyncio
async def test_heartbeat_print_true_renders():
    stub = _HeartbeatStub()
    await AutonomousAgent.aexecute_heartbeat(stub, print=True)
    # print=True does NOT force the suppression flags and does not pass the
    # silencing _print_method_default override.
    assert stub.captured_print is True
    assert "_print_method_default" not in stub.captured_kwargs
