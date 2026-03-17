"""
HITL Usage Metrics Delay Comparison Tests

Tests ALL HITL types (external_tool, confirmation, user_input) with 0s vs 10s delay.

For each HITL type, runs TWICE:
  - Run A: resume immediately (no delay)
  - Run B: wait 10 seconds before resuming

Then asserts:
  - duration(B) - duration(A) >= 9 seconds
  - model_execution_time(A) ~ model_execution_time(B) (within 3s)
  - framework_overhead(A) ~ framework_overhead(B) (within 3s)
  - pause_time(B) - pause_time(A) >= 9 seconds
  - AgentRunOutput.usage == Task._usage == Storage usage (all three match)

Run with: uv run pytest tests/smoke_tests/hitl/test_usage_metrics_hitl_delay.py -v -s
"""

import pytest
import asyncio
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.usage import TaskUsage
from upsonic.db.database import SqliteDatabase

pytestmark = pytest.mark.timeout(300)

DELAY_SECONDS = 10
DB_FILE = "usage_delay_test.db"


def cleanup():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)


# ============================================================================
# TOOLS
# ============================================================================

@tool(external_execution=True)
def send_notification(message: str) -> str:
    """Send a notification externally.

    Args:
        message: Notification message

    Returns:
        Confirmation
    """
    return f"Notification sent: {message}"


@tool(requires_confirmation=True)
def delete_file(filename: str) -> str:
    """Delete a file. Requires user confirmation.

    Args:
        filename: File to delete

    Returns:
        Deletion result
    """
    return f"Deleted {filename}"


@tool(requires_user_input=True, user_input_fields=["recipient"])
def send_email(subject: str, body: str, recipient: str) -> str:
    """Send an email. Agent provides subject/body, user provides recipient.

    Args:
        subject: Email subject
        body: Email body
        recipient: Email recipient (provided by user)

    Returns:
        Confirmation
    """
    return f"Email sent to {recipient} with subject '{subject}'"


# ============================================================================
# SNAPSHOT / COMPARISON
# ============================================================================

@dataclass
class MetricSnapshot:
    duration: Optional[float]
    model_execution_time: Optional[float]
    tool_execution_time: Optional[float]
    pause_time: Optional[float]
    upsonic_execution_time: Optional[float]
    input_tokens: int
    output_tokens: int
    requests: int


def _snap(usage: TaskUsage) -> MetricSnapshot:
    return MetricSnapshot(
        duration=usage.duration,
        model_execution_time=usage.model_execution_time,
        tool_execution_time=usage.tool_execution_time,
        pause_time=usage.pause_time,
        upsonic_execution_time=usage.upsonic_execution_time,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        requests=usage.requests,
    )


def _fmt(v):
    return f"{v:.4f}" if v is not None else "None"


def _print_snap(label: str, snap: MetricSnapshot):
    print(f"\n  [{label}]")
    print(f"    duration             = {_fmt(snap.duration)}")
    print(f"    model_execution_time = {_fmt(snap.model_execution_time)}")
    print(f"    tool_execution_time  = {_fmt(snap.tool_execution_time)}")
    print(f"    pause_time           = {_fmt(snap.pause_time)}")
    print(f"    upsonic_exec_time    = {_fmt(snap.upsonic_execution_time)}")
    print(f"    input_tokens         = {snap.input_tokens}")
    print(f"    output_tokens        = {snap.output_tokens}")
    print(f"    requests             = {snap.requests}")


def _compare_snaps(a: MetricSnapshot, b: MetricSnapshot, label_a: str, label_b: str, tolerance: float = 0.5):
    """Assert two snapshots match within tolerance."""
    print(f"\n  Comparing {label_a} vs {label_b} (tolerance={tolerance}s)")

    if a.duration is not None and b.duration is not None:
        diff = abs(a.duration - b.duration)
        print(f"    duration diff       = {diff:.4f}s")
        assert diff < tolerance, f"duration mismatch: {a.duration:.4f} vs {b.duration:.4f}"

    if a.model_execution_time is not None and b.model_execution_time is not None:
        diff = abs(a.model_execution_time - b.model_execution_time)
        print(f"    model_exec diff     = {diff:.4f}s")
        assert diff < tolerance, f"model_execution_time mismatch"

    a_tool = a.tool_execution_time or 0.0
    b_tool = b.tool_execution_time or 0.0
    diff = abs(a_tool - b_tool)
    print(f"    tool_exec diff      = {diff:.4f}s")
    assert diff < tolerance, f"tool_execution_time mismatch: {a_tool:.4f} vs {b_tool:.4f}"

    a_pause = a.pause_time or 0.0
    b_pause = b.pause_time or 0.0
    diff = abs(a_pause - b_pause)
    print(f"    pause_time diff     = {diff:.4f}s")
    assert diff < tolerance, f"pause_time mismatch: {a_pause:.4f} vs {b_pause:.4f}"

    assert a.input_tokens == b.input_tokens, f"input_tokens mismatch: {a.input_tokens} vs {b.input_tokens}"
    assert a.output_tokens == b.output_tokens, f"output_tokens mismatch: {a.output_tokens} vs {b.output_tokens}"
    assert a.requests == b.requests, f"requests mismatch: {a.requests} vs {b.requests}"

    print(f"    tokens/requests     = MATCH")
    print(f"    COMPARISON PASSED")


# ============================================================================
# HITL REQUIREMENT RESOLVERS
# ============================================================================

def _resolve_external_tool(requirement):
    te = requirement.tool_execution
    result = send_notification(**te.tool_args)
    te.result = result


def _resolve_confirmation(requirement):
    requirement.confirm()


def _resolve_user_input(requirement):
    if requirement.user_input_schema:
        for field_dict in requirement.user_input_schema:
            if isinstance(field_dict, dict) and field_dict.get("value") is None:
                name = field_dict["name"]
                field_dict["value"] = f"test_{name}@example.com"
    requirement.tool_execution.answered = True


# ============================================================================
# GENERIC DELAY FLOW
# ============================================================================

async def _run_hitl_delay_flow(
    hitl_type: str,
    task_desc: str,
    tools: list,
    expected_pause_reason: str,
    resolve_fn,
    delay_before_resume: float,
    label: str,
    session_id: str,
) -> tuple:
    """
    Run a HITL pause/resume flow and return (output_snap, task_snap, storage_snap).
    All three sources of usage must match.
    """
    print(f"\n{'='*80}")
    print(f"  {label}: {hitl_type} delay={delay_before_resume}s")
    print(f"{'='*80}")

    cleanup()

    db = SqliteDatabase(db_file=DB_FILE, session_id=session_id, user_id="test_user", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name=f"agent_{label}", db=db)
    task = Task(description=task_desc, tools=tools)

    # First run -> should pause
    output = await agent.print_do_async(task, return_output=True)
    assert output.is_paused, f"[{label}] Expected paused, got {output.status}"
    assert output.pause_reason == expected_pause_reason, (
        f"[{label}] Expected {expected_pause_reason} pause, got {output.pause_reason}"
    )

    # Resolve the requirement
    for req in output.active_requirements:
        resolve_fn(req)

    # >>> THE DELAY <<<
    if delay_before_resume > 0:
        print(f"\n  [{label}] Sleeping {delay_before_resume}s before resume...")
        await asyncio.sleep(delay_before_resume)
        print(f"  [{label}] Awake. Resuming now.")

    # Resume with SAME agent
    result = await agent.continue_run_async(
        run_id=output.run_id,
        return_output=True,
        requirements=output.requirements,
    )
    assert result.is_complete, f"[{label}] Expected complete, got {result.status}"

    # === SOURCE 1: AgentRunOutput.usage ===
    output_usage = result.usage
    assert output_usage is not None, f"[{label}] AgentRunOutput.usage is None"
    output_snap = _snap(output_usage)

    # === SOURCE 2: Task._usage ===
    task_usage = task.usage
    assert task_usage is not None, f"[{label}] Task.usage is None"
    task_snap = _snap(task_usage)

    # === SOURCE 3: Storage ===
    session = db.storage.get_session(session_id=session_id)
    assert session is not None, f"[{label}] Session not in storage"
    stored_output = session.runs[result.run_id].output
    stored_usage = stored_output.usage
    assert stored_usage is not None, f"[{label}] Stored usage is None"
    if isinstance(stored_usage, dict):
        stored_usage = TaskUsage.from_dict(stored_usage)
    storage_snap = _snap(stored_usage)

    # Print all three
    _print_snap(f"{label} AgentRunOutput", output_snap)
    _print_snap(f"{label} Task", task_snap)
    _print_snap(f"{label} Storage", storage_snap)

    # === ASSERT ALL THREE MATCH ===
    _compare_snaps(output_snap, task_snap, "AgentRunOutput", "Task")
    _compare_snaps(output_snap, storage_snap, "AgentRunOutput", "Storage")

    print(f"\n  [{label}] ALL THREE SOURCES MATCH!")

    return output_snap, task_snap, storage_snap


def _assert_delay_diffs(snap_a: MetricSnapshot, snap_b: MetricSnapshot, label: str):
    """Assert that snap_b has ~10s more duration and pause_time than snap_a."""
    duration_diff = (snap_b.duration or 0) - (snap_a.duration or 0)
    model_diff = abs((snap_b.model_execution_time or 0) - (snap_a.model_execution_time or 0))
    pause_diff = (snap_b.pause_time or 0) - (snap_a.pause_time or 0)
    overhead_diff = abs((snap_b.upsonic_execution_time or 0) - (snap_a.upsonic_execution_time or 0))

    print(f"\n  [{label}] DELAY DIFFS:")
    print(f"    duration diff    = {_fmt(duration_diff)}")
    print(f"    model_exec diff  = {_fmt(model_diff)}")
    print(f"    pause_time diff  = {_fmt(pause_diff)}")
    print(f"    overhead diff    = {_fmt(overhead_diff)}")

    assert duration_diff >= 9.0, f"[{label}] duration diff should be >= 9s (got {duration_diff:.2f}s)"
    assert model_diff < 3.0, f"[{label}] model_exec diff should be < 3s (got {model_diff:.2f}s)"
    assert pause_diff >= 9.0, f"[{label}] pause_time diff should be >= 9s (got {pause_diff:.2f}s)"
    assert overhead_diff < 3.0, f"[{label}] overhead diff should be < 3s (got {overhead_diff:.2f}s)"

    print(f"    ALL DELAY ASSERTIONS PASSED!")


# ============================================================================
# TEST 1: External Tool - 0s vs 10s delay
# ============================================================================

@pytest.mark.asyncio
async def test_external_tool_delay_comparison():
    """External tool pause/resume: 0s vs 10s delay.
    Verifies all three usage sources match and delay is reflected correctly."""

    out_a, task_a, store_a = await _run_hitl_delay_flow(
        hitl_type="external_tool",
        task_desc="Send a notification with message 'Delay test'.",
        tools=[send_notification],
        expected_pause_reason="external_tool",
        resolve_fn=_resolve_external_tool,
        delay_before_resume=0,
        label="EXT_NO_DELAY",
        session_id="ext_no_delay",
    )

    out_b, task_b, store_b = await _run_hitl_delay_flow(
        hitl_type="external_tool",
        task_desc="Send a notification with message 'Delay test'.",
        tools=[send_notification],
        expected_pause_reason="external_tool",
        resolve_fn=_resolve_external_tool,
        delay_before_resume=DELAY_SECONDS,
        label="EXT_10S_DELAY",
        session_id="ext_10s_delay",
    )

    # Assert delay diffs for all three sources
    _assert_delay_diffs(out_a, out_b, "AgentRunOutput")
    _assert_delay_diffs(task_a, task_b, "Task")
    _assert_delay_diffs(store_a, store_b, "Storage")

    print("\n  EXTERNAL TOOL DELAY TEST PASSED!")


# ============================================================================
# TEST 2: Confirmation - 0s vs 10s delay
# ============================================================================

@pytest.mark.asyncio
async def test_confirmation_delay_comparison():
    """Confirmation pause/resume: 0s vs 10s delay.
    Verifies all three usage sources match and delay is reflected correctly."""

    out_a, task_a, store_a = await _run_hitl_delay_flow(
        hitl_type="confirmation",
        task_desc="Delete the file named 'temp.txt'.",
        tools=[delete_file],
        expected_pause_reason="confirmation",
        resolve_fn=_resolve_confirmation,
        delay_before_resume=0,
        label="CONF_NO_DELAY",
        session_id="conf_no_delay",
    )

    out_b, task_b, store_b = await _run_hitl_delay_flow(
        hitl_type="confirmation",
        task_desc="Delete the file named 'temp.txt'.",
        tools=[delete_file],
        expected_pause_reason="confirmation",
        resolve_fn=_resolve_confirmation,
        delay_before_resume=DELAY_SECONDS,
        label="CONF_10S_DELAY",
        session_id="conf_10s_delay",
    )

    _assert_delay_diffs(out_a, out_b, "AgentRunOutput")
    _assert_delay_diffs(task_a, task_b, "Task")
    _assert_delay_diffs(store_a, store_b, "Storage")

    print("\n  CONFIRMATION DELAY TEST PASSED!")


# ============================================================================
# TEST 3: User Input - 0s vs 10s delay
# ============================================================================

@pytest.mark.asyncio
async def test_user_input_delay_comparison():
    """User input pause/resume: 0s vs 10s delay.
    Verifies all three usage sources match and delay is reflected correctly."""

    out_a, task_a, store_a = await _run_hitl_delay_flow(
        hitl_type="user_input",
        task_desc="Send an email with subject 'Test' and body 'Hello'.",
        tools=[send_email],
        expected_pause_reason="user_input",
        resolve_fn=_resolve_user_input,
        delay_before_resume=0,
        label="INPUT_NO_DELAY",
        session_id="input_no_delay",
    )

    out_b, task_b, store_b = await _run_hitl_delay_flow(
        hitl_type="user_input",
        task_desc="Send an email with subject 'Test' and body 'Hello'.",
        tools=[send_email],
        expected_pause_reason="user_input",
        resolve_fn=_resolve_user_input,
        delay_before_resume=DELAY_SECONDS,
        label="INPUT_10S_DELAY",
        session_id="input_10s_delay",
    )

    _assert_delay_diffs(out_a, out_b, "AgentRunOutput")
    _assert_delay_diffs(task_a, task_b, "Task")
    _assert_delay_diffs(store_a, store_b, "Storage")

    print("\n  USER INPUT DELAY TEST PASSED!")


# ============================================================================
# CLEANUP
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    yield
    cleanup()


if __name__ == "__main__":
    asyncio.run(test_external_tool_delay_comparison())
