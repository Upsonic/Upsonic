"""
Usage Metrics Storage Verification Tests

Verifies that usage metrics from ALL THREE SOURCES are identical:
  1. AgentRunOutput.usage (in-memory run output)
  2. Task._usage (task class attribute)
  3. Storage (SqliteDatabase session.runs[run_id].output.usage)

Cases:
  HITL:
    1. External tool pause/resume (no delay) -> all three match
    2. External tool with 10s delay -> all three match + delay reflected
    3. Confirmation pause/resume -> all three match
    4. User input pause/resume -> all three match
  Non-HITL:
    5. agent.print_do_async(task) -> all three match
    6. agent.astream(task) -> storage usage is valid

Run with: uv run pytest tests/smoke_tests/hitl/test_usage_storage_verification.py -v -s
"""

import pytest
import asyncio
import os
from dataclasses import dataclass
from typing import Optional

from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.usage import TaskUsage
from upsonic.db.database import SqliteDatabase

pytestmark = pytest.mark.timeout(300)

DB_FILE = "usage_storage_verification_test.db"
DELAY_SECONDS = 10


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
class UsageSnapshot:
    duration: Optional[float]
    model_execution_time: Optional[float]
    tool_execution_time: Optional[float]
    pause_time: Optional[float]
    upsonic_execution_time: Optional[float]
    input_tokens: int
    output_tokens: int
    requests: int


def _snap(usage: TaskUsage) -> UsageSnapshot:
    return UsageSnapshot(
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


def _print_usage(label: str, snap: UsageSnapshot):
    print(f"\n  [{label}]")
    print(f"    duration             = {_fmt(snap.duration)}")
    print(f"    model_execution_time = {_fmt(snap.model_execution_time)}")
    print(f"    tool_execution_time  = {_fmt(snap.tool_execution_time)}")
    print(f"    pause_time           = {_fmt(snap.pause_time)}")
    print(f"    upsonic_exec_time    = {_fmt(snap.upsonic_execution_time)}")
    print(f"    input_tokens         = {snap.input_tokens}")
    print(f"    output_tokens        = {snap.output_tokens}")
    print(f"    requests             = {snap.requests}")


def _compare_snaps(a: UsageSnapshot, b: UsageSnapshot, label_a: str, label_b: str, tolerance: float = 0.5):
    """Assert two usage snapshots match within tolerance."""
    print(f"\n  Comparing {label_a} vs {label_b} (tolerance={tolerance}s)")

    if a.duration is not None and b.duration is not None:
        diff = abs(a.duration - b.duration)
        print(f"    duration diff       = {diff:.4f}s")
        assert diff < tolerance, f"duration mismatch: {a.duration:.4f} vs {b.duration:.4f} (diff={diff:.4f})"

    if a.model_execution_time is not None and b.model_execution_time is not None:
        diff = abs(a.model_execution_time - b.model_execution_time)
        print(f"    model_exec diff     = {diff:.4f}s")
        assert diff < tolerance, f"model_execution_time mismatch"

    a_tool = a.tool_execution_time or 0.0
    b_tool = b.tool_execution_time or 0.0
    diff = abs(a_tool - b_tool)
    print(f"    tool_exec diff      = {diff:.4f}s")
    assert diff < tolerance, f"tool_execution_time mismatch"

    a_pause = a.pause_time or 0.0
    b_pause = b.pause_time or 0.0
    diff = abs(a_pause - b_pause)
    print(f"    pause_time diff     = {diff:.4f}s")
    assert diff < tolerance, f"pause_time mismatch"

    assert a.input_tokens == b.input_tokens, f"input_tokens mismatch: {a.input_tokens} vs {b.input_tokens}"
    assert a.output_tokens == b.output_tokens, f"output_tokens mismatch: {a.output_tokens} vs {b.output_tokens}"
    assert a.requests == b.requests, f"requests mismatch: {a.requests} vs {b.requests}"

    print(f"    tokens/requests     = MATCH")
    print(f"    COMPARISON PASSED")


def _get_three_sources(output, task, db, session_id, run_id, label):
    """Extract usage from all three sources and return snapshots."""
    # SOURCE 1: AgentRunOutput.usage
    output_usage = output.usage
    assert output_usage is not None, f"[{label}] AgentRunOutput.usage is None"
    out_snap = _snap(output_usage)

    # SOURCE 2: Task._usage
    task_usage = task.usage
    assert task_usage is not None, f"[{label}] Task.usage is None"
    task_snap = _snap(task_usage)

    # SOURCE 3: Storage
    session = db.storage.get_session(session_id=session_id)
    assert session is not None, f"[{label}] Session not in storage"
    stored_output = session.runs[run_id].output
    stored_usage = stored_output.usage
    assert stored_usage is not None, f"[{label}] Stored usage is None"
    if isinstance(stored_usage, dict):
        stored_usage = TaskUsage.from_dict(stored_usage)
    store_snap = _snap(stored_usage)

    return out_snap, task_snap, store_snap


def _assert_three_match(out_snap, task_snap, store_snap, label):
    """Assert all three usage sources match."""
    _print_usage(f"{label} AgentRunOutput", out_snap)
    _print_usage(f"{label} Task", task_snap)
    _print_usage(f"{label} Storage", store_snap)

    _compare_snaps(out_snap, task_snap, "AgentRunOutput", "Task")
    _compare_snaps(out_snap, store_snap, "AgentRunOutput", "Storage")

    print(f"\n  [{label}] ALL THREE SOURCES MATCH!")


# ============================================================================
# HITL RESOLVERS
# ============================================================================

def _resolve_external_tool(req):
    te = req.tool_execution
    result = send_notification(**te.tool_args)
    te.result = result


def _resolve_confirmation(req):
    req.confirm()


def _resolve_user_input(req):
    if req.user_input_schema:
        for field_dict in req.user_input_schema:
            if isinstance(field_dict, dict) and field_dict.get("value") is None:
                name = field_dict["name"]
                field_dict["value"] = f"test_{name}@example.com"
    req.tool_execution.answered = True


# ============================================================================
# TEST 1: HITL External Tool - Three-source match (no delay)
# ============================================================================

@pytest.mark.asyncio
async def test_hitl_external_tool_three_sources_match():
    """External tool pause/resume: AgentRunOutput == Task == Storage."""
    cleanup()

    session_id = "ext_no_delay"
    db = SqliteDatabase(db_file=DB_FILE, session_id=session_id, user_id="test_user", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="ext_agent", db=db)
    task = Task(
        description="Send a notification with the message 'Storage test'.",
        tools=[send_notification],
    )

    output = await agent.print_do_async(task, return_output=True)
    assert output.is_paused and output.pause_reason == "external_tool"

    for req in output.active_requirements:
        _resolve_external_tool(req)

    result = await agent.continue_run_async(run_id=output.run_id, return_output=True, requirements=output.requirements)
    assert result.is_complete

    out_snap, task_snap, store_snap = _get_three_sources(result, task, db, session_id, result.run_id, "EXT")
    _assert_three_match(out_snap, task_snap, store_snap, "EXT")

    print("\n  TEST PASSED: External tool - all three sources match")


# ============================================================================
# TEST 2: HITL Confirmation - Three-source match (no delay)
# ============================================================================

@pytest.mark.asyncio
async def test_hitl_confirmation_three_sources_match():
    """Confirmation pause/resume: AgentRunOutput == Task == Storage."""
    cleanup()

    session_id = "conf_no_delay"
    db = SqliteDatabase(db_file=DB_FILE, session_id=session_id, user_id="test_user", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="conf_agent", db=db)
    task = Task(
        description="Delete the file named 'temp.txt'.",
        tools=[delete_file],
    )

    output = await agent.print_do_async(task, return_output=True)
    assert output.is_paused and output.pause_reason == "confirmation"

    for req in output.active_requirements:
        _resolve_confirmation(req)

    result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
    assert result.is_complete

    out_snap, task_snap, store_snap = _get_three_sources(result, task, db, session_id, result.run_id, "CONF")
    _assert_three_match(out_snap, task_snap, store_snap, "CONF")

    print("\n  TEST PASSED: Confirmation - all three sources match")


# ============================================================================
# TEST 3: HITL User Input - Three-source match (no delay)
# ============================================================================

@pytest.mark.asyncio
async def test_hitl_user_input_three_sources_match():
    """User input pause/resume: AgentRunOutput == Task == Storage."""
    cleanup()

    session_id = "input_no_delay"
    db = SqliteDatabase(db_file=DB_FILE, session_id=session_id, user_id="test_user", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="input_agent", db=db)
    task = Task(
        description="Send an email with subject 'Test' and body 'Hello'.",
        tools=[send_email],
    )

    output = await agent.print_do_async(task, return_output=True)
    assert output.is_paused and output.pause_reason == "user_input"

    for req in output.active_requirements:
        _resolve_user_input(req)

    result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
    assert result.is_complete

    out_snap, task_snap, store_snap = _get_three_sources(result, task, db, session_id, result.run_id, "INPUT")
    _assert_three_match(out_snap, task_snap, store_snap, "INPUT")

    print("\n  TEST PASSED: User input - all three sources match")


# ============================================================================
# TEST 4: HITL External Tool with 10s delay - Three-source match + delay
# ============================================================================

@pytest.mark.asyncio
async def test_hitl_external_tool_delay_three_sources():
    """External tool with 0s vs 10s delay: all three sources match for BOTH runs,
    and 10s delay is reflected in duration/pause_time but NOT model_time/overhead."""
    # --- Run A: no delay ---
    cleanup()
    sid_a = "ext_delay_a"
    db_a = SqliteDatabase(db_file=DB_FILE, session_id=sid_a, user_id="test_user", full_session_memory=True)
    agent_a = Agent("openai/gpt-4o-mini", name="ext_delay_a", db=db_a)
    task_a = Task(description="Send a notification with message 'Delay A'.", tools=[send_notification])

    out_a = await agent_a.print_do_async(task_a, return_output=True)
    assert out_a.is_paused
    for req in out_a.active_requirements:
        _resolve_external_tool(req)
    result_a = await agent_a.continue_run_async(run_id=out_a.run_id, return_output=True, requirements=out_a.requirements)
    assert result_a.is_complete

    out_snap_a, task_snap_a, store_snap_a = _get_three_sources(result_a, task_a, db_a, sid_a, result_a.run_id, "NO_DELAY")
    _assert_three_match(out_snap_a, task_snap_a, store_snap_a, "NO_DELAY")

    # --- Run B: 10s delay ---
    cleanup()
    sid_b = "ext_delay_b"
    db_b = SqliteDatabase(db_file=DB_FILE, session_id=sid_b, user_id="test_user", full_session_memory=True)
    agent_b = Agent("openai/gpt-4o-mini", name="ext_delay_b", db=db_b)
    task_b = Task(description="Send a notification with message 'Delay B'.", tools=[send_notification])

    out_b = await agent_b.print_do_async(task_b, return_output=True)
    assert out_b.is_paused
    for req in out_b.active_requirements:
        _resolve_external_tool(req)

    print(f"\n  Sleeping {DELAY_SECONDS}s before resume...")
    await asyncio.sleep(DELAY_SECONDS)
    print("  Awake. Resuming now.")

    result_b = await agent_b.continue_run_async(run_id=out_b.run_id, return_output=True, requirements=out_b.requirements)
    assert result_b.is_complete

    out_snap_b, task_snap_b, store_snap_b = _get_three_sources(result_b, task_b, db_b, sid_b, result_b.run_id, "10S_DELAY")
    _assert_three_match(out_snap_b, task_snap_b, store_snap_b, "10S_DELAY")

    # --- Delay assertions across all three sources ---
    for snap_a, snap_b, source_label in [
        (out_snap_a, out_snap_b, "AgentRunOutput"),
        (task_snap_a, task_snap_b, "Task"),
        (store_snap_a, store_snap_b, "Storage"),
    ]:
        dur_diff = (snap_b.duration or 0) - (snap_a.duration or 0)
        model_diff = abs((snap_b.model_execution_time or 0) - (snap_a.model_execution_time or 0))
        pause_diff = (snap_b.pause_time or 0) - (snap_a.pause_time or 0)
        overhead_diff = abs((snap_b.upsonic_execution_time or 0) - (snap_a.upsonic_execution_time or 0))

        print(f"\n  [{source_label}] DELAY DIFFS:")
        print(f"    duration diff    = {_fmt(dur_diff)}")
        print(f"    model_exec diff  = {_fmt(model_diff)}")
        print(f"    pause_time diff  = {_fmt(pause_diff)}")
        print(f"    overhead diff    = {_fmt(overhead_diff)}")

        assert dur_diff >= 9.0, f"[{source_label}] duration diff should be >= 9s (got {dur_diff:.2f}s)"
        assert model_diff < 3.0, f"[{source_label}] model_exec diff should be < 3s"
        assert pause_diff >= 9.0, f"[{source_label}] pause_time diff should be >= 9s (got {pause_diff:.2f}s)"
        assert overhead_diff < 3.0, f"[{source_label}] overhead diff should be < 3s"

    print("\n  TEST PASSED: External tool delay - all three sources match with correct delay")


# ============================================================================
# TEST 5: Non-HITL print_do_async - Three-source match
# ============================================================================

@pytest.mark.asyncio
async def test_non_hitl_print_do_three_sources_match():
    """Non-HITL print_do_async: AgentRunOutput == Task == Storage."""
    cleanup()

    session_id = "non_hitl_print_do"
    db = SqliteDatabase(db_file=DB_FILE, session_id=session_id, user_id="test_user", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="non_hitl_agent", db=db)
    task = Task(description="What is 2 + 2? Answer with just the number.")

    output = await agent.print_do_async(task, return_output=True)
    assert output.is_complete

    out_snap, task_snap, store_snap = _get_three_sources(output, task, db, session_id, output.run_id, "PRINT_DO")
    _assert_three_match(out_snap, task_snap, store_snap, "PRINT_DO")

    # Non-HITL sanity
    assert out_snap.duration is not None and out_snap.duration > 0
    assert out_snap.input_tokens > 0
    assert out_snap.output_tokens > 0
    assert out_snap.requests > 0
    assert out_snap.model_execution_time is not None and out_snap.model_execution_time > 0
    assert (out_snap.pause_time or 0) == 0, "Non-HITL should have no pause_time"

    print("\n  TEST PASSED: Non-HITL print_do - all three sources match")


# ============================================================================
# TEST 6: Non-HITL astream - Storage usage valid
# ============================================================================

@pytest.mark.asyncio
async def test_non_hitl_stream_storage_usage():
    """Non-HITL astream: verify stored usage has valid metrics.
    Note: HITL is NOT supported in streaming mode."""
    cleanup()

    session_id = "non_hitl_stream"
    db = SqliteDatabase(db_file=DB_FILE, session_id=session_id, user_id="test_user", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="stream_agent", db=db)
    task = Task(description="What is 3 + 5? Answer with just the number.")

    collected_text = []
    async for chunk in agent.astream(task):
        if isinstance(chunk, str):
            collected_text.append(chunk)

    full_text = "".join(collected_text)
    print(f"  Streamed text: {full_text!r}")
    assert len(full_text) > 0

    # Verify stored usage
    session = db.storage.get_session(session_id=session_id)
    assert session is not None
    assert len(session.runs) > 0

    run_id = list(session.runs.keys())[-1]
    stored_output = session.runs[run_id].output
    stored_usage = stored_output.usage
    assert stored_usage is not None

    if isinstance(stored_usage, dict):
        stored_usage = TaskUsage.from_dict(stored_usage)

    snap = _snap(stored_usage)
    _print_usage("STREAM STORED", snap)

    assert snap.duration is not None and snap.duration > 0
    assert snap.input_tokens > 0
    assert snap.output_tokens > 0
    assert snap.requests > 0
    assert snap.model_execution_time is not None and snap.model_execution_time > 0
    assert (snap.pause_time or 0) == 0

    print("\n  TEST PASSED: Non-HITL stream storage usage is valid")


# ============================================================================
# CLEANUP
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    yield
    cleanup()


if __name__ == "__main__":
    asyncio.run(test_hitl_external_tool_three_sources_match())
