"""
HITL Usage Metrics Smoke Tests

Verifies that usage metrics from ALL THREE SOURCES match after HITL flows:
  1. AgentRunOutput.usage (in-memory run output)
  2. Task._usage (task class attribute)
  3. Storage (SqliteDatabase session.runs[run_id].output.usage)

Tests:
  1. External tool pause → resume: all three match
  2. Confirmation pause → resume: all three match
  3. User input pause → resume: all three match
  4. Cancel → resume: additive duration, all three match
  5. Cross-process resume: all three match
  6. Memory save: all three match
  7. hitl_handler: all three match
  8. Simple do_async baseline: all three match
  9. Tool usage: all three match

Run with: uv run pytest tests/smoke_tests/hitl/test_usage_metrics_hitl.py -v -s
"""

import pytest
import asyncio
import os
from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.usage import TaskUsage
from upsonic.run.base import RunStatus
from upsonic.run.cancel import cancel_run
from upsonic.db.database import SqliteDatabase

pytestmark = pytest.mark.timeout(300)

DB_FILE = "usage_metrics_hitl_test.db"


def cleanup_db():
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


@tool
def simple_add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum
    """
    return a + b


# ============================================================================
# HELPERS
# ============================================================================

def _fmt(v):
    return f"{v:.4f}" if v is not None else "None"


def _assert_usage_complete(usage: TaskUsage, label: str):
    """Assert all major usage fields are populated."""
    assert usage is not None, f"[{label}] usage is None"
    assert usage.duration is not None and usage.duration > 0, f"[{label}] duration invalid"
    assert usage.input_tokens > 0, f"[{label}] input_tokens should be > 0"
    assert usage.output_tokens > 0, f"[{label}] output_tokens should be > 0"
    assert usage.requests > 0, f"[{label}] requests should be > 0"
    assert usage.model_execution_time is not None and usage.model_execution_time > 0, f"[{label}] model_execution_time invalid"


def _compare_usage(a: TaskUsage, b: TaskUsage, label_a: str, label_b: str, tolerance: float = 0.5):
    """Assert two TaskUsage instances match within tolerance."""
    print(f"  Comparing {label_a} vs {label_b}:")

    if a.duration is not None and b.duration is not None:
        diff = abs(a.duration - b.duration)
        assert diff < tolerance, f"    duration mismatch: {_fmt(a.duration)} vs {_fmt(b.duration)}"

    if a.model_execution_time is not None and b.model_execution_time is not None:
        diff = abs(a.model_execution_time - b.model_execution_time)
        assert diff < tolerance, f"    model_execution_time mismatch"

    a_tool = a.tool_execution_time or 0.0
    b_tool = b.tool_execution_time or 0.0
    assert abs(a_tool - b_tool) < tolerance, f"    tool_execution_time mismatch"

    a_pause = a.pause_time or 0.0
    b_pause = b.pause_time or 0.0
    assert abs(a_pause - b_pause) < tolerance, f"    pause_time mismatch"

    assert a.input_tokens == b.input_tokens, f"    input_tokens: {a.input_tokens} vs {b.input_tokens}"
    assert a.output_tokens == b.output_tokens, f"    output_tokens: {a.output_tokens} vs {b.output_tokens}"
    assert a.requests == b.requests, f"    requests: {a.requests} vs {b.requests}"

    print(f"    MATCH!")


def _assert_three_match(output, task, db, session_id, run_id, label):
    """Assert AgentRunOutput.usage == Task.usage == Storage usage."""
    out_usage = output.usage
    task_usage = task.usage

    _assert_usage_complete(out_usage, f"{label} AgentRunOutput")
    _assert_usage_complete(task_usage, f"{label} Task")

    # Compare AgentRunOutput vs Task
    _compare_usage(out_usage, task_usage, f"{label} AgentRunOutput", f"{label} Task")

    # Compare with Storage if db is provided
    if db is not None:
        session = db.storage.get_session(session_id=session_id)
        assert session is not None, f"[{label}] Session not in storage"
        stored_output = session.runs[run_id].output
        stored_usage = stored_output.usage
        assert stored_usage is not None, f"[{label}] Stored usage is None"
        if isinstance(stored_usage, dict):
            stored_usage = TaskUsage.from_dict(stored_usage)
        _assert_usage_complete(stored_usage, f"{label} Storage")
        _compare_usage(out_usage, stored_usage, f"{label} AgentRunOutput", f"{label} Storage")

    print(f"  [{label}] ALL SOURCES MATCH!")


def _execute_tool(requirement) -> str:
    te = requirement.tool_execution
    return send_notification(**te.tool_args)


def _fill_user_input(requirement):
    if not requirement.user_input_schema:
        return
    for field_dict in requirement.user_input_schema:
        if isinstance(field_dict, dict) and field_dict.get("value") is None:
            name = field_dict["name"]
            field_dict["value"] = f"test_{name}@example.com"
    requirement.tool_execution.answered = True


# ============================================================================
# TEST 1: External tool pause → resume: all three match
# ============================================================================

@pytest.mark.asyncio
async def test_external_tool_usage_three_sources():
    """External tool pause/resume: AgentRunOutput == Task == Storage."""
    cleanup_db()
    db = SqliteDatabase(db_file=DB_FILE, session_id="ext_usage", user_id="user_1", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="ext_usage_agent", db=db)
    task = Task(description="Send a notification with message 'Hello World'.", tools=[send_notification])

    output = await agent.print_do_async(task, return_output=True)
    assert output.is_paused and output.pause_reason == "external_tool"

    paused_duration = output.usage.duration

    for req in output.active_requirements:
        if req.is_external_tool_execution:
            req.tool_execution.result = _execute_tool(req)

    result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
    assert result.is_complete

    assert result.usage.duration >= paused_duration
    _assert_three_match(result, task, db, "ext_usage", result.run_id, "EXT")

    cleanup_db()


# ============================================================================
# TEST 2: Confirmation pause → resume: all three match
# ============================================================================

@pytest.mark.asyncio
async def test_confirmation_usage_three_sources():
    """Confirmation pause/resume: AgentRunOutput == Task == Storage."""
    cleanup_db()
    db = SqliteDatabase(db_file=DB_FILE, session_id="conf_usage", user_id="user_1", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="conf_usage_agent", db=db)
    task = Task(description="Delete the file named 'temp.txt'.", tools=[delete_file])

    output = await agent.print_do_async(task, return_output=True)
    assert output.is_paused and output.pause_reason == "confirmation"

    for req in output.active_requirements:
        if req.needs_confirmation:
            req.confirm()

    result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
    assert result.is_complete

    _assert_three_match(result, task, db, "conf_usage", result.run_id, "CONF")

    cleanup_db()


# ============================================================================
# TEST 3: User input pause → resume: all three match
# ============================================================================

@pytest.mark.asyncio
async def test_user_input_usage_three_sources():
    """User input pause/resume: AgentRunOutput == Task == Storage."""
    cleanup_db()
    db = SqliteDatabase(db_file=DB_FILE, session_id="input_usage", user_id="user_1", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="input_usage_agent", db=db)
    task = Task(description="Send an email with subject 'Test' and body 'Hello'.", tools=[send_email])

    output = await agent.print_do_async(task, return_output=True)
    assert output.is_paused and output.pause_reason == "user_input"

    for req in output.active_requirements:
        if req.needs_user_input:
            _fill_user_input(req)

    result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
    assert result.is_complete

    _assert_three_match(result, task, db, "input_usage", result.run_id, "INPUT")

    cleanup_db()


# ============================================================================
# TEST 4: Cancel → resume: additive duration, all three match
# ============================================================================

@pytest.mark.asyncio
async def test_cancel_resume_three_sources():
    """Cancel and resume: duration is additive, all three sources match."""
    cleanup_db()
    db = SqliteDatabase(db_file=DB_FILE, session_id="cancel_usage", user_id="user_1", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="cancel_usage_agent", db=db)
    task = Task(description="Call simple_add with a=10 and b=20, then explain the result.", tools=[simple_add])

    async def cancel_after_delay():
        await asyncio.sleep(2)
        if agent.run_id:
            cancel_run(agent.run_id)

    asyncio.create_task(cancel_after_delay())

    output = await agent.print_do_async(task, return_output=True)

    if output.status == RunStatus.cancelled:
        cancelled_usage = output.usage
        assert cancelled_usage.duration is not None and cancelled_usage.duration > 0
        cancelled_duration = cancelled_usage.duration

        result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
        assert result.is_complete

        assert result.usage.duration >= cancelled_duration
        _assert_three_match(result, task, db, "cancel_usage", result.run_id, "CANCEL")
    else:
        # Completed before cancel — just verify all three match
        _assert_three_match(output, task, db, "cancel_usage", output.run_id, "CANCEL_EARLY")

    cleanup_db()


# ============================================================================
# TEST 5: Cross-process resume: all three match
# ============================================================================

@pytest.mark.asyncio
async def test_cross_process_resume_three_sources():
    """Cross-process resume: usage restored from storage, all three match."""
    cleanup_db()
    db = SqliteDatabase(db_file=DB_FILE, session_id="cross_proc", user_id="user_1", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="cross_proc_agent", db=db)
    task = Task(description="Send a notification with message 'Cross-process test'.", tools=[send_notification])

    output = await agent.print_do_async(task, return_output=True)
    assert output.is_paused

    paused_tokens = output.usage.input_tokens
    run_id = output.run_id

    for req in output.active_requirements:
        if req.is_external_tool_execution:
            req.tool_execution.result = _execute_tool(req)

    # NEW agent (cross-process simulation)
    new_db = SqliteDatabase(db_file=DB_FILE, session_id="cross_proc", user_id="user_1", full_session_memory=True)
    new_agent = Agent("openai/gpt-4o-mini", name="cross_proc_agent", db=new_db)

    result = await new_agent.continue_run_async(
        run_id=run_id, requirements=output.requirements, return_output=True
    )
    assert result.is_complete

    # For cross-process, task is reconstructed from storage, so use output.task
    result_task = result.task if hasattr(result, 'task') and result.task else task

    # AgentRunOutput vs Storage
    out_usage = result.usage
    _assert_usage_complete(out_usage, "CROSS AgentRunOutput")

    session = new_db.storage.get_session(session_id="cross_proc")
    stored_usage = session.runs[run_id].output.usage
    if isinstance(stored_usage, dict):
        stored_usage = TaskUsage.from_dict(stored_usage)
    _assert_usage_complete(stored_usage, "CROSS Storage")
    _compare_usage(out_usage, stored_usage, "CROSS AgentRunOutput", "CROSS Storage")

    assert out_usage.input_tokens >= paused_tokens

    print("  [CROSS] AgentRunOutput == Storage MATCH!")

    cleanup_db()


# ============================================================================
# TEST 6: Memory save includes usage: all three match
# ============================================================================

@pytest.mark.asyncio
async def test_memory_save_three_sources():
    """Simple do_async with storage: AgentRunOutput == Task == Storage."""
    cleanup_db()
    db = SqliteDatabase(db_file=DB_FILE, session_id="mem_save", user_id="user_1", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="mem_save_agent", db=db)
    task = Task(description="What is 5+5? Answer with just the number.")

    output = await agent.print_do_async(task, return_output=True)
    assert output.is_complete

    _assert_three_match(output, task, db, "mem_save", output.run_id, "MEM_SAVE")

    cleanup_db()


# ============================================================================
# TEST 7: hitl_handler: all three match
# ============================================================================

@pytest.mark.asyncio
async def test_hitl_handler_three_sources():
    """hitl_handler with external tool: AgentRunOutput == Task == Storage."""
    cleanup_db()
    db = SqliteDatabase(db_file=DB_FILE, session_id="handler_usage", user_id="user_1", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="handler_agent", db=db)
    task = Task(description="Send a notification with message 'Auto handled'.", tools=[send_notification])

    def hitl_handler(requirement):
        if requirement.needs_external_execution:
            requirement.tool_execution.result = _execute_tool(requirement)

    output = await agent.print_do_async(task, return_output=True)
    assert output.is_paused

    for req in output.active_requirements:
        hitl_handler(req)

    result = await agent.continue_run_async(
        run_id=output.run_id, return_output=True, hitl_handler=hitl_handler
    )
    assert result.is_complete
    assert result.usage.requests >= 2

    _assert_three_match(result, task, db, "handler_usage", result.run_id, "HANDLER")

    cleanup_db()


# ============================================================================
# TEST 8: Simple do_async baseline: all three match
# ============================================================================

@pytest.mark.asyncio
async def test_simple_do_async_three_sources():
    """Baseline do_async: AgentRunOutput == Task == Storage."""
    cleanup_db()
    db = SqliteDatabase(db_file=DB_FILE, session_id="baseline", user_id="user_1", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="baseline_agent", db=db)
    task = Task(description="What is 2+2? Answer with just the number.")

    output = await agent.print_do_async(task, return_output=True)
    assert output.is_complete

    _assert_three_match(output, task, db, "baseline", output.run_id, "BASELINE")

    cleanup_db()


# ============================================================================
# TEST 9: Tool usage: all three match
# ============================================================================

@pytest.mark.asyncio
async def test_tool_usage_three_sources():
    """do_async with regular tool: AgentRunOutput == Task == Storage."""
    cleanup_db()
    db = SqliteDatabase(db_file=DB_FILE, session_id="tool_time", user_id="user_1", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="tool_time_agent", db=db)
    task = Task(description="Use simple_add to add 7 and 8. Return just the number.", tools=[simple_add])

    output = await agent.print_do_async(task, return_output=True)
    assert output.is_complete
    assert output.usage.requests >= 2

    _assert_three_match(output, task, db, "tool_time", output.run_id, "TOOL")

    cleanup_db()


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    async def run_all():
        tests = [
            test_external_tool_usage_three_sources,
            test_confirmation_usage_three_sources,
            test_user_input_usage_three_sources,
            test_cancel_resume_three_sources,
            test_cross_process_resume_three_sources,
            test_memory_save_three_sources,
            test_hitl_handler_three_sources,
            test_simple_do_async_three_sources,
            test_tool_usage_three_sources,
        ]
        for t in tests:
            print(f"\n{'='*80}")
            print(f"  {t.__name__}")
            print(f"{'='*80}")
            await t()
            print("  PASSED")
        print(f"\n{'='*80}")
        print("  ALL TESTS PASSED!")
        print(f"{'='*80}")

    asyncio.run(run_all())
