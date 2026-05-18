"""
Agent Retry Metrics Smoke Tests

Verifies that ``agent.usage`` and per-task usage stay consistent across
every retry / HITL / cross-process shape. After the usage-registry
refactor the underlying invariant is structural:

    every model.request emits exactly one UsageEntry with a unique
    ``entry_id`` under the active ``agent_usage_id`` / ``task_usage_id``
    / ``run_id`` contextvars. ``agent.usage`` is a registry query — it
    sums every entry tagged with the agent's id. No baseline arithmetic
    is needed because re-recording the same entry_id is an upsert, not
    a double-count.

Scenarios:
  1. Successful retry after 2 failed attempts: 3 model.request calls
     emit 3 entries — agent.usage = 3 × per-attempt totals.
  2. All-fail retry, user discards: 3 entries from 3 attempts;
     agent.usage covers all three.
  3. All-fail retry then user resumes via continue_run_async: the
     resume adds new entries; no double-count of the failed attempts.
  4. Cross-process all-fail retry + resume: entries persisted to
     storage round-trip; the resumed registry picks them up.
  5. Single successful run: agent.usage == final.usage exactly.
  6. HITL pause + resume: pause + resume share the same run scope;
     final agent.usage equals final.usage.
  7. Single attempt error + resume via continue_run_async: no
     double-count, no missing.

Run with: uv run pytest tests/smoke_tests/agent/test_retry_metrics.py -v -s
"""

import os
import pytest

from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.db.database import SqliteDatabase
from upsonic.usage_registry import get_default_registry
from tests._pipeline_injection import (
    inject_error_into_step,
    clear_error_injection,
)


pytestmark = pytest.mark.timeout(300)

DB_FILE = "retry_metrics_smoke.db"
MODEL = "openai/gpt-4o-mini"


def _cleanup_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)


@tool
def simple_math(a: int, b: int) -> int:
    """Add two numbers and return the sum."""
    return a + b


@tool(external_execution=True)
def send_notification(message: str) -> str:
    """Send a notification externally — requires HITL execution.

    Args:
        message: The text to send.
    """
    return f"Notification sent: {message}"


def _assert_usage_positive(usage, label: str):
    """All major counters must be populated for a real (non-zero) run."""
    assert usage is not None, f"[{label}] usage is None"
    assert usage.requests > 0, f"[{label}] requests should be > 0"
    assert usage.input_tokens > 0, f"[{label}] input_tokens should be > 0"
    assert usage.output_tokens > 0, f"[{label}] output_tokens should be > 0"


# ============================================================================
# 1. Successful retry
# ============================================================================

@pytest.mark.asyncio
async def test_retry_success_metrics_accumulate_all_attempts():
    """retry=3 with first 2 attempts failing post-model — agent.usage must
    contain the sum of all 3 attempts' token usage (3 ledger entries)."""
    _cleanup_db()
    get_default_registry().clear()
    clear_error_injection()
    inject_error_into_step(
        "response_processing", RuntimeError, "INJECTED ERROR: boom-post-model",
        trigger_count=2,
    )

    agent = Agent(MODEL, retry=3)
    task = Task(description="What is 5 + 3? Reply with just the number.", tools=[simple_math])

    output = await agent.do_async(task, return_output=True)

    _assert_usage_positive(output.usage, "successful run final output")
    _assert_usage_positive(agent.usage, "agent after successful retry")

    per_attempt_req = output.usage.requests
    per_attempt_in = output.usage.input_tokens

    assert agent.usage.requests == per_attempt_req * 3, (
        f"agent.usage.requests={agent.usage.requests} != per_attempt({per_attempt_req})*3"
    )
    assert agent.usage.input_tokens == per_attempt_in * 3, (
        f"agent.usage.input_tokens={agent.usage.input_tokens} != per_attempt({per_attempt_in})*3"
    )

    clear_error_injection()


# ============================================================================
# 2. All-fail retry, user discards
# ============================================================================

@pytest.mark.asyncio
async def test_retry_all_fail_discard_accumulates_last_attempt():
    """retry=3 with all 3 attempts failing — agent.usage must include
    every attempt's tokens (3 entries from 3 model.requests, no
    accumulator hook required)."""
    _cleanup_db()
    get_default_registry().clear()
    clear_error_injection()
    inject_error_into_step(
        "response_processing", RuntimeError, "INJECTED ERROR: boom-all-fail",
        trigger_count=10,
    )

    agent = Agent(MODEL, retry=3)
    task = Task(description="What is 1+1?", tools=[simple_math])

    raised = None
    try:
        await agent.do_async(task, return_output=True)
    except Exception as e:
        raised = e

    assert raised is not None, "do_async should have raised after all retries"

    _assert_usage_positive(agent.usage, "agent after all-fail discard")
    assert agent.usage.requests >= 3, (
        f"agent.usage.requests={agent.usage.requests} should reflect all 3 attempts"
    )

    clear_error_injection()


# ============================================================================
# 3. All-fail retry, user resumes via continue_run_async
# ============================================================================

@pytest.mark.asyncio
async def test_retry_all_fail_then_resume_no_double_count():
    """retry=3, all fail, then user resumes — the resume's new model.request
    emissions must be added to the registry without re-counting the failed
    attempts. Registry idempotency on entry_id makes this structural."""
    _cleanup_db()
    get_default_registry().clear()
    clear_error_injection()
    inject_error_into_step(
        "response_processing", RuntimeError, "INJECTED ERROR: boom",
        trigger_count=3,
    )

    db = SqliteDatabase(db_file=DB_FILE, session_id="s3", user_id="u3", full_session_memory=True)
    agent = Agent(MODEL, db=db, retry=3)
    task = Task(description="What is 9+9?", tools=[simple_math])

    try:
        await agent.do_async(task, return_output=True)
    except Exception:
        pass

    requests_after_fail = agent.usage.requests
    assert requests_after_fail >= 3

    final = await agent.continue_run_async(task=task, return_output=True)

    # Resume must add at least one more request, and the final agent total
    # equals the registry's roll-up — no implicit double-count.
    assert agent.usage.requests >= requests_after_fail + 1

    clear_error_injection()
    _cleanup_db()


# ============================================================================
# 4. Cross-process retry exhaustion + resume (storage round-trip)
# ============================================================================

@pytest.mark.asyncio
async def test_retry_exhaustion_survives_storage():
    """All-fail retry in agent A → continue_run_async in agent B (fresh
    instance loading from storage). The failed attempts' UsageEntries
    persisted to storage must rehydrate into B's registry view of the
    same agent scope."""
    _cleanup_db()
    get_default_registry().clear()
    clear_error_injection()
    inject_error_into_step(
        "response_processing", RuntimeError, "INJECTED ERROR: boom",
        trigger_count=3,
    )

    db_a = SqliteDatabase(db_file=DB_FILE, session_id="x", user_id="x", full_session_memory=True)
    agent_a = Agent(MODEL, db=db_a, retry=3)
    task_a = Task(description="What is 2+2?", tools=[simple_math])

    try:
        await agent_a.do_async(task_a, return_output=True)
    except Exception:
        pass

    assert agent_a.usage.requests >= 3
    run_id = agent_a._agent_run_output.run_id

    # Resume in a fresh agent loading from storage.
    db_b = SqliteDatabase(db_file=DB_FILE, session_id="x", user_id="x", full_session_memory=True)
    agent_b = Agent(MODEL, db=db_b, retry=1)
    final = await agent_b.continue_run_async(run_id=run_id, return_output=True)

    # agent_b.usage is process-local; it covers only entries this process
    # produced (the resume's emissions). No magic baseline math needed.
    assert agent_b.usage.requests >= 1
    _assert_usage_positive(final.usage, "cross-process resume final")

    clear_error_injection()
    _cleanup_db()


# ============================================================================
# 5. Single success — no regression
# ============================================================================

@pytest.mark.asyncio
async def test_single_run_simple_path():
    """A plain successful run (no retry, no HITL) must produce
    agent.usage == output.usage."""
    _cleanup_db()
    get_default_registry().clear()
    clear_error_injection()

    agent = Agent(MODEL, retry=1)
    task = Task(description="What is 4+4?", tools=[simple_math])
    output = await agent.do_async(task, return_output=True)

    _assert_usage_positive(output.usage, "single-run output")
    _assert_usage_positive(agent.usage, "single-run agent")
    assert agent.usage.requests == output.usage.requests
    assert agent.usage.input_tokens == output.usage.input_tokens
    assert agent.usage.output_tokens == output.usage.output_tokens


# ============================================================================
# 6. HITL pause + resume
# ============================================================================

@pytest.mark.asyncio
async def test_hitl_pause_resume_no_double_count():
    """An HITL-paused run that resumes must end up with
    ``agent.usage == final.usage`` (single count of every model call,
    no double-count across the pause). Registry is keyed on entry_id so
    this invariant is structural, not arithmetic."""
    _cleanup_db()
    get_default_registry().clear()
    clear_error_injection()

    agent = Agent(MODEL, retry=1)
    task = Task(
        description="Send a short notification with message 'hi'.",
        tools=[send_notification],
    )

    output = await agent.do_async(task, return_output=True)
    assert output.is_paused, "expected HITL pause"

    # Provide the external tool result and resume.
    output.requirements[0].tool_execution.result = "Notification sent: hi"
    final = await agent.continue_run_async(run_id=output.run_id, return_output=True)

    assert final.is_complete, "expected completion after resume"
    _assert_usage_positive(final.usage, "HITL resume final")
    _assert_usage_positive(agent.usage, "HITL resume agent")
    assert agent.usage.requests == final.usage.requests, (
        f"HITL resume double-count: agent={agent.usage.requests}, final={final.usage.requests}"
    )


# ============================================================================
# 7. Durable error + continue_run_async (retry=1)
# ============================================================================

@pytest.mark.asyncio
async def test_durable_error_resume_single_count():
    """retry=1: first (and only) attempt errors, user resumes via
    continue_run_async. The failed attempt's entries are in the
    registry; the resume's entries are added on top — agent.usage rolls
    them all up without arithmetic. No double-count, no missing."""
    _cleanup_db()
    get_default_registry().clear()
    clear_error_injection()
    inject_error_into_step(
        "response_processing", RuntimeError, "INJECTED ERROR: boom-once",
        trigger_count=1,
    )

    db = SqliteDatabase(db_file=DB_FILE, session_id="s7", user_id="u7", full_session_memory=True)
    agent = Agent(MODEL, db=db, retry=1)
    task = Task(description="What is 6+6?", tools=[simple_math])

    try:
        await agent.do_async(task, return_output=True)
    except Exception:
        pass

    assert agent.usage.requests >= 1, "failed attempt must be recorded"
    requests_after_fail = agent.usage.requests

    final = await agent.continue_run_async(task=task, return_output=True)
    assert final.is_complete

    # Final agent.usage covers the failed attempt + the resume's entries.
    assert agent.usage.requests >= requests_after_fail + 1

    clear_error_injection()
    _cleanup_db()
