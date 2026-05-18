"""
Agent Retry Metrics Smoke Tests

Verifies that ``agent.usage`` (lifetime cumulative) and ``output.usage`` /
``task._usage`` (per-run cumulative) stay consistent across all retry
shapes. Specifically guards the baseline mechanism that lets the
``@retryable`` exhaustion hook accumulate the final failed attempt's
usage without double-counting on a subsequent ``continue_run_async``
resume.

Scenarios:
  1. Successful retry: failed attempts captured by retry block,
     successful final attempt captured by finally. agent.usage == sum.
  2. All-fail retry, user discards: retry block captures attempts 1..N-1,
     ``_on_retries_exhausted`` hook captures attempt N. agent.usage ==
     sum of all attempts.
  3. All-fail retry, user resumes via continue_run_async: hook captures
     attempt N and records a baseline; the resume's final accumulation
     adds only the delta (post-resume work). No double-count.
  4. Cross-process all-fail retry + resume: baseline survives storage
     round-trip; new agent's accumulation uses the baseline correctly.
  5. Successful run with no retries: baseline still works for a single
     accumulation (no regression).
  6. Retry attempt followed by HITL pause on a later attempt:
     interleaving with HITL semantics stays consistent.

Run with: uv run pytest tests/smoke_tests/agent/test_retry_metrics.py -v -s
"""

import asyncio
import os
import pytest

from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.usage import TaskUsage
from upsonic.db.database import SqliteDatabase
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


# ----------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Assertions
# ----------------------------------------------------------------------------

def _assert_usage_positive(usage: TaskUsage, label: str):
    """All major counters must be populated for a real (non-zero) run."""
    assert usage is not None, f"[{label}] usage is None"
    assert usage.requests > 0, f"[{label}] requests should be > 0"
    assert usage.input_tokens > 0, f"[{label}] input_tokens should be > 0"
    assert usage.output_tokens > 0, f"[{label}] output_tokens should be > 0"


def _assert_agent_usage_at_least(agent, lower_bound: TaskUsage, label: str):
    """agent.usage counters must be >= lower_bound counters."""
    assert agent.usage is not None, f"[{label}] agent.usage is None"
    assert agent.usage.requests >= lower_bound.requests, (
        f"[{label}] agent.usage.requests={agent.usage.requests} < {lower_bound.requests}"
    )
    assert agent.usage.input_tokens >= lower_bound.input_tokens, (
        f"[{label}] agent.usage.input_tokens={agent.usage.input_tokens} < {lower_bound.input_tokens}"
    )
    assert agent.usage.output_tokens >= lower_bound.output_tokens, (
        f"[{label}] agent.usage.output_tokens={agent.usage.output_tokens} < {lower_bound.output_tokens}"
    )


def _assert_resume_grew_by_delta(
    agent_before_resume: TaskUsage,
    agent_after_resume: TaskUsage,
    final_usage: TaskUsage,
    baseline: TaskUsage,
):
    """After resume, ``agent.usage`` must have grown by exactly the delta
    of work done after the baseline was captured.

    Specifically:
        agent_after_resume - agent_before_resume == final.usage - baseline

    If this fails:
      - growth > delta  → previously-counted portion was double-counted
      - growth < delta  → some post-resume work was missed
    """
    agent_grow = agent_after_resume.requests - agent_before_resume.requests
    expected_grow = final_usage.requests - baseline.requests
    assert agent_grow == expected_grow, (
        f"agent.usage grew by {agent_grow} requests during resume; "
        f"expected {expected_grow} (final.usage − baseline). "
        f"agent_before={agent_before_resume.requests}, "
        f"agent_after={agent_after_resume.requests}, "
        f"final={final_usage.requests}, baseline={baseline.requests}"
    )


# ============================================================================
# 1. Successful retry
# ============================================================================

@pytest.mark.asyncio
async def test_retry_success_metrics_accumulate_all_attempts():
    """retry=3 with first 2 attempts failing post-model — agent.usage must
    contain the sum of all 3 attempts' token usage."""
    _cleanup_db()
    clear_error_injection()
    inject_error_into_step(
        "response_processing", RuntimeError, "boom-post-model", trigger_count=2
    )

    agent = Agent(MODEL, retry=3)
    task = Task(description="What is 5 + 3? Reply with just the number.", tools=[simple_math])

    output = await agent.do_async(task, return_output=True)

    _assert_usage_positive(output.usage, "successful run final output")
    _assert_usage_positive(agent.usage, "agent after successful retry")

    per_attempt_req = output.usage.requests
    per_attempt_in = output.usage.input_tokens

    # agent.usage must contain 3× per-attempt counters (attempts 1+2 captured
    # by retry block, attempt 3 by finally).
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
    """retry=3 with all 3 attempts failing — agent.usage must include the
    LAST failed attempt's tokens (captured by _on_retries_exhausted hook)."""
    _cleanup_db()
    clear_error_injection()
    inject_error_into_step(
        "response_processing", RuntimeError, "boom-all-fail", trigger_count=10
    )

    agent = Agent(MODEL, retry=3)
    task = Task(description="What is 1+1?", tools=[simple_math])

    raised = None
    try:
        await agent.do_async(task, return_output=True)
    except Exception as e:
        raised = e

    assert raised is not None, "do_async should have raised after all retries"

    # The last failed attempt's task._usage was preserved on the task.
    _assert_usage_positive(task._usage, "task after all-fail")

    per_attempt_req = task._usage.requests
    per_attempt_in = task._usage.input_tokens

    # agent.usage must contain 3× per-attempt (retry block: 1, 2; hook: 3).
    _assert_usage_positive(agent.usage, "agent after all-fail discard")
    assert agent.usage.requests == per_attempt_req * 3, (
        f"agent.usage.requests={agent.usage.requests} != per_attempt({per_attempt_req})*3 "
        f"(last attempt not captured by _on_retries_exhausted hook?)"
    )
    assert agent.usage.input_tokens == per_attempt_in * 3, (
        f"agent.usage.input_tokens={agent.usage.input_tokens} != per_attempt({per_attempt_in})*3"
    )

    clear_error_injection()


# ============================================================================
# 3. All-fail retry, user resumes via continue_run_async
# ============================================================================

@pytest.mark.asyncio
async def test_retry_all_fail_then_resume_no_double_count():
    """retry=3, all fail, then user resumes — the last-attempt usage captured
    by the exhaustion hook MUST NOT be double-counted on resume's finalize.
    The baseline-on-output prevents this."""
    _cleanup_db()
    clear_error_injection()
    inject_error_into_step(
        "response_processing", RuntimeError, "boom", trigger_count=3
    )

    db = SqliteDatabase(db_file=DB_FILE, session_id="s3", user_id="u3", full_session_memory=True)
    agent = Agent(MODEL, db=db, retry=3)
    task = Task(description="What is 9+9?", tools=[simple_math])

    try:
        await agent.do_async(task, return_output=True)
    except Exception:
        pass

    # After exhaustion: agent.usage has 3× per-attempt; output has baseline set.
    agent_usage_after_fail = TaskUsage.from_dict(agent.usage.to_dict())
    output_after_fail = agent._agent_run_output
    baseline_at_exhaustion = output_after_fail._agent_usage_baseline
    assert baseline_at_exhaustion is not None, (
        "_on_retries_exhausted must set _agent_usage_baseline on the output"
    )
    # Baseline should equal the last attempt's task._usage values.
    assert baseline_at_exhaustion.requests == task._usage.requests
    assert baseline_at_exhaustion.input_tokens == task._usage.input_tokens

    # Now resume
    final = await agent.continue_run_async(task=task, return_output=True)

    # agent.usage must grow by exactly (final.usage − baseline) on resume.
    _assert_resume_grew_by_delta(
        agent_before_resume=agent_usage_after_fail,
        agent_after_resume=agent.usage,
        final_usage=final.usage,
        baseline=baseline_at_exhaustion,
    )

    clear_error_injection()
    _cleanup_db()


# ============================================================================
# 4. Cross-process retry exhaustion + resume (storage-backed baseline)
# ============================================================================

@pytest.mark.asyncio
async def test_retry_exhaustion_baseline_survives_storage():
    """All-fail retry in agent A → continue_run_async in agent B (fresh
    instance loading from storage). The baseline written by the exhaustion
    hook on output must round-trip through storage so that B's resume does
    not double-count attempts captured in A."""
    _cleanup_db()
    clear_error_injection()
    inject_error_into_step(
        "response_processing", RuntimeError, "boom", trigger_count=3
    )

    db_a = SqliteDatabase(db_file=DB_FILE, session_id="x", user_id="x", full_session_memory=True)
    agent_a = Agent(MODEL, db=db_a, retry=3)
    task_a = Task(description="What is 2+2?", tools=[simple_math])

    try:
        await agent_a.do_async(task_a, return_output=True)
    except Exception:
        pass

    assert agent_a.usage is not None
    run_id = agent_a._agent_run_output.run_id
    baseline_in_memory = agent_a._agent_run_output._agent_usage_baseline
    assert baseline_in_memory is not None, "baseline missing in-memory after exhaustion"

    # Verify storage round-trip preserves the baseline.
    session = db_a.storage.get_session(session_id="x")
    stored_output = session.runs[run_id].output
    stored_baseline = getattr(stored_output, "_agent_usage_baseline", None)
    assert stored_baseline is not None, "baseline not persisted to storage"
    # Storage may keep it as dict or TaskUsage; normalize.
    if isinstance(stored_baseline, dict):
        stored_baseline = TaskUsage.from_dict(stored_baseline)
    assert stored_baseline.requests == baseline_in_memory.requests
    assert stored_baseline.input_tokens == baseline_in_memory.input_tokens

    # Resume in a fresh agent loading from storage.
    db_b = SqliteDatabase(db_file=DB_FILE, session_id="x", user_id="x", full_session_memory=True)
    agent_b = Agent(MODEL, db=db_b, retry=1)
    final = await agent_b.continue_run_async(run_id=run_id, return_output=True)

    # agent_b.usage is process-local; should contain only delta from baseline.
    # (Pre-exhaustion attempts are not in agent_b — they were in agent_a only.)
    assert agent_b.usage is not None
    delta_req = final.usage.requests - stored_baseline.requests
    delta_in = final.usage.input_tokens - stored_baseline.input_tokens
    assert agent_b.usage.requests == delta_req, (
        f"agent_b.usage.requests={agent_b.usage.requests} != delta {delta_req} "
        "(baseline didn't survive round-trip or wasn't honored)"
    )
    assert agent_b.usage.input_tokens == delta_in, (
        f"agent_b.usage.input_tokens={agent_b.usage.input_tokens} != delta {delta_in}"
    )

    clear_error_injection()
    _cleanup_db()


# ============================================================================
# 5. Single success — baseline does not interfere with normal flow
# ============================================================================

@pytest.mark.asyncio
async def test_single_run_no_regression_with_baseline():
    """A plain successful run (no retry, no HITL) must still produce
    agent.usage == output.usage. The new baseline plumbing must not
    perturb the simple single-accumulation case."""
    _cleanup_db()
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
# 6. HITL pause + resume — baseline mechanism must not perturb HITL semantics
# ============================================================================

@pytest.mark.asyncio
async def test_hitl_pause_resume_no_baseline_double_count():
    """An HITL-paused run that resumes must end up with
    ``agent.usage == final.usage`` (single count). The pause's
    ``_finalize_agent_usage`` was skipped (is_paused=True) and left
    ``_agent_usage_baseline=None``; on resume, the finally accumulates the
    full cumulative output.usage exactly once. No double-count from the
    new baseline plumbing."""
    _cleanup_db()
    clear_error_injection()

    agent = Agent(MODEL, retry=1)
    task = Task(
        description="Send a short notification with message 'hi'.",
        tools=[send_notification],
    )

    output = await agent.do_async(task, return_output=True)
    assert output.is_paused, "expected HITL pause"
    # During pause, _finalize_agent_usage skipped — agent.usage stays None.
    assert agent.usage is None, (
        f"agent.usage should be None during pause, got {agent.usage}"
    )

    # Provide the external tool result and resume.
    output.requirements[0].tool_execution.result = "Notification sent: hi"
    final = await agent.continue_run_async(run_id=output.run_id, return_output=True)

    assert final.is_complete, "expected completion after resume"
    _assert_usage_positive(final.usage, "HITL resume final")
    _assert_usage_positive(agent.usage, "HITL resume agent")
    # Resume's finally is the ONLY accumulation; agent must equal final exactly.
    assert agent.usage.requests == final.usage.requests, (
        f"HITL resume double-count: agent={agent.usage.requests}, final={final.usage.requests}"
    )
    assert agent.usage.input_tokens == final.usage.input_tokens
    assert agent.usage.output_tokens == final.usage.output_tokens


# ============================================================================
# 7. Durable error + continue_run_async (retry=1) — single-count after resume
# ============================================================================

@pytest.mark.asyncio
async def test_durable_error_resume_single_count():
    """retry=1: first (and only) attempt errors, user resumes via
    continue_run_async. The pause-on-error suppression of accumulation in
    the failed attempt's finally + the resume's full accumulation must
    produce ``agent.usage == final.usage`` (no double-count, no missing)."""
    _cleanup_db()
    clear_error_injection()
    inject_error_into_step(
        "response_processing", RuntimeError, "boom-once", trigger_count=1
    )

    db = SqliteDatabase(db_file=DB_FILE, session_id="s7", user_id="u7", full_session_memory=True)
    agent = Agent(MODEL, db=db, retry=1)
    task = Task(description="What is 6+6?", tools=[simple_math])

    try:
        await agent.do_async(task, return_output=True)
    except Exception:
        pass

    # With retry=1, @retryable's exhaustion hook fires after the single
    # failed attempt — agent.usage already has u_1 with baseline=u_1 set.
    assert agent.usage is not None, "exhaustion hook must capture u_1"
    baseline = agent._agent_run_output._agent_usage_baseline
    assert baseline is not None

    final = await agent.continue_run_async(task=task, return_output=True)
    assert final.is_complete

    # With retry=1, the only accumulator on the agent side is this single
    # run (captured by the exhaustion hook + the resume's baseline delta).
    # Therefore ``agent.usage == final.usage`` — no double-count, no missing.
    assert agent.usage.requests == final.usage.requests, (
        f"agent.usage ({agent.usage.requests}) != final.usage ({final.usage.requests})"
    )
    assert agent.usage.input_tokens == final.usage.input_tokens
    assert agent.usage.output_tokens == final.usage.output_tokens

    clear_error_injection()
    _cleanup_db()
