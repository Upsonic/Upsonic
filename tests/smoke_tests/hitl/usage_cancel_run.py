"""
Cancel Run Usage Examples

Demonstrates how to cancel a running agent execution and resume.
Shows variants for direct call mode with HITL continuation.

Note: HITL continuation (continue_run_async) only supports direct call mode.
Streaming mode is not supported for continuation.
"""

import asyncio
import os
import time
from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.db.database import SqliteDatabase
from upsonic.run.base import RunStatus
from upsonic.run.cancel import cancel_run
from upsonic.session.agent import AgentSession


def cleanup_db():
    """Clean up test database files."""
    if os.path.exists("cancel.db"):
        os.remove("cancel.db")


@tool
def long_running_task(seconds: int) -> str:
    """A task that takes time to complete."""
    time.sleep(seconds)
    return f"Completed after {seconds} seconds"


# ============================================================================
# VARIANT 1: Direct Call with run_id - Same Agent
# ============================================================================

async def cancel_direct_call_with_run_id_same_agent():
    """
    Direct call mode with cancel and continue_run_async using run_id and same agent.
    
    Uses agent.run_id which is available during execution to cancel.
    """
    cleanup_db()
    db = SqliteDatabase(db_file="cancel.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db)
    task = Task(
        description="Call long_running_task with 10 seconds.",
        tools=[long_running_task]
    )
    
    async def cancel_after_delay():
        await asyncio.sleep(2)
        if agent.run_id:
            cancel_run(agent.run_id)
            print(f"  Cancelled run: {agent.run_id}")
    
    asyncio.create_task(cancel_after_delay())
    
    output = await agent.do_async(task, return_output=True)
    
    if output.status == RunStatus.cancelled:
        print(f"  Run was cancelled, resuming...")
        result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
        return result
    
    return output


async def cancel_direct_call_with_task_same_agent():
    """
    Direct call mode with cancel and continue_run_async using task and same agent.
    
    Uses the task object for continuation (in-memory context).
    """
    cleanup_db()
    db = SqliteDatabase(db_file="cancel.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db)
    task = Task(
        description="Call long_running_task with 10 seconds.",
        tools=[long_running_task]
    )
    
    async def cancel_after_delay():
        await asyncio.sleep(2)
        if agent.run_id:
            cancel_run(agent.run_id)
            print(f"  Cancelled run: {agent.run_id}")
    
    asyncio.create_task(cancel_after_delay())
    
    output = await agent.do_async(task, return_output=True)
    
    if output.status == RunStatus.cancelled:
        print(f"  Run was cancelled, resuming with task...")
        result = await agent.continue_run_async(task=task, return_output=True)
        return result
    
    return output


# ============================================================================
# VARIANT 2: Direct Call with run_id - New Agent (Cross-process resumption)
# ============================================================================

async def cancel_direct_call_with_run_id_new_agent():
    """
    Direct call mode with cancel and continue_run_async using run_id and new agent.
    
    Simulates cross-process resumption where a new agent instance loads 
    the cancelled run from storage using run_id.
    """
    cleanup_db()
    db = SqliteDatabase(db_file="cancel.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db)
    task = Task(
        description="Call long_running_task with 10 seconds.",
        tools=[long_running_task]
    )
    
    async def cancel_after_delay():
        await asyncio.sleep(2)
        if agent.run_id:
            cancel_run(agent.run_id)
            print(f"  Cancelled run: {agent.run_id}")
    
    asyncio.create_task(cancel_after_delay())
    
    output = await agent.do_async(task, return_output=True)
    run_id = output.run_id
    
    if output.status == RunStatus.cancelled:
        print(f"  Run was cancelled, creating new agent to resume...")
        
        # New agent loads from storage
        new_db = SqliteDatabase(db_file="cancel.db", session_id="session_1", user_id="user_1")
        new_agent = Agent("openai/gpt-4o-mini", db=new_db)
        result = await new_agent.continue_run_async(run_id=run_id, return_output=True)
        return result
    
    return output


# ============================================================================
# VARIANT 3: Using cancel_run function directly
# ============================================================================

async def cancel_using_cancel_run_function():
    """
    Cancel using the cancel_run function from upsonic.run.cancel.
    
    This approach uses the standalone cancel_run function which can be
    called from anywhere if you have the run_id.
    """
    cleanup_db()
    db = SqliteDatabase(db_file="cancel.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db)
    task = Task(
        description="Call long_running_task with 10 seconds.",
        tools=[long_running_task]
    )
    
    async def cancel_after_delay():
        await asyncio.sleep(2)
        if agent.run_id:
            cancel_run(agent.run_id)
            print(f"  Cancelled run using cancel_run(): {agent.run_id}")
    
    asyncio.create_task(cancel_after_delay())
    
    output = await agent.do_async(task, return_output=True)
    
    if output.status == RunStatus.cancelled:
        print(f"  Run was cancelled, resuming...")
        result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
        return result
    
    return output


# ============================================================================
# VARIANT 4: Immediate cancellation pattern with retry
# ============================================================================

async def cancel_with_immediate_check():
    """
    Pattern for checking cancellation status and resuming immediately.
    """
    cleanup_db()
    db = SqliteDatabase(db_file="cancel.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db)
    task = Task(
        description="Call long_running_task with 10 seconds.",
        tools=[long_running_task]
    )
    
    async def cancel_after_delay():
        await asyncio.sleep(2)
        if agent.run_id:
            cancel_run(agent.run_id)
            print(f"  Cancelled run: {agent.run_id}")
    
    cancel_task = asyncio.create_task(cancel_after_delay())
    
    output = await agent.do_async(task, return_output=True)
    
    # Try to cancel the cancel task if it hasn't run yet
    cancel_task.cancel()
    
    if output.is_cancelled:
        print(f"  Run {output.run_id} was cancelled. Resuming...")
        result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
        print(f"  Resumed run completed with status: {result.status}")
        return result
    
    return output


# ============================================================================
# VARIANT 5: Multiple resume attempts pattern
# ============================================================================

async def cancel_with_multiple_resume_attempts():
    """
    Pattern for handling multiple cancellations and resume attempts.
    This simulates a scenario where the run may be cancelled multiple times.
    """
    cleanup_db()
    db = SqliteDatabase(db_file="cancel.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db)
    task = Task(
        description="Call long_running_task with 8 seconds.",
        tools=[long_running_task]
    )
    
    max_attempts = 3
    attempt = 0
    output = None
    
    while attempt < max_attempts:
        attempt += 1
        print(f"  Attempt {attempt}...")
        
        if output is None:
            # First run - set up cancellation
            async def cancel_after_delay():
                await asyncio.sleep(1.5)
                if agent.run_id:
                    cancel_run(agent.run_id)
                    print(f"    Cancelled on attempt {attempt}")
            
            asyncio.create_task(cancel_after_delay())
            output = await agent.do_async(task, return_output=True)
        else:
            # Resume from previous cancelled run
            output = await agent.continue_run_async(run_id=output.run_id, return_output=True)
        
        if output.is_complete:
            print(f"  Run completed successfully on attempt {attempt}")
            return output
        elif output.is_cancelled:
            print(f"  Run cancelled on attempt {attempt}, will retry...")
            continue
        else:
            print(f"  Run ended with status {output.status}")
            break
    
    return output


# ============================================================================
# VARIANT 6: Verify resumption from cut-off point
# ============================================================================

async def cancel_verify_resume_from_cutoff():
    """
    Verify that when we resume a cancelled run, we continue from where
    we left off, NOT from the beginning.
    
    Uses debug=True and checks step_results in both AgentRunOutput and AgentRunContext.
    """
    cleanup_db()
    db = SqliteDatabase(db_file="cancel.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db, debug=True)
    task = Task(
        description="Call long_running_task with 10 seconds.",
        tools=[long_running_task]
    )
    
    print("\n  === INITIAL RUN ===")
    
    async def cancel_after_delay():
        await asyncio.sleep(2)
        if agent.run_id:
            cancel_run(agent.run_id)
            print(f"\n  [CANCEL] Cancelled run: {agent.run_id}")
    
    asyncio.create_task(cancel_after_delay())
    
    output = await agent.do_async(task, return_output=True)
    
    print(f"\n  === AFTER CANCEL ===")
    print(f"  Output status: {output.status}")
    print(f"  Output step_results count: {len(output.step_results)}")
    
    # Get context from agent
    context = agent._agent_run_context
    print(f"  Context step_results count: {len(context.step_results)}")
    
    # Check if they are the same
    print(f"\n  === STEP_RESULTS COMPARISON ===")
    print(f"  Output step_results is Context step_results: {output.step_results is context.step_results}")
    print(f"  Output step_results == Context step_results: {len(output.step_results) == len(context.step_results)}")
    
    # List step results at cancel
    print(f"\n  === STEP RESULTS AT CANCEL ===")
    for i, sr in enumerate(output.step_results):
        print(f"    [{i}] {sr.name}: {sr.status}")
    
    steps_before_resume = len(output.step_results)
    last_step_name_before = output.step_results[-1].name if output.step_results else None
    last_step_status_before = output.step_results[-1].status if output.step_results else None
    
    print(f"\n  Steps before resume: {steps_before_resume}")
    print(f"  Last step name: {last_step_name_before}")
    print(f"  Last step status: {last_step_status_before}")
    
    if output.status == RunStatus.cancelled:
        print(f"\n  === RESUMING ===")
        result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
        
        print(f"\n  === AFTER RESUME ===")
        print(f"  Result status: {result.status}")
        print(f"  Result step_results count: {len(result.step_results)}")
        
        # Get updated context
        context_after = agent._agent_run_context
        print(f"  Context step_results count: {len(context_after.step_results)}")
        
        # Check if they are the same
        print(f"\n  === STEP_RESULTS COMPARISON AFTER RESUME ===")
        print(f"  Result step_results is Context step_results: {result.step_results is context_after.step_results}")
        
        # List all step results after resume
        print(f"\n  === ALL STEP RESULTS AFTER RESUME ===")
        for i, sr in enumerate(result.step_results):
            # Mark steps that existed before resume
            marker = " (before cancel)" if i < steps_before_resume else " (after resume)"
            print(f"    [{i}] {sr.name}: {sr.status}{marker}")
        
        steps_after_resume = len(result.step_results)
        new_steps = steps_after_resume - steps_before_resume
        
        print(f"\n  === VERIFICATION ===")
        print(f"  Steps before resume: {steps_before_resume}")
        print(f"  Steps after resume: {steps_after_resume}")
        print(f"  New steps added: {new_steps}")
        
        if new_steps > 0:
            print(f"  ✅ VERIFIED: {new_steps} new steps were added after resume (continued from cut-off)")
        else:
            print(f"  ⚠️ No new steps added - may have been re-run from beginning")
        
        # Verify completion
        assert result.is_complete, f"Expected complete, got {result.status}"
        
        return result
    
    return output


# ============================================================================
# TEST RUNNER
# ============================================================================

async def run_all_tests():
    """Run all cancel run test variants."""
    
    print("\n" + "="*80)
    print("TEST 1: Direct Call with run_id - Same Agent")
    print("="*80)
    result = await cancel_direct_call_with_run_id_same_agent()
    assert result.is_complete, f"TEST 1 FAILED: Expected complete, got {result.status}"
    print("TEST 1 PASSED")
    
    print("\n" + "="*80)
    print("TEST 2: Direct Call with task - Same Agent")
    print("="*80)
    result = await cancel_direct_call_with_task_same_agent()
    assert result.is_complete, f"TEST 2 FAILED: Expected complete, got {result.status}"
    print("TEST 2 PASSED")
    
    print("\n" + "="*80)
    print("TEST 3: Direct Call with run_id - New Agent (Cross-process)")
    print("="*80)
    result = await cancel_direct_call_with_run_id_new_agent()
    assert result.is_complete, f"TEST 3 FAILED: Expected complete, got {result.status}"
    print("TEST 3 PASSED")
    
    print("\n" + "="*80)
    print("TEST 4: Using cancel_run function directly")
    print("="*80)
    result = await cancel_using_cancel_run_function()
    assert result.is_complete, f"TEST 4 FAILED: Expected complete, got {result.status}"
    print("TEST 4 PASSED")
    
    print("\n" + "="*80)
    print("TEST 5: Immediate cancellation with check")
    print("="*80)
    result = await cancel_with_immediate_check()
    assert result.is_complete, f"TEST 5 FAILED: Expected complete, got {result.status}"
    print("TEST 5 PASSED")
    
    print("\n" + "="*80)
    print("TEST 6: Multiple resume attempts")
    print("="*80)
    result = await cancel_with_multiple_resume_attempts()
    assert result.is_complete, f"TEST 6 FAILED: Expected complete, got {result.status}"
    print("TEST 6 PASSED")
    
    print("\n" + "="*80)
    print("TEST 7: Verify resumption from cut-off point (CRITICAL)")
    print("="*80)
    result = await cancel_verify_resume_from_cutoff()
    assert result.is_complete, f"TEST 7 FAILED: Expected complete, got {result.status}"
    print("TEST 7 PASSED")
    
    # Cleanup
    cleanup_db()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
