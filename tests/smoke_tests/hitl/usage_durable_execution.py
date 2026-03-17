"""
Durable Execution Usage Examples

Demonstrates how to use durable execution with automatic recovery from errors.
Shows variants for direct call mode with HITL continuation.

Note: HITL continuation (continue_run_async) only supports direct call mode.
Streaming mode is not supported for continuation.
"""

import pytest
import asyncio
import os
from upsonic import Agent, Task
from upsonic.db.database import SqliteDatabase
from upsonic.agent.pipeline.step import inject_error_into_step, clear_error_injection

pytestmark = pytest.mark.timeout(300)


def cleanup_db():
    """Clean up test database files."""
    if os.path.exists("durable.db"):
        os.remove("durable.db")


def cleanup_all():
    """Clean up database and error injection."""
    cleanup_db()
    clear_error_injection()


# ============================================================================
# VARIANT 1: Direct Call with run_id - Same Agent
# ============================================================================

async def durable_direct_call_with_run_id_same_agent():
    """
    Direct call mode with error recovery using run_id and same agent instance.
    
    Injects error into model_execution step, catches it, then recovers using run_id.
    """
    cleanup_all()
    
    # Inject error that will trigger once (first attempt fails, second succeeds)
    inject_error_into_step("model_execution", RuntimeError, "Simulated model failure", trigger_count=1)
    
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db, retry=1, print=True)  # retry=1 to disable internal retries
    task = Task("What is 7 + 7? Reply with just the number.")
    
    try:
        result = await agent.do_async(task, return_output=True)
        # If no error, clear injection and return
        clear_error_injection()
        return result
    except Exception as e:
        print(f"  Error caught: {e}")
        agent_output = getattr(agent, '_agent_run_output', None)
        if agent_output:
            run_id = agent_output.run_id
            print(f"  Recovering with run_id: {run_id}")
            result = await agent.continue_run_async(run_id=run_id, return_output=True)
            clear_error_injection()
            return result
        clear_error_injection()
        raise e


async def durable_direct_call_with_task_same_agent():
    """
    Direct call mode with error recovery using task and same agent instance.
    
    Injects error, catches it, then recovers using task object.
    """
    cleanup_all()
    
    inject_error_into_step("model_execution", RuntimeError, "Simulated model failure", trigger_count=1)
    
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db, retry=1, print=True)  # retry=1 to disable internal retries
    task = Task("What is 7 + 7? Reply with just the number.")
    
    try:
        result = await agent.do_async(task, return_output=True)
        clear_error_injection()
        return result
    except Exception as e:
        print(f"  Error caught: {e}")
        print(f"  Recovering with task...")
        result = await agent.continue_run_async(task=task, return_output=True)
        clear_error_injection()
        return result


# ============================================================================
# VARIANT 2: Direct Call with run_id - New Agent (Cross-process resumption)
# ============================================================================

async def durable_direct_call_with_run_id_new_agent():
    """
    Direct call mode with error recovery using run_id and new agent instance.
    
    Simulates cross-process resumption where a new agent instance loads
    the errored run from storage using run_id.
    """
    cleanup_all()
    
    inject_error_into_step("model_execution", RuntimeError, "Simulated model failure", trigger_count=1)
    
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db, retry=1, print=True)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    run_id = None
    
    try:
        result = await agent.do_async(task, return_output=True)
        clear_error_injection()
        return result
    except Exception as e:
        print(f"  Error caught: {e}")
        agent_output = getattr(agent, '_agent_run_output', None)
        if agent_output:
            run_id = agent_output.run_id
        
        if run_id:
            print(f"  Creating new agent to recover with run_id: {run_id}")
            new_db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
            new_agent = Agent("openai/gpt-4o-mini", db=new_db, retry=1, print=True)
            result = await new_agent.continue_run_async(run_id=run_id, return_output=True)
            clear_error_injection()
            return result
        clear_error_injection()
        raise


# ============================================================================
# VARIANT 2A: Direct Call with task - New Agent (Cross-process resumption)
# ============================================================================

async def durable_direct_call_with_task_new_agent():
    """
    Direct call mode with error recovery using task and new agent instance.
    
    Simulates cross-process resumption where a new agent instance uses
    the task object for continuation.
    """
    cleanup_all()
    
    inject_error_into_step("model_execution", RuntimeError, "Simulated model failure", trigger_count=1)
    
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db, retry=1, print=True)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    try:
        result = await agent.do_async(task, return_output=True)
        clear_error_injection()
        return result
    except Exception as e:
        print(f"  Error caught: {e}")
        print(f"  Creating new agent to recover with task...")
        new_db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
        new_agent = Agent("openai/gpt-4o-mini", db=new_db, retry=1, print=True)
        result = await new_agent.continue_run_async(task=task, return_output=True)
        clear_error_injection()
        return result


# ============================================================================
# VARIANT 3: Retry with exponential backoff
# ============================================================================

async def durable_with_retry_backoff():
    """
    Pattern for retrying with exponential backoff on errors using run_id.
    
    Injects error that triggers twice, requiring multiple retries.
    """
    cleanup_all()
    
    # Error triggers twice - first two attempts fail, third succeeds
    inject_error_into_step("model_execution", RuntimeError, "Simulated model failure", trigger_count=2)
    
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db, retry=1, print=True)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    max_retries = 3
    base_delay = 0.1  # Short delay for testing
    result = None
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                result = await agent.do_async(task, return_output=True)
            else:
                agent_output = getattr(agent, '_agent_run_output', None)
                run_id = agent_output.run_id if agent_output else None
                if run_id:
                    result = await agent.continue_run_async(run_id=run_id, return_output=True)
                else:
                    result = await agent.do_async(task, return_output=True)
            
            if result.is_complete:
                print(f"  Completed on attempt {attempt + 1}")
                clear_error_injection()
                return result
            elif result.is_error:
                delay = base_delay * (2 ** attempt)
                print(f"  Error status on attempt {attempt + 1}, retrying in {delay}s...")
                await asyncio.sleep(delay)
                continue
            else:
                clear_error_injection()
                return result
                
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  Exception on attempt {attempt + 1}: {type(e).__name__}, retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                clear_error_injection()
                raise
    
    clear_error_injection()
    return result


async def durable_with_retry_backoff_task():
    """
    Pattern for retrying with exponential backoff on errors using task.
    
    Injects error that triggers twice, requiring multiple retries.
    """
    cleanup_all()
    
    inject_error_into_step("model_execution", RuntimeError, "Simulated model failure", trigger_count=2)
    
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db, retry=1, print=True)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    max_retries = 3
    base_delay = 0.1
    result = None
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                result = await agent.do_async(task, return_output=True)
            else:
                result = await agent.continue_run_async(task=task, return_output=True)
            
            if result.is_complete:
                print(f"  Completed on attempt {attempt + 1}")
                clear_error_injection()
                return result
            elif result.is_error:
                delay = base_delay * (2 ** attempt)
                print(f"  Error status on attempt {attempt + 1}, retrying in {delay}s...")
                await asyncio.sleep(delay)
                continue
            else:
                clear_error_injection()
                return result
                
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  Exception on attempt {attempt + 1}: {type(e).__name__}, retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                clear_error_injection()
                raise
    
    clear_error_injection()
    return result


# ============================================================================
# VARIANT 4: Check output status for error recovery
# ============================================================================

async def durable_with_status_check():
    """
    Pattern for checking output status and recovering from errors.
    
    This approach uses error status in the output rather than exceptions.
    Injects error that sets error status instead of throwing.
    """
    cleanup_all()
    
    # For this test, we inject a single error to verify the recovery path
    inject_error_into_step("response_processing", RuntimeError, "Simulated processing failure", trigger_count=1)
    
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db, retry=1, print=True)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    try:
        output = await agent.do_async(task, return_output=True)
        
        if output.is_error:
            print(f"  Run {output.run_id} errored: {output.error_details}")
            print("  Attempting recovery...")
            
            result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
            
            if result.is_complete:
                print("  Recovery successful!")
                clear_error_injection()
                return result
            else:
                print(f"  Recovery failed with status: {result.status}")
                clear_error_injection()
                return result
        
        clear_error_injection()
        return output
    except Exception as e:
        # Error was thrown as exception instead of status
        print(f"  Error thrown as exception: {e}")
        agent_output = getattr(agent, '_agent_run_output', None)
        if agent_output and agent_output.run_id:
            result = await agent.continue_run_async(run_id=agent_output.run_id, return_output=True)
            clear_error_injection()
            return result
        clear_error_injection()
        raise


async def durable_with_status_check_task():
    """
    Pattern for checking output status and recovering from errors using task.
    """
    cleanup_all()
    
    inject_error_into_step("response_processing", RuntimeError, "Simulated processing failure", trigger_count=1)
    
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db, retry=1, print=True)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    try:
        output = await agent.do_async(task, return_output=True)
        
        if output.is_error:
            print(f"  Run {output.run_id} errored: {output.error_details}")
            print("  Attempting recovery with task...")
            
            result = await agent.continue_run_async(task=task, return_output=True)
            
            if result.is_complete:
                print("  Recovery successful!")
                clear_error_injection()
                return result
            else:
                print(f"  Recovery failed with status: {result.status}")
                clear_error_injection()
                return result
        
        clear_error_injection()
        return output
    except Exception as e:
        print(f"  Error thrown as exception: {e}")
        result = await agent.continue_run_async(task=task, return_output=True)
        clear_error_injection()
        return result


# ============================================================================
# VARIANT 5: Cross-process recovery with run_id
# ============================================================================

async def durable_cross_process_recovery():
    """
    Pattern for recovering a run across process restarts using run_id.
    
    This simulates loading a previously errored run from storage.
    """
    cleanup_all()
    
    inject_error_into_step("model_execution", RuntimeError, "Simulated model failure", trigger_count=1)
    
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db, retry=1, print=True)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    try:
        result = await agent.do_async(task, return_output=True)
        clear_error_injection()
        return result
    except Exception as e:
        print(f"  Error caught: {e}")
        agent_output = getattr(agent, '_agent_run_output', None)
        if agent_output:
            run_id = agent_output.run_id
            print(f"  Run {run_id} failed. Creating new agent to recover...")
            
            # New agent loads from storage using run_id
            new_db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
            new_agent = Agent("openai/gpt-4o-mini", db=new_db, retry=1, print=True)
            result = await new_agent.continue_run_async(run_id=run_id, return_output=True)
            clear_error_injection()
            return result
        clear_error_injection()
        raise


async def durable_cross_process_recovery_task():
    """
    Pattern for recovering a run across process restarts using task.
    
    This simulates loading a previously errored run from storage using task object.
    """
    cleanup_all()
    
    inject_error_into_step("model_execution", RuntimeError, "Simulated model failure", trigger_count=1)
    
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db, retry=1, print=True)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    try:
        result = await agent.do_async(task, return_output=True)
        clear_error_injection()
        return result
    except Exception as e:
        print(f"  Error caught: {e}")
        print("  Creating new agent to recover with task...")
        
        # New agent uses task for continuation
        new_db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
        new_agent = Agent("openai/gpt-4o-mini", db=new_db, retry=1, print=True)
        result = await new_agent.continue_run_async(task=task, return_output=True)
        clear_error_injection()
        return result


# ============================================================================
# VARIANT 6: Error in different pipeline steps
# ============================================================================

async def durable_error_in_tool_setup_step():
    """
    Test recovery from error in tool setup step.
    """
    cleanup_all()
    
    inject_error_into_step("tool_setup", RuntimeError, "Simulated tool setup failure", trigger_count=1)
    
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db, retry=1, print=True)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    try:
        result = await agent.do_async(task, return_output=True)
        clear_error_injection()
        return result
    except Exception as e:
        print(f"  Tool setup error caught: {e}")
        agent_output = getattr(agent, '_agent_run_output', None)
        if agent_output and agent_output.run_id:
            result = await agent.continue_run_async(run_id=agent_output.run_id, return_output=True)
            clear_error_injection()
            return result
        clear_error_injection()
        raise


# ============================================================================
# PYTEST TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_durable_direct_call_with_run_id_same_agent():
    """Test: Direct Call with run_id - Same Agent (with injected error)"""
    result = await durable_direct_call_with_run_id_same_agent()
    assert result.is_complete, f"Expected complete, got {result.status}"


@pytest.mark.asyncio
async def test_durable_direct_call_with_task_same_agent():
    """Test: Direct Call with task - Same Agent (with injected error)"""
    result = await durable_direct_call_with_task_same_agent()
    assert result.is_complete, f"Expected complete, got {result.status}"


@pytest.mark.asyncio
async def test_durable_direct_call_with_run_id_new_agent():
    """Test: Direct Call with run_id - New Agent (with injected error)"""
    result = await durable_direct_call_with_run_id_new_agent()
    assert result.is_complete, f"Expected complete, got {result.status}"


@pytest.mark.asyncio
async def test_durable_direct_call_with_task_new_agent():
    """Test: Direct Call with task - New Agent (with injected error)"""
    result = await durable_direct_call_with_task_new_agent()
    assert result.is_complete, f"Expected complete, got {result.status}"


@pytest.mark.asyncio
async def test_durable_with_retry_backoff():
    """Test: Retry with exponential backoff - run_id (2 failures, 3 attempts)"""
    result = await durable_with_retry_backoff()
    assert result.is_complete, f"Expected complete, got {result.status}"


@pytest.mark.asyncio
async def test_durable_with_retry_backoff_task():
    """Test: Retry with exponential backoff - task (2 failures, 3 attempts)"""
    result = await durable_with_retry_backoff_task()
    assert result.is_complete, f"Expected complete, got {result.status}"


@pytest.mark.asyncio
async def test_durable_with_status_check():
    """Test: Check output status for error recovery - run_id"""
    result = await durable_with_status_check()
    assert result.is_complete, f"Expected complete, got {result.status}"


@pytest.mark.asyncio
async def test_durable_with_status_check_task():
    """Test: Check output status for error recovery - task"""
    result = await durable_with_status_check_task()
    assert result.is_complete, f"Expected complete, got {result.status}"


@pytest.mark.asyncio
async def test_durable_cross_process_recovery():
    """Test: Cross-process recovery - run_id (with injected error)"""
    result = await durable_cross_process_recovery()
    assert result.is_complete, f"Expected complete, got {result.status}"


@pytest.mark.asyncio
async def test_durable_cross_process_recovery_task():
    """Test: Cross-process recovery - task (with injected error)"""
    result = await durable_cross_process_recovery_task()
    assert result.is_complete, f"Expected complete, got {result.status}"


@pytest.mark.asyncio
async def test_durable_error_in_tool_setup_step():
    """Test: Error in tool setup step"""
    result = await durable_error_in_tool_setup_step()
    assert result.is_complete, f"Expected complete, got {result.status}"


# ============================================================================
# TEST RUNNER (for manual execution)
# ============================================================================

async def run_all_tests():
    """Run all durable execution test variants."""
    
    print("\n" + "="*80)
    print("TEST 1: Direct Call with run_id - Same Agent (with injected error)")
    print("="*80)
    result = await durable_direct_call_with_run_id_same_agent()
    assert result.is_complete, f"TEST 1 FAILED: Expected complete, got {result.status}"
    print("TEST 1 PASSED")
    
    print("\n" + "="*80)
    print("TEST 2: Direct Call with task - Same Agent (with injected error)")
    print("="*80)
    result = await durable_direct_call_with_task_same_agent()
    assert result.is_complete, f"TEST 2 FAILED: Expected complete, got {result.status}"
    print("TEST 2 PASSED")
    
    print("\n" + "="*80)
    print("TEST 3: Direct Call with run_id - New Agent (with injected error)")
    print("="*80)
    result = await durable_direct_call_with_run_id_new_agent()
    assert result.is_complete, f"TEST 3 FAILED: Expected complete, got {result.status}"
    print("TEST 3 PASSED")
    
    print("\n" + "="*80)
    print("TEST 3A: Direct Call with task - New Agent (with injected error)")
    print("="*80)
    result = await durable_direct_call_with_task_new_agent()
    assert result.is_complete, f"TEST 3A FAILED: Expected complete, got {result.status}"
    print("TEST 3A PASSED")
    
    print("\n" + "="*80)
    print("TEST 4: Retry with exponential backoff - run_id (2 failures, 3 attempts)")
    print("="*80)
    result = await durable_with_retry_backoff()
    assert result.is_complete, f"TEST 4 FAILED: Expected complete, got {result.status}"
    print("TEST 4 PASSED")
    
    print("\n" + "="*80)
    print("TEST 4A: Retry with exponential backoff - task (2 failures, 3 attempts)")
    print("="*80)
    result = await durable_with_retry_backoff_task()
    assert result.is_complete, f"TEST 4A FAILED: Expected complete, got {result.status}"
    print("TEST 4A PASSED")
    
    print("\n" + "="*80)
    print("TEST 5: Check output status for error recovery - run_id")
    print("="*80)
    result = await durable_with_status_check()
    assert result.is_complete, f"TEST 5 FAILED: Expected complete, got {result.status}"
    print("TEST 5 PASSED")
    
    print("\n" + "="*80)
    print("TEST 5A: Check output status for error recovery - task")
    print("="*80)
    result = await durable_with_status_check_task()
    assert result.is_complete, f"TEST 5A FAILED: Expected complete, got {result.status}"
    print("TEST 5A PASSED")
    
    print("\n" + "="*80)
    print("TEST 6: Cross-process recovery - run_id (with injected error)")
    print("="*80)
    result = await durable_cross_process_recovery()
    assert result.is_complete, f"TEST 6 FAILED: Expected complete, got {result.status}"
    print("TEST 6 PASSED")
    
    print("\n" + "="*80)
    print("TEST 6A: Cross-process recovery - task (with injected error)")
    print("="*80)
    result = await durable_cross_process_recovery_task()
    assert result.is_complete, f"TEST 6A FAILED: Expected complete, got {result.status}"
    print("TEST 6A PASSED")
    
    print("\n" + "="*80)
    print("TEST 8: Error in tool setup step")
    print("="*80)
    result = await durable_error_in_tool_setup_step()
    assert result.is_complete, f"TEST 8 FAILED: Expected complete, got {result.status}"
    print("TEST 8 PASSED")
    
    # Final cleanup
    cleanup_all()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
