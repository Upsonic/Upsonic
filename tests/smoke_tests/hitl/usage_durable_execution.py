"""
Durable Execution Usage Examples

Demonstrates how to use durable execution with automatic recovery from errors.
Shows variants for direct call mode with HITL continuation.

Note: HITL continuation (continue_run_async) only supports direct call mode.
Streaming mode is not supported for continuation.
"""

import asyncio
from upsonic import Agent, Task
from upsonic.db.database import SqliteDatabase
from upsonic.run.base import RunStatus


# ============================================================================
# VARIANT 1: Direct Call with run_id - Same Agent
# ============================================================================

async def durable_direct_call_with_run_id_same_agent():
    """
    Direct call mode with error recovery using run_id and same agent instance.
    
    The run_id is obtained from the output when an error occurs.
    """
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    try:
        result = await agent.do_async(task, return_output=True)
        return result
    except Exception as e:
        if agent._agent_run_output:
            run_id = agent._agent_run_output.run_id
            result = await agent.continue_run_async(run_id=run_id, return_output=True)
            return result
        raise e


async def durable_direct_call_with_task_same_agent():
    """
    Direct call mode with error recovery using task and same agent instance.
    
    Uses in-memory context for continuation.
    """
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    try:
        result = await agent.do_async(task, return_output=True)
        return result
    except Exception:
        result = await agent.continue_run_async(task=task, return_output=True)
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
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    run_id = None
    
    try:
        result = await agent.do_async(task, return_output=True)
        return result
    except Exception:
        if agent._agent_run_output:
            run_id = agent._agent_run_output.run_id
        
        if run_id:
            new_agent = Agent("openai/gpt-4o-mini", db=db)
            result = await new_agent.continue_run_async(run_id=run_id, return_output=True)
            return result
        raise


# ============================================================================
# VARIANT 3: Retry with exponential backoff
# ============================================================================

async def durable_with_retry_backoff():
    """
    Pattern for retrying with exponential backoff on errors.
    """
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                result = await agent.do_async(task, return_output=True)
            else:
                run_id = agent._agent_run_output.run_id if agent._agent_run_output else None
                if run_id:
                    result = await agent.continue_run_async(run_id=run_id, return_output=True)
                else:
                    result = await agent.do_async(task, return_output=True)
            
            if result.is_complete:
                return result
            elif result.is_error:
                delay = base_delay * (2 ** attempt)
                print(f"Error on attempt {attempt + 1}, retrying in {delay}s...")
                await asyncio.sleep(delay)
                continue
            else:
                return result
                
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Exception on attempt {attempt + 1}: {e}, retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                raise


# ============================================================================
# VARIANT 4: Check output status for error recovery
# ============================================================================

async def durable_with_status_check():
    """
    Pattern for checking output status and recovering from errors.
    
    This approach checks the output status after do_async returns
    (some errors may be caught internally and returned as error status).
    """
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db)
    task = Task("What is 7 + 7? Reply with just the number.")
    
    output = await agent.do_async(task, return_output=True)
    
    if output.is_error:
        print(f"Run {output.run_id} errored: {output.error_details}")
        print("Attempting recovery...")
        
        result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
        
        if result.is_complete:
            print("Recovery successful!")
            return result
        else:
            print(f"Recovery failed with status: {result.status}")
            return result
    
    return output


# ============================================================================
# VARIANT 5: Persistent retry across process restarts
# ============================================================================

async def durable_cross_process_recovery(run_id_to_resume: str = None):
    """
    Pattern for recovering a run across process restarts.
    
    This simulates loading a previously errored run from storage.
    In a real application, run_id would be stored externally (e.g., database).
    
    Args:
        run_id_to_resume: If provided, resume this run. Otherwise, start fresh.
    """
    db = SqliteDatabase(db_file="durable.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", db=db)
    
    if run_id_to_resume:
        print(f"Resuming run {run_id_to_resume} from storage...")
        result = await agent.continue_run_async(run_id=run_id_to_resume, return_output=True)
        return result
    else:
        task = Task("What is 7 + 7? Reply with just the number.")
        
        try:
            result = await agent.do_async(task, return_output=True)
            return result
        except Exception:
            if agent._agent_run_output:
                run_id = agent._agent_run_output.run_id
                print(f"Run {run_id} failed. Save this ID for later recovery.")
                return agent._agent_run_output
            raise

all_tests = [
    durable_direct_call_with_run_id_same_agent,
    durable_direct_call_with_task_same_agent,
    durable_direct_call_with_run_id_new_agent,
    durable_with_retry_backoff,
    durable_with_status_check,
    durable_cross_process_recovery,
]

async def run_all_tests():
    for test in all_tests:
        print(f"Running test {test.__name__}...")
        result = await test()
        print(f"Test {test.__name__} {'PASSED' if result.is_complete else 'FAILED'}")
        print(f"Result: {result}")
        print("="*80)

if __name__ == "__main__":
    asyncio.run(run_all_tests())
