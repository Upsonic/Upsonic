"""
External Tool Execution Usage Examples

Demonstrates how to use external tool execution with HITL (Human-in-the-Loop).
External tools pause execution waiting for external completion before resuming.

Note: HITL continuation (continue_run_async) only supports direct call mode.
Streaming mode is not supported for continuation.
"""

import asyncio
import os
from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.db.database import SqliteDatabase
from upsonic.session.agent import AgentSession


def cleanup_db():
    """Clean up test database files."""
    if os.path.exists("external.db"):
        os.remove("external.db")


# ============================================================================
# EXTERNAL TOOLS DEFINITION
# ============================================================================

@tool(external_execution=True)
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email - requires external execution.
    
    Args:
        to: Email recipient
        subject: Email subject
        body: Email body content
        
    Returns:
        Confirmation message
    """
    pass


@tool(external_execution=True)
def execute_database_query(query: str) -> str:
    """
    Execute a database query - requires external execution.
    
    Args:
        query: SQL query to execute
        
    Returns:
        Query results
    """
    pass


@tool(external_execution=True)
def call_external_api(endpoint: str, payload: dict = None) -> dict:
    """
    Call an external API - requires external execution.
    
    Args:
        endpoint: API endpoint URL
        payload: Request payload
        
    Returns:
        API response
    """
    pass


# ============================================================================
# EXTERNAL TOOL EXECUTOR
# ============================================================================

def execute_tool_externally(requirement) -> str:
    """
    Execute an external tool based on the requirement.
    
    In a real application, this would call actual external services.
    """
    tool_exec = requirement.tool_execution
    tool_name = tool_exec.tool_name
    tool_args = tool_exec.tool_args
    
    if tool_name == "send_email":
        return f"Email sent successfully to {tool_args.get('to', 'unknown')}"
    
    elif tool_name == "execute_database_query":
        query = tool_args.get("query", "")
        return f"Query executed: {query} | Results: 10 rows returned"
    
    elif tool_name == "call_external_api":
        endpoint = tool_args.get("endpoint", "https://api.example.com")
        return str({"status": "success", "data": {"message": f"API called at {endpoint}"}})
    
    else:
        return f"Executed external tool: {tool_name}"


# ============================================================================
# VARIANT 1: Direct Call with run_id - Same Agent
# ============================================================================

async def external_direct_call_with_run_id_same_agent():
    """
    Direct call mode with external tool using run_id and same agent instance.
    
    The run_id is obtained from the output of do_async.
    """
    agent = Agent("openai/gpt-4o-mini", name="external_tool_agent")
    task = Task(
        description="Send an email to test@example.com with subject 'Hello' and body 'Test message'.",
        tools=[send_email]
    )
    
    output = await agent.do_async(task, return_output=True)
    
    for requirement in output.active_requirements:
        if requirement.is_external_tool_execution:
            result = execute_tool_externally(requirement)
            requirement.tool_execution.result = result
    
    result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
    return result


async def external_direct_call_with_task_same_agent():
    """
    Direct call mode with external tool using task and same agent instance.
    
    Uses in-memory context for continuation.
    """
    agent = Agent("openai/gpt-4o-mini", name="external_tool_agent")
    task = Task(
        description="Send an email to test@example.com with subject 'Hello' and body 'Test message'.",
        tools=[send_email]
    )
    
    output = await agent.do_async(task, return_output=True)
    
    for requirement in output.active_requirements:
        if requirement.is_external_tool_execution:
            result = execute_tool_externally(requirement)
            requirement.tool_execution.result = result
    
    result = await agent.continue_run_async(task=task, return_output=True)
    return result


# ============================================================================
# VARIANT 2: Direct Call with run_id - New Agent (Cross-process resumption)
# ============================================================================

async def external_direct_call_with_run_id_new_agent():
    """
    Direct call mode with external tool using run_id and new agent instance.
    
    Simulates cross-process resumption where a new agent instance loads
    the paused run from storage using run_id.
    
    IMPORTANT: When using a new agent, you must:
    1. Load the session from storage
    2. Set tool results on the requirements from storage (not in-memory)
    3. Save the session back to storage
    4. Then the new agent can load and resume
    """
    cleanup_db()
    db = SqliteDatabase(db_file="external.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", name="external_tool_agent", db=db, debug=True)
    task = Task(
        description="Send an email to test@example.com with subject 'Hello' and body 'Test message'.",
        tools=[send_email]
    )
    
    output = await agent.do_async(task, return_output=True)
    run_id = output.run_id
    
    # CRITICAL: Load session from storage and set results on STORAGE requirements
    session = await db.storage.read_async("session_1", AgentSession)
    if session and session.runs and run_id in session.runs:
        run_data = session.runs[run_id]
        for req in run_data.context.requirements:
            if req.is_external_tool_execution and not req.is_resolved():
                # Execute the tool
                result = execute_tool_externally(req)
                req.tool_execution.result = result
        
        # Save the updated session back to storage
        await db.storage.upsert_agent_session_async(session)
    
    # New agent will load from storage with the tool result already set
    new_agent = Agent("openai/gpt-4o-mini", name="external_tool_agent", db=db)
    result = await new_agent.continue_run_async(run_id=run_id, return_output=True)
    return result


# ============================================================================
# VARIANT 3: Using external_tool_executor parameter
# ============================================================================

async def external_with_executor_callback():
    """
    Use the external_tool_executor parameter to handle tools automatically.
    
    When provided, if the agent pauses again with NEW external tool requirements,
    the executor is called automatically for each requirement.
    """
    agent = Agent("openai/gpt-4o-mini", name="external_tool_agent")
    task = Task(
        description="Send an email to test@example.com with subject 'Hello' and body 'Test message'.",
        tools=[send_email]
    )
    
    output = await agent.do_async(task, return_output=True)
    
    for requirement in output.active_requirements:
        if requirement.is_external_tool_execution:
            result = execute_tool_externally(requirement)
            requirement.tool_execution.result = result
    
    result = await agent.continue_run_async(
        run_id=output.run_id, 
        return_output=True,
        external_tool_executor=execute_tool_externally
    )
    return result


# ============================================================================
# VARIANT 4: Multiple External Tools in Single Run
# ============================================================================

async def external_multiple_tools_direct_call():
    """
    Multiple external tools executed in a single run with direct call mode.
    
    Uses a loop to handle sequential external tool pauses.
    """
    agent = Agent("openai/gpt-4o-mini", name="external_tool_agent")
    task = Task(
        description=(
            "First, send an email to admin@example.com with subject 'Report' and body 'Monthly report'. "
            "Then query the database with 'SELECT * FROM users'. "
            "Finally, call the external API at https://api.example.com/data."
        ),
        tools=[send_email, execute_database_query, call_external_api]
    )
    
    output = await agent.do_async(task, return_output=True)
    
    while output.active_requirements:
        for requirement in output.active_requirements:
            if requirement.is_external_tool_execution:
                result = execute_tool_externally(requirement)
                requirement.tool_execution.result = result
        
        output = await agent.continue_run_async(
            run_id=output.run_id, 
            return_output=True
        )
    
    return output


async def external_multiple_tools_with_executor():
    """
    Multiple external tools with automatic executor callback.
    
    The external_tool_executor handles all subsequent tool pauses automatically.
    """
    agent = Agent("openai/gpt-4o-mini", name="external_tool_agent")
    task = Task(
        description=(
            "First, send an email to admin@example.com with subject 'Report' and body 'Monthly report'. "
            "Then query the database with 'SELECT * FROM users'. "
            "Finally, call the external API at https://api.example.com/data."
        ),
        tools=[send_email, execute_database_query, call_external_api]
    )
    
    output = await agent.do_async(task, return_output=True)
    
    for requirement in output.active_requirements:
        if requirement.is_external_tool_execution:
            result = execute_tool_externally(requirement)
            requirement.tool_execution.result = result
    
    result = await agent.continue_run_async(
        run_id=output.run_id, 
        return_output=True,
        external_tool_executor=execute_tool_externally
    )
    
    return result


# ============================================================================
# VARIANT 5: Cross-process External Tool Handling
# ============================================================================

async def external_cross_process_handling():
    """
    Pattern for handling external tools across process restarts.
    
    Demonstrates the full flow:
    1. Process A: do_async returns with paused status and requirements
    2. External system executes tools and stores results to storage
    3. Process B: Creates new agent, loads run by run_id, and continues
    """
    cleanup_db()
    
    # STEP 1: Initial run pauses for external tool
    db = SqliteDatabase(db_file="external.db", session_id="session_1", user_id="user_1")
    agent = Agent("openai/gpt-4o-mini", name="external_tool_agent", db=db)
    task = Task(
        description="Send an email to test@example.com with subject 'Test' and body 'Hello'.",
        tools=[send_email]
    )
    
    output = await agent.do_async(task, return_output=True)
    run_id = output.run_id
    
    if output.is_paused and output.active_requirements:
        print(f"Run {run_id} paused for external tools:")
        for req in output.active_requirements:
            if req.tool_execution:
                print(f"  - Tool: {req.tool_execution.tool_name}")
                print(f"    Call ID: {req.tool_execution.tool_call_id}")
                print(f"    Args: {req.tool_execution.tool_args}")
    
    # STEP 2: Load from storage, set tool results, save back
    print(f"\nExecuting external tools and saving results to storage...")
    session = await db.storage.read_async("session_1", AgentSession)
    if session and session.runs and run_id in session.runs:
        run_data = session.runs[run_id]
        for req in run_data.context.requirements:
            if req.is_external_tool_execution and not req.is_resolved():
                result = execute_tool_externally(req)
                req.tool_execution.result = result
                print(f"  Set result for {req.tool_execution.tool_name}: {result}")
        
        await db.storage.upsert_agent_session_async(session)
    
    # STEP 3: New agent instance resumes from storage (simulates Process B)
    print(f"\nCreating new agent to resume run {run_id}...")
    new_db = SqliteDatabase(db_file="external.db", session_id="session_1", user_id="user_1")
    new_agent = Agent("openai/gpt-4o-mini", name="external_tool_agent", db=new_db)
    
    result = await new_agent.continue_run_async(run_id=run_id, return_output=True)
    print(f"Final result: {result.content}")
    return result


async def run_all_tests():
    """Run all external tool execution test variants."""
    
    print("\n" + "="*80)
    print("TEST 1: Direct Call with run_id - Same Agent")
    print("="*80)
    result = await external_direct_call_with_run_id_same_agent()
    assert result.is_complete, f"TEST 1 FAILED: Expected complete, got {result.status}"
    print("TEST 1 PASSED")
    
    print("\n" + "="*80)
    print("TEST 2: Direct Call with task - Same Agent")
    print("="*80)
    result = await external_direct_call_with_task_same_agent()
    assert result.is_complete, f"TEST 2 FAILED: Expected complete, got {result.status}"
    print("TEST 2 PASSED")
    
    print("\n" + "="*80)
    print("TEST 3: Direct Call with run_id - New Agent (Cross-process)")
    print("="*80)
    result = await external_direct_call_with_run_id_new_agent()
    assert result.is_complete, f"TEST 3 FAILED: Expected complete, got {result.status}"
    print("TEST 3 PASSED")
    
    print("\n" + "="*80)
    print("TEST 4: Using external_tool_executor parameter")
    print("="*80)
    result = await external_with_executor_callback()
    assert result.is_complete, f"TEST 4 FAILED: Expected complete, got {result.status}"
    print("TEST 4 PASSED")
    
    print("\n" + "="*80)
    print("TEST 5: Multiple External Tools - Direct Call")
    print("="*80)
    result = await external_multiple_tools_direct_call()
    assert result.is_complete, f"TEST 5 FAILED: Expected complete, got {result.status}"
    print("TEST 5 PASSED")
    
    print("\n" + "="*80)
    print("TEST 6: Multiple External Tools - With Executor")
    print("="*80)
    result = await external_multiple_tools_with_executor()
    assert result.is_complete, f"TEST 6 FAILED: Expected complete, got {result.status}"
    print("TEST 6 PASSED")
    
    print("\n" + "="*80)
    print("TEST 7: Cross-process External Tool Handling")
    print("="*80)
    result = await external_cross_process_handling()
    assert result.is_complete, f"TEST 7 FAILED: Expected complete, got {result.status}"
    print("TEST 7 PASSED")
    
    # Cleanup
    cleanup_db()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
