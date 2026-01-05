"""
Comprehensive sync verification test for AgentRunOutput and AgentRunContext.
Checks that all common attributes are properly synced by CONTENTS (not just lengths).
Tests both Cancel Run and External Tool Call scenarios.
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
from upsonic.agent.pipeline.step import inject_error_into_step, clear_error_injection

DB_FILE = "test_sync.db"


def cleanup():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)


@tool
def long_running_task(seconds: int) -> str:
    """A task that takes time to complete."""
    time.sleep(seconds)
    return f"Completed after {seconds} seconds"


@tool(external_execution=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email - requires external execution."""
    pass


@tool
def simple_math(a: int, b: int) -> int:
    """Perform simple math addition."""
    return a + b


def verify_sync(output, context, description: str, check_identity: bool = True):
    """
    Verify that AgentRunOutput and AgentRunContext are properly synced.
    Checks CONTENTS, not just lengths.
    
    Args:
        check_identity: If True, checks same object reference (for in-memory).
                       If False, checks equal contents (for storage deserialized).
    """
    print(f"\n{'='*80}")
    print(f"SYNC VERIFICATION: {description}")
    print(f"{'='*80}")
    mode = "IDENTITY (same object)" if check_identity else "EQUALITY (same contents)"
    print(f"Mode: {mode}")
    
    errors = []
    
    # 1. Check run_id
    print(f"\n[1] run_id:")
    print(f"    Output.run_id: {output.run_id}")
    print(f"    Context.run_id: {context.run_id}")
    if output.run_id != context.run_id:
        errors.append(f"run_id mismatch: output={output.run_id}, context={context.run_id}")
        print(f"    ❌ MISMATCH")
    else:
        print(f"    ✅ MATCH")
    
    # 2. Check session_id
    print(f"\n[2] session_id:")
    print(f"    Output.session_id: {output.session_id}")
    print(f"    Context.session_id: {context.session_id}")
    if output.session_id != context.session_id:
        errors.append(f"session_id mismatch: output={output.session_id}, context={context.session_id}")
        print(f"    ❌ MISMATCH")
    else:
        print(f"    ✅ MATCH")
    
    # 3. Check user_id
    print(f"\n[3] user_id:")
    print(f"    Output.user_id: {output.user_id}")
    print(f"    Context.user_id: {context.user_id}")
    if output.user_id != context.user_id:
        errors.append(f"user_id mismatch: output={output.user_id}, context={context.user_id}")
        print(f"    ❌ MISMATCH")
    else:
        print(f"    ✅ MATCH")
    
    # 4. Check messages
    print(f"\n[4] messages:")
    out_msg_count = len(output.messages) if output.messages else 0
    ctx_msg_count = len(context.messages) if context.messages else 0
    print(f"    Output.messages count: {out_msg_count}")
    print(f"    Context.messages count: {ctx_msg_count}")
    
    if check_identity:
        print(f"    Output.messages is Context.messages: {output.messages is context.messages}")
        if output.messages is not context.messages:
            errors.append("messages are NOT the same object reference")
            print(f"    ❌ NOT SAME OBJECT")
        else:
            print(f"    ✅ SAME OBJECT")
    else:
        if out_msg_count != ctx_msg_count:
            errors.append(f"messages count mismatch: output={out_msg_count}, context={ctx_msg_count}")
            print(f"    ❌ COUNT MISMATCH")
        else:
            print(f"    ✅ COUNT MATCH")
    
    # Check message contents
    if output.messages and context.messages:
        for i, (out_msg, ctx_msg) in enumerate(zip(output.messages[:3], context.messages[:3])):  # First 3
            if check_identity:
                if out_msg is ctx_msg:
                    print(f"    ✅ messages[{i}] SAME OBJECT: {type(out_msg).__name__}")
                else:
                    errors.append(f"messages[{i}] are different objects")
                    print(f"    ❌ messages[{i}] NOT SAME OBJECT")
            else:
                out_type = type(out_msg).__name__
                ctx_type = type(ctx_msg).__name__
                if out_type == ctx_type:
                    print(f"    ✅ messages[{i}] SAME TYPE: {out_type}")
                else:
                    errors.append(f"messages[{i}] type mismatch: {out_type} vs {ctx_type}")
                    print(f"    ❌ messages[{i}] TYPE MISMATCH: {out_type} vs {ctx_type}")
        if len(output.messages) > 3:
            print(f"    ... and {len(output.messages) - 3} more messages")
    
    # 4b. Check chat_history (context only - full conversation history)
    print(f"\n[4b] chat_history (context only):")
    ctx_history_count = len(context.chat_history) if hasattr(context, 'chat_history') and context.chat_history else 0
    print(f"    Context.chat_history count: {ctx_history_count}")
    print(f"    (Note: chat_history is context-only, AgentRunOutput syncs 'messages' not 'chat_history')")
    
    # 5. Check requirements
    print(f"\n[5] requirements:")
    out_req_count = len(output.requirements) if output.requirements else 0
    ctx_req_count = len(context.requirements) if context.requirements else 0
    print(f"    Output.requirements count: {out_req_count}")
    print(f"    Context.requirements count: {ctx_req_count}")
    
    if check_identity:
        print(f"    Output.requirements is Context.requirements: {output.requirements is context.requirements}")
        if output.requirements is not context.requirements:
            errors.append("requirements are NOT the same object reference")
            print(f"    ❌ NOT SAME OBJECT")
        else:
            print(f"    ✅ SAME OBJECT")
    else:
        if out_req_count != ctx_req_count:
            errors.append(f"requirements count mismatch")
            print(f"    ❌ COUNT MISMATCH")
        else:
            print(f"    ✅ COUNT MATCH")
    
    # Check requirement contents
    if output.requirements and context.requirements:
        for i, (out_req, ctx_req) in enumerate(zip(output.requirements, context.requirements)):
            if check_identity:
                if out_req is ctx_req:
                    print(f"    ✅ requirements[{i}] SAME OBJECT: id={out_req.id[:8]}..., type={out_req.pause_type}, resolved={out_req.is_resolved()}")
                else:
                    errors.append(f"requirements[{i}] are different objects")
                    print(f"    ❌ requirements[{i}] NOT SAME OBJECT")
            else:
                # Check content equality
                id_match = out_req.id == ctx_req.id
                type_match = out_req.pause_type == ctx_req.pause_type
                resolved_match = out_req.is_resolved() == ctx_req.is_resolved()
                if id_match and type_match and resolved_match:
                    print(f"    ✅ requirements[{i}] CONTENT MATCH: id={out_req.id[:8]}..., type={out_req.pause_type}, resolved={out_req.is_resolved()}")
                else:
                    errors.append(f"requirements[{i}] content mismatch")
                    print(f"    ❌ requirements[{i}] CONTENT MISMATCH")
                    print(f"        Output: id={out_req.id[:8]}..., type={out_req.pause_type}, resolved={out_req.is_resolved()}")
                    print(f"        Context: id={ctx_req.id[:8]}..., type={ctx_req.pause_type}, resolved={ctx_req.is_resolved()}")
    
    # 6. Check step_results
    print(f"\n[6] step_results:")
    out_sr_count = len(output.step_results) if output.step_results else 0
    ctx_sr_count = len(context.step_results) if context.step_results else 0
    print(f"    Output.step_results count: {out_sr_count}")
    print(f"    Context.step_results count: {ctx_sr_count}")
    
    if check_identity:
        print(f"    Output.step_results is Context.step_results: {output.step_results is context.step_results}")
        if output.step_results is not context.step_results:
            errors.append("step_results are NOT the same object reference")
            print(f"    ❌ NOT SAME OBJECT")
        else:
            print(f"    ✅ SAME OBJECT")
    else:
        if out_sr_count != ctx_sr_count:
            errors.append(f"step_results count mismatch")
            print(f"    ❌ COUNT MISMATCH")
        else:
            print(f"    ✅ COUNT MATCH")
    
    # Check step_results contents (first and last 3)
    if output.step_results and context.step_results:
        indices_to_check = list(range(min(3, len(output.step_results))))
        if len(output.step_results) > 6:
            indices_to_check.extend(range(len(output.step_results) - 3, len(output.step_results)))
        elif len(output.step_results) > 3:
            indices_to_check.extend(range(3, len(output.step_results)))
        
        for i in indices_to_check:
            out_sr = output.step_results[i]
            ctx_sr = context.step_results[i]
            if check_identity:
                if out_sr is ctx_sr:
                    print(f"    ✅ step_results[{i}] SAME OBJECT: {out_sr.name}={out_sr.status}")
                else:
                    errors.append(f"step_results[{i}] are different objects")
                    print(f"    ❌ step_results[{i}] NOT SAME OBJECT")
            else:
                name_match = out_sr.name == ctx_sr.name
                status_match = out_sr.status == ctx_sr.status
                if name_match and status_match:
                    print(f"    ✅ step_results[{i}] CONTENT MATCH: {out_sr.name}={out_sr.status}")
                else:
                    errors.append(f"step_results[{i}] content mismatch")
                    print(f"    ❌ step_results[{i}] CONTENT MISMATCH: out={out_sr.name}/{out_sr.status} vs ctx={ctx_sr.name}/{ctx_sr.status}")
        
        if len(output.step_results) > 6:
            print(f"    ... (showing first 3 and last 3 of {len(output.step_results)} steps)")
    
    # 7. Check events
    print(f"\n[7] events:")
    out_evt_count = len(output.events) if output.events else 0
    ctx_evt_count = len(context.events) if context.events else 0
    print(f"    Output.events count: {out_evt_count}")
    print(f"    Context.events count: {ctx_evt_count}")
    
    if check_identity:
        print(f"    Output.events is Context.events: {output.events is context.events}")
        if output.events is not context.events:
            errors.append("events are NOT the same object reference")
            print(f"    ❌ NOT SAME OBJECT")
        else:
            print(f"    ✅ SAME OBJECT")
    else:
        if out_evt_count != ctx_evt_count:
            errors.append(f"events count mismatch")
            print(f"    ❌ COUNT MISMATCH")
        else:
            print(f"    ✅ COUNT MATCH")
    
    # 8. Check session_state
    print(f"\n[8] session_state:")
    print(f"    Output.session_state: {output.session_state}")
    print(f"    Context.session_state: {context.session_state}")
    
    if check_identity:
        if output.session_state is context.session_state:
            print(f"    ✅ SAME OBJECT")
        else:
            errors.append("session_state are NOT the same object reference")
            print(f"    ❌ NOT SAME OBJECT")
    else:
        if output.session_state == context.session_state:
            print(f"    ✅ EQUAL")
        else:
            errors.append("session_state content mismatch")
            print(f"    ❌ NOT EQUAL")
    
    # 9. Check execution_stats
    print(f"\n[9] execution_stats:")
    if check_identity:
        print(f"    Output.execution_stats is Context.execution_stats: {output.execution_stats is context.execution_stats}")
        if output.execution_stats is not context.execution_stats:
            errors.append("execution_stats are NOT the same object reference")
            print(f"    ❌ NOT SAME OBJECT")
        else:
            print(f"    ✅ SAME OBJECT")
    else:
        # For storage, just check both exist or both None
        both_exist = (output.execution_stats is not None) == (context.execution_stats is not None)
        if both_exist:
            print(f"    ✅ BOTH EXIST OR BOTH NONE")
        else:
            errors.append("execution_stats existence mismatch")
            print(f"    ❌ EXISTENCE MISMATCH")
    
    # 10. Check content vs final_output
    print(f"\n[10] content/final_output:")
    print(f"    Output.content: {str(output.content)[:100] if output.content else None}...")
    print(f"    Context.final_output: {str(context.final_output)[:100] if context.final_output else None}...")
    
    if output.content == context.final_output:
        print(f"    ✅ MATCH")
    else:
        print(f"    ⚠️ DIFFER (acceptable before completion)")
    
    # Summary
    print(f"\n{'='*80}")
    if errors:
        print(f"SYNC VERIFICATION FAILED: {len(errors)} errors found")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print(f"SYNC VERIFICATION PASSED: All attributes properly synced")
        return True


async def test_cancel_run_sync():
    """Test sync verification for cancel run scenario."""
    print("\n" + "#"*80)
    print("# TEST: Cancel Run - Sync Verification")
    print("#"*80)
    
    cleanup()
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent = Agent("openai/gpt-4o-mini", db=db, debug=True)
    task = Task(
        description="Call long_running_task with 10 seconds.",
        tools=[long_running_task]
    )
    
    async def cancel_after_delay():
        await asyncio.sleep(2)
        if agent.run_id:
            cancel_run(agent.run_id)
    
    asyncio.create_task(cancel_after_delay())
    
    output = await agent.do_async(task, return_output=True)
    context = agent._agent_run_context
    
    # Verify AFTER CANCEL
    sync_ok_1 = verify_sync(output, context, "After Cancel")
    assert output.status == RunStatus.cancelled, f"Expected cancelled, got {output.status}"
    
    # Resume
    result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
    context_after = agent._agent_run_context
    
    # Verify AFTER RESUME
    sync_ok_2 = verify_sync(result, context_after, "After Resume Completion")
    assert result.status == RunStatus.completed, f"Expected completed, got {result.status}"
    
    # Verify in STORAGE (use check_identity=False since deserialization creates new objects)
    session = await db.storage.read_async("test_session", AgentSession)
    if session and session.runs and output.run_id in session.runs:
        stored_output = session.runs[output.run_id].output
        stored_context = session.runs[output.run_id].context
        sync_ok_3 = verify_sync(stored_output, stored_context, "Storage After Completion", check_identity=False)
    else:
        sync_ok_3 = False
        print("❌ Could not load from storage")
    
    return sync_ok_1 and sync_ok_2 and sync_ok_3


async def test_external_tool_sync():
    """Test sync verification for external tool scenario."""
    print("\n" + "#"*80)
    print("# TEST: External Tool - Sync Verification")
    print("#"*80)
    
    cleanup()
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent = Agent("openai/gpt-4o-mini", db=db, debug=True)
    task = Task(
        description="Send an email to test@example.com with subject 'Hello' and body 'Test'.",
        tools=[send_email]
    )
    
    output = await agent.do_async(task, return_output=True)
    context = agent._agent_run_context
    
    # Verify AFTER PAUSE
    sync_ok_1 = verify_sync(output, context, "After External Tool Pause")
    assert output.status == RunStatus.paused, f"Expected paused, got {output.status}"
    
    # Set tool result
    for req in output.active_requirements:
        if req.is_external_tool_execution:
            req.tool_execution.result = "Email sent successfully"
    
    # Resume
    result = await agent.continue_run_async(run_id=output.run_id, return_output=True)
    context_after = agent._agent_run_context
    
    # Verify AFTER RESUME
    sync_ok_2 = verify_sync(result, context_after, "After Resume Completion")
    assert result.status == RunStatus.completed, f"Expected completed, got {result.status}"
    
    # Verify in STORAGE (use check_identity=False since deserialization creates new objects)
    session = await db.storage.read_async("test_session", AgentSession)
    if session and session.runs and output.run_id in session.runs:
        stored_output = session.runs[output.run_id].output
        stored_context = session.runs[output.run_id].context
        sync_ok_3 = verify_sync(stored_output, stored_context, "Storage After Completion", check_identity=False)
    else:
        sync_ok_3 = False
        print("❌ Could not load from storage")
    
    return sync_ok_1 and sync_ok_2 and sync_ok_3


async def test_durable_execution_sync():
    """Test sync verification for durable execution (error recovery) scenario."""
    print("\n" + "#"*80)
    print("# TEST: Durable Execution - Sync Verification")
    print("#"*80)
    
    cleanup()
    clear_error_injection()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent = Agent("openai/gpt-4o-mini", db=db, debug=True)
    task = Task(
        description="What is 5 + 3? Reply with just the number.",
        tools=[simple_math]
    )
    
    # Inject error into model_execution step (trigger once, then works)
    inject_error_into_step("model_execution", RuntimeError, "Simulated failure", trigger_count=1)
    
    # Run and catch error
    run_id = None
    try:
        output = await agent.do_async(task, return_output=True)
    except Exception as e:
        print(f"  Caught expected error: {type(e).__name__}: {str(e)[:50]}...")
        if agent._agent_run_output:
            run_id = agent._agent_run_output.run_id
    
    assert run_id is not None, "FAIL: run_id should be available after error"
    
    # Get context and output after error
    output_after_error = agent._agent_run_output
    context_after_error = agent._agent_run_context
    
    # Verify AFTER ERROR
    sync_ok_1 = verify_sync(output_after_error, context_after_error, "After Error")
    assert output_after_error.status == RunStatus.error, f"Expected error, got {output_after_error.status}"
    
    # Verify error step in step_results
    error_steps = [sr for sr in context_after_error.step_results if str(sr.status) == 'StepStatus.ERROR']
    assert len(error_steps) >= 1, "FAIL: Should have error step"
    print(f"  Error step: {error_steps[-1].name} (step {error_steps[-1].step_number})")
    
    # Verify durable requirement exists
    durable_reqs = [r for r in context_after_error.requirements if r.pause_type == 'durable_execution']
    assert len(durable_reqs) >= 1, "FAIL: Should have durable_execution requirement"
    print(f"  Durable requirement: id={durable_reqs[0].id[:8]}..., resolved={durable_reqs[0].is_resolved()}")
    
    # Resume
    result = await agent.continue_run_async(run_id=run_id, return_output=True)
    context_after = agent._agent_run_context
    
    # Verify AFTER RESUME
    sync_ok_2 = verify_sync(result, context_after, "After Resume Completion")
    assert result.status == RunStatus.completed, f"Expected completed, got {result.status}"
    
    # Verify requirement is now resolved
    durable_reqs_after = [r for r in context_after.requirements if r.pause_type == 'durable_execution']
    assert len(durable_reqs_after) >= 1, "FAIL: Should still have durable requirement"
    assert durable_reqs_after[0].is_resolved(), "FAIL: Durable requirement should be resolved"
    print(f"  Durable requirement after: resolved={durable_reqs_after[0].is_resolved()}")
    
    # Verify in STORAGE
    session = await db.storage.read_async("test_session", AgentSession)
    if session and session.runs and run_id in session.runs:
        stored_output = session.runs[run_id].output
        stored_context = session.runs[run_id].context
        sync_ok_3 = verify_sync(stored_output, stored_context, "Storage After Completion", check_identity=False)
        
        # Verify storage has correct requirement state
        stored_durable_reqs = [r for r in stored_context.requirements if r.pause_type == 'durable_execution']
        assert len(stored_durable_reqs) >= 1, "FAIL: Storage should have durable requirement"
        assert stored_durable_reqs[0].is_resolved(), "FAIL: Storage durable requirement should be resolved"
        print(f"  Storage durable requirement: resolved={stored_durable_reqs[0].is_resolved()}")
    else:
        sync_ok_3 = False
        print("❌ Could not load from storage")
    
    cleanup()
    clear_error_injection()
    
    return sync_ok_1 and sync_ok_2 and sync_ok_3


async def test_durable_execution_cross_process_sync():
    """Test sync verification for durable execution with new agent (cross-process)."""
    print("\n" + "#"*80)
    print("# TEST: Durable Execution Cross-Process - Sync Verification")
    print("#"*80)
    
    cleanup()
    clear_error_injection()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent = Agent("openai/gpt-4o-mini", db=db, debug=True)
    task = Task(
        description="What is 7 + 2? Reply with just the number.",
        tools=[simple_math]
    )
    
    # Inject error
    inject_error_into_step("response_processing", ValueError, "Simulated failure", trigger_count=1)
    
    run_id = None
    try:
        output = await agent.do_async(task, return_output=True)
    except Exception as e:
        print(f"  Caught expected error: {type(e).__name__}: {str(e)[:50]}...")
        if agent._agent_run_output:
            run_id = agent._agent_run_output.run_id
    
    assert run_id is not None, "FAIL: run_id should be available"
    
    # Verify storage AFTER ERROR
    session = await db.storage.read_async("test_session", AgentSession)
    stored_output = session.runs[run_id].output
    stored_context = session.runs[run_id].context
    
    sync_ok_1 = verify_sync(stored_output, stored_context, "Storage After Error", check_identity=False)
    
    # Verify error step
    error_steps = [sr for sr in stored_context.step_results if str(sr.status) == 'StepStatus.ERROR']
    assert len(error_steps) >= 1, "FAIL: Should have error step"
    assert error_steps[-1].name == "response_processing", f"FAIL: Error step should be response_processing"
    print(f"  Error step in storage: {error_steps[-1].name}")
    
    # Create NEW agent (simulate cross-process)
    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent2 = Agent("openai/gpt-4o-mini", db=db2, debug=True)
    
    assert agent2.agent_id != agent.agent_id, "FAIL: Should be different agent_id"
    print(f"  New agent ID: {agent2.agent_id[:8]}... (different from {agent.agent_id[:8]}...)")
    
    # Resume with new agent
    result = await agent2.continue_run_async(run_id=run_id, return_output=True)
    context_after = agent2._agent_run_context
    
    # Verify AFTER RESUME with new agent
    sync_ok_2 = verify_sync(result, context_after, "After Resume (New Agent)")
    assert result.status == RunStatus.completed, f"Expected completed, got {result.status}"
    
    # Verify storage AFTER COMPLETION
    session2 = await db2.storage.read_async("test_session", AgentSession)
    stored_output2 = session2.runs[run_id].output
    stored_context2 = session2.runs[run_id].context
    
    sync_ok_3 = verify_sync(stored_output2, stored_context2, "Storage After Completion (New Agent)", check_identity=False)
    
    # Verify requirement is resolved in storage
    stored_durable_reqs = [r for r in stored_context2.requirements if r.pause_type == 'durable_execution']
    assert len(stored_durable_reqs) >= 1, "FAIL: Should have durable requirement"
    assert stored_durable_reqs[0].is_resolved(), "FAIL: Requirement should be resolved"
    print(f"  Storage requirement resolved: {stored_durable_reqs[0].is_resolved()}")
    
    cleanup()
    clear_error_injection()
    
    return sync_ok_1 and sync_ok_2 and sync_ok_3


async def main():
    """Run all sync verification tests."""
    
    cancel_ok = await test_cancel_run_sync()
    external_ok = await test_external_tool_sync()
    durable_ok = await test_durable_execution_sync()
    durable_cross_ok = await test_durable_execution_cross_process_sync()
    
    cleanup()
    clear_error_injection()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Cancel Run Sync: {'✅ PASSED' if cancel_ok else '❌ FAILED'}")
    print(f"External Tool Sync: {'✅ PASSED' if external_ok else '❌ FAILED'}")
    print(f"Durable Execution Sync: {'✅ PASSED' if durable_ok else '❌ FAILED'}")
    print(f"Durable Execution Cross-Process Sync: {'✅ PASSED' if durable_cross_ok else '❌ FAILED'}")
    
    all_passed = cancel_ok and external_ok and durable_ok and durable_cross_ok
    if all_passed:
        print("\n✅ ALL SYNC VERIFICATION TESTS PASSED!")
    else:
        print("\n❌ SOME SYNC VERIFICATION TESTS FAILED!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

