"""
Comprehensive HITL Test - Storage Validation with Full Content Verification
Tests External Tool, Cancel Run, and Durable Execution with full storage attribute and content verification.
"""

import asyncio
import os
import time
from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.run.base import RunStatus
from upsonic.run.cancel import cancel_run
from upsonic.db.database import SqliteDatabase
from upsonic.session.agent import AgentSession
from upsonic.agent.pipeline.step import inject_error_into_step, clear_error_injection

DEBUG = True
DB_FILE = "test_comprehensive.db"


def cleanup():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)


@tool(external_execution=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email - requires external execution."""
    pass


@tool
def long_running_task(seconds: int) -> str:
    """A task that takes time to complete."""
    time.sleep(seconds)
    return f"Completed after {seconds} seconds"


@tool
def simple_math(a: int, b: int) -> int:
    """Perform simple math addition."""
    return a + b


def execute_tool_externally(requirement) -> str:
    tool_exec = requirement.tool_execution
    tool_name = tool_exec.tool_name
    tool_args = tool_exec.tool_args
    
    if tool_name == "send_email":
        return f"Email sent successfully to {tool_args.get('to', 'unknown')}"
    return f"Executed external tool: {tool_name}"


def verify_agent_session(session: AgentSession, run_id: str, description: str):
    """Comprehensively verify AgentSession attributes and content."""
    print(f"\n  === AgentSession Verification: {description} ===")
    
    assert session is not None, f"FAIL [{description}]: AgentSession is None"
    assert session.session_id is not None, f"FAIL [{description}]: session_id is None"
    assert isinstance(session.session_id, str), f"FAIL [{description}]: session_id should be str"
    assert len(session.session_id) > 0, f"FAIL [{description}]: session_id should not be empty"
    
    assert session.runs is not None, f"FAIL [{description}]: runs dict is None"
    assert isinstance(session.runs, dict), f"FAIL [{description}]: runs should be dict"
    assert run_id in session.runs, f"FAIL [{description}]: run_id {run_id} not in session.runs"
    
    print(f"  session_id: {session.session_id}")
    print(f"  agent_id: {session.agent_id}")
    print(f"  user_id: {session.user_id}")
    print(f"  runs count: {len(session.runs)}")
    print(f"  runs keys: {list(session.runs.keys())}")
    
    return session.runs[run_id]


def verify_run_output(output, description: str, expected_status: RunStatus, 
                      expected_pause_reason: str = None):
    """Comprehensively verify AgentRunOutput attributes and content."""
    print(f"\n  === AgentRunOutput Verification: {description} ===")
    
    assert output is not None, f"FAIL [{description}]: AgentRunOutput is None"
    
    # Verify required fields
    assert output.run_id is not None, f"FAIL [{description}]: run_id is None"
    assert isinstance(output.run_id, str), f"FAIL [{description}]: run_id should be str"
    assert len(output.run_id) > 0, f"FAIL [{description}]: run_id should not be empty"
    
    assert output.agent_id is not None, f"FAIL [{description}]: agent_id is None"
    assert isinstance(output.agent_id, str), f"FAIL [{description}]: agent_id should be str"
    
    # Verify status
    assert output.status == expected_status, f"FAIL [{description}]: Expected status {expected_status}, got {output.status}"
    
    # Verify status boolean flags
    if expected_status == RunStatus.completed:
        assert output.is_complete == True, f"FAIL [{description}]: is_complete should be True"
        assert output.is_paused == False, f"FAIL [{description}]: is_paused should be False"
        assert output.is_error == False, f"FAIL [{description}]: is_error should be False"
        assert output.is_cancelled == False, f"FAIL [{description}]: is_cancelled should be False"
    elif expected_status == RunStatus.paused:
        assert output.is_complete == False, f"FAIL [{description}]: is_complete should be False"
        assert output.is_paused == True, f"FAIL [{description}]: is_paused should be True"
        assert output.is_error == False, f"FAIL [{description}]: is_error should be False"
        assert output.is_cancelled == False, f"FAIL [{description}]: is_cancelled should be False"
    elif expected_status == RunStatus.cancelled:
        assert output.is_complete == False, f"FAIL [{description}]: is_complete should be False"
        assert output.is_paused == False, f"FAIL [{description}]: is_paused should be False"
        assert output.is_error == False, f"FAIL [{description}]: is_error should be False"
        assert output.is_cancelled == True, f"FAIL [{description}]: is_cancelled should be True"
    
    # Verify pause_reason if expected
    if expected_pause_reason:
        assert output.pause_reason == expected_pause_reason, f"FAIL [{description}]: Expected pause_reason '{expected_pause_reason}', got '{output.pause_reason}'"
    
    # Verify requirements list
    assert output.requirements is not None, f"FAIL [{description}]: requirements should not be None"
    assert isinstance(output.requirements, list), f"FAIL [{description}]: requirements should be list"
    
    # Verify messages list
    assert output.messages is not None, f"FAIL [{description}]: messages should not be None"
    assert isinstance(output.messages, list), f"FAIL [{description}]: messages should be list"
    
    # Verify step_results list
    assert output.step_results is not None, f"FAIL [{description}]: step_results should not be None"
    assert isinstance(output.step_results, list), f"FAIL [{description}]: step_results should be list"
    
    print(f"  run_id: {output.run_id}")
    print(f"  agent_id: {output.agent_id}")
    print(f"  status: {output.status}")
    print(f"  pause_reason: {output.pause_reason}")
    print(f"  content: {str(output.content)[:100] if output.content else None}...")
    print(f"  requirements count: {len(output.requirements)}")
    print(f"  active_requirements count: {len(output.active_requirements)}")
    print(f"  messages count: {len(output.messages)}")
    print(f"  step_results count: {len(output.step_results)}")
    
    return output


def verify_run_context(context, description: str, expected_run_id: str = None):
    """Comprehensively verify AgentRunContext attributes and content."""
    print(f"\n  === AgentRunContext Verification: {description} ===")
    
    assert context is not None, f"FAIL [{description}]: AgentRunContext is None"
    
    # Verify run_id
    assert context.run_id is not None, f"FAIL [{description}]: run_id is None"
    assert isinstance(context.run_id, str), f"FAIL [{description}]: run_id should be str"
    if expected_run_id:
        assert context.run_id == expected_run_id, f"FAIL [{description}]: run_id mismatch"
    
    # Verify session_id
    assert context.session_id is not None, f"FAIL [{description}]: session_id is None"
    assert isinstance(context.session_id, str), f"FAIL [{description}]: session_id should be str"
    
    # Verify requirements list
    assert context.requirements is not None, f"FAIL [{description}]: requirements should not be None"
    assert isinstance(context.requirements, list), f"FAIL [{description}]: requirements should be list"
    
    # Verify messages list
    assert context.messages is not None, f"FAIL [{description}]: messages should not be None"
    assert isinstance(context.messages, list), f"FAIL [{description}]: messages should be list"
    
    # Verify step_results list
    assert context.step_results is not None, f"FAIL [{description}]: step_results should not be None"
    assert isinstance(context.step_results, list), f"FAIL [{description}]: step_results should be list"
    
    # Verify task
    assert context.task is not None, f"FAIL [{description}]: task should not be None"
    assert hasattr(context.task, 'description'), f"FAIL [{description}]: task should have description"
    
    print(f"  run_id: {context.run_id}")
    print(f"  session_id: {context.session_id}")
    print(f"  user_id: {context.user_id}")
    print(f"  is_streaming: {context.is_streaming}")
    print(f"  tool_call_count: {context.tool_call_count}")
    print(f"  requirements count: {len(context.requirements)}")
    print(f"  messages count: {len(context.messages)}")
    print(f"  step_results count: {len(context.step_results)}")
    print(f"  task description: {context.task.description[:50]}...")
    
    return context


def verify_requirement_external_tool(req, description: str, expected_resolved: bool,
                                      expected_tool_name: str, expected_tool_args: dict,
                                      expected_result: str = None):
    """Comprehensively verify RunRequirement for external tool with full content."""
    print(f"\n  === RunRequirement (External Tool) Verification: {description} ===")
    
    assert req is not None, f"FAIL [{description}]: RunRequirement is None"
    
    # Verify basic fields
    assert req.id is not None, f"FAIL [{description}]: id is None"
    assert isinstance(req.id, str), f"FAIL [{description}]: id should be str"
    assert len(req.id) > 0, f"FAIL [{description}]: id should not be empty"
    
    assert req.pause_type == 'external_tool', f"FAIL [{description}]: pause_type should be 'external_tool', got {req.pause_type}"
    
    # Verify created_at
    assert req.created_at is not None, f"FAIL [{description}]: created_at should not be None"
    
    # Verify resolved state
    if expected_resolved:
        assert req.is_resolved() == True, f"FAIL [{description}]: Expected resolved=True, got {req.is_resolved()}"
        assert req.resolved_at is not None, f"FAIL [{description}]: resolved_at should be set when resolved"
    else:
        assert req.is_resolved() == False, f"FAIL [{description}]: Expected resolved=False, got {req.is_resolved()}"
        assert req.resolved_at is None, f"FAIL [{description}]: resolved_at should be None when not resolved"
    
    # Verify tool_execution - CONTENT CHECK
    assert req.tool_execution is not None, f"FAIL [{description}]: tool_execution should not be None"
    te = req.tool_execution
    
    assert te.tool_name == expected_tool_name, f"FAIL [{description}]: tool_name should be '{expected_tool_name}', got '{te.tool_name}'"
    assert te.tool_call_id is not None, f"FAIL [{description}]: tool_call_id should not be None"
    assert isinstance(te.tool_call_id, str), f"FAIL [{description}]: tool_call_id should be str"
    
    # Verify tool_args content
    assert te.tool_args is not None, f"FAIL [{description}]: tool_args should not be None"
    assert isinstance(te.tool_args, dict), f"FAIL [{description}]: tool_args should be dict"
    for key, expected_value in expected_tool_args.items():
        assert key in te.tool_args, f"FAIL [{description}]: tool_args missing key '{key}'"
        assert te.tool_args[key] == expected_value, f"FAIL [{description}]: tool_args['{key}'] should be '{expected_value}', got '{te.tool_args[key]}'"
    
    assert te.external_execution_required == True, f"FAIL [{description}]: external_execution_required should be True"
    
    # Verify result if expected
    if expected_result:
        assert te.result == expected_result, f"FAIL [{description}]: result should be '{expected_result}', got '{te.result}'"
    
    # Verify continuation data - CONTENT CHECK
    messages, response, agent_state = req.get_continuation_data()
    
    assert messages is not None, f"FAIL [{description}]: continuation_messages should not be None"
    assert isinstance(messages, list), f"FAIL [{description}]: continuation_messages should be list"
    assert len(messages) > 0, f"FAIL [{description}]: continuation_messages should not be empty"
    
    # Verify messages have proper structure
    for i, msg in enumerate(messages):
        assert hasattr(msg, 'parts') or isinstance(msg, dict), f"FAIL [{description}]: message[{i}] should have 'parts' or be dict"
    
    assert agent_state is not None, f"FAIL [{description}]: agent_state should not be None"
    assert isinstance(agent_state, dict), f"FAIL [{description}]: agent_state should be dict"
    assert 'tool_call_count' in agent_state, f"FAIL [{description}]: agent_state should have 'tool_call_count'"
    assert 'tool_limit_reached' in agent_state, f"FAIL [{description}]: agent_state should have 'tool_limit_reached'"
    
    # Verify step_result
    assert req.step_result is not None, f"FAIL [{description}]: step_result should not be None"
    assert req.step_result.name is not None, f"FAIL [{description}]: step_result.name should not be None"
    assert req.step_result.step_number is not None, f"FAIL [{description}]: step_result.step_number should not be None"
    
    print(f"  id: {req.id}")
    print(f"  pause_type: {req.pause_type}")
    print(f"  is_resolved(): {req.is_resolved()}")
    print(f"  resolved_at: {req.resolved_at}")
    print(f"  tool_name: {te.tool_name}")
    print(f"  tool_call_id: {te.tool_call_id}")
    print(f"  tool_args: {te.tool_args}")
    print(f"  result: {te.result}")
    print(f"  step_result.name: {req.step_result.name}")
    print(f"  step_result.step_number: {req.step_result.step_number}")
    print(f"  continuation_messages count: {len(messages)}")
    print(f"  agent_state: {agent_state}")
    
    return req


def verify_requirement_cancel(req, description: str, expected_resolved: bool):
    """Comprehensively verify RunRequirement for cancel with full content."""
    print(f"\n  === RunRequirement (Cancel) Verification: {description} ===")
    
    assert req is not None, f"FAIL [{description}]: RunRequirement is None"
    
    # Verify basic fields
    assert req.id is not None, f"FAIL [{description}]: id is None"
    assert isinstance(req.id, str), f"FAIL [{description}]: id should be str"
    assert len(req.id) > 0, f"FAIL [{description}]: id should not be empty"
    
    assert req.pause_type == 'cancel', f"FAIL [{description}]: pause_type should be 'cancel', got {req.pause_type}"
    
    # Verify created_at
    assert req.created_at is not None, f"FAIL [{description}]: created_at should not be None"
    
    # Verify resolved state
    if expected_resolved:
        assert req.is_resolved() == True, f"FAIL [{description}]: Expected resolved=True, got {req.is_resolved()}"
        assert req.resolved_at is not None, f"FAIL [{description}]: resolved_at should be set when resolved"
    else:
        assert req.is_resolved() == False, f"FAIL [{description}]: Expected resolved=False, got {req.is_resolved()}"
        assert req.resolved_at is None, f"FAIL [{description}]: resolved_at should be None when not resolved"
    
    # Verify continuation data - CONTENT CHECK
    messages, response, agent_state = req.get_continuation_data()
    
    assert messages is not None, f"FAIL [{description}]: continuation_messages should not be None"
    assert isinstance(messages, list), f"FAIL [{description}]: continuation_messages should be list"
    assert len(messages) > 0, f"FAIL [{description}]: continuation_messages should not be empty"
    
    # Verify messages have proper structure
    for i, msg in enumerate(messages):
        assert hasattr(msg, 'parts') or isinstance(msg, dict), f"FAIL [{description}]: message[{i}] should have 'parts' or be dict"
    
    assert agent_state is not None, f"FAIL [{description}]: agent_state should not be None"
    assert isinstance(agent_state, dict), f"FAIL [{description}]: agent_state should be dict"
    
    # Verify step_result
    assert req.step_result is not None, f"FAIL [{description}]: step_result should not be None"
    assert req.step_result.name is not None, f"FAIL [{description}]: step_result.name should not be None"
    assert req.step_result.step_number is not None, f"FAIL [{description}]: step_result.step_number should not be None"
    
    print(f"  id: {req.id}")
    print(f"  pause_type: {req.pause_type}")
    print(f"  is_resolved(): {req.is_resolved()}")
    print(f"  resolved_at: {req.resolved_at}")
    print(f"  step_result.name: {req.step_result.name}")
    print(f"  step_result.step_number: {req.step_result.step_number}")
    print(f"  step_result.status: {req.step_result.status}")
    print(f"  continuation_messages count: {len(messages)}")
    print(f"  agent_state: {agent_state}")
    
    return req


def verify_messages_content(messages, description: str, min_count: int = 0):
    """Verify messages list content in detail."""
    print(f"\n  === Messages Content Verification: {description} ===")
    
    assert messages is not None, f"FAIL [{description}]: messages is None"
    assert isinstance(messages, list), f"FAIL [{description}]: messages should be list"
    assert len(messages) >= min_count, f"FAIL [{description}]: Expected at least {min_count} messages, got {len(messages)}"
    
    print(f"  Total messages: {len(messages)}")
    for i, msg in enumerate(messages[:5]):  # First 5 messages
        if hasattr(msg, 'parts'):
            print(f"  message[{i}]: type={type(msg).__name__}, parts_count={len(msg.parts)}")
            for j, part in enumerate(msg.parts[:2]):  # First 2 parts
                part_type = type(part).__name__
                content_preview = ""
                if hasattr(part, 'content'):
                    content_preview = str(part.content)[:50] + "..." if len(str(part.content)) > 50 else str(part.content)
                print(f"    part[{j}]: type={part_type}, content={content_preview}")
        elif isinstance(msg, dict):
            print(f"  message[{i}]: dict with keys={list(msg.keys())[:5]}")
        else:
            print(f"  message[{i}]: type={type(msg).__name__}")
    
    if len(messages) > 5:
        print(f"  ... and {len(messages) - 5} more messages")
    
    return messages


def verify_context_messages_and_history(context, description: str, 
                                        min_messages: int = 0, 
                                        min_chat_history: int = 1):
    """
    Verify both messages and chat_history in context.
    
    - messages: Only NEW messages from THIS run
    - chat_history: Full conversation history (historical + current run)
    """
    print(f"\n  === Context Messages & History: {description} ===")
    
    # Verify messages (run-specific)
    assert hasattr(context, 'messages'), f"FAIL [{description}]: context has no 'messages' attribute"
    assert context.messages is not None, f"FAIL [{description}]: messages is None"
    assert isinstance(context.messages, list), f"FAIL [{description}]: messages should be list"
    print(f"  messages (this run only): {len(context.messages)}")
    assert len(context.messages) >= min_messages, \
        f"FAIL [{description}]: Expected at least {min_messages} messages, got {len(context.messages)}"
    
    # Verify chat_history (full history)
    assert hasattr(context, 'chat_history'), f"FAIL [{description}]: context has no 'chat_history' attribute"
    assert context.chat_history is not None, f"FAIL [{description}]: chat_history is None"
    assert isinstance(context.chat_history, list), f"FAIL [{description}]: chat_history should be list"
    print(f"  chat_history (full history): {len(context.chat_history)}")
    assert len(context.chat_history) >= min_chat_history, \
        f"FAIL [{description}]: Expected at least {min_chat_history} chat_history, got {len(context.chat_history)}"
    
    # Print some details
    for i, msg in enumerate(context.messages[:2]):
        if hasattr(msg, 'parts'):
            print(f"    messages[{i}]: type={type(msg).__name__}, parts={len(msg.parts)}")
    for i, msg in enumerate(context.chat_history[:2]):
        if hasattr(msg, 'parts'):
            print(f"    chat_history[{i}]: type={type(msg).__name__}, parts={len(msg.parts)}")
    
    return context


def verify_step_results_content(step_results, description: str, min_count: int = 1):
    """Verify step_results list content in detail."""
    print(f"\n  === Step Results Content Verification: {description} ===")
    
    assert step_results is not None, f"FAIL [{description}]: step_results is None"
    assert isinstance(step_results, list), f"FAIL [{description}]: step_results should be list"
    assert len(step_results) >= min_count, f"FAIL [{description}]: Expected at least {min_count} step_results, got {len(step_results)}"
    
    for i, sr in enumerate(step_results[:5]):  # First 5 steps
        assert sr.name is not None, f"FAIL [{description}]: step_result[{i}].name is None"
        assert sr.status is not None, f"FAIL [{description}]: step_result[{i}].status is None"
        print(f"  step[{i}]: name={sr.name}, status={sr.status}, step_number={sr.step_number}")
    
    if len(step_results) > 5:
        print(f"  ... and {len(step_results) - 5} more steps")
    
    return step_results


async def test_external_tool_comprehensive():
    """
    TEST: External Tool - Comprehensive Storage and Content Verification
    """
    print("\n" + "="*80)
    print("TEST: External Tool - Comprehensive Storage and Content Verification")
    print("="*80)
    
    cleanup()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent1 = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description="Send an email to test@example.com with subject 'Hello' and body 'Test'.",
        tools=[send_email]
    )
    
    expected_tool_args = {'to': 'test@example.com', 'subject': 'Hello', 'body': 'Test'}
    expected_result = "Email sent successfully to test@example.com"
    
    # =========================================================================
    # STEP 1: Run do_async - should pause for external tool
    # =========================================================================
    print("\n[STEP 1] Running do_async (expecting pause for external tool)...")
    output = await agent1.do_async(task, return_output=True)
    saved_run_id = output.run_id
    
    assert output.is_paused == True, "FAIL: Should be paused"
    print(f"  Run paused with run_id: {saved_run_id}")
    print("[STEP 1] PASSED")
    
    # =========================================================================
    # STEP 2: Verify storage AFTER initial pause
    # =========================================================================
    print("\n[STEP 2] Verifying storage AFTER initial pause...")
    
    session = await db.storage.read_async("test_session", AgentSession)
    run_data = verify_agent_session(session, saved_run_id, "After Pause")
    
    stored_output = verify_run_output(run_data.output, "After Pause", RunStatus.paused, 
                                       expected_pause_reason='external_tool')
    
    stored_context = verify_run_context(run_data.context, "After Pause", expected_run_id=saved_run_id)
    
    # Verify both messages and chat_history
    # - messages: Only NEW messages from THIS run (may be 0 at pause time)
    # - chat_history: Full conversation history (should have at least 1)
    verify_context_messages_and_history(stored_context, "After Pause", 
                                        min_messages=0, min_chat_history=1)
    
    # Verify step_results content
    verify_step_results_content(stored_context.step_results, "After Pause Step Results", min_count=5)
    
    # Verify requirement with full content check
    ext_reqs = [r for r in stored_context.requirements if r.pause_type == 'external_tool']
    assert len(ext_reqs) == 1, f"FAIL: Should have 1 external_tool requirement, got {len(ext_reqs)}"
    
    stored_req = verify_requirement_external_tool(
        ext_reqs[0], "After Pause", 
        expected_resolved=False,
        expected_tool_name="send_email",
        expected_tool_args=expected_tool_args,
        expected_result=None
    )
    
    print("\n[STEP 2] PASSED - Storage verified after pause with full content check")
    
    # =========================================================================
    # STEP 3: Set external tool result and save
    # =========================================================================
    print("\n[STEP 3] Setting external tool result...")
    
    result_text = execute_tool_externally(stored_req)
    assert result_text == expected_result, f"FAIL: Result should be '{expected_result}'"
    
    stored_req.tool_execution.result = result_text
    await db.storage.upsert_agent_session_async(session)
    
    # Verify result is saved
    session_after_result = await db.storage.read_async("test_session", AgentSession)
    req_after_result = [r for r in session_after_result.runs[saved_run_id].context.requirements if r.pause_type == 'external_tool'][0]
    
    assert req_after_result.tool_execution.result == expected_result, f"FAIL: Result not saved correctly"
    
    print(f"  Tool result saved: {result_text}")
    print("[STEP 3] PASSED - Tool result saved and verified")
    
    # =========================================================================
    # STEP 4: Create fresh agent and resume
    # =========================================================================
    print("\n[STEP 4] Creating fresh agent and resuming...")
    
    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent2 = Agent("openai/gpt-4o-mini", name="agent2", db=db2, debug=DEBUG)
    
    assert agent2.agent_id != agent1.agent_id, "FAIL: Should be different agent_id"
    
    result = await agent2.continue_run_async(run_id=saved_run_id, return_output=True)
    
    assert result.is_complete == True, f"FAIL: Should be complete, got {result.status}"
    assert result.status == RunStatus.completed, f"FAIL: Status should be completed"
    assert result.content is not None, "FAIL: Content should not be None"
    assert isinstance(result.content, str), "FAIL: Content should be str"
    assert len(result.content) > 0, "FAIL: Content should not be empty"
    
    print(f"  Result status: {result.status}")
    print(f"  Result content: {result.content}")
    print("[STEP 4] PASSED - Resume completed")
    
    # =========================================================================
    # STEP 5: Verify storage AFTER completion
    # =========================================================================
    print("\n[STEP 5] Verifying storage AFTER completion...")
    
    final_session = await db2.storage.read_async("test_session", AgentSession)
    final_run_data = verify_agent_session(final_session, saved_run_id, "After Completion")
    
    final_output = verify_run_output(final_run_data.output, "After Completion", RunStatus.completed)
    assert final_output.content is not None, "FAIL: content should not be None"
    assert len(final_output.content) > 0, "FAIL: content should not be empty"
    
    final_context = verify_run_context(final_run_data.context, "After Completion", expected_run_id=saved_run_id)
    
    # Verify messages and chat_history after completion
    # After completion, messages (this run) should have content, chat_history should have full history
    verify_context_messages_and_history(final_context, "After Completion", 
                                        min_messages=0, min_chat_history=1)
    
    # Verify step_results content after completion
    verify_step_results_content(final_context.step_results, "After Completion Step Results", min_count=10)
    
    # Verify requirement is now resolved with full content check
    final_ext_reqs = [r for r in final_context.requirements if r.pause_type == 'external_tool']
    assert len(final_ext_reqs) == 1, f"FAIL: Should still have 1 requirement, got {len(final_ext_reqs)}"
    
    verify_requirement_external_tool(
        final_ext_reqs[0], "After Completion", 
        expected_resolved=True,
        expected_tool_name="send_email",
        expected_tool_args=expected_tool_args,
        expected_result=expected_result
    )
    
    print("\n[STEP 5] PASSED - Storage verified after completion with full content check")
    
    # =========================================================================
    # STEP 6: Verify in-memory state
    # =========================================================================
    print("\n[STEP 6] Verifying in-memory state...")
    
    memory_context = agent2._agent_run_context
    verify_run_context(memory_context, "In-Memory Context", expected_run_id=saved_run_id)
    
    verify_context_messages_and_history(memory_context, "In-Memory", 
                                        min_messages=0, min_chat_history=1)
    verify_step_results_content(memory_context.step_results, "In-Memory Step Results", min_count=10)
    
    memory_ext_reqs = [r for r in memory_context.requirements if r.pause_type == 'external_tool']
    verify_requirement_external_tool(
        memory_ext_reqs[0], "In-Memory Requirement", 
        expected_resolved=True,
        expected_tool_name="send_email",
        expected_tool_args=expected_tool_args,
        expected_result=expected_result
    )
    
    print("[STEP 6] PASSED - In-memory state verified with full content check")
    
    cleanup()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED: External Tool - Comprehensive Storage and Content Verification")
    print("="*80)


async def test_cancel_run_comprehensive():
    """
    TEST: Cancel Run - Comprehensive Storage and Content Verification
    """
    print("\n" + "="*80)
    print("TEST: Cancel Run - Comprehensive Storage and Content Verification")
    print("="*80)
    
    cleanup()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent1 = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description="Call long_running_task with 5 seconds.",
        tools=[long_running_task]
    )
    
    # =========================================================================
    # STEP 1: Start run and cancel
    # =========================================================================
    print("\n[STEP 1] Starting run and cancelling...")
    
    async def run_task():
        return await agent1.do_async(task, return_output=True)
    
    run_task_future = asyncio.create_task(run_task())
    await asyncio.sleep(1.5)
    
    run_id = agent1.run_id
    assert run_id is not None, "FAIL: run_id should not be None"
    assert isinstance(run_id, str), "FAIL: run_id should be str"
    
    cancel_run(run_id)
    
    try:
        output = await asyncio.wait_for(run_task_future, timeout=10.0)
    except asyncio.TimeoutError:
        raise AssertionError("FAIL: Task did not complete after cancel")
    
    saved_run_id = output.run_id
    assert output.is_cancelled == True, f"FAIL: Should be cancelled, got {output.status}"
    assert output.status == RunStatus.cancelled, "FAIL: Status should be cancelled"
    
    print(f"  Run cancelled with run_id: {saved_run_id}")
    print("[STEP 1] PASSED")
    
    # =========================================================================
    # STEP 2: Verify storage AFTER cancel
    # =========================================================================
    print("\n[STEP 2] Verifying storage AFTER cancel...")
    
    session = await db.storage.read_async("test_session", AgentSession)
    run_data = verify_agent_session(session, saved_run_id, "After Cancel")
    
    stored_output = verify_run_output(run_data.output, "After Cancel", RunStatus.cancelled,
                                       expected_pause_reason='cancel')
    
    stored_context = verify_run_context(run_data.context, "After Cancel", expected_run_id=saved_run_id)
    
    # Verify messages content
    # Verify both messages and chat_history
    verify_context_messages_and_history(stored_context, "After Cancel", 
                                        min_messages=0, min_chat_history=1)
    
    # Verify step_results content
    verify_step_results_content(stored_context.step_results, "After Cancel Step Results", min_count=5)
    
    # Verify requirement with full content check
    cancel_reqs = [r for r in stored_context.requirements if r.pause_type == 'cancel']
    assert len(cancel_reqs) >= 1, f"FAIL: Should have at least 1 cancel requirement, got {len(cancel_reqs)}"
    
    verify_requirement_cancel(cancel_reqs[0], "After Cancel", expected_resolved=False)
    
    print("\n[STEP 2] PASSED - Storage verified after cancel with full content check")
    
    # =========================================================================
    # STEP 3: Create fresh agent and resume
    # =========================================================================
    print("\n[STEP 3] Creating fresh agent and resuming...")
    
    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent2 = Agent("openai/gpt-4o-mini", name="agent2", db=db2, debug=DEBUG)
    
    assert agent2.agent_id != agent1.agent_id, "FAIL: Should be different agent_id"
    
    result = await agent2.continue_run_async(run_id=saved_run_id, return_output=True)
    
    assert result.is_complete == True, f"FAIL: Should be complete, got {result.status}"
    assert result.status == RunStatus.completed, f"FAIL: Status should be completed"
    assert result.content is not None, "FAIL: Content should not be None"
    assert isinstance(result.content, str), "FAIL: Content should be str"
    assert len(result.content) > 0, "FAIL: Content should not be empty"
    
    print(f"  Result status: {result.status}")
    print(f"  Result content: {result.content}")
    print("[STEP 3] PASSED - Resume completed")
    
    # =========================================================================
    # STEP 4: Verify storage AFTER completion
    # =========================================================================
    print("\n[STEP 4] Verifying storage AFTER completion...")
    
    final_session = await db2.storage.read_async("test_session", AgentSession)
    final_run_data = verify_agent_session(final_session, saved_run_id, "After Completion")
    
    final_output = verify_run_output(final_run_data.output, "After Completion", RunStatus.completed)
    assert final_output.content is not None, "FAIL: content should not be None"
    assert len(final_output.content) > 0, "FAIL: content should not be empty"
    
    final_context = verify_run_context(final_run_data.context, "After Completion", expected_run_id=saved_run_id)
    
    # Verify messages and chat_history after completion
    verify_context_messages_and_history(final_context, "After Completion (Cancel)", 
                                        min_messages=0, min_chat_history=1)
    
    # Verify step_results content after completion
    verify_step_results_content(final_context.step_results, "After Completion Step Results", min_count=10)
    
    # Verify requirement is now resolved with full content check
    final_cancel_reqs = [r for r in final_context.requirements if r.pause_type == 'cancel']
    assert len(final_cancel_reqs) >= 1, f"FAIL: Should still have cancel requirement, got {len(final_cancel_reqs)}"
    
    verify_requirement_cancel(final_cancel_reqs[0], "After Completion", expected_resolved=True)
    
    print("\n[STEP 4] PASSED - Storage verified after completion with full content check")
    
    # =========================================================================
    # STEP 5: Verify in-memory state
    # =========================================================================
    print("\n[STEP 5] Verifying in-memory state...")
    
    memory_context = agent2._agent_run_context
    verify_run_context(memory_context, "In-Memory Context", expected_run_id=saved_run_id)
    
    verify_context_messages_and_history(memory_context, "In-Memory (Cancel)", 
                                        min_messages=0, min_chat_history=1)
    verify_step_results_content(memory_context.step_results, "In-Memory Step Results", min_count=10)
    
    memory_cancel_reqs = [r for r in memory_context.requirements if r.pause_type == 'cancel']
    verify_requirement_cancel(memory_cancel_reqs[0], "In-Memory Requirement", expected_resolved=True)
    
    print("[STEP 5] PASSED - In-memory state verified with full content check")
    
    cleanup()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Cancel Run - Comprehensive Storage and Content Verification")
    print("="*80)


def verify_requirement_durable(req, description: str, expected_resolved: bool):
    """Comprehensively verify RunRequirement for durable execution with full content."""
    print(f"\n  === RunRequirement (Durable) Verification: {description} ===")
    
    assert req is not None, f"FAIL [{description}]: RunRequirement is None"
    
    # Verify basic fields
    assert req.id is not None, f"FAIL [{description}]: id is None"
    assert isinstance(req.id, str), f"FAIL [{description}]: id should be str"
    assert len(req.id) > 0, f"FAIL [{description}]: id should not be empty"
    
    assert req.pause_type == 'durable_execution', f"FAIL [{description}]: pause_type should be 'durable_execution', got {req.pause_type}"
    
    # Verify created_at
    assert req.created_at is not None, f"FAIL [{description}]: created_at should not be None"
    
    # Verify resolved state
    if expected_resolved:
        assert req.is_resolved() == True, f"FAIL [{description}]: Expected resolved=True, got {req.is_resolved()}"
        assert req.resolved_at is not None, f"FAIL [{description}]: resolved_at should be set when resolved"
    else:
        assert req.is_resolved() == False, f"FAIL [{description}]: Expected resolved=False, got {req.is_resolved()}"
        assert req.resolved_at is None, f"FAIL [{description}]: resolved_at should be None when not resolved"
    
    # Verify continuation data - CONTENT CHECK
    messages, response, agent_state = req.get_continuation_data()
    
    assert messages is not None, f"FAIL [{description}]: continuation_messages should not be None"
    assert isinstance(messages, list), f"FAIL [{description}]: continuation_messages should be list"
    
    assert agent_state is not None, f"FAIL [{description}]: agent_state should not be None"
    assert isinstance(agent_state, dict), f"FAIL [{description}]: agent_state should be dict"
    
    # Verify step_result
    assert req.step_result is not None, f"FAIL [{description}]: step_result should not be None"
    assert req.step_result.name is not None, f"FAIL [{description}]: step_result.name should not be None"
    assert req.step_result.step_number is not None, f"FAIL [{description}]: step_result.step_number should not be None"
    
    print(f"  id: {req.id}")
    print(f"  pause_type: {req.pause_type}")
    print(f"  is_resolved(): {req.is_resolved()}")
    print(f"  resolved_at: {req.resolved_at}")
    print(f"  step_result.name: {req.step_result.name}")
    print(f"  step_result.step_number: {req.step_result.step_number}")
    print(f"  step_result.status: {req.step_result.status}")
    print(f"  continuation_messages count: {len(messages)}")
    print(f"  agent_state: {agent_state}")
    
    return req


async def test_durable_execution_model_execution_step():
    """
    TEST: Durable Execution - Error in model_execution step
    Injects error in model_execution, verifies error state, then resumes successfully.
    """
    print("\n" + "="*80)
    print("TEST: Durable Execution - Error in model_execution step")
    print("="*80)
    
    cleanup()
    clear_error_injection()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent = Agent("openai/gpt-4o-mini", db=db, debug=DEBUG)
    task = Task(
        description="What is 5 + 3? Reply with just the number.",
        tools=[simple_math]
    )
    
    # Inject error into model_execution step (trigger_count=1 means error once, then works)
    inject_error_into_step("model_execution", RuntimeError, "Simulated model execution failure", trigger_count=1)
    
    # =========================================================================
    # STEP 1: Run do_async - should raise error
    # =========================================================================
    print("\n[STEP 1] Running do_async (expecting error in model_execution)...")
    
    run_id = None
    try:
        output = await agent.do_async(task, return_output=True)
        # Should not reach here
        assert False, "FAIL: Expected exception but got output"
    except Exception as e:
        print(f"  Caught expected error: {type(e).__name__}: {str(e)[:80]}...")
        assert "INJECTED ERROR" in str(e), f"FAIL: Expected injected error, got: {e}"
        
        # Get run_id from agent's internal state
        if agent._agent_run_output:
            run_id = agent._agent_run_output.run_id
            print(f"  Run ID from agent: {run_id}")
    
    assert run_id is not None, "FAIL: run_id should be available after error"
    print("[STEP 1] PASSED - Error caught as expected")
    
    # =========================================================================
    # STEP 2: Verify storage AFTER error
    # =========================================================================
    print("\n[STEP 2] Verifying storage AFTER error...")
    
    session = await db.storage.read_async("test_session", AgentSession)
    run_data = verify_agent_session(session, run_id, "After Error")
    
    stored_output = verify_run_output(run_data.output, "After Error", RunStatus.error,
                                       expected_pause_reason='durable_execution')
    
    stored_context = verify_run_context(run_data.context, "After Error", expected_run_id=run_id)
    
    # Verify step_results - should show model_execution with ERROR status
    verify_step_results_content(stored_context.step_results, "After Error Step Results", min_count=5)
    
    # Find the error step
    error_steps = [sr for sr in stored_context.step_results if str(sr.status) == 'StepStatus.ERROR']
    print(f"  Error steps found: {len(error_steps)}")
    if error_steps:
        print(f"  Error step: {error_steps[-1].name} at step {error_steps[-1].step_number}")
    
    # Verify durable execution requirement
    durable_reqs = [r for r in stored_context.requirements if r.pause_type == 'durable_execution']
    assert len(durable_reqs) >= 1, f"FAIL: Should have durable_execution requirement, got {len(durable_reqs)}"
    
    verify_requirement_durable(durable_reqs[0], "After Error", expected_resolved=False)
    
    print("\n[STEP 2] PASSED - Storage verified after error")
    
    # =========================================================================
    # STEP 3: Resume with same agent
    # =========================================================================
    print("\n[STEP 3] Resuming with same agent...")
    
    # Error should NOT trigger again (trigger_count=1 already used)
    result = await agent.continue_run_async(run_id=run_id, return_output=True)
    
    assert result.is_complete == True, f"FAIL: Should be complete, got {result.status}"
    assert result.status == RunStatus.completed, f"FAIL: Status should be completed"
    assert result.content is not None, "FAIL: Content should not be None"
    
    print(f"  Result status: {result.status}")
    print(f"  Result content: {result.content}")
    print("[STEP 3] PASSED - Resume completed")
    
    # =========================================================================
    # STEP 4: Verify storage AFTER completion
    # =========================================================================
    print("\n[STEP 4] Verifying storage AFTER completion...")
    
    final_session = await db.storage.read_async("test_session", AgentSession)
    final_run_data = verify_agent_session(final_session, run_id, "After Completion")
    
    final_output = verify_run_output(final_run_data.output, "After Completion", RunStatus.completed)
    
    final_context = verify_run_context(final_run_data.context, "After Completion", expected_run_id=run_id)
    
    # Verify requirement is now resolved
    final_durable_reqs = [r for r in final_context.requirements if r.pause_type == 'durable_execution']
    assert len(final_durable_reqs) >= 1, f"FAIL: Should still have durable requirement, got {len(final_durable_reqs)}"
    
    verify_requirement_durable(final_durable_reqs[0], "After Completion", expected_resolved=True)
    
    print("\n[STEP 4] PASSED - Storage verified after completion")
    
    cleanup()
    clear_error_injection()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Durable Execution - model_execution step")
    print("="*80)


async def test_durable_execution_response_processing_step():
    """
    TEST: Durable Execution - Error in response_processing step
    """
    print("\n" + "="*80)
    print("TEST: Durable Execution - Error in response_processing step")
    print("="*80)
    
    cleanup()
    clear_error_injection()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent = Agent("openai/gpt-4o-mini", db=db, debug=DEBUG)
    task = Task(
        description="What is 7 + 2? Reply with just the number.",
        tools=[simple_math]
    )
    
    # Inject error into response_processing step
    inject_error_into_step("response_processing", ValueError, "Simulated response processing failure", trigger_count=1)
    
    # =========================================================================
    # STEP 1: Run do_async - should raise error
    # =========================================================================
    print("\n[STEP 1] Running do_async (expecting error in response_processing)...")
    
    run_id = None
    try:
        output = await agent.do_async(task, return_output=True)
        assert False, "FAIL: Expected exception but got output"
    except Exception as e:
        print(f"  Caught expected error: {type(e).__name__}: {str(e)[:80]}...")
        assert "INJECTED ERROR" in str(e), f"FAIL: Expected injected error, got: {e}"
        
        if agent._agent_run_output:
            run_id = agent._agent_run_output.run_id
    
    assert run_id is not None, "FAIL: run_id should be available after error"
    print("[STEP 1] PASSED")
    
    # =========================================================================
    # STEP 2: Verify error step in storage
    # =========================================================================
    print("\n[STEP 2] Verifying error step in storage...")
    
    session = await db.storage.read_async("test_session", AgentSession)
    stored_context = session.runs[run_id].context
    
    # Find error step - should be response_processing
    error_steps = [sr for sr in stored_context.step_results if str(sr.status) == 'StepStatus.ERROR']
    assert len(error_steps) >= 1, f"FAIL: Should have error step"
    print(f"  Error step: {error_steps[-1].name}")
    assert error_steps[-1].name == "response_processing", f"FAIL: Error step should be response_processing"
    
    print("[STEP 2] PASSED")
    
    # =========================================================================
    # STEP 3: Resume with NEW agent (cross-process simulation)
    # =========================================================================
    print("\n[STEP 3] Resuming with NEW agent (cross-process)...")
    
    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent2 = Agent("openai/gpt-4o-mini", db=db2, debug=DEBUG)
    
    assert agent2.agent_id != agent.agent_id, "FAIL: Should be different agent_id"
    
    result = await agent2.continue_run_async(run_id=run_id, return_output=True)
    
    assert result.is_complete == True, f"FAIL: Should be complete, got {result.status}"
    print(f"  Result: {result.content}")
    print("[STEP 3] PASSED")
    
    # =========================================================================
    # STEP 4: Verify final state
    # =========================================================================
    print("\n[STEP 4] Verifying final state...")
    
    final_session = await db2.storage.read_async("test_session", AgentSession)
    final_context = final_session.runs[run_id].context
    
    durable_reqs = [r for r in final_context.requirements if r.pause_type == 'durable_execution']
    assert len(durable_reqs) >= 1, "FAIL: Should have durable requirement"
    assert durable_reqs[0].is_resolved(), "FAIL: Requirement should be resolved"
    
    print("[STEP 4] PASSED")
    
    cleanup()
    clear_error_injection()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Durable Execution - response_processing step")
    print("="*80)


async def test_durable_execution_tool_setup_step():
    """
    TEST: Durable Execution - Error in tool_setup step
    """
    print("\n" + "="*80)
    print("TEST: Durable Execution - Error in tool_setup step")
    print("="*80)
    
    cleanup()
    clear_error_injection()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent = Agent("openai/gpt-4o-mini", db=db, debug=DEBUG)
    task = Task(
        description="What is 10 + 5? Reply with just the number.",
        tools=[simple_math]
    )
    
    # Inject error into tool_setup step
    inject_error_into_step("tool_setup", RuntimeError, "Simulated tool setup failure", trigger_count=1)
    
    print("\n[STEP 1] Running do_async (expecting error in tool_setup)...")
    
    run_id = None
    try:
        output = await agent.do_async(task, return_output=True)
        assert False, "FAIL: Expected exception but got output"
    except Exception as e:
        print(f"  Caught expected error: {type(e).__name__}: {str(e)[:80]}...")
        if agent._agent_run_output:
            run_id = agent._agent_run_output.run_id
    
    assert run_id is not None, "FAIL: run_id should be available"
    print("[STEP 1] PASSED")
    
    print("\n[STEP 2] Verifying error step...")
    session = await db.storage.read_async("test_session", AgentSession)
    stored_context = session.runs[run_id].context
    
    error_steps = [sr for sr in stored_context.step_results if str(sr.status) == 'StepStatus.ERROR']
    assert len(error_steps) >= 1, "FAIL: Should have error step"
    print(f"  Error step: {error_steps[-1].name}")
    assert error_steps[-1].name == "tool_setup", f"FAIL: Error step should be tool_setup, got {error_steps[-1].name}"
    print("[STEP 2] PASSED")
    
    print("\n[STEP 3] Resuming...")
    result = await agent.continue_run_async(run_id=run_id, return_output=True)
    assert result.is_complete == True, f"FAIL: Should be complete, got {result.status}"
    print(f"  Result: {result.content}")
    print("[STEP 3] PASSED")
    
    cleanup()
    clear_error_injection()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Durable Execution - tool_setup step")
    print("="*80)


async def test_durable_execution_message_build_step():
    """
    TEST: Durable Execution - Error in message_build step
    """
    print("\n" + "="*80)
    print("TEST: Durable Execution - Error in message_build step")
    print("="*80)
    
    cleanup()
    clear_error_injection()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent = Agent("openai/gpt-4o-mini", db=db, debug=DEBUG)
    task = Task(
        description="What is 20 + 30? Reply with just the number.",
        tools=[simple_math]
    )
    
    # Inject error into message_build step
    inject_error_into_step("message_build", RuntimeError, "Simulated message build failure", trigger_count=1)
    
    print("\n[STEP 1] Running do_async (expecting error in message_build)...")
    
    run_id = None
    try:
        output = await agent.do_async(task, return_output=True)
        assert False, "FAIL: Expected exception but got output"
    except Exception as e:
        print(f"  Caught expected error: {type(e).__name__}: {str(e)[:80]}...")
        if agent._agent_run_output:
            run_id = agent._agent_run_output.run_id
    
    assert run_id is not None, "FAIL: run_id should be available"
    print("[STEP 1] PASSED")
    
    print("\n[STEP 2] Verifying error step...")
    session = await db.storage.read_async("test_session", AgentSession)
    stored_context = session.runs[run_id].context
    
    error_steps = [sr for sr in stored_context.step_results if str(sr.status) == 'StepStatus.ERROR']
    assert len(error_steps) >= 1, "FAIL: Should have error step"
    print(f"  Error step: {error_steps[-1].name}")
    assert error_steps[-1].name == "message_build", f"FAIL: Error step should be message_build"
    print("[STEP 2] PASSED")
    
    print("\n[STEP 3] Resuming...")
    result = await agent.continue_run_async(run_id=run_id, return_output=True)
    assert result.is_complete == True, f"FAIL: Should be complete, got {result.status}"
    print(f"  Result: {result.content}")
    print("[STEP 3] PASSED")
    
    cleanup()
    clear_error_injection()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Durable Execution - message_build step")
    print("="*80)


async def test_durable_execution_multiple_retries():
    """
    TEST: Durable Execution - Multiple retry pattern
    Error triggers twice, then succeeds on third attempt.
    """
    print("\n" + "="*80)
    print("TEST: Durable Execution - Multiple Retries (error 2x, success on 3rd)")
    print("="*80)
    
    cleanup()
    clear_error_injection()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user")
    agent = Agent("openai/gpt-4o-mini", db=db, debug=DEBUG)
    task = Task(
        description="What is 100 + 200? Reply with just the number."
    )
    
    # Inject error that triggers twice
    inject_error_into_step("model_execution", RuntimeError, "Simulated failure", trigger_count=2)
    
    run_id = None
    attempt = 0
    max_attempts = 3
    result = None
    
    while attempt < max_attempts:
        attempt += 1
        print(f"\n[ATTEMPT {attempt}]")
        
        try:
            if run_id is None:
                output = await agent.do_async(task, return_output=True)
                run_id = output.run_id
                if output.is_complete:
                    result = output
                    print(f"  Completed on attempt {attempt}")
                    break
            else:
                result = await agent.continue_run_async(run_id=run_id, return_output=True)
                if result.is_complete:
                    print(f"  Completed on attempt {attempt}")
                    break
        except Exception as e:
            print(f"  Error on attempt {attempt}: {str(e)[:60]}...")
            if agent._agent_run_output:
                run_id = agent._agent_run_output.run_id
            continue
    
    assert result is not None, "FAIL: Should have result"
    assert result.is_complete, f"FAIL: Should be complete, got {result.status}"
    assert attempt == 3, f"FAIL: Should complete on attempt 3, completed on {attempt}"
    
    print(f"\n  Final result: {result.content}")
    print(f"  Completed after {attempt} attempts")
    
    # Verify storage shows resolved requirement
    session = await db.storage.read_async("test_session", AgentSession)
    final_context = session.runs[run_id].context
    durable_reqs = [r for r in final_context.requirements if r.pause_type == 'durable_execution']
    
    # Should have 2 durable requirements (one for each error)
    print(f"  Durable requirements count: {len(durable_reqs)}")
    for i, req in enumerate(durable_reqs):
        print(f"    [{i}] resolved: {req.is_resolved()}")
    
    # At least the last one should be resolved
    assert any(r.is_resolved() for r in durable_reqs), "FAIL: At least one requirement should be resolved"
    
    cleanup()
    clear_error_injection()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Durable Execution - Multiple Retries")
    print("="*80)


async def main():
    """Run all comprehensive tests."""
    
    # External Tool Tests
    await test_external_tool_comprehensive()
    
    # Cancel Run Tests
    await test_cancel_run_comprehensive()
    
    # Durable Execution Tests
    await test_durable_execution_model_execution_step()
    await test_durable_execution_response_processing_step()
    await test_durable_execution_tool_setup_step()
    await test_durable_execution_message_build_step()
    await test_durable_execution_multiple_retries()
    
    print("\n" + "="*80)
    print("ALL COMPREHENSIVE TESTS PASSED!")
    print("  - External Tool: PASSED")
    print("  - Cancel Run: PASSED")
    print("  - Durable Execution (model_execution): PASSED")
    print("  - Durable Execution (response_processing): PASSED")
    print("  - Durable Execution (tool_setup): PASSED")
    print("  - Durable Execution (message_build): PASSED")
    print("  - Durable Execution (multiple retries): PASSED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
