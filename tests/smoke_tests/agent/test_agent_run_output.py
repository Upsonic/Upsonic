"""
Comprehensive Smoke Tests for AgentRunOutput

This test suite validates all aspects of AgentRunOutput:
- Basic creation and initialization
- Properties (is_paused, is_cancelled, is_complete, etc.)
- Status methods (mark_paused, mark_completed, mark_error, etc.)
- Message tracking methods
- Usage tracking methods
- Serialization (to_dict, from_dict, to_json)
- Requirement methods
- Step result methods
- Tool-related methods

Run with: python3 -m pytest tests/smoke_tests/agent/test_agent_run_output.py -v
"""

import pytest
from typing import List, Dict, Any, Optional
from time import sleep

from upsonic.run.agent.output import AgentRunOutput
from upsonic.run.base import RunStatus
from upsonic.run.agent.input import AgentRunInput
from upsonic.tasks.tasks import Task
from upsonic.messages.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from upsonic.run.requirements import RunRequirement
from upsonic.run.tools.tools import ToolExecution
from upsonic.usage import RunUsage, RequestUsage
from upsonic.agent.pipeline.step import StepResult, StepStatus
from upsonic.run.pipeline.stats import PipelineExecutionStats




def test_basic_creation():
    """Test basic AgentRunOutput creation with minimal attributes."""
    print("\n" + "=" * 60)
    print("TEST: Basic Creation")
    print("=" * 60)
    
    output = AgentRunOutput(
        run_id="test_run_001",
        agent_id="agent_001",
        agent_name="Test Agent",
        session_id="session_001",
        user_id="user_001"
    )
    
    assert output.run_id == "test_run_001"
    assert output.agent_id == "agent_001"
    assert output.agent_name == "Test Agent"
    assert output.session_id == "session_001"
    assert output.user_id == "user_001"
    assert output.status == RunStatus.running
    assert output.is_streaming is False
    assert output.accumulated_text == ""
    assert output.chat_history == []
    assert output.messages is None
    assert output.tools is None
    assert output.requirements is None
    assert output.step_results == []
    assert output.created_at > 0
    
    print("✅ Basic creation test passed")


def test_creation_with_all_attributes():
    """Test AgentRunOutput creation with all attributes."""
    print("\n" + "=" * 60)
    print("TEST: Creation with All Attributes")
    print("=" * 60)
    
    task = Task("Test task")
    run_input = AgentRunInput(user_prompt="Test prompt")
    
    output = AgentRunOutput(
        run_id="test_run_002",
        agent_id="agent_002",
        agent_name="Full Agent",
        session_id="session_002",
        user_id="user_002",
        parent_run_id="parent_001",
        task=task,
        input=run_input,
        output="Test output",
        output_schema=str,
        thinking_content="Test thinking",
        model_name="anthropic/claude-sonnet-4-5",
        model_provider="anthropic",
        chat_history=[],
        messages=[],
        usage=RunUsage(),
        memory_message_count=5,
        tools=[],
        tool_call_count=3,
        tool_limit_reached=False,
        images=None,
        files=None,
        status=RunStatus.completed,
        requirements=[],
        step_results=[],
        execution_stats=None,
        events=[],
        metadata={"key": "value"},
        session_state={"state": "active"},
        is_streaming=True,
        accumulated_text="Accumulated",
        pause_reason=None,
        error_details=None
    )
    
    assert output.run_id == "test_run_002"
    assert output.task == task
    assert output.input == run_input
    assert output.output == "Test output"
    assert output.status == RunStatus.completed
    assert output.is_streaming is True
    assert output.accumulated_text == "Accumulated"
    assert output.metadata == {"key": "value"}
    
    print("✅ Full attribute creation test passed")




def test_status_properties():
    """Test all status-related properties."""
    print("\n" + "=" * 60)
    print("TEST: Status Properties")
    print("=" * 60)
    
    # Test is_paused
    output_paused = AgentRunOutput(run_id="test_001", status=RunStatus.paused)
    assert output_paused.is_paused is True
    assert output_paused.is_complete is False
    assert output_paused.is_cancelled is False
    assert output_paused.is_error is False
    assert output_paused.is_problematic is True
    
    # Test is_cancelled
    output_cancelled = AgentRunOutput(run_id="test_002", status=RunStatus.cancelled)
    assert output_cancelled.is_cancelled is True
    assert output_cancelled.is_complete is False
    assert output_cancelled.is_paused is False
    assert output_cancelled.is_error is False
    assert output_cancelled.is_problematic is True
    
    # Test is_complete
    output_complete = AgentRunOutput(run_id="test_003", status=RunStatus.completed)
    assert output_complete.is_complete is True
    assert output_complete.is_paused is False
    assert output_complete.is_cancelled is False
    assert output_complete.is_error is False
    assert output_complete.is_problematic is False
    
    # Test is_error
    output_error = AgentRunOutput(run_id="test_004", status=RunStatus.error)
    assert output_error.is_error is True
    assert output_error.is_complete is False
    assert output_error.is_paused is False
    assert output_error.is_cancelled is False
    assert output_error.is_problematic is True
    
    # Test running
    output_running = AgentRunOutput(run_id="test_005", status=RunStatus.running)
    assert output_running.is_paused is False
    assert output_running.is_cancelled is False
    assert output_running.is_complete is False
    assert output_running.is_error is False
    assert output_running.is_problematic is False
    
    print("✅ Status properties test passed")


def test_active_requirements_property():
    """Test active_requirements property."""
    print("\n" + "=" * 60)
    print("TEST: Active Requirements Property")
    print("=" * 60)
    
    # Test with no requirements
    output = AgentRunOutput(run_id="test_001")
    assert output.active_requirements == []
    
    # Test with requirements that need external execution
    tool_exec = ToolExecution(
        tool_name="external_tool",
        tool_call_id="call_001",
        external_execution_required=True
    )
    requirement = RunRequirement(
        id="req_001",
        tool_execution=tool_exec
    )
    requirement.tool_execution.result = None  # Not executed yet
    
    output.requirements = [requirement]
    assert len(output.active_requirements) == 1
    assert output.active_requirements[0] == requirement
    
    # Test with requirement that has result (no longer active)
    requirement.tool_execution.result = "result"
    assert len(output.active_requirements) == 0
    
    print("✅ Active requirements property test passed")


def test_tools_properties():
    """Test tool-related properties."""
    print("\n" + "=" * 60)
    print("TEST: Tools Properties")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001")
    
    # Test with no tools
    assert output.tools_requiring_confirmation == []
    assert output.tools_requiring_user_input == []
    assert output.tools_awaiting_external_execution == []
    
    # Test with tools requiring confirmation (confirmed=None means pending)
    tool1 = ToolExecution(
        tool_name="tool1",
        tool_call_id="call_001",
        requires_confirmation=True,
        confirmed=None
    )
    
    # Test with tools requiring user input
    tool2 = ToolExecution(
        tool_name="tool2",
        tool_call_id="call_002",
        requires_user_input=True,
        answered=False
    )
    
    # Test with tools awaiting external execution
    tool3 = ToolExecution(
        tool_name="tool3",
        tool_call_id="call_003",
        external_execution_required=True,
        result=None
    )
    
    output.tools = [tool1, tool2, tool3]
    
    assert len(output.tools_requiring_confirmation) == 1
    assert output.tools_requiring_confirmation[0] == tool1
    
    assert len(output.tools_requiring_user_input) == 1
    assert output.tools_requiring_user_input[0] == tool2
    
    assert len(output.tools_awaiting_external_execution) == 1
    assert output.tools_awaiting_external_execution[0] == tool3
    
    print("✅ Tools properties test passed")



def test_mark_paused():
    """Test mark_paused method."""
    print("\n" + "=" * 60)
    print("TEST: Mark Paused")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001", status=RunStatus.running)
    task = Task("Test")
    output.task = task
    
    assert output.status == RunStatus.running
    assert output.pause_reason is None
    
    output.mark_paused(reason="external_tool")
    
    assert output.status == RunStatus.paused
    assert output.pause_reason == "external_tool"
    assert output.is_paused is True
    assert output.updated_at is not None
    assert task.status == RunStatus.paused
    
    print("✅ Mark paused test passed")


def test_mark_cancelled():
    """Test mark_cancelled method."""
    print("\n" + "=" * 60)
    print("TEST: Mark Cancelled")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001", status=RunStatus.running)
    task = Task("Test")
    output.task = task
    
    output.mark_cancelled()
    
    assert output.status == RunStatus.cancelled
    assert output.is_cancelled is True
    assert output.updated_at is not None
    assert task.status == RunStatus.cancelled
    
    print("✅ Mark cancelled test passed")


def test_mark_completed():
    """Test mark_completed method."""
    print("\n" + "=" * 60)
    print("TEST: Mark Completed")
    print("=" * 60)
    
    output = AgentRunOutput(
        run_id="test_001",
        status=RunStatus.running,
        is_streaming=True,
        accumulated_text="Final text"
    )
    task = Task("Test")
    output.task = task
    
    output.mark_completed()
    
    assert output.status == RunStatus.completed
    assert output.is_complete is True
    assert output.output == "Final text"  # Should be set from accumulated_text
    assert output.updated_at is not None
    assert task.status == RunStatus.completed
    
    print("✅ Mark completed test passed")


def test_mark_error():
    """Test mark_error method."""
    print("\n" + "=" * 60)
    print("TEST: Mark Error")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001", status=RunStatus.running)
    task = Task("Test")
    output.task = task
    
    output.mark_error(error="Test error message")
    
    assert output.status == RunStatus.error
    assert output.is_error is True
    assert output.error_details == "Test error message"
    assert output.metadata is not None
    assert output.metadata.get("error") == "Test error message"
    assert output.updated_at is not None
    assert task.status == RunStatus.error
    
    print("✅ Mark error test passed")



def test_start_new_run():
    """Test start_new_run method."""
    print("\n" + "=" * 60)
    print("TEST: Start New Run")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001")
    
    # Add some historical messages
    msg1 = ModelRequest(parts=[UserPromptPart(content="Hello")])
    msg2 = ModelResponse(parts=[TextPart(content="Hi")])
    output.chat_history = [msg1, msg2]
    
    assert len(output._run_boundaries) == 0
    
    output.start_new_run()
    
    assert len(output._run_boundaries) == 1
    assert output._run_boundaries[0] == 2
    
    # Add new messages
    msg3 = ModelRequest(parts=[UserPromptPart(content="How are you?")])
    output.chat_history.append(msg3)
    
    # Start another run
    output.start_new_run()
    
    assert len(output._run_boundaries) == 2
    assert output._run_boundaries[1] == 3
    
    print("✅ Start new run test passed")


def test_finalize_run_messages():
    """Test finalize_run_messages method."""
    print("\n" + "=" * 60)
    print("TEST: Finalize Run Messages")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001")
    
    # Set up historical messages
    msg1 = ModelRequest(parts=[UserPromptPart(content="Hello")])
    msg2 = ModelResponse(parts=[TextPart(content="Hi")])
    output.chat_history = [msg1, msg2]
    
    # Mark run start
    output.start_new_run()
    
    # Add new messages during this run
    msg3 = ModelRequest(parts=[UserPromptPart(content="How are you?")])
    msg4 = ModelResponse(parts=[TextPart(content="I'm good")])
    output.chat_history.extend([msg3, msg4])
    
    # Finalize
    output.finalize_run_messages()
    
    assert output.messages is not None
    assert len(output.messages) == 2
    assert output.messages[0] == msg3
    assert output.messages[1] == msg4
    
    print("✅ Finalize run messages test passed")


def test_add_message():
    """Test add_message method."""
    print("\n" + "=" * 60)
    print("TEST: Add Message")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001")
    
    msg = ModelRequest(parts=[UserPromptPart(content="Test")])
    output.add_message(msg)
    
    assert output.messages is not None
    assert len(output.messages) == 1
    assert output.messages[0] == msg
    
    print("✅ Add message test passed")


def test_add_messages():
    """Test add_messages method."""
    print("\n" + "=" * 60)
    print("TEST: Add Messages")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001")
    
    msg1 = ModelRequest(parts=[UserPromptPart(content="Test 1")])
    msg2 = ModelResponse(parts=[TextPart(content="Test 2")])
    
    output.add_messages([msg1, msg2])
    
    assert output.messages is not None
    assert len(output.messages) == 2
    assert output.messages[0] == msg1
    assert output.messages[1] == msg2
    
    print("✅ Add messages test passed")


def test_new_messages():
    """Test new_messages method."""
    print("\n" + "=" * 60)
    print("TEST: New Messages")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001")
    
    # Test with finalized messages
    msg1 = ModelRequest(parts=[UserPromptPart(content="Test 1")])
    msg2 = ModelResponse(parts=[TextPart(content="Test 2")])
    output.messages = [msg1, msg2]
    
    new_msgs = output.new_messages()
    assert len(new_msgs) == 2
    assert new_msgs[0] == msg1
    assert new_msgs[1] == msg2
    
    # Test with run boundaries
    output.messages = None
    output.chat_history = [msg1, msg2]
    output._run_boundaries = [0]
    
    new_msgs = output.new_messages()
    assert len(new_msgs) == 2
    
    print("✅ New messages test passed")


def test_all_messages():
    """Test all_messages method."""
    print("\n" + "=" * 60)
    print("TEST: All Messages")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001")
    
    msg1 = ModelRequest(parts=[UserPromptPart(content="Test 1")])
    msg2 = ModelResponse(parts=[TextPart(content="Test 2")])
    output.messages = [msg1, msg2]
    
    all_msgs = output.all_messages()
    assert len(all_msgs) == 2
    assert all_msgs[0] == msg1
    assert all_msgs[1] == msg2
    
    # Should return a copy
    all_msgs.append(msg1)
    assert len(output.messages) == 2  # Original unchanged
    
    print("✅ All messages test passed")


def test_get_last_model_response():
    """Test get_last_model_response method."""
    print("\n" + "=" * 60)
    print("TEST: Get Last Model Response")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001")
    
    # Test with no messages
    assert output.get_last_model_response() is None
    
    # Test with messages
    msg1 = ModelRequest(parts=[UserPromptPart(content="Test")])
    msg2 = ModelResponse(parts=[TextPart(content="Response 1")])
    msg3 = ModelRequest(parts=[UserPromptPart(content="Test 2")])
    msg4 = ModelResponse(parts=[TextPart(content="Response 2")])
    
    output.messages = [msg1, msg2, msg3, msg4]
    
    last_response = output.get_last_model_response()
    assert last_response is not None
    assert last_response == msg4
    
    print("✅ Get last model response test passed")


def test_has_new_messages():
    """Test has_new_messages method."""
    print("\n" + "=" * 60)
    print("TEST: Has New Messages")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001")
    
    # Test with no messages
    assert output.has_new_messages() is False
    
    # Test with messages
    msg = ModelRequest(parts=[UserPromptPart(content="Test")])
    output.messages = [msg]
    assert output.has_new_messages() is True
    
    # Test with run boundaries
    output.messages = None
    output.chat_history = [msg]
    output._run_boundaries = [0]
    assert output.has_new_messages() is True
    
    print("✅ Has new messages test passed")



def test_usage_tracking():
    """Test usage tracking methods."""
    print("\n" + "=" * 60)
    print("TEST: Usage Tracking")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001")
    
    # Test start_usage_timer
    output.start_usage_timer()
    assert output.usage is not None
    assert isinstance(output.usage, RunUsage)

    # Test usage accumulation with RequestUsage via incr()
    request_usage = RequestUsage(
        input_tokens=100,
        output_tokens=50,
        cache_write_tokens=10,
        cache_read_tokens=5
    )
    output.usage.incr(request_usage)

    assert output.usage.input_tokens == 100
    assert output.usage.output_tokens == 50
    assert output.usage.cache_write_tokens == 10
    assert output.usage.cache_read_tokens == 5

    # Test usage accumulation with a second RequestUsage
    request_usage2 = RequestUsage(
        input_tokens=200,
        output_tokens=100
    )
    output.usage.incr(request_usage2)

    assert output.usage.input_tokens == 300  # Accumulated
    assert output.usage.output_tokens == 150
    
    # Test increment_tool_calls
    output.increment_tool_calls(3)
    assert output.usage.tool_calls == 3
    
    # Test set_usage_cost
    output.set_usage_cost(0.05)
    assert output.usage.cost == 0.05
    
    output.set_usage_cost(0.02)
    assert output.usage.cost == 0.07  # Accumulated
    
    # Test stop_usage_timer
    output.stop_usage_timer(set_duration=True)
    assert output.usage.duration is not None
    
    print("✅ Usage tracking test passed")


def test_set_usage_time_to_first_token():
    """Test set_usage_time_to_first_token method."""
    print("\n" + "=" * 60)
    print("TEST: Set Usage Time to First Token")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001")
    output.start_usage_timer()
    
    sleep(0.1)  # Small delay
    
    output.set_usage_time_to_first_token()
    
    assert output.usage is not None
    assert output.usage.time_to_first_token is not None
    assert output.usage.time_to_first_token > 0
    
    print("✅ Set usage time to first token test passed")



def test_requirement_methods():
    """Test requirement-related methods."""
    print("\n" + "=" * 60)
    print("TEST: Requirement Methods")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001")
    
    # Test add_requirement
    tool_exec = ToolExecution(
        tool_name="test_tool",
        tool_call_id="call_001",
        external_execution_required=True
    )
    requirement = RunRequirement(
        id="req_001",
        tool_execution=tool_exec
    )
    
    output.add_requirement(requirement)
    
    assert output.requirements is not None
    assert len(output.requirements) == 1
    assert output.requirements[0] == requirement
    
    # Test get_external_tool_requirements
    requirements = output.get_external_tool_requirements()
    assert len(requirements) == 1
    assert requirements[0] == requirement
    
    # Test get_external_tool_requirements_with_results
    requirements_with_results = output.get_external_tool_requirements_with_results()
    assert len(requirements_with_results) == 0  # No results yet
    
    requirement.tool_execution.result = "result"
    requirements_with_results = output.get_external_tool_requirements_with_results()
    assert len(requirements_with_results) == 1
    
    # Test has_pending_external_tools
    requirement.tool_execution.result = None
    requirement.tool_execution.external_execution_required = True
    assert output.has_pending_external_tools() is True
    
    requirement.tool_execution.result = "result"
    assert output.has_pending_external_tools() is False
    
    print("✅ Requirement methods test passed")



def test_step_result_methods():
    """Test step result-related methods."""
    print("\n" + "=" * 60)
    print("TEST: Step Result Methods")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="test_001")
    
    # Test get_step_results
    assert output.get_step_results() == []
    
    step1 = StepResult(
        step_name="Step1",
        step_index=0,
        status=StepStatus.COMPLETED
    )
    step2 = StepResult(
        step_name="Step2",
        step_index=1,
        status=StepStatus.COMPLETED
    )
    step3 = StepResult(
        step_name="Step3",
        step_index=2,
        status=StepStatus.ERROR
    )
    
    output.step_results = [step1, step2, step3]
    
    results = output.get_step_results()
    assert len(results) == 3
    
    # Test get_last_successful_step
    last_successful = output.get_last_successful_step()
    assert last_successful is not None
    assert last_successful == step2
    
    # Test get_error_step
    error_step = output.get_error_step()
    assert error_step is not None
    assert error_step == step3
    
    # Test get_problematic_step
    problematic = output.get_problematic_step()
    assert problematic is not None
    assert problematic == step3
    
    # Test get_cancelled_step
    step4 = StepResult(
        step_name="Step4",
        step_index=3,
        status=StepStatus.CANCELLED
    )
    output.step_results.append(step4)
    
    cancelled_step = output.get_cancelled_step()
    assert cancelled_step is not None
    assert cancelled_step == step4
    
    # Test get_paused_step
    step5 = StepResult(
        step_name="Step5",
        step_index=4,
        status=StepStatus.PAUSED
    )
    output.step_results.append(step5)
    
    paused_step = output.get_paused_step()
    assert paused_step is not None
    assert paused_step == step5
    
    # Test get_execution_stats
    assert output.get_execution_stats() is None
    
    stats = PipelineExecutionStats(
        total_steps=5,
        executed_steps=3
    )
    output.execution_stats = stats
    
    retrieved_stats = output.get_execution_stats()
    assert retrieved_stats is not None
    assert retrieved_stats == stats
    
    print("✅ Step result methods test passed")



def test_to_dict():
    """Test to_dict serialization."""
    print("\n" + "=" * 60)
    print("TEST: To Dict Serialization")
    print("=" * 60)
    
    task = Task("Test task")
    run_input = AgentRunInput(user_prompt="Test prompt")
    
    output = AgentRunOutput(
        run_id="test_001",
        agent_id="agent_001",
        agent_name="Test Agent",
        session_id="session_001",
        user_id="user_001",
        task=task,
        input=run_input,
        output="Test output",
        status=RunStatus.completed,
        is_streaming=False,
        accumulated_text="",
        metadata={"key": "value"},
        session_state={"state": "active"}
    )
    
    output.start_usage_timer()
    output.usage.incr(RequestUsage(input_tokens=100, output_tokens=50))

    data = output.to_dict()

    assert isinstance(data, dict)
    assert data["run_id"] == "test_001"
    assert data["agent_id"] == "agent_001"
    assert data["agent_name"] == "Test Agent"
    assert data["session_id"] == "session_001"
    assert data["user_id"] == "user_001"
    assert data["output"] == "Test output"
    assert data["metadata"] == {"key": "value"}
    assert data["session_state"] == {"state": "active"}
    assert "task" in data
    assert "input" in data
    assert "usage" in data
    
    print("✅ To dict serialization test passed")


def test_from_dict():
    """Test from_dict deserialization."""
    print("\n" + "=" * 60)
    print("TEST: From Dict Deserialization")
    print("=" * 60)
    
    task = Task("Test task")
    run_input = AgentRunInput(user_prompt="Test prompt")
    
    original = AgentRunOutput(
        run_id="test_001",
        agent_id="agent_001",
        agent_name="Test Agent",
        session_id="session_001",
        user_id="user_001",
        task=task,
        input=run_input,
        output="Test output",
        status=RunStatus.completed,
        metadata={"key": "value"}
    )
    
    data = original.to_dict()
    restored = AgentRunOutput.from_dict(data)
    
    assert restored.run_id == original.run_id
    assert restored.agent_id == original.agent_id
    assert restored.agent_name == original.agent_name
    assert restored.session_id == original.session_id
    assert restored.user_id == original.user_id
    assert restored.output == original.output
    assert restored.status == original.status
    assert restored.metadata == original.metadata
    assert restored.task is not None
    assert restored.input is not None
    
    print("✅ From dict deserialization test passed")


def test_to_json():
    """Test to_json serialization."""
    print("\n" + "=" * 60)
    print("TEST: To JSON Serialization")
    print("=" * 60)
    
    output = AgentRunOutput(
        run_id="test_001",
        agent_id="agent_001",
        output="Test output",
        status=RunStatus.completed
    )
    
    json_str = output.to_json()
    
    assert isinstance(json_str, str)
    assert "test_001" in json_str
    assert "Test output" in json_str
    
    # Test with indent
    json_str_indented = output.to_json(indent=4)
    assert isinstance(json_str_indented, str)
    assert "test_001" in json_str_indented
    assert "Test output" in json_str_indented
    
    print("✅ To JSON serialization test passed")


def test_serialization_roundtrip():
    """Test complete serialization roundtrip."""
    print("\n" + "=" * 60)
    print("TEST: Serialization Roundtrip")
    print("=" * 60)
    
    task = Task("Test task")
    run_input = AgentRunInput(user_prompt="Test prompt")
    
    original = AgentRunOutput(
        run_id="test_001",
        agent_id="agent_001",
        agent_name="Test Agent",
        session_id="session_001",
        user_id="user_001",
        parent_run_id="parent_001",
        task=task,
        input=run_input,
        output="Test output",
        output_schema=str,
        thinking_content="Test thinking",
        model_name="anthropic/claude-sonnet-4-5",
        model_provider="anthropic",
        status=RunStatus.completed,
        is_streaming=True,
        accumulated_text="Accumulated",
        metadata={"key": "value", "nested": {"inner": "data"}},
        session_state={"state": "active"},
        pause_reason=None,
        error_details=None,
        tool_call_count=5,
        tool_limit_reached=False,
        memory_message_count=10
    )
    
    # Add messages
    msg1 = ModelRequest(parts=[UserPromptPart(content="Hello")])
    msg2 = ModelResponse(parts=[TextPart(content="Hi")])
    original.chat_history = [msg1, msg2]
    original.messages = [msg1, msg2]
    
    # Add usage
    original.start_usage_timer()
    original.usage.incr(RequestUsage(input_tokens=100, output_tokens=50))
    original.increment_tool_calls(3)
    original.set_usage_cost(0.05)
    
    # Serialize and deserialize
    data = original.to_dict()
    restored = AgentRunOutput.from_dict(data)
    
    # Verify all attributes
    assert restored.run_id == original.run_id
    assert restored.agent_id == original.agent_id
    assert restored.agent_name == original.agent_name
    assert restored.session_id == original.session_id
    assert restored.user_id == original.user_id
    assert restored.parent_run_id == original.parent_run_id
    assert restored.output == original.output
    assert restored.output_schema == original.output_schema
    assert restored.thinking_content == original.thinking_content
    assert restored.model_name == original.model_name
    assert restored.model_provider == original.model_provider
    assert restored.status == original.status
    assert restored.is_streaming == original.is_streaming
    assert restored.accumulated_text == original.accumulated_text
    assert restored.metadata == original.metadata
    assert restored.session_state == original.session_state
    assert restored.tool_call_count == original.tool_call_count
    assert restored.tool_limit_reached == original.tool_limit_reached
    assert restored.memory_message_count == original.memory_message_count
    assert len(restored.chat_history) == len(original.chat_history)
    assert len(restored.messages) == len(original.messages)
    assert restored.usage is not None
    assert restored.usage.input_tokens == original.usage.input_tokens
    assert restored.usage.output_tokens == original.usage.output_tokens
    assert restored.usage.tool_calls == original.usage.tool_calls
    assert restored.usage.cost == original.usage.cost
    
    print("✅ Serialization roundtrip test passed")



def test_string_representations():
    """Test __str__ and __repr__ methods."""
    print("\n" + "=" * 60)
    print("TEST: String Representations")
    print("=" * 60)
    
    output = AgentRunOutput(
        run_id="test_001",
        output="Test output text"
    )
    
    str_repr = str(output)
    assert str_repr == "Test output text"
    
    repr_repr = repr(output)
    assert "AgentRunOutput" in repr_repr
    assert "test_001" in repr_repr
    assert "running" in repr_repr.lower() or "completed" in repr_repr.lower()
    
    print("✅ String representations test passed")



def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)
    
    # Test with None values
    output = AgentRunOutput(
        run_id="test_001",
        task=None,
        input=None,
        output=None,
        messages=None,
        tools=None,
        requirements=None
    )
    
    assert output.task is None
    assert output.input is None
    assert output.output is None
    assert output.messages is None
    assert output.tools is None
    assert output.requirements is None
    
    # Test finalize_run_messages with no chat_history
    output.finalize_run_messages()
    assert output.messages == []
    
    # Test finalize_run_messages with no run boundaries
    output.chat_history = []
    output._run_boundaries = []
    output.finalize_run_messages()
    assert output.messages == []
    
    # Test new_messages with no tracking info
    output.messages = None
    output.chat_history = []
    output._run_boundaries = []
    assert len(output.new_messages()) == 0
    
    # Test mark_error with None error
    output.mark_error(error=None)
    assert output.status == RunStatus.error
    assert output.error_details is None
    
    # Test usage methods with None usage
    output.usage = None
    output.start_usage_timer()
    assert output.usage is not None
    
    print("✅ Edge cases test passed")


# =============================================================================
# Test: chat_history vs messages vs response (Single Run)
# =============================================================================

def test_chat_history_messages_response_single_run():
    """
    Test chat_history, messages, and response in a SINGLE run scenario.
    
    In a single run:
    - chat_history: Contains all messages from this run (loaded from memory + new messages)
    - messages: Contains ONLY new messages from this run (after finalize_run_messages)
    - response: The last ModelResponse from this run
    """
    print("\n" + "=" * 60)
    print("TEST: chat_history vs messages vs response (Single Run)")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="single_run_001", session_id="session_001")
    
    # Simulate a single run with messages
    # Step 1: Load historical messages (from memory) - in real scenario, Memory class does this
    historical_request = ModelRequest(parts=[UserPromptPart(content="Hello from memory")])
    historical_response = ModelResponse(parts=[TextPart(content="Hi from memory")])
    output.chat_history = [historical_request, historical_response]
    
    # Step 2: Mark start of new run (ChatHistoryStep does this)
    output.start_new_run()
    
    # Step 3: Add new messages during this run
    new_request_1 = ModelRequest(parts=[UserPromptPart(content="What is 2+2?")])
    new_response_1 = ModelResponse(parts=[TextPart(content="2+2 equals 4")])
    new_request_2 = ModelRequest(parts=[UserPromptPart(content="What is 3+3?")])
    new_response_2 = ModelResponse(parts=[TextPart(content="3+3 equals 6")])
    
    # Add to chat_history (this happens during pipeline execution)
    output.chat_history.extend([new_request_1, new_response_1, new_request_2, new_response_2])
    
    # Step 4: Set response to last ModelResponse (this happens in ModelResponseStep)
    output.response = new_response_2
    
    # Step 5: Finalize run messages (MemorySaveStep does this)
    output.finalize_run_messages()
    
    # Verify chat_history contains ALL messages (historical + new)
    assert len(output.chat_history) == 6, f"chat_history should have 6 messages, got {len(output.chat_history)}"
    assert output.chat_history[0] == historical_request
    assert output.chat_history[1] == historical_response
    assert output.chat_history[2] == new_request_1
    assert output.chat_history[3] == new_response_1
    assert output.chat_history[4] == new_request_2
    assert output.chat_history[5] == new_response_2
    
    # Verify messages contains ONLY new messages from this run
    assert output.messages is not None, "messages should be set after finalize_run_messages"
    assert len(output.messages) == 4, f"messages should have 4 messages (only from this run), got {len(output.messages)}"
    assert output.messages[0] == new_request_1
    assert output.messages[1] == new_response_1
    assert output.messages[2] == new_request_2
    assert output.messages[3] == new_response_2
    
    # Verify response is the last ModelResponse
    assert output.response is not None, "response should be set"
    assert output.response == new_response_2, "response should be the last ModelResponse"
    assert isinstance(output.response, ModelResponse), "response should be a ModelResponse"
    
    # Verify response is the same as the last message in messages
    assert output.response == output.messages[-1], "response should match the last message in messages"
    
    # Verify new_messages() returns the same as messages
    new_msgs = output.new_messages()
    assert len(new_msgs) == len(output.messages), "new_messages() should return same count as messages"
    assert new_msgs == output.messages, "new_messages() should return same messages"
    
    print(f"  ✓ chat_history: {len(output.chat_history)} messages (historical + new)")
    print(f"  ✓ messages: {len(output.messages)} messages (only from this run)")
    print(f"  ✓ response: Last ModelResponse from this run")
    print("✅ Single run test passed")


def test_chat_history_messages_response_single_run_no_history():
    """
    Test chat_history, messages, and response in a SINGLE run with NO historical messages.
    
    When there's no memory/history:
    - chat_history: Contains only messages from this run
    - messages: Contains all messages from this run (same as chat_history)
    - response: The last ModelResponse from this run
    """
    print("\n" + "=" * 60)
    print("TEST: chat_history vs messages vs response (Single Run, No History)")
    print("=" * 60)
    
    output = AgentRunOutput(run_id="single_run_002", session_id="session_002")
    
    # No historical messages (first run in session)
    output.chat_history = []
    
    # Mark start of new run
    output.start_new_run()
    
    # Add new messages during this run
    request_1 = ModelRequest(parts=[UserPromptPart(content="Hello")])
    response_1 = ModelResponse(parts=[TextPart(content="Hi there")])
    request_2 = ModelRequest(parts=[UserPromptPart(content="How are you?")])
    response_2 = ModelResponse(parts=[TextPart(content="I'm doing well")])
    
    output.chat_history.extend([request_1, response_1, request_2, response_2])
    output.response = response_2
    output.finalize_run_messages()
    
    # Verify chat_history contains all messages from this run
    assert len(output.chat_history) == 4, f"chat_history should have 4 messages, got {len(output.chat_history)}"
    
    # Verify messages contains all messages from this run (same as chat_history when no history)
    assert output.messages is not None
    assert len(output.messages) == 4, f"messages should have 4 messages, got {len(output.messages)}"
    assert output.messages == output.chat_history, "messages should equal chat_history when no history"
    
    # Verify response is the last ModelResponse
    assert output.response == response_2
    assert output.response == output.messages[-1]
    
    print(f"  ✓ chat_history: {len(output.chat_history)} messages (all from this run)")
    print(f"  ✓ messages: {len(output.messages)} messages (same as chat_history)")
    print(f"  ✓ response: Last ModelResponse")
    print("✅ Single run (no history) test passed")


# =============================================================================
# Test: chat_history vs messages vs response (Multiple Runs)
# =============================================================================

def test_chat_history_messages_response_multiple_runs():
    """
    Test chat_history, messages, and response in MULTIPLE runs scenario.
    
    In multiple runs:
    - chat_history: Contains messages from ALL runs (accumulated across session)
    - messages: Contains ONLY messages from the CURRENT run
    - response: The last ModelResponse from the CURRENT run
    """
    print("\n" + "=" * 60)
    print("TEST: chat_history vs messages vs response (Multiple Runs)")
    print("=" * 60)
    
    # ========================================================================
    # RUN 1: First run in session
    # ========================================================================
    print("\n--- Run 1: First run in session ---")
    output_run1 = AgentRunOutput(run_id="run_001", session_id="session_003")
    
    # No historical messages
    output_run1.chat_history = []
    output_run1.start_new_run()
    
    # Messages from run 1
    run1_request = ModelRequest(parts=[UserPromptPart(content="What is AI?")])
    run1_response = ModelResponse(parts=[TextPart(content="AI is artificial intelligence")])
    output_run1.chat_history.extend([run1_request, run1_response])
    output_run1.response = run1_response
    output_run1.finalize_run_messages()
    
    print(f"  Run 1 - chat_history: {len(output_run1.chat_history)} messages")
    print(f"  Run 1 - messages: {len(output_run1.messages)} messages")
    print(f"  Run 1 - response: {output_run1.response.parts[0].content if output_run1.response else None}")
    
    # Verify run 1
    assert len(output_run1.chat_history) == 2
    assert len(output_run1.messages) == 2
    assert output_run1.response == run1_response
    
    # ========================================================================
    # RUN 2: Second run in same session (chat_history accumulates)
    # ========================================================================
    print("\n--- Run 2: Second run in same session ---")
    output_run2 = AgentRunOutput(run_id="run_002", session_id="session_003")
    
    # chat_history contains messages from run 1 (loaded from memory)
    # In real scenario, Memory class loads previous messages into chat_history
    output_run2.chat_history = output_run1.chat_history.copy()  # Memory loads this
    
    output_run2.start_new_run()
    
    # New messages from run 2
    run2_request = ModelRequest(parts=[UserPromptPart(content="What is ML?")])
    run2_response = ModelResponse(parts=[TextPart(content="ML is machine learning")])
    output_run2.chat_history.extend([run2_request, run2_response])
    output_run2.response = run2_response
    output_run2.finalize_run_messages()
    
    print(f"  Run 2 - chat_history: {len(output_run2.chat_history)} messages (run1 + run2)")
    print(f"  Run 2 - messages: {len(output_run2.messages)} messages (only run2)")
    print(f"  Run 2 - response: {output_run2.response.parts[0].content if output_run2.response else None}")
    
    # Verify run 2
    assert len(output_run2.chat_history) == 4, "chat_history should have messages from run1 + run2"
    assert output_run2.chat_history[0] == run1_request, "First message should be from run1"
    assert output_run2.chat_history[1] == run1_response, "Second message should be from run1"
    assert output_run2.chat_history[2] == run2_request, "Third message should be from run2"
    assert output_run2.chat_history[3] == run2_response, "Fourth message should be from run2"
    
    assert len(output_run2.messages) == 2, "messages should have ONLY messages from run2"
    assert output_run2.messages[0] == run2_request, "First message in messages should be from run2"
    assert output_run2.messages[1] == run2_response, "Second message in messages should be from run2"
    
    assert output_run2.response == run2_response, "response should be from run2"
    assert output_run2.response != run1_response, "response should NOT be from run1"
    
    # ========================================================================
    # RUN 3: Third run in same session
    # ========================================================================
    print("\n--- Run 3: Third run in same session ---")
    output_run3 = AgentRunOutput(run_id="run_003", session_id="session_003")
    
    # chat_history contains messages from run1 + run2 (loaded from memory)
    output_run3.chat_history = output_run2.chat_history.copy()  # Memory loads this
    
    output_run3.start_new_run()
    
    # New messages from run 3
    run3_request = ModelRequest(parts=[UserPromptPart(content="What is DL?")])
    run3_response = ModelResponse(parts=[TextPart(content="DL is deep learning")])
    output_run3.chat_history.extend([run3_request, run3_response])
    output_run3.response = run3_response
    output_run3.finalize_run_messages()
    
    print(f"  Run 3 - chat_history: {len(output_run3.chat_history)} messages (run1 + run2 + run3)")
    print(f"  Run 3 - messages: {len(output_run3.messages)} messages (only run3)")
    print(f"  Run 3 - response: {output_run3.response.parts[0].content if output_run3.response else None}")
    
    # Verify run 3
    assert len(output_run3.chat_history) == 6, "chat_history should have messages from all runs"
    assert output_run3.chat_history[0] == run1_request
    assert output_run3.chat_history[1] == run1_response
    assert output_run3.chat_history[2] == run2_request
    assert output_run3.chat_history[3] == run2_response
    assert output_run3.chat_history[4] == run3_request
    assert output_run3.chat_history[5] == run3_response
    
    assert len(output_run3.messages) == 2, "messages should have ONLY messages from run3"
    assert output_run3.messages[0] == run3_request
    assert output_run3.messages[1] == run3_response
    
    assert output_run3.response == run3_response, "response should be from run3"
    assert output_run3.response != run2_response, "response should NOT be from run2"
    assert output_run3.response != run1_response, "response should NOT be from run1"
    
    # ========================================================================
    # Verify differences between runs
    # ========================================================================
    print("\n--- Verification: Differences between runs ---")
    
    # chat_history grows across runs
    assert len(output_run1.chat_history) < len(output_run2.chat_history)
    assert len(output_run2.chat_history) < len(output_run3.chat_history)
    
    # messages is always only from current run (2 messages per run in this test)
    assert len(output_run1.messages) == 2
    assert len(output_run2.messages) == 2
    assert len(output_run3.messages) == 2
    
    # response is always from current run
    assert output_run1.response == run1_response
    assert output_run2.response == run2_response
    assert output_run3.response == run3_response
    
    # messages from different runs are different
    assert output_run1.messages != output_run2.messages
    assert output_run2.messages != output_run3.messages
    
    # chat_history from later runs contains messages from earlier runs
    assert output_run1.chat_history[0] in output_run2.chat_history
    assert output_run1.chat_history[1] in output_run2.chat_history
    assert output_run2.chat_history[0] in output_run3.chat_history
    assert output_run2.chat_history[1] in output_run3.chat_history
    
    print("  ✓ chat_history accumulates across runs")
    print("  ✓ messages contains only current run messages")
    print("  ✓ response is always from current run")
    print("✅ Multiple runs test passed")


def test_chat_history_messages_response_multiple_runs_with_tool_calls():
    """
    Test chat_history, messages, and response with tool calls across multiple runs.
    
    This tests a more complex scenario with tool calls in messages.
    """
    print("\n" + "=" * 60)
    print("TEST: chat_history vs messages vs response (Multiple Runs with Tools)")
    print("=" * 60)
    
    from upsonic.messages.messages import ToolCallPart, ToolReturnPart
    
    # Run 1: Simple question
    output_run1 = AgentRunOutput(run_id="run_tool_001", session_id="session_tool_001")
    output_run1.chat_history = []
    output_run1.start_new_run()
    
    run1_request = ModelRequest(parts=[UserPromptPart(content="Calculate 5+3")])
    run1_response = ModelResponse(parts=[
        TextPart(content="I'll calculate that for you."),
        ToolCallPart(tool_name="add_numbers", tool_call_id="call_001", args={"a": 5, "b": 3})
    ])
    run1_tool_result = ModelRequest(parts=[
        ToolReturnPart(tool_name="add_numbers", tool_call_id="call_001", content={"func": 8})
    ])
    run1_final_response = ModelResponse(parts=[TextPart(content="5+3 equals 8")])
    
    output_run1.chat_history.extend([run1_request, run1_response, run1_tool_result, run1_final_response])
    output_run1.response = run1_final_response
    output_run1.finalize_run_messages()
    
    print(f"  Run 1 - chat_history: {len(output_run1.chat_history)} messages")
    print(f"  Run 1 - messages: {len(output_run1.messages)} messages")
    
    assert len(output_run1.chat_history) == 4
    assert len(output_run1.messages) == 4
    assert output_run1.response == run1_final_response
    
    # Run 2: Another calculation (chat_history includes run1)
    output_run2 = AgentRunOutput(run_id="run_tool_002", session_id="session_tool_001")
    output_run2.chat_history = output_run1.chat_history.copy()  # Memory loads run1 messages
    output_run2.start_new_run()
    
    run2_request = ModelRequest(parts=[UserPromptPart(content="Calculate 10+20")])
    run2_response = ModelResponse(parts=[
        TextPart(content="I'll calculate that."),
        ToolCallPart(tool_name="add_numbers", tool_call_id="call_002", args={"a": 10, "b": 20})
    ])
    run2_tool_result = ModelRequest(parts=[
        ToolReturnPart(tool_name="add_numbers", tool_call_id="call_002", content={"func": 30})
    ])
    run2_final_response = ModelResponse(parts=[TextPart(content="10+20 equals 30")])
    
    output_run2.chat_history.extend([run2_request, run2_response, run2_tool_result, run2_final_response])
    output_run2.response = run2_final_response
    output_run2.finalize_run_messages()
    
    print(f"  Run 2 - chat_history: {len(output_run2.chat_history)} messages (run1 + run2)")
    print(f"  Run 2 - messages: {len(output_run2.messages)} messages (only run2)")
    
    # Verify chat_history contains all messages from both runs
    assert len(output_run2.chat_history) == 8, "chat_history should have 4 from run1 + 4 from run2"
    
    # Verify messages contains only run2 messages
    assert len(output_run2.messages) == 4, "messages should have only 4 messages from run2"
    assert output_run2.messages[0] == run2_request
    assert output_run2.messages[1] == run2_response
    assert output_run2.messages[2] == run2_tool_result
    assert output_run2.messages[3] == run2_final_response
    
    # Verify response is from run2
    assert output_run2.response == run2_final_response
    assert output_run2.response != run1_final_response
    
    # Verify chat_history structure
    assert output_run2.chat_history[0] == run1_request  # First message from run1
    assert output_run2.chat_history[3] == run1_final_response  # Last message from run1
    assert output_run2.chat_history[4] == run2_request  # First message from run2
    assert output_run2.chat_history[7] == run2_final_response  # Last message from run2
    
    print("  ✓ chat_history contains all messages including tool calls")
    print("  ✓ messages contains only current run messages including tool calls")
    print("  ✓ response is the final response from current run")
    print("✅ Multiple runs with tools test passed")


def test_chat_history_messages_response_get_last_model_response():
    """
    Test get_last_model_response() method with multiple runs.
    
    This verifies that get_last_model_response() correctly identifies
    the last ModelResponse from the current run's messages.
    """
    print("\n" + "=" * 60)
    print("TEST: get_last_model_response with Multiple Runs")
    print("=" * 60)
    
    # Run 1
    output_run1 = AgentRunOutput(run_id="run_last_001", session_id="session_last_001")
    output_run1.chat_history = []
    output_run1.start_new_run()
    
    run1_request = ModelRequest(parts=[UserPromptPart(content="Hello")])
    run1_response_1 = ModelResponse(parts=[TextPart(content="Hi")])
    run1_request_2 = ModelRequest(parts=[UserPromptPart(content="How are you?")])
    run1_response_2 = ModelResponse(parts=[TextPart(content="I'm good")])
    
    output_run1.chat_history.extend([run1_request, run1_response_1, run1_request_2, run1_response_2])
    output_run1.response = run1_response_2
    output_run1.finalize_run_messages()
    
    last_response_run1 = output_run1.get_last_model_response()
    assert last_response_run1 == run1_response_2, "Should return last response from run1"
    assert last_response_run1 == output_run1.response, "Should match response attribute"
    
    # Run 2 (with run1 history)
    output_run2 = AgentRunOutput(run_id="run_last_002", session_id="session_last_001")
    output_run2.chat_history = output_run1.chat_history.copy()
    output_run2.start_new_run()
    
    run2_request = ModelRequest(parts=[UserPromptPart(content="What's 2+2?")])
    run2_response = ModelResponse(parts=[TextPart(content="4")])
    
    output_run2.chat_history.extend([run2_request, run2_response])
    output_run2.response = run2_response
    output_run2.finalize_run_messages()
    
    last_response_run2 = output_run2.get_last_model_response()
    assert last_response_run2 == run2_response, "Should return last response from run2"
    assert last_response_run2 != run1_response_2, "Should NOT return response from run1"
    assert last_response_run2 == output_run2.response, "Should match response attribute"
    
    # Verify chat_history has both responses, but get_last_model_response returns only run2's
    assert run1_response_2 in output_run2.chat_history, "chat_history should contain run1 response"
    assert run2_response in output_run2.chat_history, "chat_history should contain run2 response"
    assert last_response_run2 == run2_response, "get_last_model_response should return run2's response"
    
    print("  ✓ get_last_model_response returns last response from current run")
    print("  ✓ get_last_model_response ignores historical responses")
    print("  ✓ get_last_model_response matches response attribute")
    print("✅ get_last_model_response test passed")


# =============================================================================
# Main Test Runner
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMPREHENSIVE AGENTRUNOUTPUT SMOKE TEST SUITE")
    print("=" * 70)
    
    pytest.main([__file__, "-v", "-s"])
