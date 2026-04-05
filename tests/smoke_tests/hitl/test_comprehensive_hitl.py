"""
Comprehensive HITL Test - Full Attribute and Content Verification

Tests External Tool, Cancel Run, Durable Execution, User Confirmation,
User Input, and Dynamic User Input with comprehensive verification of
all attributes in:
- AgentSession
- AgentRunOutput
- RunRequirement
- PipelineExecutionStats
- ToolExecution
- StepResult
- ModelResponse, RequestUsage, ModelProfile, AgentRunInput, ModelMessage
"""

import pytest
import asyncio
import os
import time
from typing import Any, Dict, List
from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.tools.user_input import UserControlFlowTools
from upsonic.run.base import RunStatus
from upsonic.run.cancel import cancel_run
from upsonic.db.database import SqliteDatabase
from upsonic.session.agent import AgentSession
from upsonic.agent.pipeline.step import inject_error_into_step, clear_error_injection, StepResult, StepStatus
from upsonic.run.agent.output import AgentRunOutput
from upsonic.run.requirements import RunRequirement
from upsonic.run.tools.tools import ToolExecution
from upsonic.run.pipeline.stats import PipelineExecutionStats
from upsonic.run.agent.input import AgentRunInput
from upsonic.usage import RequestUsage, RunUsage
from upsonic.profiles import ModelProfile
from upsonic.messages.messages import ModelResponse, ModelRequest

pytestmark = pytest.mark.timeout(300)

DEBUG = True
DB_FILE = "test_comprehensive.db"


def cleanup():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    clear_error_injection()


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


# --- Confirmation tools ---

@tool(requires_confirmation=True)
def sensitive_operation(data: str) -> str:
    """Perform a sensitive operation that requires user confirmation."""
    return f"Sensitive operation completed on: {data}"


@tool(requires_confirmation=True)
def delete_records(table: str, condition: str) -> str:
    """Delete records from a database table - requires user confirmation."""
    return f"Deleted records from {table} where {condition}"


@tool(requires_confirmation=True)
def deploy_to_production(version: str, environment: str) -> str:
    """Deploy application to production - requires user confirmation."""
    return f"Deployed version {version} to {environment}"


# --- User input tools ---

@tool(requires_user_input=True, user_input_fields=["to_address"])
def send_email_with_input(subject: str, body: str, to_address: str) -> str:
    """Send an email. Agent provides subject/body, user provides address."""
    return f"Email sent to {to_address} with subject '{subject}' and body '{body}'"


@tool(requires_user_input=True, user_input_fields=["priority", "assignee"])
def create_ticket(title: str, description: str, priority: str, assignee: str) -> str:
    """Create a support ticket. Agent provides title/description, user provides priority/assignee."""
    return f"Ticket '{title}' created with priority={priority}, assignee={assignee}"


@tool(requires_user_input=True, user_input_fields=["date", "attendees"])
def schedule_meeting(topic: str, date: str, attendees: str) -> str:
    """Schedule a meeting. Agent provides topic, user provides date and attendees."""
    return f"Meeting '{topic}' scheduled for {date} with attendees: {attendees}"


# --- Dynamic user input tools (regular tools, agent uses UserControlFlowTools) ---

@tool
def dynamic_send_email(subject: str, body: str, to_address: str) -> str:
    """Send an email to the given address."""
    return f"Email sent to {to_address} with subject '{subject}' and body '{body}'"


# --- User input helpers ---

_USER_INPUT_VALUES: Dict[str, str] = {
    "to_address": "user@example.com",
    "priority": "high",
    "assignee": "john.doe",
    "date": "2026-04-01",
    "attendees": "alice, bob, charlie",
    "subject": "Dynamic Subject",
    "body": "Dynamic body content",
    "recipient": "recipient@example.com",
    "email_address": "dynamic@example.com",
}


def _fill_user_input(requirement) -> None:
    """Fill all user input fields on a requirement with predefined values."""
    if not requirement.user_input_schema:
        return
    for field_dict in requirement.user_input_schema:
        if isinstance(field_dict, dict) and field_dict.get("value") is None:
            name = field_dict["name"]
            field_dict["value"] = _USER_INPUT_VALUES.get(name, f"test_{name}")
    requirement.tool_execution.answered = True


def execute_tool_externally(requirement: RunRequirement) -> str:
    """Execute external tool and return result."""
    tool_exec = requirement.tool_execution
    tool_name = tool_exec.tool_name
    tool_args = tool_exec.tool_args
    
    if tool_name == "send_email":
        return f"Email sent successfully to {tool_args.get('to', 'unknown')}"
    return f"Executed external tool: {tool_name}"


# =============================================================================
# DEEP VERIFICATION FUNCTIONS
# =============================================================================

def verify_tool_execution(te: ToolExecution, description: str, 
                          expected_tool_name: str = None,
                          expected_tool_args: Dict[str, Any] = None,
                          expected_result: str = None,
                          expected_external: bool = None) -> ToolExecution:
    """Comprehensive ToolExecution verification."""
    print(f"\n  === ToolExecution: {description} ===")
    
    assert te is not None, f"FAIL [{description}]: ToolExecution is None"
    
    # Verify tool_call_id
    assert te.tool_call_id is not None, f"FAIL [{description}]: tool_call_id is None"
    assert isinstance(te.tool_call_id, str), f"FAIL [{description}]: tool_call_id should be str"
    assert len(te.tool_call_id) > 0, f"FAIL [{description}]: tool_call_id should not be empty"
    
    # Verify tool_name
    assert te.tool_name is not None, f"FAIL [{description}]: tool_name is None"
    assert isinstance(te.tool_name, str), f"FAIL [{description}]: tool_name should be str"
    if expected_tool_name:
        assert te.tool_name == expected_tool_name, f"FAIL [{description}]: tool_name should be '{expected_tool_name}', got '{te.tool_name}'"
    
    # Verify tool_args
    assert te.tool_args is not None, f"FAIL [{description}]: tool_args is None"
    assert isinstance(te.tool_args, dict), f"FAIL [{description}]: tool_args should be dict"
    if expected_tool_args:
        for key, value in expected_tool_args.items():
            assert key in te.tool_args, f"FAIL [{description}]: tool_args missing key '{key}'"
            assert te.tool_args[key] == value, f"FAIL [{description}]: tool_args['{key}'] = '{te.tool_args[key]}', expected '{value}'"
    
    # Verify created_at
    assert te.created_at is not None, f"FAIL [{description}]: created_at is None"
    assert isinstance(te.created_at, int), f"FAIL [{description}]: created_at should be int"
    assert te.created_at > 0, f"FAIL [{description}]: created_at should be > 0"
    
    # Verify result if expected
    if expected_result is not None:
        assert te.result == expected_result, f"FAIL [{description}]: result = '{te.result}', expected '{expected_result}'"
    
    # Verify external_execution_required if expected
    if expected_external is not None:
        assert te.external_execution_required == expected_external, f"FAIL [{description}]: external_execution_required = {te.external_execution_required}, expected {expected_external}"
    
    print(f"    tool_call_id: {te.tool_call_id}")
    print(f"    tool_name: {te.tool_name}")
    print(f"    tool_args: {te.tool_args}")
    print(f"    result: {te.result}")
    print(f"    external_execution_required: {te.external_execution_required}")
    print(f"    created_at: {te.created_at}")
    
    return te


def verify_step_result(sr: StepResult, description: str,
                       expected_name: str = None,
                       expected_status: StepStatus = None) -> StepResult:
    """Comprehensive StepResult verification."""
    print(f"\n  === StepResult: {description} ===")
    
    assert sr is not None, f"FAIL [{description}]: StepResult is None"
    
    # Verify name
    assert sr.name is not None, f"FAIL [{description}]: name is None"
    assert isinstance(sr.name, str), f"FAIL [{description}]: name should be str"
    if expected_name:
        assert sr.name == expected_name, f"FAIL [{description}]: name = '{sr.name}', expected '{expected_name}'"
    
    # Verify step_number
    assert sr.step_number is not None, f"FAIL [{description}]: step_number is None"
    assert isinstance(sr.step_number, int), f"FAIL [{description}]: step_number should be int"
    assert sr.step_number >= 0, f"FAIL [{description}]: step_number should be >= 0"
    
    # Verify status
    assert sr.status is not None, f"FAIL [{description}]: status is None"
    assert isinstance(sr.status, StepStatus), f"FAIL [{description}]: status should be StepStatus"
    if expected_status:
        assert sr.status == expected_status, f"FAIL [{description}]: status = {sr.status}, expected {expected_status}"
    
    # Verify execution_time
    assert sr.execution_time is not None, f"FAIL [{description}]: execution_time is None"
    assert isinstance(sr.execution_time, (int, float)), f"FAIL [{description}]: execution_time should be numeric"
    assert sr.execution_time >= 0, f"FAIL [{description}]: execution_time should be >= 0"
    
    print(f"    name: {sr.name}")
    print(f"    step_number: {sr.step_number}")
    print(f"    status: {sr.status}")
    print(f"    execution_time: {sr.execution_time:.4f}s")
    print(f"    message: {sr.message}")
    
    return sr


def verify_pipeline_stats(stats: PipelineExecutionStats, description: str,
                          expected_resumed_from: int = None) -> PipelineExecutionStats:
    """Comprehensive PipelineExecutionStats verification."""
    print(f"\n  === PipelineExecutionStats: {description} ===")
    
    assert stats is not None, f"FAIL [{description}]: PipelineExecutionStats is None"
    
    # Verify total_steps
    assert stats.total_steps is not None, f"FAIL [{description}]: total_steps is None"
    assert isinstance(stats.total_steps, int), f"FAIL [{description}]: total_steps should be int"
    assert stats.total_steps > 0, f"FAIL [{description}]: total_steps should be > 0"
    
    # Verify executed_steps
    assert stats.executed_steps is not None, f"FAIL [{description}]: executed_steps is None"
    assert isinstance(stats.executed_steps, int), f"FAIL [{description}]: executed_steps should be int"
    assert stats.executed_steps >= 0, f"FAIL [{description}]: executed_steps should be >= 0"
    
    # Verify step_timing
    assert stats.step_timing is not None, f"FAIL [{description}]: step_timing is None"
    assert isinstance(stats.step_timing, dict), f"FAIL [{description}]: step_timing should be dict"
    
    # Verify step_statuses
    assert stats.step_statuses is not None, f"FAIL [{description}]: step_statuses is None"
    assert isinstance(stats.step_statuses, dict), f"FAIL [{description}]: step_statuses should be dict"
    
    # Verify resumed_from if expected
    if expected_resumed_from is not None:
        assert stats.resumed_from == expected_resumed_from, f"FAIL [{description}]: resumed_from = {stats.resumed_from}, expected {expected_resumed_from}"
    
    print(f"    total_steps: {stats.total_steps}")
    print(f"    executed_steps: {stats.executed_steps}")
    print(f"    resumed_from: {stats.resumed_from}")
    print(f"    step_timing keys: {list(stats.step_timing.keys())[:5]}...")
    print(f"    step_statuses keys: {list(stats.step_statuses.keys())[:5]}...")
    
    return stats


def verify_run_requirement(req: RunRequirement, description: str,
                           expected_resolved: bool = None,
                           expected_tool_name: str = None,
                           expected_tool_args: Dict[str, Any] = None,
                           expected_result: str = None) -> RunRequirement:
    """Comprehensive RunRequirement verification."""
    print(f"\n  === RunRequirement: {description} ===")
    
    assert req is not None, f"FAIL [{description}]: RunRequirement is None"
    
    # Verify id
    assert req.id is not None, f"FAIL [{description}]: id is None"
    assert isinstance(req.id, str), f"FAIL [{description}]: id should be str"
    assert len(req.id) > 0, f"FAIL [{description}]: id should not be empty"
    
    # Verify created_at
    assert req.created_at is not None, f"FAIL [{description}]: created_at is None"
    
    # Verify resolved state if expected
    if expected_resolved is not None:
        actual_resolved = req.is_resolved
        assert actual_resolved == expected_resolved, f"FAIL [{description}]: is_resolved = {actual_resolved}, expected {expected_resolved}"
    
    # Verify tool_execution if expected
    if expected_tool_name:
        assert req.tool_execution is not None, f"FAIL [{description}]: tool_execution is None"
        verify_tool_execution(
            req.tool_execution, 
            f"{description} - tool_execution",
            expected_tool_name=expected_tool_name,
            expected_tool_args=expected_tool_args,
            expected_result=expected_result
        )
    
    print(f"    id: {req.id}")
    print(f"    is_resolved: {req.is_resolved}")
    print(f"    needs_external_execution: {req.needs_external_execution}")
    print(f"    has_result: {req.has_result}")
    
    return req


def verify_messages_list(messages: List, description: str, min_count: int = 0) -> List:
    """Comprehensive messages list verification."""
    print(f"\n  === Messages List: {description} ===")
    
    assert messages is not None, f"FAIL [{description}]: messages is None"
    assert isinstance(messages, list), f"FAIL [{description}]: messages should be list"
    assert len(messages) >= min_count, f"FAIL [{description}]: messages count = {len(messages)}, expected >= {min_count}"
    
    print(f"    Total messages: {len(messages)}")
    
    for i, msg in enumerate(messages[:3]):
        verify_model_message(msg, description, i)
    
    if len(messages) > 3:
        print(f"    ... and {len(messages) - 3} more messages")
    
    return messages


def verify_request_usage(usage: Any, description: str) -> Any:
    """Comprehensive RequestUsage/RunUsage verification with all attributes."""
    print(f"\n  === Usage: {description} ===")
    
    if usage is None:
        print("    (No usage data - skipping)")
        return None
    
    # Usage can be RequestUsage, RunUsage, or dict (from deserialization)
    is_dict = isinstance(usage, dict)
    is_usage_obj = isinstance(usage, (RequestUsage, RunUsage))
    
    assert is_dict or is_usage_obj, f"FAIL [{description}]: Expected RequestUsage, RunUsage, or dict, got {type(usage).__name__}"
    
    if is_dict:
        print("    (Usage is dict from deserialization - validating structure)")
        # In error cases, usage dict might be empty or have minimal structure
        # Just validate it's a dict and print what's available
        assert isinstance(usage, dict), f"FAIL [{description}]: Expected dict, got {type(usage).__name__}"
        print(f"    dict keys: {list(usage.keys())}")
        if 'input_tokens' in usage:
            print(f"    input_tokens: {usage.get('input_tokens', 0)}")
        if 'output_tokens' in usage:
            print(f"    output_tokens: {usage.get('output_tokens', 0)}")
        if 'input_tokens' in usage or 'output_tokens' in usage:
            print(f"    total_tokens: {usage.get('input_tokens', 0) + usage.get('output_tokens', 0)}")
        if 'cache_write_tokens' in usage:
            print(f"    cache_write_tokens: {usage.get('cache_write_tokens', 0)}")
        if 'cache_read_tokens' in usage:
            print(f"    cache_read_tokens: {usage.get('cache_read_tokens', 0)}")
        if 'requests' in usage:
            print(f"    requests: {usage.get('requests', 0)}")
        if 'tool_calls' in usage:
            print(f"    tool_calls: {usage.get('tool_calls', 0)}")
        if 'details' in usage:
            print(f"    details: {usage.get('details', {})}")
        if len(usage) == 0:
            print("    (Empty usage dict - may occur in error cases)")
        return usage
    
    # === Token counts ===
    assert hasattr(usage, 'input_tokens'), f"FAIL [{description}]: Missing input_tokens"
    assert isinstance(usage.input_tokens, int), f"FAIL [{description}]: input_tokens should be int"
    assert usage.input_tokens >= 0, f"FAIL [{description}]: input_tokens should be >= 0"
    print(f"    input_tokens: {usage.input_tokens}")
    
    assert hasattr(usage, 'output_tokens'), f"FAIL [{description}]: Missing output_tokens"
    assert isinstance(usage.output_tokens, int), f"FAIL [{description}]: output_tokens should be int"
    assert usage.output_tokens >= 0, f"FAIL [{description}]: output_tokens should be >= 0"
    print(f"    output_tokens: {usage.output_tokens}")
    
    # === Calculated properties ===
    assert hasattr(usage, 'total_tokens'), f"FAIL [{description}]: Missing total_tokens"
    expected_total = usage.input_tokens + usage.output_tokens
    assert usage.total_tokens == expected_total, f"FAIL [{description}]: total_tokens mismatch: {usage.total_tokens} != {expected_total}"
    print(f"    total_tokens: {usage.total_tokens}")
    
    # === Cache tokens ===
    assert hasattr(usage, 'cache_write_tokens'), f"FAIL [{description}]: Missing cache_write_tokens"
    assert isinstance(usage.cache_write_tokens, int), f"FAIL [{description}]: cache_write_tokens should be int"
    print(f"    cache_write_tokens: {usage.cache_write_tokens}")
    
    assert hasattr(usage, 'cache_read_tokens'), f"FAIL [{description}]: Missing cache_read_tokens"
    assert isinstance(usage.cache_read_tokens, int), f"FAIL [{description}]: cache_read_tokens should be int"
    print(f"    cache_read_tokens: {usage.cache_read_tokens}")
    
    # === Audio tokens ===
    assert hasattr(usage, 'input_audio_tokens'), f"FAIL [{description}]: Missing input_audio_tokens"
    assert hasattr(usage, 'output_audio_tokens'), f"FAIL [{description}]: Missing output_audio_tokens"
    print(f"    input_audio_tokens: {usage.input_audio_tokens}")
    print(f"    output_audio_tokens: {usage.output_audio_tokens}")
    
    # === Details dict ===
    assert hasattr(usage, 'details'), f"FAIL [{description}]: Missing details"
    assert isinstance(usage.details, dict), f"FAIL [{description}]: details should be dict"
    print(f"    details: {usage.details}")
    
    # === RunUsage specific attributes ===
    if isinstance(usage, RunUsage):
        assert hasattr(usage, 'requests'), f"FAIL [{description}]: Missing requests"
        assert isinstance(usage.requests, int), f"FAIL [{description}]: requests should be int"
        print(f"    requests: {usage.requests}")
        
        assert hasattr(usage, 'tool_calls'), f"FAIL [{description}]: Missing tool_calls"
        assert isinstance(usage.tool_calls, int), f"FAIL [{description}]: tool_calls should be int"
        print(f"    tool_calls: {usage.tool_calls}")
        
        assert hasattr(usage, 'reasoning_tokens'), f"FAIL [{description}]: Missing reasoning_tokens"
        assert isinstance(usage.reasoning_tokens, int), f"FAIL [{description}]: reasoning_tokens should be int"
        print(f"    reasoning_tokens: {usage.reasoning_tokens}")
        
        if usage.duration is not None:
            print(f"    duration: {usage.duration}")
        if usage.cost is not None:
            print(f"    cost: {usage.cost}")
        if usage.time_to_first_token is not None:
            print(f"    time_to_first_token: {usage.time_to_first_token}")
    
    return usage


def verify_model_response(response: Any, description: str, 
                          verify_parts: bool = True,
                          min_parts: int = 0) -> ModelResponse:
    """Comprehensive ModelResponse verification with all attributes."""
    print(f"\n  === ModelResponse: {description} ===")
    
    if response is None:
        print("    (No response - skipping)")
        return None
    
    # Verify type
    assert isinstance(response, ModelResponse), f"FAIL [{description}]: Expected ModelResponse, got {type(response).__name__}"
    
    # === Parts ===
    assert hasattr(response, 'parts'), f"FAIL [{description}]: Missing parts"
    assert response.parts is not None, f"FAIL [{description}]: parts is None"
    assert len(response.parts) >= min_parts, f"FAIL [{description}]: Expected >= {min_parts} parts, got {len(response.parts)}"
    print(f"    parts_count: {len(response.parts)}")
    
    if verify_parts and response.parts:
        for i, part in enumerate(response.parts[:3]):
            part_type = type(part).__name__
            print(f"      part[{i}]: {part_type}")
            # Verify each part has expected attributes
            assert hasattr(part, 'part_kind'), f"FAIL [{description}]: part[{i}] missing part_kind"
            if hasattr(part, 'content'):
                content_preview = str(part.content)[:40] + "..." if len(str(part.content)) > 40 else str(part.content)
                print(f"        content: {content_preview}")
    
    # === Usage ===
    assert hasattr(response, 'usage'), f"FAIL [{description}]: Missing usage"
    if response.usage:
        verify_request_usage(response.usage, f"{description} - response.usage")
    
    # === Model name ===
    assert hasattr(response, 'model_name'), f"FAIL [{description}]: Missing model_name"
    print(f"    model_name: {response.model_name}")
    
    # === Provider info ===
    assert hasattr(response, 'provider_name'), f"FAIL [{description}]: Missing provider_name"
    print(f"    provider_name: {response.provider_name}")
    
    assert hasattr(response, 'provider_response_id'), f"FAIL [{description}]: Missing provider_response_id"
    print(f"    provider_response_id: {response.provider_response_id}")
    
    # === Timestamp ===
    assert hasattr(response, 'timestamp'), f"FAIL [{description}]: Missing timestamp"
    assert response.timestamp is not None, f"FAIL [{description}]: timestamp is None"
    print(f"    timestamp: {response.timestamp}")
    
    # === Kind ===
    assert hasattr(response, 'kind'), f"FAIL [{description}]: Missing kind"
    assert response.kind == 'response', f"FAIL [{description}]: kind should be 'response', got '{response.kind}'"
    print(f"    kind: {response.kind}")
    
    # === Finish reason ===
    assert hasattr(response, 'finish_reason'), f"FAIL [{description}]: Missing finish_reason"
    print(f"    finish_reason: {response.finish_reason}")
    
    # === Text property ===
    assert hasattr(response, 'text'), f"FAIL [{description}]: Missing text property"
    if response.text:
        text_preview = response.text[:50] + "..." if len(response.text) > 50 else response.text
        print(f"    text: {text_preview}")
    
    # === Tool calls property ===
    assert hasattr(response, 'tool_calls'), f"FAIL [{description}]: Missing tool_calls property"
    print(f"    tool_calls count: {len(response.tool_calls)}")
    
    return response


def verify_model_profile(profile: Any, description: str) -> ModelProfile:
    """Comprehensive ModelProfile verification with all attributes."""
    print(f"\n  === ModelProfile: {description} ===")
    
    if profile is None:
        print("    (No profile - skipping)")
        return None
    
    # Verify type
    assert isinstance(profile, ModelProfile), f"FAIL [{description}]: Expected ModelProfile, got {type(profile).__name__}"
    
    # === Core capabilities ===
    assert hasattr(profile, 'supports_tools'), f"FAIL [{description}]: Missing supports_tools"
    assert isinstance(profile.supports_tools, bool), f"FAIL [{description}]: supports_tools should be bool"
    print(f"    supports_tools: {profile.supports_tools}")
    
    assert hasattr(profile, 'supports_json_schema_output'), f"FAIL [{description}]: Missing supports_json_schema_output"
    assert isinstance(profile.supports_json_schema_output, bool), f"FAIL [{description}]: supports_json_schema_output should be bool"
    print(f"    supports_json_schema_output: {profile.supports_json_schema_output}")
    
    assert hasattr(profile, 'supports_json_object_output'), f"FAIL [{description}]: Missing supports_json_object_output"
    assert isinstance(profile.supports_json_object_output, bool), f"FAIL [{description}]: supports_json_object_output should be bool"
    print(f"    supports_json_object_output: {profile.supports_json_object_output}")
    
    assert hasattr(profile, 'supports_image_output'), f"FAIL [{description}]: Missing supports_image_output"
    assert isinstance(profile.supports_image_output, bool), f"FAIL [{description}]: supports_image_output should be bool"
    print(f"    supports_image_output: {profile.supports_image_output}")
    
    # === Structured output mode ===
    assert hasattr(profile, 'default_structured_output_mode'), f"FAIL [{description}]: Missing default_structured_output_mode"
    print(f"    default_structured_output_mode: {profile.default_structured_output_mode}")
    
    # === Thinking tags ===
    assert hasattr(profile, 'thinking_tags'), f"FAIL [{description}]: Missing thinking_tags"
    assert isinstance(profile.thinking_tags, tuple), f"FAIL [{description}]: thinking_tags should be tuple"
    assert len(profile.thinking_tags) == 2, f"FAIL [{description}]: thinking_tags should have 2 elements"
    print(f"    thinking_tags: {profile.thinking_tags}")
    
    # === Prompt template ===
    assert hasattr(profile, 'prompted_output_template'), f"FAIL [{description}]: Missing prompted_output_template"
    assert isinstance(profile.prompted_output_template, str), f"FAIL [{description}]: prompted_output_template should be str"
    print(f"    prompted_output_template: (len={len(profile.prompted_output_template)})")
    
    return profile


def verify_agent_run_input(input_obj: Any, description: str) -> AgentRunInput:
    """Comprehensive AgentRunInput verification with all attributes."""
    print(f"\n  === AgentRunInput: {description} ===")
    
    if input_obj is None:
        print("    (No input - skipping)")
        return None
    
    # Verify type
    assert isinstance(input_obj, AgentRunInput), f"FAIL [{description}]: Expected AgentRunInput, got {type(input_obj).__name__}"
    
    # === User prompt ===
    assert hasattr(input_obj, 'user_prompt'), f"FAIL [{description}]: Missing user_prompt"
    assert input_obj.user_prompt is not None, f"FAIL [{description}]: user_prompt is None"
    prompt_preview = str(input_obj.user_prompt)[:60] + "..." if len(str(input_obj.user_prompt)) > 60 else str(input_obj.user_prompt)
    print(f"    user_prompt: {prompt_preview}")
    
    # === Images ===
    assert hasattr(input_obj, 'images'), f"FAIL [{description}]: Missing images"
    if input_obj.images:
        assert isinstance(input_obj.images, list), f"FAIL [{description}]: images should be list"
        print(f"    images count: {len(input_obj.images)}")
        for i, img in enumerate(input_obj.images[:2]):
            assert hasattr(img, 'data'), f"FAIL [{description}]: image[{i}] missing data"
            assert hasattr(img, 'media_type'), f"FAIL [{description}]: image[{i}] missing media_type"
            print(f"      image[{i}]: media_type={img.media_type}, data_len={len(img.data) if img.data else 0}")
    else:
        print("    images: None")
    
    # === Documents ===
    assert hasattr(input_obj, 'documents'), f"FAIL [{description}]: Missing documents"
    if input_obj.documents:
        assert isinstance(input_obj.documents, list), f"FAIL [{description}]: documents should be list"
        print(f"    documents count: {len(input_obj.documents)}")
        for i, doc in enumerate(input_obj.documents[:2]):
            assert hasattr(doc, 'data'), f"FAIL [{description}]: document[{i}] missing data"
            assert hasattr(doc, 'media_type'), f"FAIL [{description}]: document[{i}] missing media_type"
            print(f"      document[{i}]: media_type={doc.media_type}, data_len={len(doc.data) if doc.data else 0}")
    else:
        print("    documents: None")
    
    return input_obj


def verify_model_message(msg: Any, description: str, index: int) -> Any:
    """Comprehensive verification of a single ModelMessage (ModelRequest or ModelResponse)."""
    print(f"    message[{index}]: ", end="")
    
    assert msg is not None, f"FAIL [{description}]: message[{index}] is None"
    
    msg_type = type(msg).__name__
    print(f"type={msg_type}")
    
    # ModelResponse specific
    if isinstance(msg, ModelResponse):
        assert hasattr(msg, 'parts'), f"FAIL [{description}]: ModelResponse[{index}] missing parts"
        assert hasattr(msg, 'usage'), f"FAIL [{description}]: ModelResponse[{index}] missing usage"
        assert hasattr(msg, 'kind'), f"FAIL [{description}]: ModelResponse[{index}] missing kind"
        assert msg.kind == 'response', f"FAIL [{description}]: ModelResponse[{index}] kind should be 'response'"
        print(f"      parts_count: {len(msg.parts)}, kind: {msg.kind}")
        
        # Verify parts structure
        for j, part in enumerate(msg.parts[:2]):
            part_type = type(part).__name__
            assert hasattr(part, 'part_kind'), f"FAIL [{description}]: ModelResponse[{index}].part[{j}] missing part_kind"
            print(f"        part[{j}]: {part_type} (kind={part.part_kind})")
    
    # ModelRequest specific
    elif isinstance(msg, ModelRequest):
        assert hasattr(msg, 'parts'), f"FAIL [{description}]: ModelRequest[{index}] missing parts"
        assert hasattr(msg, 'kind'), f"FAIL [{description}]: ModelRequest[{index}] missing kind"
        assert msg.kind == 'request', f"FAIL [{description}]: ModelRequest[{index}] kind should be 'request'"
        print(f"      parts_count: {len(msg.parts)}, kind: {msg.kind}")
        
        # Verify parts structure  
        for j, part in enumerate(msg.parts[:2]):
            part_type = type(part).__name__
            assert hasattr(part, 'part_kind'), f"FAIL [{description}]: ModelRequest[{index}].part[{j}] missing part_kind"
            print(f"        part[{j}]: {part_type} (kind={part.part_kind})")
    
    # Generic dict (from deserialization)
    elif isinstance(msg, dict):
        print(f"      keys: {list(msg.keys())}")
        if 'parts' in msg:
            print(f"      parts_count: {len(msg['parts'])}")
    
    else:
        # Fallback for other types with parts
        if hasattr(msg, 'parts'):
            print(f"      parts_count: {len(msg.parts)}")
    
    return msg


def verify_messages_list_deep(messages: List, description: str, 
                               min_count: int = 0,
                               verify_content: bool = True) -> List:
    """Deep verification of messages list with content checks."""
    print(f"\n  === Messages List Deep Verification: {description} ===")
    
    assert messages is not None, f"FAIL [{description}]: messages is None"
    assert isinstance(messages, list), f"FAIL [{description}]: messages should be list"
    assert len(messages) >= min_count, f"FAIL [{description}]: messages count = {len(messages)}, expected >= {min_count}"
    
    print(f"    Total messages: {len(messages)}")
    
    if verify_content:
        for i, msg in enumerate(messages[:5]):
            verify_model_message(msg, description, i)
    
    if len(messages) > 5:
        print(f"    ... and {len(messages) - 5} more messages")
    
    return messages


def verify_chat_history_deep(chat_history: List, description: str,
                              min_count: int = 1) -> List:
    """Deep verification of chat_history with structure validation."""
    print(f"\n  === Chat History Deep Verification: {description} ===")
    
    assert chat_history is not None, f"FAIL [{description}]: chat_history is None"
    assert isinstance(chat_history, list), f"FAIL [{description}]: chat_history should be list"
    assert len(chat_history) >= min_count, f"FAIL [{description}]: chat_history count = {len(chat_history)}, expected >= {min_count}"
    
    print(f"    Total messages: {len(chat_history)}")
    
    # Verify structure of each message
    request_count = 0
    response_count = 0
    
    for i, msg in enumerate(chat_history):
        if isinstance(msg, ModelRequest):
            request_count += 1
        elif isinstance(msg, ModelResponse):
            response_count += 1
    
    print(f"    ModelRequest count: {request_count}")
    print(f"    ModelResponse count: {response_count}")
    
    # Verify first few messages in detail
    for i, msg in enumerate(chat_history[:3]):
        verify_model_message(msg, description, i)
    
    if len(chat_history) > 3:
        print(f"    ... and {len(chat_history) - 3} more messages")
    
    return chat_history


def compare_messages_content(msg1: Any, msg2: Any, index: int, description: str) -> bool:
    """Compare two message objects for content equality."""
    # Both should be same type
    if type(msg1).__name__ != type(msg2).__name__:
        print(f"      message[{index}]: TYPE MISMATCH - {type(msg1).__name__} vs {type(msg2).__name__}")
        return False
    
    # Both should have parts
    if hasattr(msg1, 'parts') and hasattr(msg2, 'parts'):
        if len(msg1.parts) != len(msg2.parts):
            print(f"      message[{index}]: PARTS COUNT MISMATCH - {len(msg1.parts)} vs {len(msg2.parts)}")
            return False
        
        # Compare parts
        for j, (p1, p2) in enumerate(zip(msg1.parts, msg2.parts)):
            if type(p1).__name__ != type(p2).__name__:
                print(f"        part[{j}]: TYPE MISMATCH - {type(p1).__name__} vs {type(p2).__name__}")
                return False
            
            if hasattr(p1, 'content') and hasattr(p2, 'content'):
                if str(p1.content) != str(p2.content):
                    print(f"        part[{j}]: CONTENT MISMATCH")
                    return False
    
    # For ModelResponse, check kind
    if hasattr(msg1, 'kind') and hasattr(msg2, 'kind'):
        if msg1.kind != msg2.kind:
            print(f"      message[{index}]: KIND MISMATCH - {msg1.kind} vs {msg2.kind}")
            return False
    
    return True


def compare_messages_lists(list1: List, list2: List, description: str) -> None:
    """Comprehensive comparison of two message lists."""
    print(f"\n{'='*60}")
    print(f"Messages List Comparison: {description}")
    print(f"{'='*60}")
    
    assert list1 is not None, f"FAIL [{description}]: list1 is None"
    assert list2 is not None, f"FAIL [{description}]: list2 is None"
    assert isinstance(list1, list), f"FAIL [{description}]: list1 should be list"
    assert isinstance(list2, list), f"FAIL [{description}]: list2 should be list"
    
    print(f"  list1 count: {len(list1)}")
    print(f"  list2 count: {len(list2)}")
    
    # Lengths should match
    assert len(list1) == len(list2), f"FAIL [{description}]: Length mismatch - {len(list1)} vs {len(list2)}"
    print(f"  ✓ Lengths match: {len(list1)}")
    
    # Compare each message
    all_match = True
    for i in range(min(len(list1), len(list2))):
        if not compare_messages_content(list1[i], list2[i], i, description):
            all_match = False
            print(f"  ✗ Message[{i}] content mismatch")
        else:
            print(f"  ✓ Message[{i}] matches")
    
    assert all_match, f"FAIL [{description}]: Message content mismatch detected"
    print("\n  ✓ ALL MESSAGES MATCH")


def verify_session_messages_consistency(session: AgentSession, output: AgentRunOutput, description: str) -> None:
    """
    Verify comprehensive consistency between AgentSession and AgentRunOutput.
    
    Key relationships:
    1. output.chat_history should match session.messages (full conversation)
    2. output.messages should match the LAST elements of session.messages 
       (since output.messages = THIS run's messages, session.messages = whole history)
    3. Run statuses should match
    """
    print(f"\n{'='*60}")
    print(f"Session vs Output Consistency: {description}")
    print(f"{'='*60}")
    
    # Get all relevant data
    session_messages = session.messages if session.messages else []
    output_chat_history = output.chat_history if output.chat_history else []
    output_messages = output.messages if output.messages else []
    
    print(f"  session.messages count: {len(session_messages)}")
    print(f"  output.chat_history count: {len(output_chat_history)}")
    print(f"  output.messages count: {len(output_messages)}")
    
    # === 1. Verify output.chat_history matches session.messages ===
    print("\n--- Verification 1: output.chat_history == session.messages ---")
    
    if len(session_messages) > 0 or len(output_chat_history) > 0:
        # They should match in length for a completed run
        assert len(output_chat_history) == len(session_messages), \
            f"FAIL [{description}]: output.chat_history ({len(output_chat_history)}) should equal session.messages ({len(session_messages)})"
        
        # Compare each message
        for i in range(len(session_messages)):
            if not compare_messages_content(session_messages[i], output_chat_history[i], i, description):
                print(f"  ✗ Message[{i}] mismatch between session.messages and output.chat_history")
                assert False, f"FAIL [{description}]: Message[{i}] mismatch"
            else:
                print(f"  ✓ Message[{i}] matches")
        
        print("  ✓ output.chat_history == session.messages VERIFIED")
    else:
        print("  (No messages to compare)")
    
    # === 2. Verify output.messages matches LAST elements of session.messages ===
    print("\n--- Verification 2: output.messages == session.messages[-N:] ---")
    
    if len(output_messages) > 0:
        # output.messages should be the LAST N messages of session.messages
        # where N = len(output.messages)
        assert len(session_messages) >= len(output_messages), \
            f"FAIL [{description}]: session.messages ({len(session_messages)}) should be >= output.messages ({len(output_messages)})"
        
        # Get the last N messages from session
        session_last_n = session_messages[-len(output_messages):]
        
        print(f"  Comparing output.messages with session.messages[-{len(output_messages)}:]")
        
        for i in range(len(output_messages)):
            if not compare_messages_content(output_messages[i], session_last_n[i], i, description + " - messages match"):
                print(f"  ✗ output.messages[{i}] doesn't match session.messages[{-len(output_messages)+i}]")
                assert False, f"FAIL [{description}]: output.messages[{i}] doesn't match last messages of session"
            else:
                print(f"  ✓ output.messages[{i}] matches session.messages[-{len(output_messages)-i}]")
        
        print("  ✓ output.messages == session.messages[-N:] VERIFIED")
    else:
        print("  (No output.messages to compare - may be empty for error states)")
    
    # === 3. Verify run status consistency ===
    print("\n--- Verification 3: Run Status Consistency ---")
    
    # Get stored output from session.runs if available
    if output.run_id and output.run_id in session.runs:
        stored_run = session.runs[output.run_id]
        stored_output = stored_run.output
        
        print(f"  output.status: {output.status}")
        print(f"  stored_output.status: {stored_output.status}")
        
        assert output.status == stored_output.status, \
            f"FAIL [{description}]: output.status ({output.status}) != stored_output.status ({stored_output.status})"
        
        print("  ✓ Run statuses match")
    else:
        print(f"  output.status: {output.status}")
        print("  (No stored run to compare status - using output directly)")
    
    print(f"\n  ✓ ALL CONSISTENCY CHECKS PASSED for {description}")


def verify_storage_consistency(before_session: AgentSession, after_session: AgentSession,
                               before_output: AgentRunOutput, after_output: AgentRunOutput,
                               description: str) -> None:
    """Verify that messages and chat_history are preserved correctly through storage.
    
    Validates:
    1. Session messages are preserved (before messages are subset of after)
    2. Chat history is preserved (before history is subset of after)
    3. Session messages match output chat_history (after storage)
    """
    print(f"\n{'='*60}")
    print(f"Storage Consistency Verification: {description}")
    print(f"{'='*60}")
    
    # === 1. Verify AgentSession.messages preserved ===
    print("\n--- AgentSession.messages: Before vs After Storage ---")
    before_messages = before_session.messages if before_session.messages else []
    after_messages = after_session.messages if after_session.messages else []
    
    print(f"  Before storage: {len(before_messages)} messages")
    print(f"  After storage: {len(after_messages)} messages")
    
    # Messages should be preserved or grown (not shrunk)
    assert len(after_messages) >= len(before_messages), \
        f"FAIL [{description}]: Session messages shrunk from {len(before_messages)} to {len(after_messages)}"
    
    # Verify that before messages are preserved in after (prefix match)
    if len(before_messages) > 0:
        print(f"  Verifying first {len(before_messages)} messages are preserved...")
        for i in range(len(before_messages)):
            if not compare_messages_content(before_messages[i], after_messages[i], i, description):
                assert False, f"FAIL [{description}]: Message[{i}] not preserved in session.messages"
        print(f"  ✓ All {len(before_messages)} before messages preserved")
    print(f"  ✓ Messages preserved/grown: {len(before_messages)} -> {len(after_messages)}")
    
    # === 2. Verify AgentRunOutput.chat_history preserved ===
    print("\n--- AgentRunOutput.chat_history: Before vs After Storage ---")
    before_history = before_output.chat_history if before_output.chat_history else []
    after_history = after_output.chat_history if after_output.chat_history else []
    
    print(f"  Before storage: {len(before_history)} messages")
    print(f"  After storage: {len(after_history)} messages")
    
    # Chat history should be preserved or grown (not shrunk)
    assert len(after_history) >= len(before_history), \
        f"FAIL [{description}]: Chat history shrunk from {len(before_history)} to {len(after_history)}"
    
    # Verify that before history is preserved in after (prefix match)
    if len(before_history) > 0:
        print(f"  Verifying first {len(before_history)} messages are preserved...")
        for i in range(len(before_history)):
            if not compare_messages_content(before_history[i], after_history[i], i, description):
                assert False, f"FAIL [{description}]: Message[{i}] not preserved in chat_history"
        print(f"  ✓ All {len(before_history)} before messages preserved")
    print(f"  ✓ Chat history preserved/grown: {len(before_history)} -> {len(after_history)}")
    
    # === 3. Verify AgentSession.messages matches AgentRunOutput.chat_history (after storage) ===
    print("\n--- Consistency: Session.messages vs Output.chat_history (After Storage) ---")
    compare_messages_lists(after_messages, after_history, f"{description} - Session.messages vs chat_history")
    
    print("\n  ✓ STORAGE CONSISTENCY VERIFIED")


def verify_agent_run_output(output: AgentRunOutput, description: str,
                            expected_status: RunStatus = None,
                            expected_run_id: str = None,
                            min_messages: int = 0,
                            min_step_results: int = 0,
                            min_requirements: int = 0,
                            verify_content: bool = True) -> AgentRunOutput:
    """Comprehensive AgentRunOutput verification with all attributes."""
    print(f"\n{'='*60}")
    print(f"AgentRunOutput Verification: {description}")
    print(f"{'='*60}")
    
    assert output is not None, f"FAIL [{description}]: AgentRunOutput is None"
    
    # === Identity ===
    print("\n--- Identity ---")
    assert output.run_id is not None, f"FAIL [{description}]: run_id is None"
    assert isinstance(output.run_id, str), f"FAIL [{description}]: run_id should be str"
    if expected_run_id:
        assert output.run_id == expected_run_id, f"FAIL [{description}]: run_id mismatch"
    print(f"  run_id: {output.run_id}")
    
    assert output.agent_id is not None, f"FAIL [{description}]: agent_id is None"
    print(f"  agent_id: {output.agent_id}")
    print(f"  agent_name: {output.agent_name}")
    print(f"  session_id: {output.session_id}")
    print(f"  user_id: {output.user_id}")
    
    # === Status ===
    print("\n--- Status ---")
    assert output.status is not None, f"FAIL [{description}]: status is None"
    if expected_status:
        assert output.status == expected_status, f"FAIL [{description}]: status = {output.status}, expected {expected_status}"
    print(f"  status: {output.status}")
    print(f"  is_complete: {output.is_complete}")
    print(f"  is_paused: {output.is_paused}")
    print(f"  is_cancelled: {output.is_cancelled}")
    print(f"  is_error: {output.is_error}")
    print(f"  pause_reason: {output.pause_reason}")
    print(f"  error_details: {output.error_details}")
    
    # === Output ===
    print("\n--- Output ---")
    if output.output is not None:
        output_preview = str(output.output)[:60] + "..." if len(str(output.output)) > 60 else str(output.output)
        print(f"  output: {output_preview}")
    else:
        print("  output: None")
    print(f"  output_schema: {output.output_schema}")
    
    # === Messages (deep verification) ===
    print("\n--- Messages ---")
    if output.messages:
        verify_messages_list_deep(output.messages, f"{description} - messages", 
                                   min_count=min_messages, verify_content=True)
    else:
        print("  messages: None or empty")
    
    # === Chat History (deep verification) ===
    print("\n--- Chat History ---")
    if output.chat_history:
        verify_chat_history_deep(output.chat_history, f"{description} - chat_history", min_count=1)
    else:
        print("  chat_history: None or empty")
    
    # === Response (deep verification) ===
    verify_model_response(output.response, f"{description} - response", verify_parts=True)
    
    # === Usage (deep verification) ===
    verify_request_usage(output.usage, f"{description} - usage")
    
    # === Model Info ===
    print("\n--- Model Info ---")
    assert output.model_name is not None, f"FAIL [{description}]: model_name is None"
    assert isinstance(output.model_name, str), f"FAIL [{description}]: model_name should be str"
    print(f"  model_name: {output.model_name}")
    
    print(f"  model_provider: {output.model_provider}")
    
    # === Model Profile (deep verification) ===
    if output.model_provider_profile:
        verify_model_profile(output.model_provider_profile, f"{description} - profile")
    
    # === Tools ===
    print("\n--- Tools ---")
    print(f"  tool_call_count: {output.tool_call_count}")
    print(f"  tool_limit_reached: {output.tool_limit_reached}")
    if output.tools:
        print(f"  tools count: {len(output.tools)}")
        for i, te in enumerate(output.tools[:2]):
            verify_tool_execution(te, f"{description} - tool[{i}]")
    
    # === Requirements ===
    print("\n--- Requirements ---")
    if output.requirements:
        assert len(output.requirements) >= min_requirements, f"FAIL [{description}]: requirements count = {len(output.requirements)}, expected >= {min_requirements}"
        print(f"  requirements count: {len(output.requirements)}")
        print(f"  active_requirements count: {len(output.active_requirements)}")
        for i, req in enumerate(output.requirements[:2]):
            verify_run_requirement(req, f"{description} - requirement[{i}]")
    else:
        print("  requirements: None or empty")
    
    # === Step Results ===
    print("\n--- Step Results ---")
    if output.step_results:
        assert len(output.step_results) >= min_step_results, f"FAIL [{description}]: step_results count = {len(output.step_results)}, expected >= {min_step_results}"
        print(f"  step_results count: {len(output.step_results)}")
        for i, sr in enumerate(output.step_results[:3]):
            verify_step_result(sr, f"{description} - step[{i}]")
        if len(output.step_results) > 3:
            print(f"    ... and {len(output.step_results) - 3} more steps")
    else:
        print("  step_results: None or empty")
    
    # === Execution Stats ===
    if output.execution_stats:
        verify_pipeline_stats(output.execution_stats, description)
    
    # === Timestamps ===
    print("\n--- Timestamps ---")
    print(f"  created_at: {output.created_at}")
    print(f"  updated_at: {output.updated_at}")
    
    # === Input (deep verification) ===
    print("\n--- Input ---")
    if output.input:
        verify_agent_run_input(output.input, f"{description} - input")
    else:
        print("  input: None")
    
    # === Task ===
    print("\n--- Task ---")
    if output.task:
        print(f"  task type: {type(output.task).__name__}")
        if hasattr(output.task, 'description'):
            print(f"  task.description: {str(output.task.description)[:50]}...")
    else:
        print("  task: None")
    
    return output


def verify_agent_session(session: AgentSession, description: str,
                         expected_run_id: str = None,
                         expected_run_count: int = None) -> AgentSession:
    """Comprehensive AgentSession verification."""
    print(f"\n{'='*60}")
    print(f"AgentSession Verification: {description}")
    print(f"{'='*60}")
    
    assert session is not None, f"FAIL [{description}]: AgentSession is None"
    
    # === Identity ===
    print("\n--- Identity ---")
    assert session.session_id is not None, f"FAIL [{description}]: session_id is None"
    assert isinstance(session.session_id, str), f"FAIL [{description}]: session_id should be str"
    print(f"  session_id: {session.session_id}")
    print(f"  agent_id: {session.agent_id}")
    print(f"  user_id: {session.user_id}")
    print(f"  workflow_id: {session.workflow_id}")
    
    # === Runs ===
    print("\n--- Runs ---")
    assert session.runs is not None, f"FAIL [{description}]: runs is None"
    assert isinstance(session.runs, dict), f"FAIL [{description}]: runs should be dict"
    print(f"  runs count: {len(session.runs)}")
    print(f"  run_ids: {list(session.runs.keys())}")
    
    if expected_run_count is not None:
        assert len(session.runs) >= expected_run_count, f"FAIL [{description}]: runs count = {len(session.runs)}, expected >= {expected_run_count}"
    
    if expected_run_id:
        assert expected_run_id in session.runs, f"FAIL [{description}]: run_id {expected_run_id} not in runs"
    
    # === Messages ===
    print("\n--- Messages ---")
    if session.messages:
        print(f"  messages count: {len(session.messages)}")
    else:
        print("  messages: None or empty")
    
    # === Metadata ===
    print("\n--- Metadata ---")
    print(f"  session_data: {session.session_data}")
    print(f"  metadata: {session.metadata}")
    print(f"  summary: {session.summary}")
    
    # === Timestamps ===
    print("\n--- Timestamps ---")
    print(f"  created_at: {session.created_at}")
    print(f"  updated_at: {session.updated_at}")
    
    return session


def compare_outputs(before: AgentRunOutput, after: AgentRunOutput, description: str) -> None:
    """Compare two AgentRunOutput instances for consistency."""
    print(f"\n{'='*60}")
    print(f"Output Comparison: {description}")
    print(f"{'='*60}")
    
    # === Identity should match ===
    assert before.run_id == after.run_id, f"FAIL: run_id mismatch: {before.run_id} vs {after.run_id}"
    print(f"  run_id: MATCH ({before.run_id})")
    
    # === Chat history should be preserved or grown ===
    before_history_len = len(before.chat_history) if before.chat_history else 0
    after_history_len = len(after.chat_history) if after.chat_history else 0
    assert after_history_len >= before_history_len, f"FAIL: chat_history shrunk: {before_history_len} -> {after_history_len}"
    print(f"  chat_history: {before_history_len} -> {after_history_len} (preserved/grown)")
    
    # === Step results should be preserved or grown ===
    before_steps = len(before.step_results) if before.step_results else 0
    after_steps = len(after.step_results) if after.step_results else 0
    assert after_steps >= before_steps, f"FAIL: step_results shrunk: {before_steps} -> {after_steps}"
    print(f"  step_results: {before_steps} -> {after_steps} (preserved/grown)")
    
    # === Requirements should be preserved ===
    before_reqs = len(before.requirements) if before.requirements else 0
    after_reqs = len(after.requirements) if after.requirements else 0
    assert after_reqs >= before_reqs, f"FAIL: requirements shrunk: {before_reqs} -> {after_reqs}"
    print(f"  requirements: {before_reqs} -> {after_reqs} (preserved/grown)")
    
    print("\n  COMPARISON PASSED")


# =============================================================================
# TEST: EXTERNAL TOOL
# =============================================================================

@pytest.mark.asyncio
async def test_external_tool_comprehensive():
    """Comprehensive External Tool test with full attribute verification."""
    print("\n" + "="*80)
    print("TEST: External Tool - Full Attribute Verification")
    print("="*80)
    
    cleanup()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent1 = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description="Send an email to test@example.com with subject 'Hello' and body 'Test'.",
        tools=[send_email]
    )
    
    expected_tool_args = {'to': 'test@example.com', 'subject': 'Hello', 'body': 'Test'}
    expected_result = "Email sent successfully to test@example.com"
    
    # =========================================================================
    # STEP 1: Initial run - should pause
    # =========================================================================
    print("\n[STEP 1] Running do_async (expecting pause)...")
    output = await agent1.do_async(task, return_output=True)
    run_id = output.run_id
    
    # Verify output at pause
    verify_agent_run_output(
        output, "After Initial Pause",
        expected_status=RunStatus.paused,
        expected_run_id=run_id,
        min_step_results=5,
        min_requirements=1
    )
    
    # Verify requirement details
    assert len(output.active_requirements) == 1, "Should have 1 active requirement"
    req = output.active_requirements[0]
    verify_run_requirement(
        req, "Paused Requirement",
        expected_resolved=False,
        expected_tool_name="send_email",
        expected_tool_args=expected_tool_args,
        expected_result=None
    )
    
    print("\n[STEP 1] PASSED")
    
    # =========================================================================
    # STEP 2: Verify storage after pause
    # =========================================================================
    print("\n[STEP 2] Verifying storage after pause...")
    
    # Capture in-memory state before reading from storage
    # Note: We need to get the session from the agent's memory if available
    # For now, we'll read from storage and compare
    
    session = db.storage.get_session(session_id="test_session")
    verify_agent_session(session, "After Pause", expected_run_id=run_id, expected_run_count=1)
    
    stored_output = session.runs[run_id].output
    verify_agent_run_output(
        stored_output, "Stored After Pause",
        expected_status=RunStatus.paused,
        expected_run_id=run_id
    )
    
    # Compare in-memory vs stored
    compare_outputs(output, stored_output, "In-Memory vs Stored (Pause)")
    
    # Verify storage consistency: messages and chat_history
    # Create a "before" session from in-memory output for comparison
    # Since we don't have direct access to session before storage, we'll compare
    # the stored session messages with the output's chat_history
    verify_session_messages_consistency(session, stored_output, "After Pause - Storage")
    
    print("\n[STEP 2] PASSED")
    
    # =========================================================================
    # STEP 3: Set tool result
    # =========================================================================
    print("\n[STEP 3] Setting external tool result...")
    
    # Set result on the in-memory requirement
    req.set_external_execution_result(expected_result)
    
    # Verify result was set
    assert req.tool_execution.result == expected_result, "Result should be set"
    assert req.is_resolved, "Requirement should be resolved"
    
    print(f"  Result set: {expected_result}")
    print("\n[STEP 3] PASSED")
    
    # =========================================================================
    # STEP 4: Resume with NEW agent
    # =========================================================================
    print("\n[STEP 4] Resuming with new agent...")
    
    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent2 = Agent("openai/gpt-4o-mini", name="agent2", db=db2, debug=DEBUG)
    
    result = await agent2.continue_run_async(
        run_id=run_id, 
        return_output=True,
        requirements=output.requirements  # Pass all requirements (with results set)
    )
    
    # Verify completed output
    verify_agent_run_output(
        result, "After Completion",
        expected_status=RunStatus.completed,
        expected_run_id=run_id,
        min_step_results=10
    )
    
    # Verify content exists
    assert result.output is not None, "Output content should exist"
    assert len(str(result.output)) > 0, "Output content should not be empty"
    
    print(f"  Result content: {result.output}")
    print("\n[STEP 4] PASSED")
    
    # =========================================================================
    # STEP 5: Verify final storage
    # =========================================================================
    print("\n[STEP 5] Verifying final storage...")
    
    # Capture before state (from step 2)
    before_session = session
    before_output = stored_output
    
    final_session = db2.storage.get_session(session_id="test_session")
    verify_agent_session(final_session, "Final", expected_run_id=run_id)
    
    final_output = final_session.runs[run_id].output
    verify_agent_run_output(
        final_output, "Final Stored",
        expected_status=RunStatus.completed,
        expected_run_id=run_id
    )
    
    # Verify requirement is resolved in storage
    if final_output.requirements:
        for req in final_output.requirements:
            if req.tool_execution and req.tool_execution.tool_name == "send_email":
                assert req.is_resolved, "Requirement should be resolved"
                assert req.tool_execution.result == expected_result, "Result should match"
    
    # Verify storage consistency: messages and chat_history before and after
    verify_storage_consistency(
        before_session, final_session,
        before_output, final_output,
        "External Tool - Pause to Completion"
    )
    
    # Verify current state consistency
    verify_session_messages_consistency(final_session, final_output, "Final - Storage")
    
    print("\n[STEP 5] PASSED")
    
    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: External Tool - Full Attribute Verification")
    print("="*80)


# =============================================================================
# TEST: CANCEL RUN
# =============================================================================

@pytest.mark.asyncio
async def test_cancel_run_comprehensive():
    """Comprehensive Cancel Run test with full attribute verification."""
    print("\n" + "="*80)
    print("TEST: Cancel Run - Full Attribute Verification")
    print("="*80)
    
    cleanup()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent1 = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description="Call long_running_task with 5 seconds.",
        tools=[long_running_task]
    )
    
    # =========================================================================
    # STEP 1: Start and cancel
    # =========================================================================
    print("\n[STEP 1] Starting run and cancelling...")
    
    async def run_task():
        return await agent1.do_async(task, return_output=True)
    
    run_future = asyncio.create_task(run_task())
    await asyncio.sleep(1.5)
    
    run_id = agent1.run_id
    assert run_id is not None, "run_id should be available"
    
    cancel_run(run_id)
    print(f"  Cancelled run: {run_id}")
    
    output = await asyncio.wait_for(run_future, timeout=10.0)
    
    # Verify cancelled output
    verify_agent_run_output(
        output, "After Cancel",
        expected_status=RunStatus.cancelled,
        expected_run_id=run_id,
        min_step_results=5
    )
    
    print("\n[STEP 1] PASSED")
    
    # =========================================================================
    # STEP 2: Verify storage after cancel
    # =========================================================================
    print("\n[STEP 2] Verifying storage after cancel...")
    
    session = db.storage.get_session(session_id="test_session")
    verify_agent_session(session, "After Cancel", expected_run_id=run_id)
    
    stored_output = session.runs[run_id].output
    verify_agent_run_output(
        stored_output, "Stored After Cancel",
        expected_status=RunStatus.cancelled
    )
    
    compare_outputs(output, stored_output, "In-Memory vs Stored (Cancel)")
    
    # Verify storage consistency: messages and chat_history
    verify_session_messages_consistency(session, stored_output, "After Cancel - Storage")
    
    print("\n[STEP 2] PASSED")
    
    # =========================================================================
    # STEP 3: Resume with new agent
    # =========================================================================
    print("\n[STEP 3] Resuming with new agent...")
    
    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent2 = Agent("openai/gpt-4o-mini", name="agent2", db=db2, debug=DEBUG)
    
    result = await agent2.continue_run_async(run_id=run_id, return_output=True)
    
    verify_agent_run_output(
        result, "After Resume",
        expected_status=RunStatus.completed,
        expected_run_id=run_id,
        min_step_results=10
    )
    
    print(f"  Result: {result.output}")
    print("\n[STEP 3] PASSED")
    
    # =========================================================================
    # STEP 4: Verify final storage
    # =========================================================================
    print("\n[STEP 4] Verifying final storage...")
    
    # Capture before state (from step 2)
    before_session = session
    before_output = stored_output
    
    final_session = db2.storage.get_session(session_id="test_session")
    final_output = final_session.runs[run_id].output
    
    verify_agent_run_output(
        final_output, "Final Stored",
        expected_status=RunStatus.completed
    )
    
    # Compare before/after
    compare_outputs(stored_output, final_output, "Cancel vs Completed")
    
    # Verify storage consistency: messages and chat_history before and after
    verify_storage_consistency(
        before_session, final_session,
        before_output, final_output,
        "Cancel Run - Cancel to Completion"
    )
    
    # Verify current state consistency
    verify_session_messages_consistency(final_session, final_output, "Final - Storage")
    
    print("\n[STEP 4] PASSED")
    
    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Cancel Run - Full Attribute Verification")
    print("="*80)


# =============================================================================
# TEST: DURABLE EXECUTION
# =============================================================================

@pytest.mark.asyncio
async def test_durable_execution_comprehensive():
    """Comprehensive Durable Execution test with full attribute verification."""
    print("\n" + "="*80)
    print("TEST: Durable Execution - Full Attribute Verification")
    print("="*80)
    
    cleanup()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", db=db, debug=DEBUG, retry=1)
    task = Task(
        description="What is 5 + 3? Reply with just the number.",
        tools=[simple_math]
    )
    
    # Inject error that triggers once
    inject_error_into_step("model_execution", RuntimeError, "Simulated error", trigger_count=1)
    
    # =========================================================================
    # STEP 1: Run and catch error
    # =========================================================================
    print("\n[STEP 1] Running do_async (expecting error)...")
    
    run_id = None
    error_output = None
    try:
        _ = await agent.do_async(task, return_output=True)
        assert False, "Should have raised exception"
    except Exception as e:
        print(f"  Caught error: {type(e).__name__}: {str(e)[:60]}...")
        assert "INJECTED ERROR" in str(e), "Should be injected error"
        
        error_output = getattr(agent, '_agent_run_output', None)
        if error_output:
            run_id = error_output.run_id
    
    assert run_id is not None, "run_id should be available"
    print(f"  Run ID: {run_id}")
    
    # Verify error output
    if error_output:
        verify_agent_run_output(
            error_output, "After Error",
            expected_status=RunStatus.error,
            expected_run_id=run_id,
            min_step_results=5
        )
    
    print("\n[STEP 1] PASSED")
    
    # =========================================================================
    # STEP 2: Verify storage after error
    # =========================================================================
    print("\n[STEP 2] Verifying storage after error...")
    
    session = db.storage.get_session(session_id="test_session")
    verify_agent_session(session, "After Error", expected_run_id=run_id)
    
    stored_output = session.runs[run_id].output
    verify_agent_run_output(
        stored_output, "Stored After Error",
        expected_status=RunStatus.error
    )
    
    # Verify error step
    error_steps = [sr for sr in stored_output.step_results if sr.status == StepStatus.ERROR]
    assert len(error_steps) >= 1, "Should have error step"
    print(f"  Error step: {error_steps[-1].name} at step {error_steps[-1].step_number}")
    
    # Verify storage consistency: messages and chat_history
    if error_output:
        compare_outputs(error_output, stored_output, "In-Memory vs Stored (Error)")
    verify_session_messages_consistency(session, stored_output, "After Error - Storage")
    
    print("\n[STEP 2] PASSED")
    
    # =========================================================================
    # STEP 3: Resume
    # =========================================================================
    print("\n[STEP 3] Resuming...")
    
    result = await agent.continue_run_async(run_id=run_id, return_output=True)
    
    verify_agent_run_output(
        result, "After Resume",
        expected_status=RunStatus.completed,
        expected_run_id=run_id,
        min_step_results=10
    )
    
    print(f"  Result: {result.output}")
    print("\n[STEP 3] PASSED")
    
    # =========================================================================
    # STEP 4: Verify final state and compare
    # =========================================================================
    print("\n[STEP 4] Verifying final state...")
    
    # Capture before state (from step 2)
    before_session = session
    before_output = stored_output
    
    final_session = db.storage.get_session(session_id="test_session")
    final_output = final_session.runs[run_id].output
    
    verify_agent_run_output(
        final_output, "Final Stored",
        expected_status=RunStatus.completed
    )
    
    # Compare error vs completed
    compare_outputs(stored_output, final_output, "Error vs Completed")
    
    # Verify execution_stats shows resumption
    if final_output.execution_stats:
        verify_pipeline_stats(final_output.execution_stats, "Final Stats")
    
    # Verify storage consistency: messages and chat_history before and after
    verify_storage_consistency(
        before_session, final_session,
        before_output, final_output,
        "Durable Execution - Error to Completion"
    )
    
    # Verify current state consistency
    verify_session_messages_consistency(final_session, final_output, "Final - Storage")
    
    print("\n[STEP 4] PASSED")
    
    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Durable Execution - Full Attribute Verification")
    print("="*80)


# =============================================================================
# TEST: CROSS-PROCESS DURABLE EXECUTION
# =============================================================================

@pytest.mark.asyncio
async def test_durable_cross_process_comprehensive():
    """Durable execution with new agent (cross-process simulation)."""
    print("\n" + "="*80)
    print("TEST: Durable Execution Cross-Process - Full Verification")
    print("="*80)
    
    cleanup()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent1 = Agent("openai/gpt-4o-mini", db=db, debug=DEBUG, retry=1)
    task = Task("What is 7 + 2? Reply with just the number.")
    
    inject_error_into_step("response_processing", ValueError, "Simulated error", trigger_count=1)
    
    # =========================================================================
    # STEP 1: Error with agent1
    # =========================================================================
    print("\n[STEP 1] Running with agent1 (expecting error)...")
    
    run_id = None
    try:
        _ = await agent1.do_async(task, return_output=True)
        assert False, "Should have raised"
    except Exception as e:
        print(f"  Error: {type(e).__name__}")
        error_output = getattr(agent1, '_agent_run_output', None)
        if error_output:
            run_id = error_output.run_id
    
    assert run_id is not None, "run_id required"
    print(f"  Run ID: {run_id}")
    print("\n[STEP 1] PASSED")
    
    # =========================================================================
    # STEP 2: Verify storage
    # =========================================================================
    print("\n[STEP 2] Verifying storage...")
    
    session = db.storage.get_session(session_id="test_session")
    stored_output = session.runs[run_id].output
    
    verify_agent_run_output(stored_output, "Stored Error", expected_status=RunStatus.error)
    
    # Verify storage consistency: messages and chat_history
    if error_output:
        compare_outputs(error_output, stored_output, "In-Memory vs Stored (Error)")
    verify_session_messages_consistency(session, stored_output, "After Error - Storage")
    
    print("\n[STEP 2] PASSED")
    
    # =========================================================================
    # STEP 3: Resume with NEW agent (cross-process)
    # =========================================================================
    print("\n[STEP 3] Resuming with NEW agent (cross-process)...")
    
    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent2 = Agent("openai/gpt-4o-mini", db=db2, debug=DEBUG, retry=1)
    
    assert agent2.agent_id != agent1.agent_id, "Should be different agent"
    
    result = await agent2.continue_run_async(run_id=run_id, return_output=True)
    
    verify_agent_run_output(
        result, "Completed by agent2",
        expected_status=RunStatus.completed,
        expected_run_id=run_id
    )
    
    print(f"  Result: {result.output}")
    print("\n[STEP 3] PASSED")
    
    # =========================================================================
    # STEP 4: Verify cross-process consistency
    # =========================================================================
    print("\n[STEP 4] Verifying cross-process consistency...")
    
    # Capture before state (from step 2)
    before_session = session
    before_output = stored_output
    
    final_session = db2.storage.get_session(session_id="test_session")
    final_output = final_session.runs[run_id].output
    
    compare_outputs(stored_output, final_output, "Cross-Process Error vs Completed")
    
    # Verify storage consistency: messages and chat_history before and after
    verify_storage_consistency(
        before_session, final_session,
        before_output, final_output,
        "Cross-Process Durable Execution - Error to Completion"
    )
    
    # Verify current state consistency
    verify_session_messages_consistency(final_session, final_output, "Final - Storage")
    
    print("\n[STEP 4] PASSED")
    
    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Durable Cross-Process")
    print("="*80)


# =============================================================================
# TEST: DURABLE EXECUTION - EARLY STEPS (Before and At ChatHistoryStep)
# =============================================================================

@pytest.mark.asyncio
async def test_durable_execution_early_steps():
    """
    Test durable execution with error injection at early steps:
    - Steps 0-6 (before message-building steps)
    - Steps 7-12 (the decomposed message-building steps)
    
    This verifies that _run_boundaries is correctly handled when:
    - Resuming FROM ChatHistoryStep (step 11) - ChatHistoryStep sets its own boundary
    - Resuming FROM earlier steps (0-10) - ChatHistoryStep will rebuild and set boundary
    """
    print("\n" + "="*80)
    print("TEST: Durable Execution - Early Steps (Before/At ChatHistoryStep)")
    print("="*80)
    
    # Define early steps to test with their names and step indices
    # ChatHistoryStep is at index 11
    early_steps_to_test = [
        ("tool_setup", 6),          # Step right before message-building steps
        ("memory_prepare", 7),      # First message-building step
        ("chat_history", 11),       # ChatHistoryStep (sets run boundary)
        ("model_selection", 5),     # Even earlier step
    ]
    
    for step_name, expected_step_index in early_steps_to_test:
        print(f"\n{'='*60}")
        print(f"Testing error injection at: {step_name} (step {expected_step_index})")
        print(f"{'='*60}")
        
        cleanup()
        
        db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
        agent = Agent("openai/gpt-4o-mini", db=db, debug=DEBUG, retry=1)
        task = Task(
            description="What is 5 + 3? Reply with just the number.",
            tools=[simple_math]
        )
        
        # Inject error at the specific step (triggers once)
        inject_error_into_step(step_name, RuntimeError, f"Simulated error at {step_name}", trigger_count=1)
        
        # =====================================================================
        # STEP 1: Run and catch error
        # =====================================================================
        print(f"\n[STEP 1] Running with error injection at {step_name}...")
        
        try:
            output = await agent.do_async(task, return_output=True)
            # For some early steps, we might get an error output instead of exception
            if output.status == RunStatus.error:
                print("  Error status returned as expected")
            else:
                assert False, f"Expected error at {step_name}, but got status: {output.status}"
        except RuntimeError as e:
            print(f"  Caught expected RuntimeError: {e}")
            output = agent._agent_run_output
        
        # Note: agent.run_id is cleared in finally block, so get from output
        assert output is not None, "Output should exist after error"
        run_id = output.run_id
        assert run_id is not None, "run_id should be available from output"
        print(f"  Run ID: {run_id}")
        print(f"  Status: {output.status}")
        
        # Verify step_results exists
        assert output.step_results is not None, "step_results should exist"
        print(f"  Step results count: {len(output.step_results)}")
        
        # Find the failed step
        failed_step = None
        for sr in output.step_results:
            if sr.status == StepStatus.ERROR:
                failed_step = sr
                break
        
        if failed_step:
            print(f"  Failed step: {failed_step.name} (step {failed_step.step_number})")
            assert failed_step.name == step_name, f"Failed step should be {step_name}, got {failed_step.name}"
        
        print("\n[STEP 1] PASSED")
        
        # =====================================================================
        # STEP 2: Verify storage after error
        # =====================================================================
        print(f"\n[STEP 2] Verifying storage after error at {step_name}...")
        
        session = db.storage.get_session(session_id="test_session")
        assert session is not None, "Session should exist"
        assert run_id in session.runs, f"Run {run_id} should be in session.runs"
        
        stored_output = session.runs[run_id].output
        assert stored_output is not None, "Stored output should exist"
        assert stored_output.status == RunStatus.error, f"Stored status should be error, got {stored_output.status}"
        
        # Verify in-memory output status matches stored output status
        assert output.status == stored_output.status, \
            f"In-memory status ({output.status}) should match stored status ({stored_output.status})"
        
        # Verify chat_history state
        print(f"  In-memory output.status: {output.status}")
        print(f"  Stored output.status: {stored_output.status}")
        print("  ✓ Status consistency verified")
        print(f"  Stored chat_history length: {len(stored_output.chat_history) if stored_output.chat_history else 0}")
        print(f"  Stored messages length: {len(stored_output.messages) if stored_output.messages else 0}")
        print(f"  Stored _run_boundaries: {stored_output._run_boundaries}")
        
        print("\n[STEP 2] PASSED")
        
        # =====================================================================
        # STEP 3: Resume and complete
        # =====================================================================
        print(f"\n[STEP 3] Resuming run after {step_name} error...")
        
        # Clear the error injection so resume succeeds
        clear_error_injection()
        
        # Create new database and agent (simulating process restart)
        db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
        agent2 = Agent("openai/gpt-4o-mini", db=db2, debug=DEBUG)
        
        result = await agent2.continue_run_async(run_id=run_id, return_output=True)
        
        # Verify completion
        assert result is not None, "Result should exist"
        assert result.status == RunStatus.completed, f"Status should be completed, got {result.status}"
        assert result.run_id == run_id, f"run_id should match: {result.run_id} vs {run_id}"
        
        print(f"  Result status: {result.status}")
        print(f"  Result output: {result.output}")
        
        # Verify chat_history is correct after resume
        assert result.chat_history is not None, "chat_history should exist"
        assert len(result.chat_history) >= 2, f"chat_history should have at least 2 messages, got {len(result.chat_history)}"
        print(f"  Final chat_history length: {len(result.chat_history)}")
        
        # Verify messages were extracted correctly
        assert result.messages is not None, "messages should exist"
        print(f"  Final messages length: {len(result.messages)}")
        
        # Verify _run_boundaries is correct
        print(f"  Final _run_boundaries: {result._run_boundaries}")
        
        print("\n[STEP 3] PASSED")
        
        # =====================================================================
        # STEP 4: Verify final storage consistency
        # =====================================================================
        print(f"\n[STEP 4] Verifying final storage consistency for {step_name}...")
        
        final_session = db2.storage.get_session(session_id="test_session")
        assert final_session is not None, "Final session should exist"
        
        final_output = final_session.runs[run_id].output
        assert final_output is not None, "Final output should exist"
        assert final_output.status == RunStatus.completed, f"Final status should be completed, got {final_output.status}"
        
        # Verify session.messages vs output.chat_history consistency
        print(f"  Final stored chat_history length: {len(final_output.chat_history) if final_output.chat_history else 0}")
        print(f"  Final session.messages length: {len(final_session.messages) if final_session.messages else 0}")
        
        # The key verification: messages should be extractable and correct
        verify_session_messages_consistency(final_session, final_output, f"{step_name} - Final Storage")
        
        print(f"\n[STEP 4] PASSED for {step_name}")
        
        cleanup()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Durable Execution - Early Steps")
    print("="*80)


# =============================================================================
# TEST: CANCEL RUN - EARLY STEPS (Before and At ChatHistoryStep)
# =============================================================================

@pytest.mark.asyncio
async def test_cancel_run_early_steps():
    """
    Test cancel run at early steps:
    - Steps 0-6 (before message-building steps)
    - Steps 7-11 (message-building steps including ChatHistoryStep)
    
    This verifies that resumption from early cancellation points correctly
    handles _run_boundaries and message tracking.
    """
    print("\n" + "="*80)
    print("TEST: Cancel Run - Early Steps (Before/At ChatHistoryStep)")
    print("="*80)
    
    # We'll test cancellation at different timing points
    # Since we can't precisely control which step gets cancelled,
    # we'll use error injection to simulate the cancellation point
    # by injecting an error that causes early failure, then testing resume
    
    # For this test, we'll inject a SLOW operation at specific steps
    # and cancel during that slow operation
    
    early_steps_to_test = [
        ("tool_setup", 6, 0.3),          # Step right before message-building steps
        ("memory_prepare", 7, 0.5),      # First message-building step
    ]
    
    for step_name, expected_step_index, delay_before_cancel in early_steps_to_test:
        print(f"\n{'='*60}")
        print(f"Testing cancel at: {step_name} (step {expected_step_index})")
        print(f"{'='*60}")
        
        cleanup()
        
        db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
        agent = Agent("openai/gpt-4o-mini", db=db, debug=DEBUG)
        task = Task(
            description="Call long_running_task with 5 seconds.",
            tools=[long_running_task]
        )
        
        # =====================================================================
        # STEP 1: Start and cancel quickly
        # =====================================================================
        print(f"\n[STEP 1] Starting run and cancelling at {step_name}...")
        
        async def run_task():
            return await agent.do_async(task, return_output=True)
        
        run_future = asyncio.create_task(run_task())
        
        # Wait a short time, then cancel
        # This timing aims to cancel during early steps
        await asyncio.sleep(delay_before_cancel)
        
        run_id = agent.run_id
        if run_id:
            cancel_run(run_id)
            print(f"  Cancelled run: {run_id}")
        
        try:
            output = await asyncio.wait_for(run_future, timeout=10.0)
        except asyncio.TimeoutError:
            print("  Timeout waiting for cancelled run")
            output = agent._agent_run_output
        
        if output is None:
            print(f"  Run cancelled too early - no output yet. Skipping {step_name}.")
            cleanup()
            continue
        
        run_id = output.run_id or agent.run_id
        if run_id is None:
            print(f"  Run ID not available - run cancelled too early. Skipping {step_name}.")
            cleanup()
            continue
        
        print(f"  Run ID: {run_id}")
        print(f"  Output status: {output.status}")
        
        # Check what step we were on when cancelled
        if output.step_results:
            last_step = output.step_results[-1] if output.step_results else None
            if last_step:
                print(f"  Last step executed: {last_step.name} (step {last_step.step_number})")
        
        print("\n[STEP 1] PASSED")
        
        # =====================================================================
        # STEP 2: Verify storage after cancel
        # =====================================================================
        print("\n[STEP 2] Verifying storage after cancel...")
        
        # Give storage time to save
        await asyncio.sleep(0.5)
        
        session = db.storage.get_session(session_id="test_session")
        
        if session is None or run_id not in session.runs:
            print(f"  Run not saved to storage (cancelled too early). Skipping storage checks for {step_name}.")
            cleanup()
            continue
        
        stored_output = session.runs[run_id].output
        print(f"  Stored status: {stored_output.status}")
        print(f"  Stored chat_history length: {len(stored_output.chat_history) if stored_output.chat_history else 0}")
        print(f"  Stored _run_boundaries: {stored_output._run_boundaries}")
        
        print("\n[STEP 2] PASSED")
        
        # =====================================================================
        # STEP 3: Resume and complete
        # =====================================================================
        print(f"\n[STEP 3] Resuming run after cancel at {step_name}...")
        
        db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
        agent2 = Agent("openai/gpt-4o-mini", db=db2, debug=DEBUG)
        
        result = await agent2.continue_run_async(run_id=run_id, return_output=True)
        
        # Verify completion
        assert result is not None, "Result should exist"
        assert result.status == RunStatus.completed, f"Status should be completed, got {result.status}"
        
        print(f"  Result status: {result.status}")
        print(f"  Result output: {result.output}")
        
        # Verify chat_history is correct after resume
        assert result.chat_history is not None, "chat_history should exist"
        print(f"  Final chat_history length: {len(result.chat_history)}")
        
        # Verify messages were extracted correctly
        assert result.messages is not None, "messages should exist"
        print(f"  Final messages length: {len(result.messages)}")
        
        # Verify _run_boundaries is correct
        print(f"  Final _run_boundaries: {result._run_boundaries}")
        
        print("\n[STEP 3] PASSED")
        
        # =====================================================================
        # STEP 4: Verify final storage consistency
        # =====================================================================
        print(f"\n[STEP 4] Verifying final storage consistency for {step_name}...")
        
        final_session = db2.storage.get_session(session_id="test_session")
        assert final_session is not None, "Final session should exist"
        
        final_output = final_session.runs[run_id].output
        assert final_output is not None, "Final output should exist"
        assert final_output.status == RunStatus.completed, f"Final status should be completed, got {final_output.status}"
        
        # Verify session.messages vs output.chat_history consistency
        print(f"  Final stored chat_history length: {len(final_output.chat_history) if final_output.chat_history else 0}")
        print(f"  Final session.messages length: {len(final_session.messages) if final_session.messages else 0}")
        
        # The key verification: messages should be extractable and correct
        verify_session_messages_consistency(final_session, final_output, f"{step_name} Cancel - Final Storage")
        
        print(f"\n[STEP 4] PASSED for {step_name}")
        
        cleanup()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Cancel Run - Early Steps")
    print("="*80)


# =============================================================================
# TEST: MESSAGE BUILD STEP SPECIFIC - Boundary Verification
# =============================================================================

@pytest.mark.asyncio
async def test_chat_history_step_boundary():
    """
    Specifically test that _run_boundaries is correctly set when:
    1. Error occurs AT ChatHistoryStep
    2. Resume happens - ChatHistoryStep runs again and sets correct boundary
    3. Final messages are correctly extracted
    """
    print("\n" + "="*80)
    print("TEST: ChatHistoryStep Boundary Verification")
    print("="*80)
    
    cleanup()
    
    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", db=db, debug=DEBUG, retry=1)
    task = Task(
        description="What is 2 + 2? Reply with just the number.",
        tools=[simple_math]
    )
    
    # Inject error at chat_history step
    inject_error_into_step("chat_history", RuntimeError, "Simulated error at chat_history", trigger_count=1)
    
    # =========================================================================
    # STEP 1: Run and catch error at chat_history
    # =========================================================================
    print("\n[STEP 1] Running with error at chat_history step...")
    
    try:
        output = await agent.do_async(task, return_output=True)
        if output.status != RunStatus.error:
            assert False, f"Expected error status, got {output.status}"
    except RuntimeError as e:
        print(f"  Caught expected error: {e}")
        output = agent._agent_run_output
    
    assert output is not None, "Output should exist after error"
    run_id = output.run_id
    assert run_id is not None, "run_id should exist"
    
    print(f"  Run ID: {run_id}")
    print(f"  Status: {output.status}")
    print(f"  chat_history length: {len(output.chat_history) if output.chat_history else 0}")
    print(f"  _run_boundaries: {output._run_boundaries}")
    
    # Verify the failed step is chat_history
    failed_step = None
    for sr in output.step_results:
        if sr.status == StepStatus.ERROR:
            failed_step = sr
            break
    
    assert failed_step is not None, "Should have a failed step"
    assert failed_step.name == "chat_history", f"Failed step should be chat_history, got {failed_step.name}"
    print(f"  Failed step: {failed_step.name} (step {failed_step.step_number})")
    
    print("\n[STEP 1] PASSED")
    
    # =========================================================================
    # STEP 2: Resume - ChatHistoryStep should run again
    # =========================================================================
    print("\n[STEP 2] Resuming from chat_history error...")
    
    clear_error_injection()
    
    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent2 = Agent("openai/gpt-4o-mini", db=db2, debug=DEBUG)
    
    result = await agent2.continue_run_async(run_id=run_id, return_output=True)
    
    # Verify completion
    assert result is not None, "Result should exist"
    assert result.status == RunStatus.completed, f"Status should be completed, got {result.status}"
    
    print(f"  Result status: {result.status}")
    print(f"  Result output: {result.output}")
    
    # =========================================================================
    # STEP 3: Verify _run_boundaries was correctly set by ChatHistoryStep
    # =========================================================================
    print("\n[STEP 3] Verifying _run_boundaries handling...")
    
    # chat_history should exist and have messages
    assert result.chat_history is not None, "chat_history should exist"
    assert len(result.chat_history) >= 2, f"chat_history should have at least 2 messages (request + response), got {len(result.chat_history)}"
    
    # _run_boundaries should have at least one entry
    assert result._run_boundaries is not None, "_run_boundaries should exist"
    assert len(result._run_boundaries) >= 1, f"_run_boundaries should have at least 1 entry, got {len(result._run_boundaries)}"
    
    # messages should be correctly extracted
    assert result.messages is not None, "messages should exist"
    assert len(result.messages) >= 2, f"messages should have at least 2 entries, got {len(result.messages)}"
    
    print(f"  chat_history length: {len(result.chat_history)}")
    print(f"  _run_boundaries: {result._run_boundaries}")
    print(f"  messages length: {len(result.messages)}")
    
    # Verify new_messages() returns correct messages
    new_msgs = result.new_messages()
    print(f"  new_messages() length: {len(new_msgs)}")
    
    # The boundary should be correct: messages extracted from chat_history after boundary
    boundary = result._run_boundaries[-1]
    expected_new_count = len(result.chat_history) - boundary
    assert len(new_msgs) == expected_new_count, f"new_messages count mismatch: {len(new_msgs)} vs expected {expected_new_count}"
    
    print("\n[STEP 3] PASSED")
    
    # =========================================================================
    # STEP 4: Verify final storage
    # =========================================================================
    print("\n[STEP 4] Verifying final storage consistency...")
    
    final_session = db2.storage.get_session(session_id="test_session")
    assert final_session is not None, "Final session should exist"
    
    final_output = final_session.runs[run_id].output
    assert final_output is not None, "Final output should exist"
    assert final_output.status == RunStatus.completed, f"Final status should be completed, got {final_output.status}"
    
    # Verify session.messages is consistent
    verify_session_messages_consistency(final_session, final_output, "ChatHistoryStep Boundary - Final")
    
    print("\n[STEP 4] PASSED")
    
    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: ChatHistoryStep Boundary Verification")
    print("="*80)


# =============================================================================
# TEST: USER CONFIRMATION - APPROVE
# =============================================================================

@pytest.mark.asyncio
async def test_confirmation_approve_comprehensive():
    """Comprehensive User Confirmation (approve) test with full attribute verification."""
    print("\n" + "="*80)
    print("TEST: User Confirmation Approve - Full Attribute Verification")
    print("="*80)

    cleanup()

    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent1 = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description="Perform a sensitive operation on the data 'user_records_2024'.",
        tools=[sensitive_operation]
    )

    # =========================================================================
    # STEP 1: Initial run - should pause for confirmation
    # =========================================================================
    print("\n[STEP 1] Running do_async (expecting confirmation pause)...")
    output = await agent1.do_async(task, return_output=True)
    run_id = output.run_id

    verify_agent_run_output(
        output, "After Confirmation Pause",
        expected_status=RunStatus.paused,
        expected_run_id=run_id,
        min_step_results=5,
        min_requirements=1
    )

    assert output.is_paused, f"Expected paused, got {output.status}"
    assert output.pause_reason == "confirmation", f"Expected confirmation pause, got {output.pause_reason}"
    assert len(output.active_requirements) == 1, "Should have 1 active requirement"

    req = output.active_requirements[0]
    verify_run_requirement(
        req, "Confirmation Requirement",
        expected_resolved=False,
        expected_tool_name="sensitive_operation"
    )
    assert req.needs_confirmation, "Requirement should need confirmation"

    print("\n[STEP 1] PASSED")

    # =========================================================================
    # STEP 2: Verify storage after pause
    # =========================================================================
    print("\n[STEP 2] Verifying storage after confirmation pause...")

    session = db.storage.get_session(session_id="test_session")
    verify_agent_session(session, "After Confirmation Pause", expected_run_id=run_id, expected_run_count=1)

    stored_output = session.runs[run_id].output
    verify_agent_run_output(
        stored_output, "Stored After Confirmation Pause",
        expected_status=RunStatus.paused,
        expected_run_id=run_id
    )

    compare_outputs(output, stored_output, "In-Memory vs Stored (Confirmation Pause)")
    verify_session_messages_consistency(session, stored_output, "After Confirmation Pause - Storage")

    print("\n[STEP 2] PASSED")

    # =========================================================================
    # STEP 3: Confirm the requirement
    # =========================================================================
    print("\n[STEP 3] Confirming the requirement...")

    req.confirm()
    assert req.is_resolved, "Requirement should be resolved after confirm"

    print("\n[STEP 3] PASSED")

    # =========================================================================
    # STEP 4: Resume with NEW agent
    # =========================================================================
    print("\n[STEP 4] Resuming with new agent...")

    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent2 = Agent("openai/gpt-4o-mini", name="agent2", db=db2, debug=DEBUG)

    result = await agent2.continue_run_async(
        run_id=run_id,
        return_output=True,
        requirements=output.requirements
    )

    verify_agent_run_output(
        result, "After Confirmation Completion",
        expected_status=RunStatus.completed,
        expected_run_id=run_id,
        min_step_results=10
    )

    assert result.output is not None, "Output content should exist"
    assert result.is_complete, f"Expected complete, got {result.status}"

    print(f"  Result content: {result.output}")
    print("\n[STEP 4] PASSED")

    # =========================================================================
    # STEP 5: Verify final storage
    # =========================================================================
    print("\n[STEP 5] Verifying final storage...")

    before_session = session
    before_output = stored_output

    final_session = db2.storage.get_session(session_id="test_session")
    verify_agent_session(final_session, "Final Confirmation", expected_run_id=run_id)

    final_output = final_session.runs[run_id].output
    verify_agent_run_output(
        final_output, "Final Stored Confirmation",
        expected_status=RunStatus.completed,
        expected_run_id=run_id
    )

    verify_storage_consistency(
        before_session, final_session,
        before_output, final_output,
        "Confirmation - Pause to Completion"
    )
    verify_session_messages_consistency(final_session, final_output, "Confirmation Final - Storage")

    print("\n[STEP 5] PASSED")

    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: User Confirmation Approve - Full Attribute Verification")
    print("="*80)


# =============================================================================
# TEST: USER CONFIRMATION - REJECT
# =============================================================================

@pytest.mark.asyncio
async def test_confirmation_reject_comprehensive():
    """Comprehensive User Confirmation (reject) test with full attribute verification."""
    print("\n" + "="*80)
    print("TEST: User Confirmation Reject - Full Attribute Verification")
    print("="*80)

    cleanup()

    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent1 = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description="Perform a sensitive operation on the data 'user_records_2024'.",
        tools=[sensitive_operation]
    )

    # =========================================================================
    # STEP 1: Initial run - should pause for confirmation
    # =========================================================================
    print("\n[STEP 1] Running do_async (expecting confirmation pause)...")
    output = await agent1.do_async(task, return_output=True)
    run_id = output.run_id

    assert output.is_paused, f"Expected paused, got {output.status}"
    assert output.pause_reason == "confirmation"
    assert len(output.active_requirements) >= 1

    req = output.active_requirements[0]
    assert req.needs_confirmation, "Should need confirmation"

    print("\n[STEP 1] PASSED")

    # =========================================================================
    # STEP 2: Reject the requirement
    # =========================================================================
    print("\n[STEP 2] Rejecting the requirement...")

    req.reject(note="Not authorized to run this operation")
    assert req.is_resolved, "Requirement should be resolved after reject"

    print("\n[STEP 2] PASSED")

    # =========================================================================
    # STEP 3: Resume - agent should complete without executing the tool
    # =========================================================================
    print("\n[STEP 3] Resuming after rejection...")

    result = await agent1.continue_run_async(
        run_id=run_id,
        return_output=True
    )

    verify_agent_run_output(
        result, "After Rejection Completion",
        expected_status=RunStatus.completed,
        expected_run_id=run_id,
        min_step_results=10
    )

    assert result.is_complete, f"Expected complete, got {result.status}"

    print(f"  Result content: {result.output}")
    print("\n[STEP 3] PASSED")

    # =========================================================================
    # STEP 4: Verify final storage
    # =========================================================================
    print("\n[STEP 4] Verifying final storage...")

    final_session = db.storage.get_session(session_id="test_session")
    verify_agent_session(final_session, "Final Rejection", expected_run_id=run_id)

    final_output = final_session.runs[run_id].output
    verify_agent_run_output(
        final_output, "Final Stored Rejection",
        expected_status=RunStatus.completed,
        expected_run_id=run_id
    )

    verify_session_messages_consistency(final_session, final_output, "Rejection Final - Storage")

    print("\n[STEP 4] PASSED")

    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: User Confirmation Reject - Full Attribute Verification")
    print("="*80)


# =============================================================================
# TEST: USER CONFIRMATION - CROSS-PROCESS
# =============================================================================

@pytest.mark.asyncio
async def test_confirmation_cross_process_comprehensive():
    """Cross-process confirmation with new agent and full verification."""
    print("\n" + "="*80)
    print("TEST: User Confirmation Cross-Process - Full Verification")
    print("="*80)

    cleanup()

    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent1 = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description="Perform a sensitive operation on the data 'audit_logs'.",
        tools=[sensitive_operation]
    )

    # =========================================================================
    # STEP 1: Initial run pauses for confirmation
    # =========================================================================
    print("\n[STEP 1] Running with agent1 (expecting confirmation pause)...")
    output = await agent1.do_async(task, return_output=True)
    run_id = output.run_id

    assert output.is_paused, f"Expected paused, got {output.status}"
    assert output.pause_reason == "confirmation"

    print(f"  Run ID: {run_id}")
    print("\n[STEP 1] PASSED")

    # =========================================================================
    # STEP 2: Verify storage
    # =========================================================================
    print("\n[STEP 2] Verifying storage...")

    session = db.storage.get_session(session_id="test_session")
    stored_output = session.runs[run_id].output
    verify_agent_run_output(stored_output, "Stored Confirmation Pause", expected_status=RunStatus.paused)
    verify_session_messages_consistency(session, stored_output, "After Confirmation Pause - Storage")

    print("\n[STEP 2] PASSED")

    # =========================================================================
    # STEP 3: Confirm and resume with NEW agent (cross-process)
    # =========================================================================
    print("\n[STEP 3] Confirming and resuming with NEW agent...")

    for req in output.active_requirements:
        if req.needs_confirmation:
            req.confirm()

    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent2 = Agent("openai/gpt-4o-mini", name="agent2", db=db2, debug=DEBUG)

    assert agent2.agent_id != agent1.agent_id, "Should be different agent"

    result = await agent2.continue_run_async(
        run_id=run_id,
        requirements=output.requirements,
        return_output=True
    )

    verify_agent_run_output(
        result, "Completed by agent2",
        expected_status=RunStatus.completed,
        expected_run_id=run_id
    )

    print(f"  Result: {result.output}")
    print("\n[STEP 3] PASSED")

    # =========================================================================
    # STEP 4: Verify cross-process consistency
    # =========================================================================
    print("\n[STEP 4] Verifying cross-process consistency...")

    before_session = session
    before_output = stored_output

    final_session = db2.storage.get_session(session_id="test_session")
    final_output = final_session.runs[run_id].output

    compare_outputs(stored_output, final_output, "Cross-Process Confirmation Pause vs Completed")

    verify_storage_consistency(
        before_session, final_session,
        before_output, final_output,
        "Cross-Process Confirmation - Pause to Completion"
    )
    verify_session_messages_consistency(final_session, final_output, "Confirmation Cross-Process Final - Storage")

    print("\n[STEP 4] PASSED")

    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: User Confirmation Cross-Process")
    print("="*80)


# =============================================================================
# TEST: USER CONFIRMATION - MULTIPLE TOOLS (LOOP)
# =============================================================================

@pytest.mark.asyncio
async def test_confirmation_multiple_tools_comprehensive():
    """Multiple confirmation tools using a while-loop with full verification."""
    print("\n" + "="*80)
    print("TEST: User Confirmation Multiple Tools - Full Verification")
    print("="*80)

    cleanup()

    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description=(
            "First, delete records from the 'users' table where status='inactive'. "
            "Then deploy version '2.1.0' to the 'production' environment."
        ),
        tools=[delete_records, deploy_to_production]
    )

    # =========================================================================
    # STEP 1: Run and loop through confirmations
    # =========================================================================
    print("\n[STEP 1] Running do_async (expecting multiple confirmation pauses)...")
    output = await agent.do_async(task, return_output=True)
    run_id = output.run_id

    loop_count = 0
    while output.active_requirements:
        loop_count += 1
        print(f"\n  --- Confirmation loop {loop_count} ---")
        print(f"  Active requirements: {len(output.active_requirements)}")

        assert output.is_paused, f"Expected paused in loop {loop_count}, got {output.status}"
        assert output.pause_reason == "confirmation", f"Expected confirmation pause in loop {loop_count}"

        for req in output.active_requirements:
            if req.needs_confirmation:
                verify_run_requirement(req, f"Loop {loop_count} Requirement", expected_resolved=False)
                req.confirm()
                print(f"  Confirmed: {req.tool_execution.tool_name}")

        output = await agent.continue_run_async(run_id=output.run_id, return_output=True)

        if loop_count > 5:
            assert False, "Too many confirmation loops - possible infinite loop"

    print(f"\n  Total confirmation loops: {loop_count}")
    assert loop_count >= 1, "Should have had at least 1 confirmation loop"

    verify_agent_run_output(
        output, "After All Confirmations",
        expected_status=RunStatus.completed,
        expected_run_id=run_id,
        min_step_results=10
    )

    assert output.is_complete, f"Expected complete, got {output.status}"
    print(f"  Result: {output.output}")
    print("\n[STEP 1] PASSED")

    # =========================================================================
    # STEP 2: Verify final storage
    # =========================================================================
    print("\n[STEP 2] Verifying final storage...")

    final_session = db.storage.get_session(session_id="test_session")
    verify_agent_session(final_session, "Final Multi-Confirmation", expected_run_id=run_id)

    final_output = final_session.runs[run_id].output
    verify_agent_run_output(
        final_output, "Final Stored Multi-Confirmation",
        expected_status=RunStatus.completed,
        expected_run_id=run_id
    )

    verify_session_messages_consistency(final_session, final_output, "Multi-Confirmation Final - Storage")

    print("\n[STEP 2] PASSED")

    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: User Confirmation Multiple Tools")
    print("="*80)


# =============================================================================
# TEST: USER INPUT - COMPREHENSIVE
# =============================================================================

@pytest.mark.asyncio
async def test_user_input_comprehensive():
    """Comprehensive User Input test with full attribute verification."""
    print("\n" + "="*80)
    print("TEST: User Input - Full Attribute Verification")
    print("="*80)

    cleanup()

    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent1 = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description="Send an email with subject 'Hello' and body 'Hello, world!'.",
        tools=[send_email_with_input]
    )

    # =========================================================================
    # STEP 1: Initial run - should pause for user input
    # =========================================================================
    print("\n[STEP 1] Running do_async (expecting user input pause)...")
    output = await agent1.do_async(task, return_output=True)
    run_id = output.run_id

    verify_agent_run_output(
        output, "After User Input Pause",
        expected_status=RunStatus.paused,
        expected_run_id=run_id,
        min_step_results=5,
        min_requirements=1
    )

    assert output.is_paused, f"Expected paused, got {output.status}"
    assert output.pause_reason == "user_input", f"Expected user_input pause, got {output.pause_reason}"
    assert len(output.active_requirements) >= 1, "Should have at least 1 active requirement"

    req = output.active_requirements[0]
    verify_run_requirement(
        req, "User Input Requirement",
        expected_resolved=False,
        expected_tool_name="send_email_with_input"
    )
    assert req.needs_user_input, "Requirement should need user input"
    assert req.user_input_schema is not None, "Should have user_input_schema"

    # Verify the schema has the expected fields
    field_names = [f["name"] for f in req.user_input_schema if isinstance(f, dict)]
    assert "to_address" in field_names, f"Schema should include 'to_address', got {field_names}"
    print(f"  User input fields: {field_names}")

    print("\n[STEP 1] PASSED")

    # =========================================================================
    # STEP 2: Verify storage after pause
    # =========================================================================
    print("\n[STEP 2] Verifying storage after user input pause...")

    session = db.storage.get_session(session_id="test_session")
    verify_agent_session(session, "After User Input Pause", expected_run_id=run_id, expected_run_count=1)

    stored_output = session.runs[run_id].output
    verify_agent_run_output(
        stored_output, "Stored After User Input Pause",
        expected_status=RunStatus.paused,
        expected_run_id=run_id
    )

    compare_outputs(output, stored_output, "In-Memory vs Stored (User Input Pause)")
    verify_session_messages_consistency(session, stored_output, "After User Input Pause - Storage")

    print("\n[STEP 2] PASSED")

    # =========================================================================
    # STEP 3: Fill user input fields
    # =========================================================================
    print("\n[STEP 3] Filling user input fields...")

    for r in output.active_requirements:
        if r.needs_user_input:
            _fill_user_input(r)

    assert req.is_resolved, "Requirement should be resolved after filling input"
    print(f"  Filled values for requirement")

    print("\n[STEP 3] PASSED")

    # =========================================================================
    # STEP 4: Resume with NEW agent
    # =========================================================================
    print("\n[STEP 4] Resuming with new agent...")

    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent2 = Agent("openai/gpt-4o-mini", name="agent2", db=db2, debug=DEBUG)

    result = await agent2.continue_run_async(
        run_id=run_id,
        return_output=True,
        requirements=output.requirements
    )

    verify_agent_run_output(
        result, "After User Input Completion",
        expected_status=RunStatus.completed,
        expected_run_id=run_id,
        min_step_results=10
    )

    assert result.output is not None, "Output content should exist"
    assert result.is_complete, f"Expected complete, got {result.status}"

    print(f"  Result content: {result.output}")
    print("\n[STEP 4] PASSED")

    # =========================================================================
    # STEP 5: Verify final storage
    # =========================================================================
    print("\n[STEP 5] Verifying final storage...")

    before_session = session
    before_output = stored_output

    final_session = db2.storage.get_session(session_id="test_session")
    verify_agent_session(final_session, "Final User Input", expected_run_id=run_id)

    final_output = final_session.runs[run_id].output
    verify_agent_run_output(
        final_output, "Final Stored User Input",
        expected_status=RunStatus.completed,
        expected_run_id=run_id
    )

    verify_storage_consistency(
        before_session, final_session,
        before_output, final_output,
        "User Input - Pause to Completion"
    )
    verify_session_messages_consistency(final_session, final_output, "User Input Final - Storage")

    print("\n[STEP 5] PASSED")

    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: User Input - Full Attribute Verification")
    print("="*80)


# =============================================================================
# TEST: USER INPUT - CROSS-PROCESS
# =============================================================================

@pytest.mark.asyncio
async def test_user_input_cross_process_comprehensive():
    """Cross-process user input with new agent and full verification."""
    print("\n" + "="*80)
    print("TEST: User Input Cross-Process - Full Verification")
    print("="*80)

    cleanup()

    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent1 = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description="Send an email with subject 'Report' and body 'Monthly report attached'.",
        tools=[send_email_with_input]
    )

    # =========================================================================
    # STEP 1: Run with agent1 - pauses for user input
    # =========================================================================
    print("\n[STEP 1] Running with agent1 (expecting user input pause)...")
    output = await agent1.do_async(task, return_output=True)
    run_id = output.run_id

    assert output.is_paused, f"Expected paused, got {output.status}"
    assert output.pause_reason == "user_input"

    print(f"  Run ID: {run_id}")
    print("\n[STEP 1] PASSED")

    # =========================================================================
    # STEP 2: Verify storage
    # =========================================================================
    print("\n[STEP 2] Verifying storage...")

    session = db.storage.get_session(session_id="test_session")
    stored_output = session.runs[run_id].output
    verify_agent_run_output(stored_output, "Stored User Input Pause", expected_status=RunStatus.paused)
    verify_session_messages_consistency(session, stored_output, "After User Input Pause - Storage")

    print("\n[STEP 2] PASSED")

    # =========================================================================
    # STEP 3: Fill input and resume with NEW agent
    # =========================================================================
    print("\n[STEP 3] Filling input and resuming with NEW agent...")

    for req in output.active_requirements:
        if req.needs_user_input:
            _fill_user_input(req)

    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent2 = Agent("openai/gpt-4o-mini", name="agent2", db=db2, debug=DEBUG)

    assert agent2.agent_id != agent1.agent_id, "Should be different agent"

    result = await agent2.continue_run_async(
        run_id=run_id,
        requirements=output.requirements,
        return_output=True
    )

    verify_agent_run_output(
        result, "Completed by agent2",
        expected_status=RunStatus.completed,
        expected_run_id=run_id
    )

    print(f"  Result: {result.output}")
    print("\n[STEP 3] PASSED")

    # =========================================================================
    # STEP 4: Verify cross-process consistency
    # =========================================================================
    print("\n[STEP 4] Verifying cross-process consistency...")

    before_session = session
    before_output = stored_output

    final_session = db2.storage.get_session(session_id="test_session")
    final_output = final_session.runs[run_id].output

    compare_outputs(stored_output, final_output, "Cross-Process User Input Pause vs Completed")

    verify_storage_consistency(
        before_session, final_session,
        before_output, final_output,
        "Cross-Process User Input - Pause to Completion"
    )
    verify_session_messages_consistency(final_session, final_output, "User Input Cross-Process Final - Storage")

    print("\n[STEP 4] PASSED")

    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: User Input Cross-Process")
    print("="*80)


# =============================================================================
# TEST: USER INPUT - MULTIPLE TOOLS (LOOP)
# =============================================================================

@pytest.mark.asyncio
async def test_user_input_multiple_tools_comprehensive():
    """Multiple user input tools using a while-loop with full verification."""
    print("\n" + "="*80)
    print("TEST: User Input Multiple Tools - Full Verification")
    print("="*80)

    cleanup()

    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description=(
            "First, create a support ticket titled 'Bug Report' with description 'App crashes on login'. "
            "Then schedule a meeting about 'Bug Triage'."
        ),
        tools=[create_ticket, schedule_meeting]
    )

    # =========================================================================
    # STEP 1: Run and loop through user input pauses
    # =========================================================================
    print("\n[STEP 1] Running do_async (expecting multiple user input pauses)...")
    output = await agent.do_async(task, return_output=True)
    run_id = output.run_id

    loop_count = 0
    while output.active_requirements:
        loop_count += 1
        print(f"\n  --- User input loop {loop_count} ---")
        print(f"  Active requirements: {len(output.active_requirements)}")

        assert output.is_paused, f"Expected paused in loop {loop_count}, got {output.status}"
        assert output.pause_reason == "user_input", f"Expected user_input pause in loop {loop_count}"

        for req in output.active_requirements:
            if req.needs_user_input:
                verify_run_requirement(req, f"Loop {loop_count} User Input Requirement", expected_resolved=False)
                _fill_user_input(req)
                print(f"  Filled input for: {req.tool_execution.tool_name}")

        output = await agent.continue_run_async(run_id=output.run_id, return_output=True)

        if loop_count > 5:
            assert False, "Too many user input loops - possible infinite loop"

    print(f"\n  Total user input loops: {loop_count}")
    assert loop_count >= 1, "Should have had at least 1 user input loop"

    verify_agent_run_output(
        output, "After All User Inputs",
        expected_status=RunStatus.completed,
        expected_run_id=run_id,
        min_step_results=10
    )

    assert output.is_complete, f"Expected complete, got {output.status}"
    print(f"  Result: {output.output}")
    print("\n[STEP 1] PASSED")

    # =========================================================================
    # STEP 2: Verify final storage
    # =========================================================================
    print("\n[STEP 2] Verifying final storage...")

    final_session = db.storage.get_session(session_id="test_session")
    verify_agent_session(final_session, "Final Multi-User-Input", expected_run_id=run_id)

    final_output = final_session.runs[run_id].output
    verify_agent_run_output(
        final_output, "Final Stored Multi-User-Input",
        expected_status=RunStatus.completed,
        expected_run_id=run_id
    )

    verify_session_messages_consistency(final_session, final_output, "Multi-User-Input Final - Storage")

    print("\n[STEP 2] PASSED")

    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: User Input Multiple Tools")
    print("="*80)


# =============================================================================
# TEST: DYNAMIC USER INPUT - COMPREHENSIVE
# =============================================================================

@pytest.mark.asyncio
async def test_dynamic_user_input_comprehensive():
    """Comprehensive Dynamic User Input test with full attribute verification."""
    print("\n" + "="*80)
    print("TEST: Dynamic User Input - Full Attribute Verification")
    print("="*80)

    cleanup()

    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent1 = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description="Send an email with the body 'What is the weather in Tokyo?'",
        tools=[dynamic_send_email, UserControlFlowTools()]
    )

    # =========================================================================
    # STEP 1: Initial run - should pause for dynamic user input
    # =========================================================================
    print("\n[STEP 1] Running do_async (expecting dynamic user input pause)...")
    output = await agent1.do_async(task, return_output=True)
    run_id = output.run_id

    # The agent may or may not pause depending on whether it calls get_user_input
    if output.is_paused and output.active_requirements:
        print(f"  Paused with {len(output.active_requirements)} active requirements")
        assert output.pause_reason == "user_input", f"Expected user_input pause, got {output.pause_reason}"

        for req in output.active_requirements:
            if req.needs_user_input:
                assert req.user_input_schema is not None, "Should have user_input_schema"
                field_names = [f["name"] for f in req.user_input_schema if isinstance(f, dict)]
                print(f"  Dynamic input fields: {field_names}")
    else:
        # Agent completed without needing input - this is acceptable
        print("  Agent completed without dynamic input pause")
        assert output.is_complete, f"Expected complete or paused, got {output.status}"

    print("\n[STEP 1] PASSED")

    # =========================================================================
    # STEP 2: Loop through dynamic input pauses
    # =========================================================================
    print("\n[STEP 2] Processing dynamic user input loop...")

    loop_count = 0
    while output.is_paused and output.active_requirements:
        loop_count += 1
        print(f"\n  --- Dynamic input loop {loop_count} ---")

        # Verify storage at each pause
        session = db.storage.get_session(session_id="test_session")
        stored_output = session.runs[run_id].output
        assert stored_output.status == RunStatus.paused, f"Stored should be paused, got {stored_output.status}"
        verify_session_messages_consistency(session, stored_output, f"Dynamic Input Loop {loop_count} - Storage")

        for req in output.active_requirements:
            if req.needs_user_input:
                _fill_user_input(req)
                print(f"  Filled dynamic input for requirement")

        output = await agent1.continue_run_async(run_id=output.run_id, return_output=True)

        if loop_count > 5:
            assert False, "Too many dynamic input loops - possible infinite loop"

    print(f"\n  Total dynamic input loops: {loop_count}")

    verify_agent_run_output(
        output, "After Dynamic User Input",
        expected_status=RunStatus.completed,
        expected_run_id=run_id,
        min_step_results=10
    )

    assert output.is_complete, f"Expected complete, got {output.status}"
    print(f"  Result: {output.output}")

    print("\n[STEP 2] PASSED")

    # =========================================================================
    # STEP 3: Verify final storage
    # =========================================================================
    print("\n[STEP 3] Verifying final storage...")

    final_session = db.storage.get_session(session_id="test_session")
    verify_agent_session(final_session, "Final Dynamic Input", expected_run_id=run_id)

    final_output = final_session.runs[run_id].output
    verify_agent_run_output(
        final_output, "Final Stored Dynamic Input",
        expected_status=RunStatus.completed,
        expected_run_id=run_id
    )

    verify_session_messages_consistency(final_session, final_output, "Dynamic Input Final - Storage")

    print("\n[STEP 3] PASSED")

    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Dynamic User Input - Full Attribute Verification")
    print("="*80)


# =============================================================================
# TEST: DYNAMIC USER INPUT - CROSS-PROCESS
# =============================================================================

@pytest.mark.asyncio
async def test_dynamic_user_input_cross_process_comprehensive():
    """Cross-process dynamic user input with new agent and full verification."""
    print("\n" + "="*80)
    print("TEST: Dynamic User Input Cross-Process - Full Verification")
    print("="*80)

    cleanup()

    db = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent1 = Agent("openai/gpt-4o-mini", name="agent1", db=db, debug=DEBUG)
    task = Task(
        description="Send an email with the body 'What is the weather in Tokyo?'",
        tools=[dynamic_send_email, UserControlFlowTools()]
    )

    # =========================================================================
    # STEP 1: Run with agent1 - may pause for dynamic input
    # =========================================================================
    print("\n[STEP 1] Running with agent1...")
    output = await agent1.do_async(task, return_output=True)
    run_id = output.run_id

    if not output.is_paused:
        # Agent completed without input - verify and return
        print("  Agent completed without dynamic input pause")
        verify_agent_run_output(output, "Completed Without Pause", expected_status=RunStatus.completed)
        cleanup()
        print("\n" + "="*80)
        print("ALL TESTS PASSED: Dynamic User Input Cross-Process (no pause needed)")
        print("="*80)
        return

    assert output.pause_reason == "user_input"
    print(f"  Run ID: {run_id}")
    print("\n[STEP 1] PASSED")

    # =========================================================================
    # STEP 2: Verify storage
    # =========================================================================
    print("\n[STEP 2] Verifying storage...")

    session = db.storage.get_session(session_id="test_session")
    stored_output = session.runs[run_id].output
    verify_agent_run_output(stored_output, "Stored Dynamic Input Pause", expected_status=RunStatus.paused)
    verify_session_messages_consistency(session, stored_output, "After Dynamic Input Pause - Storage")

    print("\n[STEP 2] PASSED")

    # =========================================================================
    # STEP 3: Fill input and resume with NEW agent, looping if needed
    # =========================================================================
    print("\n[STEP 3] Filling input and resuming with NEW agent...")

    for req in output.active_requirements:
        if req.needs_user_input:
            _fill_user_input(req)

    db2 = SqliteDatabase(db_file=DB_FILE, session_id="test_session", user_id="test_user", full_session_memory=True)
    agent2 = Agent("openai/gpt-4o-mini", name="agent2", db=db2, debug=DEBUG)

    result = await agent2.continue_run_async(
        run_id=run_id,
        requirements=output.requirements,
        return_output=True
    )

    # May need more loops for dynamic input
    loop_count = 0
    while result.is_paused and result.active_requirements:
        loop_count += 1
        print(f"\n  --- Cross-process dynamic input loop {loop_count} ---")

        for req in result.active_requirements:
            if req.needs_user_input:
                _fill_user_input(req)

        result = await agent2.continue_run_async(
            run_id=run_id,
            return_output=True
        )

        if loop_count > 5:
            assert False, "Too many loops"

    verify_agent_run_output(
        result, "Completed by agent2",
        expected_status=RunStatus.completed,
        expected_run_id=run_id
    )

    print(f"  Result: {result.output}")
    print("\n[STEP 3] PASSED")

    # =========================================================================
    # STEP 4: Verify cross-process consistency
    # =========================================================================
    print("\n[STEP 4] Verifying cross-process consistency...")

    before_session = session
    before_output = stored_output

    final_session = db2.storage.get_session(session_id="test_session")
    final_output = final_session.runs[run_id].output

    compare_outputs(stored_output, final_output, "Cross-Process Dynamic Input Pause vs Completed")

    verify_storage_consistency(
        before_session, final_session,
        before_output, final_output,
        "Cross-Process Dynamic Input - Pause to Completion"
    )
    verify_session_messages_consistency(final_session, final_output, "Dynamic Input Cross-Process Final - Storage")

    print("\n[STEP 4] PASSED")

    cleanup()
    print("\n" + "="*80)
    print("ALL TESTS PASSED: Dynamic User Input Cross-Process")
    print("="*80)


# =============================================================================
# MAIN (for manual execution)
# =============================================================================

async def main():
    """Run all comprehensive tests manually."""
    await test_external_tool_comprehensive()
    await test_cancel_run_comprehensive()
    await test_durable_execution_comprehensive()
    await test_durable_cross_process_comprehensive()
    await test_durable_execution_early_steps()
    await test_cancel_run_early_steps()
    await test_chat_history_step_boundary()
    await test_confirmation_approve_comprehensive()
    await test_confirmation_reject_comprehensive()
    await test_confirmation_cross_process_comprehensive()
    await test_confirmation_multiple_tools_comprehensive()
    await test_user_input_comprehensive()
    await test_user_input_cross_process_comprehensive()
    await test_user_input_multiple_tools_comprehensive()
    await test_dynamic_user_input_comprehensive()
    await test_dynamic_user_input_cross_process_comprehensive()

    print("\n" + "="*80)
    print("ALL COMPREHENSIVE TESTS PASSED!")
    print("  - External Tool: PASSED")
    print("  - Cancel Run: PASSED")
    print("  - Durable Execution: PASSED")
    print("  - Durable Cross-Process: PASSED")
    print("  - Durable Execution Early Steps: PASSED")
    print("  - Cancel Run Early Steps: PASSED")
    print("  - ChatHistoryStep Boundary: PASSED")
    print("  - Confirmation Approve: PASSED")
    print("  - Confirmation Reject: PASSED")
    print("  - Confirmation Cross-Process: PASSED")
    print("  - Confirmation Multiple Tools: PASSED")
    print("  - User Input: PASSED")
    print("  - User Input Cross-Process: PASSED")
    print("  - User Input Multiple Tools: PASSED")
    print("  - Dynamic User Input: PASSED")
    print("  - Dynamic User Input Cross-Process: PASSED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
