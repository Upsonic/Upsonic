"""
Smoke test for Agent safety policies.

Tests:
- user_policy: Validates user input
- agent_policy: Validates agent output
- tool_policy_pre: Validates tools at registration (pre-execution)
- tool_policy_post: Validates tool calls before execution (post-execution)

Success criteria: We must see related logs in the terminal for each policy type.
"""

import pytest
from upsonic import Agent, Task
from upsonic.tools import tool
from upsonic.safety_engine.policies.adult_content_policies import AdultContentBlockPolicy
from upsonic.safety_engine.policies.pii_policies import PIIBlockPolicy
from upsonic.safety_engine.policies.tool_safety_policies import (
    HarmfulToolBlockPolicy,
    MaliciousToolCallBlockPolicy
)
from io import StringIO
from contextlib import redirect_stdout

pytestmark = pytest.mark.timeout(60)


@tool
def safe_tool(message: str) -> str:
    """A safe tool that returns a message."""
    return f"Tool executed: {message}"


@pytest.mark.asyncio
async def test_user_policy():
    """Test that user_policy validates user input and logs appear."""
    # Test with prebuilt AdultContentBlockPolicy and debug=True to see logs
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent",
        user_policy=AdultContentBlockPolicy,
        debug=True  # Enable debug to see policy logs
    )
    
    # Verify policy is set
    assert agent.user_policy is not None, "user_policy should be set"
    assert agent.user_policy_manager.has_policies(), "user_policy_manager should have policies"
    
    # Capture stdout to check logs
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        # Use content that might trigger the policy
        task = Task(description="Tell me about adult entertainment industry")
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result exists
    assert result is not None, "Result should not be None"
    
    # Verify policy logs appear in output
    # user_policy logs should show "Safety Policy Triggered" and "User Input Check"
    assert "Safety Policy Triggered" in output or "Safety Policy" in output, f"user_policy log should appear. Output: {output[:1000]}"
    assert "User Input Check" in output or "Policy Name" in output, f"user_policy check type should appear. Output: {output[:1000]}"
    assert "Adult Content" in output or "AdultContentBlockPolicy" in output, f"Policy name should appear in logs. Output: {output[:1000]}"


@pytest.mark.asyncio
async def test_agent_policy():
    """Test that agent_policy validates agent output and logs appear."""
    # Test with prebuilt PIIBlockPolicy and debug=True to see logs
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent",
        agent_policy=PIIBlockPolicy,
        debug=True  # Enable debug to see policy logs
    )
    
    # Verify policy is set
    assert agent.agent_policy is not None, "agent_policy should be set"
    assert agent.agent_policy_manager.has_policies(), "agent_policy_manager should have policies"
    
    # Capture stdout to check logs
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        # Ask agent to generate content that might contain PII
        task = Task(description="Generate a sample email address and phone number for testing")
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result exists
    assert result is not None, "Result should not be None"
    
    # Verify agent_policy logs appear in output
    # agent_policy logs should show "Safety Policy Triggered" and "Agent Output Check"
    assert "Safety Policy Triggered" in output or "Safety Policy" in output, f"agent_policy log should appear. Output: {output[:1000]}"
    assert "Agent Output Check" in output or "Policy Name" in output, f"agent_policy check type should appear. Output: {output[:1000]}"
    assert "PII" in output or "PIIBlockPolicy" in output, f"Policy name should appear in logs. Output: {output[:1000]}"


@pytest.mark.asyncio
async def test_tool_policy_pre():
    """Test that tool_policy_pre validates tools at registration and logs appear."""
    # Create a potentially harmful tool to trigger the policy
    @tool
    def delete_file(filepath: str) -> str:
        """Delete a file from the system."""
        import os
        if os.path.exists(filepath):
            os.remove(filepath)
            return f"Deleted {filepath}"
        return f"File {filepath} not found"
    
    # Test with prebuilt HarmfulToolBlockPolicy and debug=True to see logs
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent",
        tool_policy_pre=HarmfulToolBlockPolicy,
        debug=True  # Enable debug to see policy logs
    )
    
    # Verify policy is set
    assert agent.tool_policy_pre is not None, "tool_policy_pre should be set"
    assert agent.tool_policy_pre_manager.has_policies(), "tool_policy_pre_manager should have policies"
    
    # Capture stdout to check logs
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        task = Task(
            description="Use the delete_file tool",
            tools=[delete_file]
        )
        
        try:
            result = await agent.do_async(task)
        except Exception:
            # Tool might be blocked, which is expected
            result = None
    
    output = output_buffer.getvalue()
    
    # Verify tool_policy_pre logs appear in output
    # tool_policy_pre logs should show "Tool Safety" and "Pre-Execution" or "Tool Validation"
    tool_safety_logs = (
        "Tool Safety" in output or
        "tool safety" in output.lower() or
        "Pre-Execution" in output or
        "Tool Validation" in output
    )
    assert tool_safety_logs, f"tool_policy_pre log should appear. Output: {output[:1000]}"
    assert "delete_file" in output or "Tool Name" in output, f"Tool name should appear in logs. Output: {output[:1000]}"


@pytest.mark.asyncio
async def test_tool_policy_post():
    """Test that tool_policy_post validates tool calls before execution and logs appear."""
    # Create a tool that might trigger malicious call detection
    @tool
    def run_command(command: str) -> str:
        """Execute a shell command."""
        import subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout
    
    # Test with prebuilt MaliciousToolCallBlockPolicy and debug=True to see logs
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent",
        tool_policy_post=MaliciousToolCallBlockPolicy,
        debug=True  # Enable debug to see policy logs
    )
    
    # Verify policy is set
    assert agent.tool_policy_post is not None, "tool_policy_post should be set"
    assert agent.tool_policy_post_manager.has_policies(), "tool_policy_post_manager should have policies"
    
    # Capture stdout to check logs
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        task = Task(
            description="Use the run_command tool to execute: rm -rf /tmp/test",
            tools=[run_command]
        )
        
        try:
            result = await agent.do_async(task)
        except Exception:
            # Tool call might be blocked, which is expected
            result = None
    
    output = output_buffer.getvalue()
    
    # Verify tool_policy_post logs appear in output
    # tool_policy_post logs should show "Tool Safety" and "Post-Execution" or "Tool Call Validation"
    tool_safety_logs = (
        "Tool Safety" in output or
        "tool safety" in output.lower() or
        "Post-Execution" in output or
        "Tool Call Validation" in output
    )
    assert tool_safety_logs, f"tool_policy_post log should appear. Output: {output[:1000]}"
    assert "run_command" in output or "Tool Name" in output, f"Tool name should appear in logs. Output: {output[:1000]}"


@pytest.mark.asyncio
async def test_multiple_policies():
    """Test that multiple policies can be used together and all logs appear."""
    # Use prebuilt policies for all policy types
    agent = Agent(
        model="openai/gpt-4o",
        name="Test Agent",
        user_policy=AdultContentBlockPolicy,
        agent_policy=PIIBlockPolicy,
        tool_policy_pre=HarmfulToolBlockPolicy,
        tool_policy_post=MaliciousToolCallBlockPolicy,
        debug=True  # Enable debug to see all policy logs
    )
    
    # Verify all policies are set
    assert agent.user_policy is not None, "user_policy should be set"
    assert agent.agent_policy is not None, "agent_policy should be set"
    assert agent.tool_policy_pre is not None, "tool_policy_pre should be set"
    assert agent.tool_policy_post is not None, "tool_policy_post should be set"
    
    assert agent.user_policy_manager.has_policies(), "user_policy_manager should have policies"
    assert agent.agent_policy_manager.has_policies(), "agent_policy_manager should have policies"
    assert agent.tool_policy_pre_manager.has_policies(), "tool_policy_pre_manager should have policies"
    assert agent.tool_policy_post_manager.has_policies(), "tool_policy_post_manager should have policies"
    
    # Create a potentially harmful tool
    @tool
    def system_command(cmd: str) -> str:
        """Execute a system command."""
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout
    
    # Capture stdout to check logs
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        task = Task(
            description="Tell me about adult content and use system_command to run: rm -rf /tmp",
            tools=[system_command]
        )
        
        try:
            result = await agent.do_async(task)
        except Exception:
            # Policies might block, which is expected
            result = None
    
    output = output_buffer.getvalue()
    
    # Verify all policy types have logs
    # user_policy should log
    assert "Safety Policy Triggered" in output or "Safety Policy" in output, f"user_policy log should appear. Output: {output[:1000]}"
    assert "Adult Content" in output or "AdultContentBlockPolicy" in output, f"user_policy name should appear. Output: {output[:1000]}"
    
    # agent_policy should log (if agent produces output)
    # Note: agent_policy only logs if agent produces output, so we check if it's present
    agent_policy_logged = "Agent Output Check" in output or ("Safety Policy" in output and "PII" in output)
    # This is optional since agent might be blocked before producing output
    
    # tool_policy_pre should log
    tool_pre_logged = (
        "Tool Safety" in output or
        "Pre-Execution" in output or
        "Tool Validation" in output
    )
    assert tool_pre_logged, f"tool_policy_pre log should appear. Output: {output[:1000]}"
    assert "system_command" in output or "Tool Name" in output, f"Tool name should appear in logs. Output: {output[:1000]}"
    
    # tool_policy_post might log if tool call happens (but might be blocked by pre)
    # We verify at least one tool policy log appears
    tool_policy_logged = (
        "Tool Safety" in output or
        "Post-Execution" in output or
        "Tool Call Validation" in output or
        "Pre-Execution" in output
    )
    assert tool_policy_logged, f"At least one tool policy log should appear. Output: {output[:1000]}"

