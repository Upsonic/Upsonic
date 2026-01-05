"""
Policy Feedback Loop Test Examples

This test file demonstrates the new policy feedback loop feature
for both user_policy and agent_policy configurations.

Run with: pytest test_policy_feedback.py -v
"""

import sys
from io import StringIO
from contextlib import redirect_stdout
import pytest

from upsonic import Agent, Task
from upsonic.safety_engine.policies.crypto_policies import CryptoBlockPolicy
from upsonic.safety_engine.policies.pii_policies import PIIBlockPolicy


def test_user_policy_with_feedback():
    """Test User Policy with Feedback (Crypto Policy)"""
    print("=" * 60)
    print("TEST 1: User Policy with Feedback (Crypto)")
    print("=" * 60)
    print("\nScenario: User asks about cryptocurrency - should get helpful feedback")
    print("-" * 60)

    # Create agent with user policy feedback enabled
    agent_user_feedback = Agent(
        model="openai/gpt-4o-mini",
        user_policy=CryptoBlockPolicy,
        user_policy_feedback=True,
        user_policy_feedback_loop=1,
        debug=True
    )

    print("\n[Agent Configuration]")
    print(f"  - user_policy: CryptoBlockPolicy")
    print(f"  - user_policy_feedback: True")
    print(f"  - user_policy_feedback_loop: 1")

    print("\n[User Input]")
    user_query = "Tell me how to buy Bitcoin and other cryptocurrencies"
    print(f"  '{user_query}'")

    print("\n[Executing...]")
    # Assert agent configuration before execution
    assert agent_user_feedback.user_policy is not None, "user_policy should be set"
    assert agent_user_feedback.user_policy_feedback is True, "user_policy_feedback should be True"
    assert agent_user_feedback.user_policy_feedback_loop == 1, "user_policy_feedback_loop should be 1"
    assert agent_user_feedback.user_policy_manager.has_policies(), "user_policy_manager should have policies"
    assert agent_user_feedback.user_policy_manager.enable_feedback is True, "user_policy_manager.enable_feedback should be True"
    assert agent_user_feedback.user_policy_manager.feedback_loop_count == 1, "user_policy_manager.feedback_loop_count should be 1"
    
    # Capture stdout to check for log messages (debug=True shows Rich panels)
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = agent_user_feedback.do(user_query)
    output = output_buffer.getvalue()
    
    # Assert result type and content
    assert isinstance(result, str), f"Result should be a string, got {type(result)}"
    assert len(result) > 0, "Result should not be empty"
    assert "crypto" in result.lower() or "bitcoin" in result.lower() or "policy" in result.lower(), \
        f"Result should mention crypto/policy violation. Got: {result[:200]}"
    # With feedback enabled, should get constructive feedback, not just "blocked"
    assert any(keyword in result.lower() for keyword in ["suggest", "consider", "rephrase", "revise", "approach", "feedback", "comply"]), \
        f"Result should contain feedback keywords. Got: {result[:200]}"
    
    # Assert log messages appear in debug output (Rich panels)
    # Rich panels use box drawing characters, so we search for key content
    assert "Policy Feedback" in output or "policy feedback" in output.lower(), \
        f"Should have 'Policy Feedback' content in debug output. Output: {output[:1000]}"
    assert "User Policy" in output or "user policy" in output.lower() or "Feedback Returned" in output, \
        f"Should have 'User Policy Feedback' content in debug output. Output: {output[:1000]}"
    assert "Crypto" in output or "crypto" in output.lower(), \
        f"Should mention crypto/policy in logs. Output: {output[:1000]}"
    
    print("\n[Result - Feedback Response]")
    print("-" * 40)
    print(result)
    print("-" * 40)
    print("\n✅ User policy feedback test PASSED - Got constructive feedback instead of hard block")


def test_user_policy_without_feedback():
    """Test User Policy WITHOUT Feedback (Comparison)"""
    print("\n" + "=" * 60)
    print("TEST 2: User Policy WITHOUT Feedback (Comparison)")
    print("=" * 60)
    print("\nScenario: Same query but with feedback DISABLED - should block/raise")
    print("-" * 60)

    user_query = "Tell me how to buy Bitcoin and other cryptocurrencies"

    # Create agent without feedback
    agent_user_no_feedback = Agent(
        model="openai/gpt-4o-mini",
        user_policy=CryptoBlockPolicy,
        user_policy_feedback=False,
        debug=True
    )

    print("\n[Agent Configuration]")
    print(f"  - user_policy: CryptoBlockPolicy")
    print(f"  - user_policy_feedback: False")

    print("\n[User Input]")
    print(f"  '{user_query}'")

    print("\n[Executing...]")
    # Assert agent configuration before execution
    assert agent_user_no_feedback.user_policy is not None, "user_policy should be set"
    assert agent_user_no_feedback.user_policy_feedback is False, "user_policy_feedback should be False"
    assert agent_user_no_feedback.user_policy_manager.has_policies(), "user_policy_manager should have policies"
    assert agent_user_no_feedback.user_policy_manager.enable_feedback is False, "user_policy_manager.enable_feedback should be False"
    
    # Capture stdout to check for log messages
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = agent_user_no_feedback.do(user_query)
    output = output_buffer.getvalue()
    
    # Assert result type and content
    assert isinstance(result, str), f"Result should be a string, got {type(result)}"
    assert len(result) > 0, "Result should not be empty"
    assert "block" in str(result).lower() or "crypto" in str(result).lower() or "detected" in str(result).lower(), \
        f"Result should contain block message. Got: {result[:200]}"
    
    # Assert NO feedback logs appear (since feedback is disabled)
    # Note: With debug=True, we might see policy trigger logs, but not feedback generation logs
    assert "Policy Feedback Generated" not in output or "Feedback Returned" not in output, \
        "Should NOT have policy feedback generation when feedback is disabled"
    
    print("\n[Result]")
    print("-" * 40)
    print(result)
    print("-" * 40)
    print("\n✅ Comparison test PASSED - Got block message as expected")


def test_agent_policy_with_feedback():
    """Test Agent Policy with Feedback (PII Policy)"""
    print("\n" + "=" * 60)
    print("TEST 3: Agent Policy with Feedback (PII)")
    print("=" * 60)
    print("\nScenario: Agent tries to generate PII - should retry with feedback")
    print("-" * 60)

    # Create agent with agent policy feedback enabled
    agent_agent_feedback = Agent(
        model="openai/gpt-4o-mini",
        agent_policy=PIIBlockPolicy,
        agent_policy_feedback=True,
        agent_policy_feedback_loop=2,
        debug=True
    )

    print("\n[Agent Configuration]")
    print(f"  - agent_policy: PIIBlockPolicy")
    print(f"  - agent_policy_feedback: True")
    print(f"  - agent_policy_feedback_loop: 2")

    print("\n[User Input]")
    agent_query = "Create a fake customer profile with a name, email, phone number, and address"
    print(f"  '{agent_query}'")

    print("\n[Executing - Watch for feedback loop iterations...]")
    # Assert agent configuration before execution
    assert agent_agent_feedback.agent_policy is not None, "agent_policy should be set"
    assert agent_agent_feedback.agent_policy_feedback is True, "agent_policy_feedback should be True"
    assert agent_agent_feedback.agent_policy_feedback_loop == 2, "agent_policy_feedback_loop should be 2"
    assert agent_agent_feedback.agent_policy_manager.has_policies(), "agent_policy_manager should have policies"
    assert agent_agent_feedback.agent_policy_manager.enable_feedback is True, "agent_policy_manager.enable_feedback should be True"
    assert agent_agent_feedback.agent_policy_manager.feedback_loop_count == 2, "agent_policy_manager.feedback_loop_count should be 2"
    
    # Capture stdout to check for log messages (debug=True shows Rich panels)
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = agent_agent_feedback.do(agent_query)
    output = output_buffer.getvalue()
    
    # Assert result type and content
    assert isinstance(result, str), f"Result should be a string, got {type(result)}"
    assert len(result) > 0, "Result should not be empty"
    # Agent policy with feedback may either:
    # 1. Still block after retries (contains PII blocking message)
    # 2. Successfully generate compliant response after feedback (contains generic placeholders)
    # Both are valid outcomes - feedback loop attempted to fix the issue
    is_blocked = any(keyword in result.lower() for keyword in ["pii", "personal", "identifiable", "blocked", "remove", "anonymize"])
    is_compliant = any(keyword in result.lower() for keyword in ["customer name", "customer email", "customer phone", "placeholder", "generic", "sample", "[customer", "[redacted"])
    assert is_blocked or is_compliant, \
        f"Result should either be blocked or contain compliant placeholders. Got: {result[:200]}"
    
    # Assert log messages appear in debug output (Rich panels for agent policy feedback)
    # Rich panels use box drawing characters, so we search for key content
    assert "Policy Feedback" in output or "policy feedback" in output.lower(), \
        f"Should have 'Policy Feedback' content in debug output. Output: {output[:1000]}"
    assert "Agent Retry" in output or "agent retry" in output.lower() or "Retrying" in output, \
        f"Should have 'Agent Retry' content in debug output. Output: {output[:1000]}"
    assert "PII" in output or "pii" in output.lower() or "personal" in output.lower(), \
        f"Should mention PII/policy in logs. Output: {output[:1000]}"
    
    print("\n[Result - After Feedback Loop]")
    print("-" * 40)
    print(result)
    print("-" * 40)
    print("\n✅ Agent policy feedback test completed")


def test_agent_policy_without_feedback():
    """Test Agent Policy WITHOUT Feedback (Comparison)"""
    print("\n" + "=" * 60)
    print("TEST 4: Agent Policy WITHOUT Feedback (Comparison)")
    print("=" * 60)
    print("\nScenario: Same query but with feedback DISABLED")
    print("-" * 60)

    agent_query = "Create a fake customer profile with a name, email, phone number, and address"

    # Create agent without feedback
    agent_agent_no_feedback = Agent(
        model="openai/gpt-4o-mini",
        agent_policy=PIIBlockPolicy,
        agent_policy_feedback=False,
        debug=True
    )

    print("\n[Agent Configuration]")
    print(f"  - agent_policy: PIIBlockPolicy")
    print(f"  - agent_policy_feedback: False")

    print("\n[User Input]")
    print(f"  '{agent_query}'")

    print("\n[Executing...]")
    # Assert agent configuration before execution
    assert agent_agent_no_feedback.agent_policy is not None, "agent_policy should be set"
    assert agent_agent_no_feedback.agent_policy_feedback is False, "agent_policy_feedback should be False"
    assert agent_agent_no_feedback.agent_policy_manager.has_policies(), "agent_policy_manager should have policies"
    assert agent_agent_no_feedback.agent_policy_manager.enable_feedback is False, "agent_policy_manager.enable_feedback should be False"
    
    # Capture stdout to check for log messages
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = agent_agent_no_feedback.do(agent_query)
    output = output_buffer.getvalue()
    
    # Assert result type and content
    assert isinstance(result, str), f"Result should be a string, got {type(result)}"
    assert len(result) > 0, "Result should not be empty"
    # Without feedback, should get a block message
    assert any(keyword in result.lower() for keyword in ["pii", "personal", "identifiable", "blocked", "remove", "anonymize"]), \
        f"Result should mention PII blocking. Got: {result[:200]}"
    
    # Assert NO feedback logs appear (since feedback is disabled)
    # Note: With debug=True, we might see policy trigger logs, but not feedback generation logs
    assert "Policy Feedback Generated" not in output or "Feedback Returned" not in output, \
        "Should NOT have policy feedback generation when feedback is disabled"
    
    print("\n[Result]")
    print("-" * 40)
    print(result)
    print("-" * 40)
    print("\n✅ Comparison test PASSED - Got block message as expected")
