"""
Simple Policy Feedback Loop Test - With Debug Output
"""

import sys
from io import StringIO
from contextlib import redirect_stdout
import pytest

from upsonic import Agent
from upsonic.safety_engine.policies.crypto_policies import CryptoBlockPolicy
from upsonic.safety_engine.policies.pii_policies import PIIBlockPolicy


def test_user_policy_with_feedback_debug():
    """Test User Policy with Feedback (DEBUG=True)"""
    print("=" * 70)
    print("TEST 1: User Policy with Feedback (DEBUG=True)")
    print("=" * 70)

    agent1 = Agent(
        model="openai/gpt-4o-mini",
        user_policy=CryptoBlockPolicy,
        user_policy_feedback=True,
        user_policy_feedback_loop=1,
        debug=True  # Enable debug to see new printing functions
    )

    # Assert agent configuration
    assert agent1.user_policy is not None, "user_policy should be set"
    assert agent1.user_policy_feedback is True, "user_policy_feedback should be True"
    assert agent1.user_policy_feedback_loop == 1, "user_policy_feedback_loop should be 1"
    assert agent1.user_policy_manager.has_policies(), "user_policy_manager should have policies"
    assert agent1.user_policy_manager.enable_feedback is True, "user_policy_manager.enable_feedback should be True"
    assert agent1.user_policy_manager.feedback_loop_count == 1, "user_policy_manager.feedback_loop_count should be 1"

    # Capture stdout to check for log messages (debug=True shows Rich panels)
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result1 = agent1.do("Tell me how to buy Bitcoin")
    output1 = output_buffer.getvalue()

    # Assert result type and content
    assert isinstance(result1, str), f"Result should be a string, got {type(result1)}"
    assert len(result1) > 0, "Result should not be empty"
    assert "crypto" in result1.lower() or "bitcoin" in result1.lower() or "policy" in result1.lower(), \
        f"Result should mention crypto/policy violation. Got: {result1[:200]}"
    # With feedback enabled, should get constructive feedback, not just "blocked"
    assert any(keyword in result1.lower() for keyword in ["suggest", "consider", "rephrase", "revise", "approach", "feedback"]), \
        f"Result should contain feedback keywords. Got: {result1[:200]}"

    # Assert log messages appear in debug output (Rich panels)
    # Rich panels use box drawing characters, so we search for key content
    assert "Policy Feedback" in output1 or "policy feedback" in output1.lower(), \
        f"Should have 'Policy Feedback' content in debug output. Output: {output1[:1000]}"
    assert "User Policy" in output1 or "user policy" in output1.lower() or "Feedback Returned" in output1, \
        f"Should have 'User Policy Feedback' content in debug output. Output: {output1[:1000]}"
    assert "Crypto" in output1 or "crypto" in output1.lower(), \
        f"Should mention crypto/policy in logs. Output: {output1[:1000]}"

    print("\n[FINAL RESULT]:")
    print("-" * 70)
    print(result1)
    print("-" * 70)
    print("✅ TEST 1 PASSED: User policy feedback working correctly")


def test_agent_policy_with_feedback_debug():
    """Test Agent Policy with Feedback (DEBUG=True)"""
    print("\n" + "=" * 70)
    print("TEST 2: Agent Policy with Feedback (DEBUG=True)")
    print("=" * 70)

    agent2 = Agent(
        model="openai/gpt-4o-mini",
        agent_policy=PIIBlockPolicy,
        agent_policy_feedback=True,
        agent_policy_feedback_loop=2,
        debug=True  # Enable debug to see new printing functions
    )

    # Assert agent configuration
    assert agent2.agent_policy is not None, "agent_policy should be set"
    assert agent2.agent_policy_feedback is True, "agent_policy_feedback should be True"
    assert agent2.agent_policy_feedback_loop == 2, "agent_policy_feedback_loop should be 2"
    assert agent2.agent_policy_manager.has_policies(), "agent_policy_manager should have policies"
    assert agent2.agent_policy_manager.enable_feedback is True, "agent_policy_manager.enable_feedback should be True"
    assert agent2.agent_policy_manager.feedback_loop_count == 2, "agent_policy_manager.feedback_loop_count should be 2"

    # Capture stdout to check for log messages (debug=True shows Rich panels)
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result2 = agent2.do("Generate a sample customer record with name and email")
    output2 = output_buffer.getvalue()

    # Assert result type and content
    assert isinstance(result2, str), f"Result should be a string, got {type(result2)}"
    assert len(result2) > 0, "Result should not be empty"
    # Agent policy with feedback may either:
    # 1. Still block after retries (contains PII blocking message)
    # 2. Successfully generate compliant response after feedback (contains generic placeholders)
    # Both are valid outcomes - feedback loop attempted to fix the issue
    is_blocked = any(keyword in result2.lower() for keyword in ["pii", "personal", "identifiable", "blocked", "remove", "anonymize"])
    is_compliant = any(keyword in result2.lower() for keyword in ["customer name", "customer email", "customer phone", "placeholder", "generic", "sample", "[customer", "[redacted"])
    assert is_blocked or is_compliant, \
        f"Result should either be blocked or contain compliant placeholders. Got: {result2[:200]}"

    # Assert log messages appear in debug output (Rich panels for agent policy feedback)
    # Rich panels use box drawing characters, so we search for key content
    assert "Policy Feedback" in output2 or "policy feedback" in output2.lower(), \
        f"Should have 'Policy Feedback' content in debug output. Output: {output2[:1000]}"
    assert "Agent Retry" in output2 or "agent retry" in output2.lower() or "Retrying" in output2, \
        f"Should have 'Agent Retry' content in debug output. Output: {output2[:1000]}"
    assert "PII" in output2 or "pii" in output2.lower() or "personal" in output2.lower(), \
        f"Should mention PII/policy in logs. Output: {output2[:1000]}"

    print("\n[FINAL RESULT]:")
    print("-" * 70)
    print(result2)
    print("-" * 70)
    print("✅ TEST 2 PASSED: Agent policy feedback working correctly")
