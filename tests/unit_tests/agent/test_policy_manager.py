"""
Tests for PolicyManager

This module contains tests for the PolicyManager class, verifying its initialization,
policy execution, and handling of multiple policies.
"""

import pytest

from upsonic.agent.policy_manager import PolicyManager, PolicyResult
from upsonic.safety_engine.base import Policy, RuleBase, ActionBase
from upsonic.safety_engine.models import PolicyInput, RuleOutput, PolicyOutput
from upsonic.safety_engine.exceptions import DisallowedOperation


# ============================================================================
# Test Fixtures
# ============================================================================


class MockRule(RuleBase):
    """Mock rule for testing."""

    name = "Mock Rule"
    description = "A mock rule for testing"

    def __init__(self, confidence: float = 0.0, content_type: str = "SAFE"):
        super().__init__()
        self.confidence = confidence
        self.content_type = content_type

    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process the input and return a rule result."""
        return RuleOutput(
            confidence=self.confidence,
            content_type=self.content_type,
            details=f"Mock rule detected {self.content_type}",
            triggered_keywords=[] if self.confidence == 0.0 else ["test"],
        )


class MockAction(ActionBase):
    """Mock action for testing."""

    name = "Mock Action"
    description = "A mock action for testing"

    def __init__(self, action_taken: str = "ALLOW"):
        super().__init__()
        self.action_taken = action_taken

    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute the action based on rule result."""
        if self.action_taken == "BLOCK":
            return self.raise_block_error("Content blocked by mock policy")
        elif self.action_taken == "REPLACE":
            return self.replace_triggered_keywords(replacement="[REPLACED]")
        elif self.action_taken == "ANONYMIZE":
            return self.anonymize_triggered_keywords()
        else:
            return self.allow_content()


def create_mock_policy(
    name: str, confidence: float = 0.0, action_taken: str = "ALLOW"
) -> Policy:
    """Helper function to create a mock policy."""
    rule = MockRule(
        confidence=confidence, content_type="TEST" if confidence > 0 else "SAFE"
    )
    action = MockAction(action_taken=action_taken)
    return Policy(
        name=name,
        description=f"Mock policy: {name}",
        rule=rule,
        action=action,
    )


# ============================================================================
# Test PolicyManager Initialization
# ============================================================================


class TestPolicyManagerInitialization:
    """Test suite for PolicyManager initialization."""

    def test_policy_manager_initialization(self):
        """Test PolicyManager initialization."""
        manager = PolicyManager()

        assert manager.policies == []
        assert manager.debug is False
        assert manager.has_policies() is False

    def test_policy_manager_initialization_single_policy(self):
        """Test PolicyManager initialization with a single policy."""
        policy = create_mock_policy("Test Policy")
        manager = PolicyManager(policies=policy)

        assert len(manager.policies) == 1
        assert manager.policies[0].name == "Test Policy"
        assert manager.has_policies() is True

    def test_policy_manager_initialization_multiple_policies(self):
        """Test PolicyManager initialization with multiple policies."""
        policy1 = create_mock_policy("Policy 1")
        policy2 = create_mock_policy("Policy 2")
        manager = PolicyManager(policies=[policy1, policy2])

        assert len(manager.policies) == 2
        assert manager.policies[0].name == "Policy 1"
        assert manager.policies[1].name == "Policy 2"
        assert manager.has_policies() is True

    def test_policy_manager_initialization_with_debug(self):
        """Test PolicyManager initialization with debug enabled."""
        manager = PolicyManager(debug=True)

        assert manager.debug is True

    def test_policy_manager_repr(self):
        """Test PolicyManager string representation."""
        policy1 = create_mock_policy("Policy 1")
        policy2 = create_mock_policy("Policy 2")
        manager = PolicyManager(policies=[policy1, policy2])

        repr_str = repr(manager)
        assert "PolicyManager" in repr_str
        assert "Policy 1" in repr_str
        assert "Policy 2" in repr_str


# ============================================================================
# Test User Policy Application
# ============================================================================


class TestPolicyManagerUserPolicy:
    """Test suite for user policy application."""

    @pytest.mark.asyncio
    async def test_policy_manager_user_policy(self):
        """Test user policy application."""
        # Test allow
        policy = create_mock_policy("User Policy", confidence=0.0, action_taken="ALLOW")
        manager = PolicyManager(policies=policy)

        policy_input = PolicyInput(input_texts=["This is safe content"])
        result = await manager.execute_policies_async(policy_input, "User Input Check")

        assert result.action_taken == "ALLOW"
        assert result.was_blocked is False
        assert result.should_block() is False
        assert len(result.triggered_policies) == 0
        assert result.disallowed_exception is None

        # Test block
        policy = create_mock_policy("User Policy", confidence=1.0, action_taken="BLOCK")
        manager = PolicyManager(policies=policy)

        policy_input = PolicyInput(input_texts=["This is blocked content"])
        result = await manager.execute_policies_async(policy_input, "User Input Check")

        assert result.action_taken == "BLOCK"
        assert result.was_blocked is True
        assert result.should_block() is True
        assert len(result.triggered_policies) == 1
        assert result.triggered_policies[0] == "User Policy"
        assert "blocked" in result.message.lower()
        assert result.disallowed_exception is None

    @pytest.mark.asyncio
    async def test_policy_manager_user_policy_replace(self):
        """Test user policy that replaces content."""
        policy = create_mock_policy(
            "User Policy", confidence=1.0, action_taken="REPLACE"
        )
        manager = PolicyManager(policies=policy)

        policy_input = PolicyInput(input_texts=["This contains test content"])
        result = await manager.execute_policies_async(policy_input, "User Input Check")

        assert result.action_taken == "REPLACE"
        assert result.was_blocked is False
        assert result.should_block() is False
        assert len(result.triggered_policies) == 1
        assert result.final_output is not None

    @pytest.mark.asyncio
    async def test_policy_manager_user_policy_anonymize(self):
        """Test user policy that anonymizes content."""
        policy = create_mock_policy(
            "User Policy", confidence=1.0, action_taken="ANONYMIZE"
        )
        manager = PolicyManager(policies=policy)

        policy_input = PolicyInput(input_texts=["This contains test content"])
        result = await manager.execute_policies_async(policy_input, "User Input Check")

        assert result.action_taken == "ANONYMIZE"
        assert result.was_blocked is False
        assert result.should_block() is False
        assert len(result.triggered_policies) == 1


# ============================================================================
# Test Agent Policy Application
# ============================================================================


class TestPolicyManagerAgentPolicy:
    """Test suite for agent policy application."""

    @pytest.mark.asyncio
    async def test_policy_manager_agent_policy(self):
        """Test agent policy application."""
        # Test allow
        policy = create_mock_policy(
            "Agent Policy", confidence=0.0, action_taken="ALLOW"
        )
        manager = PolicyManager(policies=policy)

        policy_input = PolicyInput(input_texts=["This is safe agent output"])
        result = await manager.execute_policies_async(
            policy_input, "Agent Output Check"
        )

        assert result.action_taken == "ALLOW"
        assert result.was_blocked is False
        assert result.should_block() is False
        assert len(result.triggered_policies) == 0

        # Test block
        policy = create_mock_policy(
            "Agent Policy", confidence=1.0, action_taken="BLOCK"
        )
        manager = PolicyManager(policies=policy)

        policy_input = PolicyInput(input_texts=["This is blocked agent output"])
        result = await manager.execute_policies_async(
            policy_input, "Agent Output Check"
        )

        assert result.action_taken == "BLOCK"
        assert result.was_blocked is True
        assert result.should_block() is True
        assert len(result.triggered_policies) == 1
        assert result.triggered_policies[0] == "Agent Policy"

    @pytest.mark.asyncio
    async def test_policy_manager_agent_policy_disallowed_exception(self):
        """Test agent policy that raises DisallowedOperation."""
        # Create a policy that raises DisallowedOperation
        rule = MockRule(confidence=1.0, content_type="DISALLOWED")
        action = MockAction(action_taken="BLOCK")

        # Override execute_async to raise DisallowedOperation
        async def execute_async_raises(policy_input):
            raise DisallowedOperation("Operation not allowed")

        policy = Policy(
            name="Disallow Policy",
            description="Policy that disallows operations",
            rule=rule,
            action=action,
        )
        policy.execute_async = execute_async_raises

        manager = PolicyManager(policies=policy)

        policy_input = PolicyInput(input_texts=["This should be disallowed"])
        result = await manager.execute_policies_async(
            policy_input, "Agent Output Check"
        )

        assert result.action_taken == "DISALLOWED_EXCEPTION"
        assert result.was_blocked is True
        assert result.should_block() is True
        assert result.disallowed_exception is not None
        assert "disallowed" in result.message.lower()
        assert len(result.triggered_policies) == 1


# ============================================================================
# Test Multiple Policies
# ============================================================================


class TestPolicyManagerMultiplePolicies:
    """Test suite for multiple policies."""

    @pytest.mark.asyncio
    async def test_policy_manager_multiple_policies(self):
        """Test multiple policies."""
        # Test all allow
        policy1 = create_mock_policy("Policy 1", confidence=0.0, action_taken="ALLOW")
        policy2 = create_mock_policy("Policy 2", confidence=0.0, action_taken="ALLOW")
        manager = PolicyManager(policies=[policy1, policy2])

        policy_input = PolicyInput(input_texts=["Safe content"])
        result = await manager.execute_policies_async(policy_input, "Policy Check")

        assert result.action_taken == "ALLOW"
        assert result.was_blocked is False
        assert len(result.triggered_policies) == 0

        # Test first blocks
        policy1 = create_mock_policy("Policy 1", confidence=1.0, action_taken="BLOCK")
        policy2 = create_mock_policy("Policy 2", confidence=1.0, action_taken="REPLACE")
        manager = PolicyManager(policies=[policy1, policy2])

        policy_input = PolicyInput(input_texts=["Blocked content"])
        result = await manager.execute_policies_async(policy_input, "Policy Check")

        # First policy blocks, so second should not execute
        assert result.action_taken == "BLOCK"
        assert result.was_blocked is True
        assert len(result.triggered_policies) == 1
        assert result.triggered_policies[0] == "Policy 1"

        # Test sequential replace
        policy1 = create_mock_policy("Policy 1", confidence=1.0, action_taken="REPLACE")
        policy2 = create_mock_policy("Policy 2", confidence=1.0, action_taken="REPLACE")
        manager = PolicyManager(policies=[policy1, policy2])

        policy_input = PolicyInput(input_texts=["Content to replace"])
        result = await manager.execute_policies_async(policy_input, "Policy Check")

        assert result.action_taken == "REPLACE"
        assert result.was_blocked is False
        assert len(result.triggered_policies) == 2
        assert "Policy 1" in result.triggered_policies
        assert "Policy 2" in result.triggered_policies

        # Test mixed actions (replace then block)
        policy1 = create_mock_policy("Policy 1", confidence=1.0, action_taken="REPLACE")
        policy2 = create_mock_policy("Policy 2", confidence=1.0, action_taken="BLOCK")
        manager = PolicyManager(policies=[policy1, policy2])

        policy_input = PolicyInput(input_texts=["Content"])
        result = await manager.execute_policies_async(policy_input, "Policy Check")

        # Second policy blocks, so result should be BLOCK
        assert result.action_taken == "BLOCK"
        assert result.was_blocked is True
        assert len(result.triggered_policies) == 2

    @pytest.mark.asyncio
    async def test_policy_manager_multiple_policies_error_handling(self):
        """Test that errors in one policy don't stop other policies."""
        # Create a policy that raises an exception
        rule = MockRule(confidence=0.0, content_type="SAFE")
        action = MockAction(action_taken="ALLOW")

        async def execute_async_raises(policy_input):
            raise Exception("Unexpected error")

        policy1 = Policy(
            name="Error Policy",
            description="Policy that raises an error",
            rule=rule,
            action=action,
        )
        policy1.execute_async = execute_async_raises

        policy2 = create_mock_policy("Policy 2", confidence=1.0, action_taken="BLOCK")
        manager = PolicyManager(policies=[policy1, policy2], debug=False)

        policy_input = PolicyInput(input_texts=["Content"])
        result = await manager.execute_policies_async(policy_input, "Policy Check")

        # Second policy should still execute and block
        assert result.action_taken == "BLOCK"
        assert result.was_blocked is True
        assert len(result.triggered_policies) == 1
        assert result.triggered_policies[0] == "Policy 2"

    @pytest.mark.asyncio
    async def test_policy_manager_no_policies(self):
        """Test PolicyManager with no policies returns allow result."""
        manager = PolicyManager()

        policy_input = PolicyInput(input_texts=["Any content"])
        result = await manager.execute_policies_async(policy_input, "Policy Check")

        assert result.action_taken == "ALLOW"
        assert result.was_blocked is False
        assert len(result.triggered_policies) == 0


# ============================================================================
# Test PolicyResult
# ============================================================================


class TestPolicyResult:
    """Test suite for PolicyResult class."""

    def test_policy_result_initialization(self):
        """Test PolicyResult initialization."""
        result = PolicyResult()

        assert result.action_taken == "ALLOW"
        assert result.final_output is None
        assert result.message == ""
        assert result.triggered_policies == []
        assert result.rule_outputs == []
        assert result.was_blocked is False
        assert result.disallowed_exception is None

    def test_policy_result_should_block(self):
        """Test PolicyResult should_block method."""
        result = PolicyResult()
        assert result.should_block() is False

        result.was_blocked = True
        assert result.should_block() is True

        result.was_blocked = False
        result.disallowed_exception = DisallowedOperation("Test")
        assert result.should_block() is True

    def test_policy_result_get_final_message(self):
        """Test PolicyResult get_final_message method."""
        result = PolicyResult()
        assert result.get_final_message() == "Content processed by policies"

        result.message = "Custom message"
        assert result.get_final_message() == "Custom message"

        result.disallowed_exception = DisallowedOperation("Operation not allowed")
        assert "disallowed" in result.get_final_message().lower()


# ============================================================================
# Test setup_policy_models
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
