"""
Policy Manager - Handles multiple safety policies for agent execution.

This module provides a clean, modular way to manage and execute multiple
safety policies (both user input and agent output policies).
"""

from typing import List, Optional, Union, Tuple
from upsonic.safety_engine.base import Policy
from upsonic.safety_engine.models import PolicyInput, RuleOutput, PolicyOutput
from upsonic.safety_engine.exceptions import DisallowedOperation


class PolicyResult:
    """Aggregated result from multiple policy executions."""
    
    def __init__(self):
        self.action_taken: str = "ALLOW"  # ALLOW, BLOCK, REPLACE, ANONYMIZE, DISALLOWED_EXCEPTION
        self.final_output: Optional[str] = None
        self.message: str = ""
        self.triggered_policies: List[str] = []
        self.rule_outputs: List[RuleOutput] = []
        self.was_blocked: bool = False
        self.disallowed_exception: Optional[DisallowedOperation] = None
    
    def should_block(self) -> bool:
        """Check if content should be blocked."""
        return self.was_blocked or self.disallowed_exception is not None
    
    def get_final_message(self) -> str:
        """Get the final message to return to user."""
        if self.disallowed_exception:
            return f"Operation disallowed: {str(self.disallowed_exception)}"
        return self.message or "Content processed by policies"


class PolicyManager:
    """
    Manages execution of multiple safety policies.
    
    This class handles:
    - Executing multiple policies in sequence
    - Aggregating results from all policies
    - Applying the most restrictive action
    - Proper async execution
    
    Usage:
        ```python
        manager = PolicyManager(policies=[policy1, policy2])
        result = await manager.execute_policies_async(policy_input, "User Input Check")
        
        if result.should_block():
            # Handle blocking
            pass
        ```
    """
    
    def __init__(
        self,
        policies: Optional[Union[Policy, List[Policy]]] = None,
        debug: bool = False
    ):
        """
        Initialize the policy manager.
        
        Args:
            policies: Single policy or list of policies to manage
            debug: Enable debug logging
        """
        self.debug = debug
        
        # Normalize to list
        if policies is None:
            self.policies: List[Policy] = []
        elif isinstance(policies, list):
            self.policies = policies
        else:
            self.policies = [policies]
    
    def has_policies(self) -> bool:
        """Check if any policies are configured."""
        return len(self.policies) > 0
    
    async def execute_policies_async(
        self,
        policy_input: PolicyInput,
        check_type: str = "Policy Check"
    ) -> PolicyResult:
        """
        Execute all policies asynchronously and aggregate results.
        
        This method:
        1. Executes each policy in sequence
        2. Applies the most restrictive action across all policies
        3. Aggregates transformations (replacements/anonymizations)
        4. Handles exceptions properly
        
        Args:
            policy_input: Input to evaluate
            check_type: Type of check (for logging)
        
        Returns:
            PolicyResult: Aggregated result from all policies
        """
        result = PolicyResult()
        
        if not self.has_policies():
            return result
        
        # Current content being processed (may be transformed by policies)
        current_texts = policy_input.input_texts or []
        
        for policy in self.policies:
            try:
                # Create policy input with current (possibly transformed) texts
                current_input = PolicyInput(
                    input_texts=current_texts,
                    input_images=policy_input.input_images,
                    input_videos=policy_input.input_videos,
                    input_audio=policy_input.input_audio,
                    input_files=policy_input.input_files,
                    extra_data=policy_input.extra_data
                )
                
                # Execute policy
                rule_output, action_output, policy_output = await policy.execute_async(current_input)
                action_taken = policy_output.action_output.get("action_taken", "UNKNOWN")
                
                # Store rule output for logging
                if rule_output.confidence > 0.0:
                    result.rule_outputs.append(rule_output)
                    result.triggered_policies.append(policy.name)
                
                if self.debug and rule_output.confidence > 0.0:
                    from upsonic.utils.printing import policy_triggered
                    policy_triggered(
                        policy_name=policy.name,
                        check_type=check_type,
                        action_taken=action_taken,
                        rule_output=rule_output
                    )
                
                # Handle action taken
                if action_taken == "BLOCK":
                    # BLOCK is the most restrictive action - stop immediately
                    result.action_taken = "BLOCK"
                    result.was_blocked = True
                    result.final_output = policy_output.output_texts[0] if policy_output.output_texts else f"Content blocked by policy: {policy.name}"
                    result.message = result.final_output
                    break
                
                elif action_taken in ["REPLACE", "ANONYMIZE"]:
                    # Apply transformation - content continues to next policy
                    if policy_output.output_texts:
                        current_texts = policy_output.output_texts
                    
                    # Keep track of most restrictive non-blocking action
                    if result.action_taken == "ALLOW":
                        result.action_taken = action_taken
                    
                    # Store the transformed text
                    result.final_output = current_texts[0] if current_texts else ""
                
            except DisallowedOperation as e:
                # DisallowedOperation is like BLOCK - stop immediately
                result.action_taken = "DISALLOWED_EXCEPTION"
                result.was_blocked = True
                result.disallowed_exception = e
                result.message = f"Operation disallowed by policy '{policy.name}': {str(e)}"
                result.triggered_policies.append(policy.name)
                
                # Create mock rule output for logging
                if self.debug:
                    mock_rule_output = RuleOutput(
                        confidence=1.0,
                        content_type="DISALLOWED_OPERATION",
                        details=str(e)
                    )
                    result.rule_outputs.append(mock_rule_output)
                    
                    from upsonic.utils.printing import policy_triggered
                    policy_triggered(
                        policy_name=policy.name,
                        check_type=check_type,
                        action_taken="DISALLOWED_EXCEPTION",
                        rule_output=mock_rule_output
                    )
                break
            
            except Exception as e:
                # Unexpected error - log but continue with other policies
                if self.debug:
                    from upsonic.utils.printing import warning_log
                    warning_log(f"Policy '{policy.name}' execution failed: {str(e)}", "PolicyManager")
                continue
        
        # Set final output if not blocked and transformations were applied
        if not result.should_block() and result.action_taken in ["REPLACE", "ANONYMIZE"]:
            if not result.final_output and current_texts:
                result.final_output = current_texts[0]
            result.message = f"Content {result.action_taken.lower()}d by {len(result.triggered_policies)} policy(ies)"
        
        return result
    
    def setup_policy_models(self, model) -> None:
        """
        Setup model references for all policies.
        
        This ensures policies have access to the agent's model for LLM operations.
        
        Args:
            model: The model instance to set on policies
        """
        for policy in self.policies:
            if hasattr(policy, 'base_llm') and policy.base_llm is None:
                from upsonic.safety_engine.llm.upsonic_llm import UpsonicLLMProvider
                policy.base_llm = UpsonicLLMProvider(
                    agent_name=f"Policy Base Agent ({policy.name})",
                    model=model
                )
    
    def __repr__(self) -> str:
        """String representation of the policy manager."""
        policy_names = [p.name for p in self.policies]
        return f"PolicyManager(policies={policy_names})"

