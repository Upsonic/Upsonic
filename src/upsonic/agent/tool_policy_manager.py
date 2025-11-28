"""
Tool Policy Manager - Handles tool safety policies for agent execution.

This module provides a dedicated manager for tool safety validation,
parallel to PolicyManager but specialized for tool-specific checks.
"""

from typing import List, Optional, Union, Dict, Any
from upsonic.safety_engine.base import Policy
from upsonic.safety_engine.models import PolicyInput, RuleOutput, PolicyOutput
from upsonic.safety_engine.exceptions import DisallowedOperation


class ToolPolicyResult:
    """Aggregated result from tool policy execution."""
    
    def __init__(self):
        self.action_taken: str = "ALLOW"  # ALLOW, BLOCK, DISALLOWED_EXCEPTION
        self.is_safe: bool = True
        self.message: str = ""
        self.triggered_policies: List[str] = []
        self.rule_outputs: List[RuleOutput] = []
        self.was_blocked: bool = False
        self.disallowed_exception: Optional[DisallowedOperation] = None
        self.threat_details: Dict[str, Any] = {}
    
    def should_block(self) -> bool:
        """Check if tool should be blocked."""
        return self.was_blocked or self.disallowed_exception is not None
    
    def get_final_message(self) -> str:
        """Get the final message to return."""
        if self.disallowed_exception:
            return f"Tool blocked: {str(self.disallowed_exception)}"
        return self.message or "Tool validation completed"


class ToolPolicyManager:
    """
    Manages execution of tool safety policies.
    
    This class handles:
    - Validating tools at registration (pre-execution)
    - Validating tool calls before execution (post-execution)
    - Aggregating results from multiple tool policies
    - Proper async execution
    
    Usage:
        ```python
        manager = ToolPolicyManager(policies=[policy1, policy2])
        result = await manager.execute_tool_validation_async(tool_info, "Pre-Execution")
        
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
        Initialize the tool policy manager.
        
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
        """Check if any tool policies are configured."""
        return len(self.policies) > 0
    
    async def execute_tool_validation_async(
        self,
        tool_info: Dict[str, Any],
        check_type: str = "Tool Validation"
    ) -> ToolPolicyResult:
        """
        Execute all tool validation policies asynchronously.
        
        This method validates a tool during registration (pre-execution).
        
        Args:
            tool_info: Tool information (name, description, parameters)
            check_type: Type of check (for logging)
        
        Returns:
            ToolPolicyResult: Aggregated validation result
        """
        result = ToolPolicyResult()
        
        if not self.has_policies():
            return result
        
        for policy in self.policies:
            try:
                # Create policy input with tool information in extra_data
                policy_input = PolicyInput(
                    input_texts=[],
                    extra_data={
                        "tool_name": tool_info.get("name", "unknown"),
                        "tool_description": tool_info.get("description", ""),
                        "tool_parameters_schema": tool_info.get("parameters", {}),
                        "tool_metadata": tool_info.get("metadata", {})
                    }
                )
                
                # Execute policy
                rule_output, action_output, policy_output = await policy.execute_async(policy_input)
                action_taken = policy_output.action_output.get("action_taken", "UNKNOWN")
                
                # Store rule output for logging
                if rule_output.confidence > 0.0:
                    result.rule_outputs.append(rule_output)
                    result.triggered_policies.append(policy.name)
                    result.threat_details[policy.name] = {
                        "content_type": rule_output.content_type,
                        "details": rule_output.details,
                        "confidence": rule_output.confidence
                    }
                
                if self.debug and rule_output.confidence > 0.0:
                    from upsonic.utils.printing import tool_safety_check
                    tool_safety_check(
                        tool_name=tool_info.get("name", "unknown"),
                        validation_type=check_type,
                        status="BLOCKED" if action_taken == "BLOCK" else "ALLOWED",
                        details=rule_output.details,
                        confidence=rule_output.confidence
                    )
                
                # Handle action taken
                if action_taken == "BLOCK":
                    # BLOCK is the most restrictive - stop immediately
                    result.action_taken = "BLOCK"
                    result.was_blocked = True
                    result.is_safe = False
                    result.message = policy_output.output_texts[0] if policy_output.output_texts else f"Tool blocked by policy: {policy.name}"
                    break
                
            except DisallowedOperation as e:
                # DisallowedOperation - stop immediately
                result.action_taken = "DISALLOWED_EXCEPTION"
                result.was_blocked = True
                result.is_safe = False
                result.disallowed_exception = e
                result.message = f"Tool disallowed by policy '{policy.name}': {str(e)}"
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
                    warning_log(f"Tool policy '{policy.name}' execution failed: {str(e)}", "ToolPolicyManager")
                continue
        
        return result
    
    async def execute_tool_call_validation_async(
        self,
        tool_call_info: Dict[str, Any],
        check_type: str = "Tool Call Validation"
    ) -> ToolPolicyResult:
        """
        Execute all tool call validation policies asynchronously.
        
        This method validates a specific tool call before execution (post-execution).
        
        Args:
            tool_call_info: Tool call information (name, description, arguments)
            check_type: Type of check (for logging)
        
        Returns:
            ToolPolicyResult: Aggregated validation result
        """
        result = ToolPolicyResult()
        
        if not self.has_policies():
            return result
        
        for policy in self.policies:
            try:
                # Create policy input with tool call information
                policy_input = PolicyInput(
                    input_texts=[],
                    extra_data={
                        "tool_name": tool_call_info.get("name", "unknown"),
                        "tool_description": tool_call_info.get("description", ""),
                        "tool_parameters_schema": tool_call_info.get("parameters", {}),
                        "tool_call_args": tool_call_info.get("arguments", {}),
                        "tool_call_id": tool_call_info.get("call_id", "")
                    }
                )
                
                # Execute policy
                rule_output, action_output, policy_output = await policy.execute_async(policy_input)
                action_taken = policy_output.action_output.get("action_taken", "UNKNOWN")
                
                # Store rule output for logging
                if rule_output.confidence > 0.0:
                    result.rule_outputs.append(rule_output)
                    result.triggered_policies.append(policy.name)
                    result.threat_details[policy.name] = {
                        "content_type": rule_output.content_type,
                        "details": rule_output.details,
                        "confidence": rule_output.confidence
                    }
                
                if self.debug and rule_output.confidence > 0.0:
                    from upsonic.utils.printing import tool_safety_check
                    tool_safety_check(
                        tool_name=tool_call_info.get("name", "unknown"),
                        validation_type=check_type,
                        status="BLOCKED" if action_taken == "BLOCK" else "ALLOWED",
                        details=rule_output.details,
                        confidence=rule_output.confidence
                    )
                
                # Handle action taken
                if action_taken == "BLOCK":
                    result.action_taken = "BLOCK"
                    result.was_blocked = True
                    result.is_safe = False
                    result.message = policy_output.output_texts[0] if policy_output.output_texts else f"Tool call blocked by policy: {policy.name}"
                    break
                
            except DisallowedOperation as e:
                result.action_taken = "DISALLOWED_EXCEPTION"
                result.was_blocked = True
                result.is_safe = False
                result.disallowed_exception = e
                result.message = f"Tool call disallowed by policy '{policy.name}': {str(e)}"
                result.triggered_policies.append(policy.name)
                
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
                if self.debug:
                    from upsonic.utils.printing import warning_log
                    warning_log(f"Tool call policy '{policy.name}' execution failed: {str(e)}", "ToolPolicyManager")
                continue
        
        return result
    
    def setup_policy_models(self, model) -> None:
        """
        Setup model references for all tool policies.
        
        This ensures policies have access to the agent's model for LLM operations.
        
        Args:
            model: The model instance to set on policies
        """
        for policy in self.policies:
            # Setup text_finder_llm for rules that use LLM-based detection
            if hasattr(policy, 'rule') and hasattr(policy.rule, 'text_finder_llm') and policy.rule.text_finder_llm is None:
                from upsonic.safety_engine.llm.upsonic_llm import UpsonicLLMProvider
                policy.rule.text_finder_llm = UpsonicLLMProvider(
                    agent_name=f"Tool Safety Analyzer ({policy.name})",
                    model=model
                )
            
            # Setup base_llm for actions that use LLM
            if hasattr(policy, 'action') and hasattr(policy.action, 'base_llm') and not hasattr(policy.action, 'base_llm'):
                from upsonic.safety_engine.llm.upsonic_llm import UpsonicLLMProvider
                policy.action.base_llm = UpsonicLLMProvider(
                    agent_name=f"Tool Safety Action Agent ({policy.name})",
                    model=model
                )
    
    def __repr__(self) -> str:
        """String representation of the tool policy manager."""
        policy_names = [p.name for p in self.policies]
        return f"ToolPolicyManager(policies={policy_names})"
