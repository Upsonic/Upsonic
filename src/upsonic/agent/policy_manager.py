"""
Policy Manager - Handles multiple safety policies for agent execution.

This module provides a clean, modular way to manage and execute multiple
safety policies (both user input and agent output policies).

Supports feedback loop mechanism where policies can provide constructive
feedback before applying blocking actions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
from upsonic.safety_engine.base import Policy
from upsonic.safety_engine.models import PolicyInput, RuleOutput
from upsonic.safety_engine.exceptions import DisallowedOperation

if TYPE_CHECKING:
    from upsonic.safety_engine.llm.upsonic_llm import UpsonicLLMProvider
    from upsonic.usage import RunUsage
    from upsonic.tasks.tasks import Task
    from upsonic.agent.agent import Agent


@dataclass
class PolicyScope:
    """Resolved scope flags for a single policy execution."""
    description: bool
    context: bool
    system_prompt: bool
    chat_history: bool
    tool_outputs: bool


def resolve_policy_scope(
    policy: Policy,
    task: "Task",
    agent: "Agent",
) -> PolicyScope:
    """Resolve scope flags with Policy > Task > Agent priority."""
    def _resolve(policy_attr: str, task_attr: str, agent_attr: str) -> bool:
        val = getattr(policy, policy_attr, None)
        if val is not None:
            return val
        val = getattr(task, task_attr, None)
        if val is not None:
            return val
        return getattr(agent, agent_attr, True)

    return PolicyScope(
        description=_resolve("apply_to_description", "policy_apply_to_description", "user_policy_apply_to_description"),
        context=_resolve("apply_to_context", "policy_apply_to_context", "user_policy_apply_to_context"),
        system_prompt=_resolve("apply_to_system_prompt", "policy_apply_to_system_prompt", "user_policy_apply_to_system_prompt"),
        chat_history=_resolve("apply_to_chat_history", "policy_apply_to_chat_history", "user_policy_apply_to_chat_history"),
        tool_outputs=_resolve("apply_to_tool_outputs", "policy_apply_to_tool_outputs", "user_policy_apply_to_tool_outputs"),
    )


class PolicyResult:
    """Aggregated result from multiple policy executions.
    
    Extended with feedback loop support to track retry state and feedback messages.
    Also includes transformation_map for reversible anonymization support.
    """
    
    def __init__(self):
        self.action_taken: str = "ALLOW"  # ALLOW, BLOCK, REPLACE, ANONYMIZE, DISALLOWED_EXCEPTION
        self.final_output: Optional[str] = None
        self.message: str = ""
        self.triggered_policies: List[str] = []
        self.rule_outputs: List[RuleOutput] = []
        self.was_blocked: bool = False
        self.disallowed_exception: Optional[DisallowedOperation] = None

        # Feedback loop support
        self.feedback_message: Optional[str] = None  # The feedback if generated
        self.requires_retry: bool = False  # Whether agent should retry with feedback
        self.original_content: Optional[str] = None  # Content that violated policy
        self.violated_policy_name: Optional[str] = None  # Name of the first violated policy
        self.violation_reason: Optional[str] = None  # Reason for violation
        
        # Transformation map for reversible anonymization
        # Maps anonymous values back to originals: {idx: {"original": "...", "anonymous": "...", "pii_type": "..."}}
        self.transformation_map: Optional[dict] = None
        
        self.output_texts: Optional[List[str]] = None
        self.source_keys: Optional[List[Tuple[str, Optional[int]]]] = None
    
    def should_block(self) -> bool:
        """Check if content should be blocked.
        
        Note: If requires_retry is True, this still returns True to indicate
        the content is not allowed in its current form.
        """
        return self.was_blocked or self.disallowed_exception is not None

    def should_retry_with_feedback(self) -> bool:
        """Check if a retry should be attempted with feedback."""
        return self.requires_retry and self.feedback_message is not None
    
    def get_final_message(self) -> str:
        """Get the final message to return to user."""
        if self.feedback_message:
            return self.feedback_message
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
    - Feedback loop mechanism for retry with constructive feedback
    
    Usage:
        ```python
        # Basic usage
        manager = PolicyManager(policies=[policy1, policy2])
        result = await manager.execute_policies_async(policy_input, "User Input Check")
        
        if result.should_block():
            # Handle blocking
            pass
        
        # With feedback loop
        manager = PolicyManager(
            policies=[policy1],
            enable_feedback=True,
            feedback_loop_count=3,
            policy_type="agent_policy"
        )
        result = await manager.execute_policies_async(policy_input, "Agent Output Check")
        
        if result.should_retry_with_feedback():
            # Re-execute with feedback message
            pass
        ```
    """
    
    def __init__(
        self,
        policies: Optional[Union[Policy, List[Policy]]] = None,
        debug: bool = False,
        enable_feedback: bool = False,
        feedback_loop_count: int = 1,
        policy_type: str = "user_policy"
    ):
        """
        Initialize the policy manager.
        
        Args:
            policies: Single policy or list of policies to manage
            debug: Enable debug logging
            enable_feedback: Enable feedback generation on policy violation
            feedback_loop_count: Maximum number of feedback retry attempts (default 1)
            policy_type: Either "user_policy" or "agent_policy" (affects feedback generation)
        """
        self.debug = debug
        self.enable_feedback = enable_feedback
        self.feedback_loop_count = feedback_loop_count
        self.policy_type = policy_type
        
        # Track current retry count for feedback loop
        self._current_retry_count: int = 0
        
        # LLM provider for feedback generation (will be set up by setup_policy_models)
        self._feedback_llm: Optional["UpsonicLLMProvider"] = None
        
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

    def reset_retry_count(self) -> None:
        """Reset the retry count for a new execution cycle."""
        self._current_retry_count = 0
    
    def increment_retry_count(self) -> None:
        """Increment the retry count after a feedback retry."""
        self._current_retry_count += 1
    
    def can_retry(self) -> bool:
        """Check if more retries are available."""
        return self._current_retry_count < self.feedback_loop_count
    
    async def execute_policies_async(
        self,
        policy_input: PolicyInput,
        check_type: str = "Policy Check",
        source_keys: Optional[List[Tuple[str, Optional[int]]]] = None,
        task: Optional["Task"] = None,
        agent: Optional["Agent"] = None,
    ) -> PolicyResult:
        """
        Execute all policies asynchronously and aggregate results.
        
        This method:
        1. Executes each policy in sequence
        2. Applies the most restrictive action across all policies
        3. Aggregates transformations (replacements/anonymizations)
        4. Handles exceptions properly
        5. Generates feedback if enabled and policy violation detected
        
        Args:
            policy_input: Input to evaluate
            check_type: Type of check (for logging)
        
        Returns:
            PolicyResult: Aggregated result from all policies
        """
        result = PolicyResult()
        
        if not self.has_policies():
            return result
        
        scoped_mode: bool = task is not None and agent is not None and source_keys is not None
        
        current_texts: List[str] = list(policy_input.input_texts or [])
        original_content: str = " ".join(current_texts) if current_texts else ""
        accumulated_map: Optional[dict] = (
            dict(policy_input.existing_transformation_map)
            if getattr(policy_input, 'existing_transformation_map', None)
            else None
        )
        for policy in self.policies:
            try:
                if scoped_mode:
                    scope: PolicyScope = resolve_policy_scope(policy, task, agent)
                    
                    filtered_indices: List[int] = []
                    filtered_texts: List[str] = []
                    for idx, (source_type, _sub_idx) in enumerate(source_keys):
                        if getattr(scope, source_type, True):
                            filtered_indices.append(idx)
                            filtered_texts.append(current_texts[idx])
                    
                    if not filtered_texts:
                        continue
                    
                    per_policy_input = PolicyInput(
                        input_texts=filtered_texts,
                        existing_transformation_map=accumulated_map,
                        input_images=policy_input.input_images,
                        input_videos=policy_input.input_videos,
                        input_audio=policy_input.input_audio,
                        input_files=policy_input.input_files,
                        extra_data=policy_input.extra_data,
                    )
                else:
                    filtered_indices = None
                    per_policy_input = PolicyInput(
                        input_texts=current_texts,
                        existing_transformation_map=accumulated_map,
                        input_images=policy_input.input_images,
                        input_videos=policy_input.input_videos,
                        input_audio=policy_input.input_audio,
                        input_files=policy_input.input_files,
                        extra_data=policy_input.extra_data,
                    )
                
                rule_output, action_output, policy_output = await policy.execute_async(per_policy_input)
                action_taken = policy_output.action_output.get("action_taken", "UNKNOWN") if policy_output.action_output else "UNKNOWN"
                
                # Store rule output for logging
                if rule_output.confidence > 0.0:
                    result.rule_outputs.append(rule_output)
                    result.triggered_policies.append(policy.name)
                
                # Always print policy check result in debug mode (including when policy passes)
                if self.debug:
                    from upsonic.utils.printing import policy_triggered
                    policy_triggered(
                        policy_name=policy.name,
                        check_type=check_type,
                        action_taken=action_taken,
                        rule_output=rule_output
                    )
                
                # Handle action taken (any action except ALLOW triggers potential feedback)
                if action_taken == "BLOCK":
                    # BLOCK is the most restrictive action - stop immediately
                    result.action_taken = "BLOCK"
                    result.was_blocked = True
                    result.final_output = policy_output.output_texts[0] if policy_output.output_texts else f"Content blocked by policy: {policy.name}"
                    result.message = result.final_output
                    result.original_content = original_content
                    result.violated_policy_name = policy.name
                    result.violation_reason = rule_output.details
                    
                    # Generate feedback if enabled and can retry
                    await self._generate_feedback_if_enabled(result, action_taken)
                    break
                
                elif action_taken in ["REPLACE", "ANONYMIZE"]:
                    if policy_output.output_texts:
                        if filtered_indices is not None:
                            for i, full_idx in enumerate(filtered_indices):
                                if i < len(policy_output.output_texts):
                                    current_texts[full_idx] = policy_output.output_texts[i]
                        else:
                            current_texts = policy_output.output_texts
                    else:
                        if self.debug:
                            from upsonic.utils.printing import warning_log
                            warning_log(
                                f"Policy '{policy.name}' returned action '{action_taken}' but no output_texts. "
                                "Using original content.",
                                "PolicyManager"
                            )
                    
                    if result.action_taken == "ALLOW":
                        result.action_taken = action_taken
                        result.original_content = original_content
                        result.violated_policy_name = policy.name
                        result.violation_reason = rule_output.details
                    
                    result.final_output = current_texts[0] if current_texts else ""
                    
                    if policy_output.transformation_map:
                        if result.transformation_map is None:
                            result.transformation_map = {}
                        base_idx = len(result.transformation_map)
                        for key, value in policy_output.transformation_map.items():
                            result.transformation_map[base_idx + key] = value
                        accumulated_map = dict(result.transformation_map)
                    
                    await self._generate_feedback_if_enabled(result, action_taken)
                
            except DisallowedOperation as e:
                # DisallowedOperation is like BLOCK - stop immediately
                result.action_taken = "DISALLOWED_EXCEPTION"
                result.was_blocked = True
                result.disallowed_exception = e
                result.message = f"Operation disallowed by policy '{policy.name}': {str(e)}"
                result.triggered_policies.append(policy.name)
                result.original_content = original_content
                result.violated_policy_name = policy.name
                result.violation_reason = str(e)
                
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
                await self._generate_feedback_if_enabled(result, "RAISE")

                break
            
            except Exception as e:
                # Unexpected error - log but continue with other policies
                if self.debug:
                    from upsonic.utils.printing import warning_log
                    warning_log(f"Policy '{policy.name}' execution failed: {str(e)}", "PolicyManager")
                continue
        
        result.output_texts = current_texts
        result.source_keys = source_keys
        
        if not result.should_block() and result.action_taken in ["REPLACE", "ANONYMIZE"]:
            if not result.final_output and current_texts:
                result.final_output = current_texts[0]
            result.message = f"Content {result.action_taken.lower()}d by {len(result.triggered_policies)} policy(ies)"
        
        return result

    async def _generate_feedback_if_enabled(
        self, 
        result: PolicyResult, 
        action_taken: str
    ) -> None:
        """
        Generate feedback message if feedback is enabled and retries are available.
        
        This sets the result's feedback_message and requires_retry fields.
        
        Args:
            result: The PolicyResult to update with feedback
            action_taken: The action that was taken (BLOCK, REPLACE, ANONYMIZE, RAISE)
        """
        if not self.enable_feedback:
            return
        
        if not self.can_retry():
            # No more retries available - don't generate feedback
            return
        
        if not result.original_content or not result.violated_policy_name:
            return
        
        # Initialize feedback LLM if needed
        if self._feedback_llm is None:
            from upsonic.safety_engine.llm.upsonic_llm import UpsonicLLMProvider
            self._feedback_llm = UpsonicLLMProvider(
                agent_name=f"Policy Feedback Agent ({self.policy_type})"
            )
        
        try:
            # Generate feedback using LLM
            feedback = await self._feedback_llm.generate_policy_feedback_async(
                original_content=result.original_content,
                policy_name=result.violated_policy_name,
                violation_reason=result.violation_reason or "Policy violation detected",
                policy_type=self.policy_type,
                action_type=action_taken,
                language="en"  # TODO: Make this configurable
            )
            
            result.feedback_message = feedback
            result.requires_retry = True
            
            if self.debug:
                from upsonic.utils.printing import policy_feedback_generated
                policy_feedback_generated(
                    policy_type=self.policy_type,
                    policy_name=result.violated_policy_name or "Unknown",
                    feedback_message=feedback,
                    retry_count=self._current_retry_count + 1,
                    max_retries=self.feedback_loop_count,
                    violation_reason=result.violation_reason
                )
                
        except Exception as e:
            if self.debug:
                from upsonic.utils.printing import warning_log
                warning_log(f"Failed to generate policy feedback: {str(e)}", "PolicyManager")
            # Don't set requires_retry if feedback generation failed
    
    def setup_policy_models(self, model) -> None:
        """
        Setup model references for all policies and feedback LLM.
        
        This ensures policies have access to the agent's model for LLM operations.
        
        Args:
            model: The model instance to set on policies
        """
        from upsonic.safety_engine.llm.upsonic_llm import UpsonicLLMProvider
        for policy in self.policies:
            if hasattr(policy, 'base_llm') and policy.base_llm is None:
                policy.base_llm = UpsonicLLMProvider(
                    agent_name=f"Policy Base Agent ({policy.name})",
                    model=model
                )
        # Also setup feedback LLM with the model if feedback is enabled
        if self.enable_feedback:
            self._feedback_llm = UpsonicLLMProvider(
                agent_name=f"Policy Feedback Agent ({self.policy_type})",
                model=model
            )
    
    def drain_accumulated_usage(self) -> Optional["RunUsage"]:
        """Drain and return accumulated sub-agent usage from all LLM providers used by policies.
        
        Returns:
            Aggregated RunUsage from all policy LLM calls, or None if no usage.
        """
        from upsonic.usage import RunUsage
        aggregated: Optional[RunUsage] = None
        
        # Drain feedback LLM usage
        if self._feedback_llm is not None:
            feedback_usage = self._feedback_llm.drain_accumulated_usage()
            if feedback_usage is not None:
                if aggregated is None:
                    aggregated = RunUsage()
                aggregated.incr(feedback_usage)
        
        # Drain usage from individual policy LLM providers
        for policy in self.policies:
            for attr_name in ('base_llm', 'text_finder_llm'):
                llm_provider = getattr(policy, attr_name, None)
                if llm_provider is not None and hasattr(llm_provider, 'drain_accumulated_usage'):
                    provider_usage = llm_provider.drain_accumulated_usage()
                    if provider_usage is not None:
                        if aggregated is None:
                            aggregated = RunUsage()
                        aggregated.incr(provider_usage)
        
        return aggregated
    
    def __repr__(self) -> str:
        """String representation of the policy manager."""
        policy_names = [p.name for p in self.policies]
        return f"PolicyManager(policies={policy_names}, enable_feedback={self.enable_feedback}, feedback_loop_count={self.feedback_loop_count})"

