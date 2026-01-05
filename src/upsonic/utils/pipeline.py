"""Pipeline utility functions for step management."""

from typing import Any, List


def find_step_index_by_name(steps: List[Any], step_name: str) -> int:
    """
    Find step index by step name.
    
    Args:
        steps: List of Step instances
        step_name: Name of the step to find
        
    Returns:
        Index of the step (0-based)
        
    Raises:
        ValueError: If step not found
    """
    for i, step in enumerate(steps):
        if hasattr(step, 'name') and step.name == step_name:
            return i
    raise ValueError(f"Step '{step_name}' not found in pipeline")


def get_message_build_step_index() -> int:
    """
    Get the standard index of MessageBuildStep in the pipeline.
    
    MessageBuildStep is always at index 8 in the standard agent pipeline:
    0: InitializationStep
    1: StorageConnectionStep
    2: CacheCheckStep
    3: UserPolicyStep
    4: LLMManagerStep
    5: ModelSelectionStep
    6: ValidationStep
    7: ToolSetupStep
    8: MessageBuildStep
    9: ModelExecutionStep  <-- External tool resumption point
    ...
    
    Returns:
        The index of MessageBuildStep (8)
    """
    return 8


def get_model_execution_step_index() -> int:
    """
    Get the standard index of ModelExecutionStep in the pipeline.
    
    ModelExecutionStep is always at index 9 in the standard agent pipeline.
    This is the correct resumption point for external tool continuation since
    messages are already injected by _inject_external_tool_results.
    
    Returns:
        The index of ModelExecutionStep (9)
    """
    return 9

