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


def get_chat_history_step_index() -> int:
    """
    Get the standard index of ChatHistoryStep in the direct pipeline.
    
    ChatHistoryStep is at index 11 in the standard agent pipeline:
    0: InitializationStep
    1: StorageConnectionStep
    2: CacheCheckStep
    3: UserPolicyStep
    4: LLMManagerStep
    5: ModelSelectionStep
    6: ToolSetupStep
    7: MemoryPrepareStep
    8: SystemPromptBuildStep
    9: ContextBuildStep
    10: UserInputBuildStep
    11: ChatHistoryStep
    12: MessageAssemblyStep
    13: CallManagerSetupStep
    14: ModelExecutionStep  <-- External tool resumption point
    ...
    
    Returns:
        The index of ChatHistoryStep (11)
    """
    return 11


def get_message_assembly_step_index() -> int:
    """
    Get the standard index of MessageAssemblyStep in the direct pipeline.
    
    Returns:
        The index of MessageAssemblyStep (12)
    """
    return 12


def get_call_manager_setup_step_index() -> int:
    """
    Get the standard index of CallManagerSetupStep in the direct pipeline.
    
    Returns:
        The index of CallManagerSetupStep (13)
    """
    return 13


def get_model_execution_step_index() -> int:
    """
    Get the standard index of ModelExecutionStep in the direct pipeline.
    
    ModelExecutionStep is at index 14 in the standard agent pipeline.
    This is the correct resumption point for external tool continuation since
    messages are already injected by _inject_external_tool_results.
    
    Returns:
        The index of ModelExecutionStep (14)
    """
    return 14
