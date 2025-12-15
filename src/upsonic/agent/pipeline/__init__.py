"""
Agent Pipeline Architecture

This module provides a comprehensive pipeline system for agent execution.
The pipeline architecture breaks down the agent's execution into discrete,
manageable steps that can be easily understood, tested, and extended.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context import StepContext
    from .step import Step, StepResult, StepStatus
    from .manager import PipelineManager
    from .steps import (
        InitializationStep,
        CacheCheckStep,
        UserPolicyStep,
        StorageConnectionStep,
        LLMManagerStep,
        ModelSelectionStep,
        ValidationStep,
        ToolSetupStep,
        MessageBuildStep,
        ModelExecutionStep,
        ResponseProcessingStep,
        ReflectionStep,
        CallManagementStep,
        TaskManagementStep,
        MemoryMessageTrackingStep,
        ReliabilityStep,
        AgentPolicyStep,
        CacheStorageStep,
        FinalizationStep,
        StreamModelExecutionStep,
        StreamMemoryMessageTrackingStep,
        StreamFinalizationStep,
    )

def _get_pipeline_base_classes():
    """Lazy import of base pipeline classes."""
    from .context import StepContext
    from .step import Step, StepResult, StepStatus
    from .manager import PipelineManager
    
    return {
        'StepContext': StepContext,
        'Step': Step,
        'StepResult': StepResult,
        'StepStatus': StepStatus,
        'PipelineManager': PipelineManager,
    }

def _get_pipeline_step_classes():
    """Lazy import of pipeline step classes."""
    from .steps import (
        InitializationStep,
        CacheCheckStep,
        UserPolicyStep,
        StorageConnectionStep,
        LLMManagerStep,
        ModelSelectionStep,
        ValidationStep,
        ToolSetupStep,
        MessageBuildStep,
        ModelExecutionStep,
        ResponseProcessingStep,
        ReflectionStep,
        CallManagementStep,
        TaskManagementStep,
        MemoryMessageTrackingStep,
        ReliabilityStep,
        AgentPolicyStep,
        CacheStorageStep,
        FinalizationStep,
        StreamModelExecutionStep,
        StreamMemoryMessageTrackingStep,
        StreamFinalizationStep,
    )
    
    return {
        'InitializationStep': InitializationStep,
        'CacheCheckStep': CacheCheckStep,
        'UserPolicyStep': UserPolicyStep,
        'StorageConnectionStep': StorageConnectionStep,
        'LLMManagerStep': LLMManagerStep,
        'ModelSelectionStep': ModelSelectionStep,
        'ValidationStep': ValidationStep,
        'ToolSetupStep': ToolSetupStep,
        'MessageBuildStep': MessageBuildStep,
        'ModelExecutionStep': ModelExecutionStep,
        'ResponseProcessingStep': ResponseProcessingStep,
        'ReflectionStep': ReflectionStep,
        'CallManagementStep': CallManagementStep,
        'TaskManagementStep': TaskManagementStep,
        'MemoryMessageTrackingStep': MemoryMessageTrackingStep,
        'ReliabilityStep': ReliabilityStep,
        'AgentPolicyStep': AgentPolicyStep,
        'CacheStorageStep': CacheStorageStep,
        'FinalizationStep': FinalizationStep,
        'StreamModelExecutionStep': StreamModelExecutionStep,
        'StreamMemoryMessageTrackingStep': StreamMemoryMessageTrackingStep,
        'StreamFinalizationStep': StreamFinalizationStep,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    # Base classes
    base_classes = _get_pipeline_base_classes()
    if name in base_classes:
        return base_classes[name]
    
    # Step classes
    step_classes = _get_pipeline_step_classes()
    if name in step_classes:
        return step_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    "StepContext",
    "Step",
    "StepResult",
    "StepStatus",
    "PipelineManager",
    "InitializationStep",
    "CacheCheckStep",
    "UserPolicyStep",
    "StorageConnectionStep",
    "LLMManagerStep",
    "ModelSelectionStep",
    "ValidationStep",
    "ToolSetupStep",
    "MessageBuildStep",
    "ModelExecutionStep",
    "ResponseProcessingStep",
    "ReflectionStep",
    "CallManagementStep",
    "TaskManagementStep",
    "MemoryMessageTrackingStep",
    "ReliabilityStep",
    "AgentPolicyStep",
    "CacheStorageStep",
    "FinalizationStep",
    # Streaming-specific steps
    "StreamModelExecutionStep",
    "StreamMemoryMessageTrackingStep",
    "StreamFinalizationStep",
]

