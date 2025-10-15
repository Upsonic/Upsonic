"""
Agent Pipeline Architecture

This module provides a comprehensive pipeline system for agent execution.
The pipeline architecture breaks down the agent's execution into discrete,
manageable steps that can be easily understood, tested, and extended.
"""

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
    # Streaming-specific steps
    StreamModelExecutionStep,
    StreamMemoryMessageTrackingStep,
    StreamFinalizationStep,
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

