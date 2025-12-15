"""
Agent module for the Upsonic AI Agent Framework.
This module provides agent classes for executing tasks and managing AI interactions.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from .agent import Agent
    from .base import BaseAgent
    from .run_result import AgentRunResult, OutputDataT, StreamRunResult
    from .deepagent import DeepAgent
    from .events import (
        AgentEvent,
        PipelineStartEvent,
        PipelineEndEvent,
        StepStartEvent,
        StepEndEvent,
        AgentInitializedEvent,
        CacheCheckEvent,
        CacheHitEvent,
        CacheMissEvent,
        PolicyCheckEvent,
        PolicyFeedbackEvent,
        ModelSelectedEvent,
        ToolsConfiguredEvent,
        MessagesBuiltEvent,
        ModelRequestStartEvent,
        ModelResponseEvent,
        ToolCallEvent,
        ToolResultEvent,
        ExternalToolPauseEvent,
        ReflectionEvent,
        MemoryUpdateEvent,
        ReliabilityEvent,
        CacheStoredEvent,
        ExecutionCompleteEvent,
        TextDeltaEvent,
        TextCompleteEvent,
        ThinkingDeltaEvent,
        ToolCallDeltaEvent,
        FinalOutputEvent,
        convert_llm_event_to_agent_event,
        AgentStreamEvent,
    )
def _get_agent_classes():
    """Lazy import of agent classes."""
    from .agent import Agent
    from .base import BaseAgent
    from .run_result import AgentRunResult, OutputDataT, StreamRunResult
    from .deepagent import DeepAgent
    
    return {
        'Agent': Agent,
        'BaseAgent': BaseAgent,
        'AgentRunResult': AgentRunResult,
        'OutputDataT': OutputDataT,
        'StreamRunResult': StreamRunResult,
        'DeepAgent': DeepAgent,
    }
def _get_event_classes():
    """Lazy import of event classes."""
    from .events import (
        AgentEvent,
        PipelineStartEvent,
        PipelineEndEvent,
        StepStartEvent,
        StepEndEvent,
        AgentInitializedEvent,
        CacheCheckEvent,
        CacheHitEvent,
        CacheMissEvent,
        PolicyCheckEvent,
        PolicyFeedbackEvent,
        ModelSelectedEvent,
        ToolsConfiguredEvent,
        MessagesBuiltEvent,
        ModelRequestStartEvent,
        ModelResponseEvent,
        ToolCallEvent,
        ToolResultEvent,
        ExternalToolPauseEvent,
        ReflectionEvent,
        MemoryUpdateEvent,
        ReliabilityEvent,
        CacheStoredEvent,
        ExecutionCompleteEvent,
        TextDeltaEvent,
        TextCompleteEvent,
        ThinkingDeltaEvent,
        ToolCallDeltaEvent,
        FinalOutputEvent,
        convert_llm_event_to_agent_event,
        AgentStreamEvent,
    )
    
    return {
        'AgentEvent': AgentEvent,
        'PipelineStartEvent': PipelineStartEvent,
        'PipelineEndEvent': PipelineEndEvent,
        'StepStartEvent': StepStartEvent,
        'StepEndEvent': StepEndEvent,
        'AgentInitializedEvent': AgentInitializedEvent,
        'CacheCheckEvent': CacheCheckEvent,
        'CacheHitEvent': CacheHitEvent,
        'CacheMissEvent': CacheMissEvent,
        'PolicyCheckEvent': PolicyCheckEvent,
        'PolicyFeedbackEvent': PolicyFeedbackEvent,
        'ModelSelectedEvent': ModelSelectedEvent,
        'ToolsConfiguredEvent': ToolsConfiguredEvent,
        'MessagesBuiltEvent': MessagesBuiltEvent,
        'ModelRequestStartEvent': ModelRequestStartEvent,
        'ModelResponseEvent': ModelResponseEvent,
        'ToolCallEvent': ToolCallEvent,
        'ToolResultEvent': ToolResultEvent,
        'ExternalToolPauseEvent': ExternalToolPauseEvent,
        'ReflectionEvent': ReflectionEvent,
        'MemoryUpdateEvent': MemoryUpdateEvent,
        'ReliabilityEvent': ReliabilityEvent,
        'CacheStoredEvent': CacheStoredEvent,
        'ExecutionCompleteEvent': ExecutionCompleteEvent,
        'TextDeltaEvent': TextDeltaEvent,
        'TextCompleteEvent': TextCompleteEvent,
        'ThinkingDeltaEvent': ThinkingDeltaEvent,
        'ToolCallDeltaEvent': ToolCallDeltaEvent,
        'FinalOutputEvent': FinalOutputEvent,
        'convert_llm_event_to_agent_event': convert_llm_event_to_agent_event,
        'AgentStreamEvent': AgentStreamEvent,
    }
def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    # Agent classes
    agent_classes = _get_agent_classes()
    if name in agent_classes:
        return agent_classes[name]
    
    # Event classes
    event_classes = _get_event_classes()
    if name in event_classes:
        return event_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module. "
        f"For example: from upsonic.agent.pipeline import PipelineManager"
    )
__all__ = [
    # Agent classes
    'Agent',
    'BaseAgent',
    'AgentRunResult',
    'OutputDataT',
    'DeepAgent',
    'StreamRunResult',
    
    # Base event
    'AgentEvent',
    
    # Pipeline-level events
    'PipelineStartEvent',
    'PipelineEndEvent',
    
    # Step-level events
    'StepStartEvent',
    'StepEndEvent',
    
    # Step-specific events
    'AgentInitializedEvent',
    'CacheCheckEvent',
    'CacheHitEvent', 
    'CacheMissEvent',
    'PolicyCheckEvent',
    'PolicyFeedbackEvent',
    'ModelSelectedEvent',
    'ToolsConfiguredEvent',
    'MessagesBuiltEvent',
    'ModelRequestStartEvent',
    'ModelResponseEvent',
    'ToolCallEvent',
    'ToolResultEvent',
    'ExternalToolPauseEvent',
    'ReflectionEvent',
    'MemoryUpdateEvent',
    'ReliabilityEvent',
    'CacheStoredEvent',
    'ExecutionCompleteEvent',
    
    # LLM stream event wrappers
    'TextDeltaEvent',
    'TextCompleteEvent',
    'ThinkingDeltaEvent',
    'ToolCallDeltaEvent',
    'FinalOutputEvent',
    
    # Helper function
    'convert_llm_event_to_agent_event',
    
    # Type alias
    'AgentStreamEvent',
]