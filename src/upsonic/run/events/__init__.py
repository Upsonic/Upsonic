"""
Run Events Module

This module provides event emission and event classes for agent execution.
"""

import importlib
from typing import Any

_lazy_imports = {}


def _lazy_import(module_name: str, class_name: str = None):
    """Lazy import function to defer heavy imports until actually needed."""
    def _import():
        if module_name not in _lazy_imports:
            _lazy_imports[module_name] = importlib.import_module(module_name)
        
        if class_name:
            return getattr(_lazy_imports[module_name], class_name)
        return _lazy_imports[module_name]
    
    return _import


def _get_EventEmitter():
    return _lazy_import("upsonic.run.events.emitter", "EventEmitter")()


# Event class getters
def _get_event_class(class_name: str):
    return _lazy_import("upsonic.run.events.events", class_name)()


_EVENT_CLASSES = [
    "AgentEvent",
    "AgentRunEvent",
    "RunEvent",  # Alias for AgentRunEvent
    "PipelineStartEvent",
    "PipelineEndEvent",
    "StepStartEvent",
    "StepEndEvent",
    "AgentInitializedEvent",
    "StorageConnectionEvent",
    "CacheCheckEvent",
    "CacheHitEvent",
    "CacheMissEvent",
    "PolicyCheckEvent",
    "PolicyFeedbackEvent",
    "LLMPreparedEvent",
    "ModelSelectedEvent",
    "ValidationEvent",
    "ToolsConfiguredEvent",
    "MessagesBuiltEvent",
    "ModelRequestStartEvent",
    "ModelResponseEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "ExternalToolPauseEvent",
    "ReflectionEvent",
    "MemoryUpdateEvent",
    "CultureUpdateEvent",
    "ReliabilityEvent",
    "CacheStoredEvent",
    "ExecutionCompleteEvent",
    "AgentStreamEvent",
    "TextDeltaEvent",
    "TextCompleteEvent",
    "ThinkingDeltaEvent",
    "ToolCallDeltaEvent",
    "FinalOutputEvent",
    "PartStartEvent",
    "PartDeltaEvent",
    "PartEndEvent",
    "FinalResultEvent",
    "RunStartedEvent",
    "RunCompletedEvent",
    "RunCancelledEvent",
    "RunPausedEvent",
    "convert_llm_event_to_agent_event",
]


def __getattr__(name: str) -> Any:
    """Lazy loading of event module components."""
    
    if name == "EventEmitter":
        return _get_EventEmitter()
    
    if name in _EVENT_CLASSES:
        return _get_event_class(name)
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
    )


__all__ = [
    "EventEmitter",
] + _EVENT_CLASSES
