"""
Run Module

This module provides run management capabilities for Upsonic agents,
including run cancellation, status tracking, and the new run architecture
with AgentRunInput, AgentRunContext, and AgentRunOutput.
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


# Cancel module
def _get_RunCancellationManager():
    return _lazy_import("upsonic.run.cancel", "RunCancellationManager")()


def _get_register_run():
    return _lazy_import("upsonic.run.cancel", "register_run")()


def _get_cancel_run():
    return _lazy_import("upsonic.run.cancel", "cancel_run")()


def _get_cleanup_run():
    return _lazy_import("upsonic.run.cancel", "cleanup_run")()


def _get_raise_if_cancelled():
    return _lazy_import("upsonic.run.cancel", "raise_if_cancelled")()


def _get_is_cancelled():
    return _lazy_import("upsonic.run.cancel", "is_cancelled")()


# Base module
def _get_RunStatus():
    return _lazy_import("upsonic.run.base", "RunStatus")()


# Agent module
def _get_AgentRunInput():
    return _lazy_import("upsonic.run.agent.input", "AgentRunInput")()


def _get_AgentRunContext():
    return _lazy_import("upsonic.run.agent.context", "AgentRunContext")()


def _get_DurableCheckpoint():
    return _lazy_import("upsonic.run.agent.context", "DurableCheckpoint")()


def _get_AgentRunOutput():
    return _lazy_import("upsonic.run.agent.output", "AgentRunOutput")()


# Requirements module
def _get_RunRequirement():
    return _lazy_import("upsonic.run.requirements", "RunRequirement")()


# Tools module
def _get_ToolExecution():
    return _lazy_import("upsonic.run.tools.tools", "ToolExecution")()


# Events module
def _get_EventEmitter():
    return _lazy_import("upsonic.run.events.emitter", "EventEmitter")()


def __getattr__(name: str) -> Any:
    """Lazy loading of run module components."""
    
    # Cancel module
    if name == "RunCancellationManager":
        return _get_RunCancellationManager()
    elif name == "register_run":
        return _get_register_run()
    elif name == "cancel_run":
        return _get_cancel_run()
    elif name == "cleanup_run":
        return _get_cleanup_run()
    elif name == "raise_if_cancelled":
        return _get_raise_if_cancelled()
    elif name == "is_cancelled":
        return _get_is_cancelled()
    
    # Base module
    elif name == "RunStatus":
        return _get_RunStatus()
    
    # Agent module
    elif name == "AgentRunInput":
        return _get_AgentRunInput()
    elif name == "AgentRunContext":
        return _get_AgentRunContext()
    elif name == "DurableCheckpoint":
        return _get_DurableCheckpoint()
    elif name == "AgentRunOutput":
        return _get_AgentRunOutput()
    
    # Requirements module
    elif name == "RunRequirement":
        return _get_RunRequirement()
    
    # Tools module
    elif name == "ToolExecution":
        return _get_ToolExecution()
    
    # Events module
    elif name == "EventEmitter":
        return _get_EventEmitter()
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
    )


__all__ = [
    # Cancel module
    "RunCancellationManager",
    "register_run",
    "cancel_run",
    "cleanup_run",
    "raise_if_cancelled",
    "is_cancelled",
    # Base module
    "RunStatus",
    # Agent module
    "AgentRunInput",
    "AgentRunContext",
    "DurableCheckpoint",
    "AgentRunOutput",
    # Requirements module
    "RunRequirement",
    # Tools module
    "ToolExecution",
    # Events module
    "EventEmitter",
]
