from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph import (
        Graph,
        State,
        TaskNode,
        TaskChain,
        DecisionFunc,
        DecisionLLM,
        DecisionResponse,
        task,
        node,
        create_graph,
    )

def _get_graph_classes():
    """Lazy import of graph classes."""
    from .graph import (
        Graph,
        State,
        TaskNode,
        TaskChain,
        DecisionFunc,
        DecisionLLM,
        DecisionResponse,
        task,
        node,
        create_graph,
    )
    
    return {
        'Graph': Graph,
        'State': State,
        'TaskNode': TaskNode,
        'TaskChain': TaskChain,
        'DecisionFunc': DecisionFunc,
        'DecisionLLM': DecisionLLM,
        'DecisionResponse': DecisionResponse,
        'task': task,
        'node': node,
        'create_graph': create_graph,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    graph_classes = _get_graph_classes()
    if name in graph_classes:
        return graph_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = [
    'Graph',
    'State',
    'TaskNode',
    'TaskChain',
    'DecisionFunc',
    'DecisionLLM',
    'DecisionResponse',
    'task',
    'node',
    'create_graph',
]
