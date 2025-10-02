"""
Reflection module for self-evaluation and improvement.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import (
        ReflectionAction,
        EvaluationCriteria,
        EvaluationResult,
        ReflectionConfig,
        ReflectionState,
        ReflectionPrompts
    )
    from .processor import ReflectionProcessor

def _get_model_classes():
    """Lazy import of model classes."""
    from .models import (
        ReflectionAction,
        EvaluationCriteria,
        EvaluationResult,
        ReflectionConfig,
        ReflectionState,
        ReflectionPrompts
    )
    
    return {
        'ReflectionAction': ReflectionAction,
        'EvaluationCriteria': EvaluationCriteria,
        'EvaluationResult': EvaluationResult,
        'ReflectionConfig': ReflectionConfig,
        'ReflectionState': ReflectionState,
        'ReflectionPrompts': ReflectionPrompts,
    }

def _get_processor_classes():
    """Lazy import of processor classes."""
    from .processor import ReflectionProcessor
    
    return {
        'ReflectionProcessor': ReflectionProcessor,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    model_classes = _get_model_classes()
    if name in model_classes:
        return model_classes[name]
    
    processor_classes = _get_processor_classes()
    if name in processor_classes:
        return processor_classes[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'ReflectionAction',
    'EvaluationCriteria', 
    'EvaluationResult',
    'ReflectionConfig',
    'ReflectionState',
    'ReflectionPrompts',
    'ReflectionProcessor'
]
