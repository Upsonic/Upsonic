from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .accuracy import AccuracyEvaluator
    from .performance import PerformanceEvaluator
    from .reliability import ReliabilityEvaluator
    from .models import (
        EvaluationScore, AccuracyEvaluationResult, ToolCallCheck, 
        ReliabilityEvaluationResult, PerformanceRunResult, PerformanceEvaluationResult
    )

def _get_evaluator_classes():
    """Lazy import of evaluator classes."""
    from .accuracy import AccuracyEvaluator
    from .performance import PerformanceEvaluator
    from .reliability import ReliabilityEvaluator
    
    return {
        'AccuracyEvaluator': AccuracyEvaluator,
        'PerformanceEvaluator': PerformanceEvaluator,
        'ReliabilityEvaluator': ReliabilityEvaluator,
    }

def _get_model_classes():
    """Lazy import of model classes."""
    from .models import (
        EvaluationScore, AccuracyEvaluationResult, ToolCallCheck, 
        ReliabilityEvaluationResult, PerformanceRunResult, PerformanceEvaluationResult
    )
    
    return {
        'EvaluationScore': EvaluationScore,
        'AccuracyEvaluationResult': AccuracyEvaluationResult,
        'ToolCallCheck': ToolCallCheck,
        'ReliabilityEvaluationResult': ReliabilityEvaluationResult,
        'PerformanceRunResult': PerformanceRunResult,
        'PerformanceEvaluationResult': PerformanceEvaluationResult,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    evaluator_classes = _get_evaluator_classes()
    if name in evaluator_classes:
        return evaluator_classes[name]
    
    model_classes = _get_model_classes()
    if name in model_classes:
        return model_classes[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
__all__ = [
    "AccuracyEvaluator",
    "PerformanceEvaluator",
    "ReliabilityEvaluator",
    "ToolCallCheck",
    "PerformanceRunResult",
    "PerformanceEvaluationResult",
    "ReliabilityEvaluationResult",
    "EvaluationScore",
    "AccuracyEvaluationResult",
]