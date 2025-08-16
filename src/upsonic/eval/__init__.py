# Import the main evaluator class to make it directly available at the package level.
# This allows users to write `from upsonic.evals import AccuracyEvaluator`
from .accuracy import AccuracyEvaluator
from .performance import PerformanceEvaluator
from .reliability import ReliabilityEvaluator

# Optionally, you can also expose the data models if you think users will need to interact with them directly.
from .models import EvaluationScore, AccuracyEvaluationResult, ToolCallCheck, ReliabilityEvaluationResult, PerformanceRunResult, PerformanceEvaluationResult


# Define what gets imported when a user writes `from upsonic.evals import *`
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