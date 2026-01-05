"""Pipeline execution statistics tracking."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class PipelineExecutionStats:
    """
    Pipeline execution statistics for tracking and debugging.
    
    Tracks step timing, statuses, and resumption information for
    complete visibility into pipeline execution.
    
    Attributes:
        total_steps: Total number of steps in the pipeline
        executed_steps: Number of steps actually executed
        resumed_from: Step index we resumed from (for continuations), None for fresh runs
        step_timing: Mapping of step_name -> execution_time in seconds
        step_statuses: Mapping of step_name -> status string
    """
    
    total_steps: int
    executed_steps: int = 0
    resumed_from: Optional[int] = None
    step_timing: Dict[str, float] = field(default_factory=dict)
    step_statuses: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_steps": self.total_steps,
            "executed_steps": self.executed_steps,
            "resumed_from": self.resumed_from,
            "step_timing": self.step_timing,
            "step_statuses": self.step_statuses,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineExecutionStats":
        """Reconstruct from dictionary."""
        return cls(
            total_steps=data.get("total_steps", 0),
            executed_steps=data.get("executed_steps", 0),
            resumed_from=data.get("resumed_from"),
            step_timing=data.get("step_timing", {}),
            step_statuses=data.get("step_statuses", {}),
        )
    
    def serialize(self) -> bytes:
        """Serialize to bytes for storage."""
        import cloudpickle
        return cloudpickle.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data: bytes) -> "PipelineExecutionStats":
        """Deserialize from bytes."""
        import cloudpickle
        dict_data = cloudpickle.loads(data)
        return cls.from_dict(dict_data)

