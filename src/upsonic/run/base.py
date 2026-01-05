from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.agent.pipeline.step import StepStatus
    
from enum import Enum


class RunStatus(str, Enum):
    """Status enum for Agent/Team/Workflow runs."""
    
    running = "RUNNING"
    completed = "COMPLETED"
    paused = "PAUSED"
    cancelled = "CANCELLED"
    error = "ERROR"
    
    @staticmethod
    def from_step_status(step_status: "StepStatus") -> "RunStatus":
        """
        Map StepStatus to RunStatus.
        
        Args:
            step_status: Step status to convert
            
        Returns:
            Corresponding RunStatus
        """
        from upsonic.agent.pipeline.step import StepStatus
        
        mapping = {
            StepStatus.RUNNING: RunStatus.running,
            StepStatus.COMPLETED: RunStatus.completed,
            StepStatus.PAUSED: RunStatus.paused,
            StepStatus.CANCELLED: RunStatus.cancelled,
            StepStatus.ERROR: RunStatus.error,
            StepStatus.SKIPPED: RunStatus.completed,
        }
        return mapping.get(step_status, RunStatus.running)



