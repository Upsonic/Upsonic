"""
Team module for multi-agent operations using the Upsonic client.
"""

from .team import Team
from .context_sharing import ContextSharing
from .task_assignment import TaskAssignment
from .result_combiner import ResultCombiner

__all__ = [
    'Team',
    'ContextSharing', 
    'TaskAssignment',
    'ResultCombiner'
]
