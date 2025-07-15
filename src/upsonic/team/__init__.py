"""
Team module for multi-agent operations using the Upsonic client.
"""

from upsonic.team.team import Team
from upsonic.team.context_sharing import ContextSharing
from upsonic.team.task_assignment import TaskAssignment
from upsonic.team.result_combiner import ResultCombiner

__all__ = [
    'Team',
    'ContextSharing', 
    'TaskAssignment',
    'ResultCombiner'
]
