"""
Context sharing module for managing context between tasks in multi-agent workflows.
"""

from typing import List, Any
from upsonic.tasks.tasks import Task


class ContextSharing:
    """Handles context sharing and management between tasks in multi-agent workflows."""
    
    @staticmethod
    def enhance_task_context(
        current_task: Task, 
        all_tasks: List[Task], 
        task_index: int, 
        agent_configurations: List[Any], 
        completed_results: List[Task]
    ) -> None:
        """
        Enhance a task's context with all relevant information from the workflow.
        
        Args:
            current_task: The task to enhance
            all_tasks: All tasks in the workflow
            task_index: Index of the current task
            agent_configurations: Available agent configurations
            completed_results: Previously completed tasks with results
        """
        # Initialize context if needed
        if not hasattr(current_task, 'context') or current_task.context is None:
            current_task.context = []
        elif not isinstance(current_task.context, list):
            current_task.context = [current_task.context]
        
        # Add all other tasks to context (excluding the current task itself)
        other_tasks = [task for i, task in enumerate(all_tasks) if i != task_index]
        current_task.context.extend(other_tasks)
        
        # Add agent configurations to context
        current_task.context.extend(agent_configurations)
        
    
    @staticmethod
    def build_selection_context(
        current_task: Task, 
        all_tasks: List[Task], 
        task_index: int, 
        agent_configurations: List[Any], 
        completed_results: List[Task]
    ) -> List[Any]:
        """
        Build context for agent selection process.
        
        Args:
            current_task: The task for which to select an agent
            all_tasks: All tasks in the workflow
            task_index: Index of the current task
            agent_configurations: Available agent configurations
            completed_results: Previously completed tasks with results
            
        Returns:
            List of context items for agent selection
        """
        context = [current_task]  # Current task first
        context += [task for i, task in enumerate(all_tasks) if i != task_index]  # All other tasks
        context += agent_configurations  # Available agents
        
        return context 