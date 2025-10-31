"""
Task assignment module for selecting appropriate agents for tasks in multi-agent workflows.
"""

from pydantic import BaseModel

from typing import List, Any, Optional, Dict
from upsonic.tasks.tasks import Task

from upsonic.agent.agent import Agent


class TaskAssignment:
    """Handles task assignment and agent selection in multi-agent workflows."""
    
    def __init__(self):
        """Initialize the task assignment handler."""
        pass
    
    def prepare_agents_registry(self, agent_configurations: List[Any]) -> tuple[Dict[str, Any], List[str]]:
        """
        Prepare a registry of agents indexed by their names.
        
        Args:
            agent_configurations: List of agent configurations
            
        Returns:
            Tuple of (agents_dict, agent_names_list)
        """
        agents_registry = {}
        
        for agent in agent_configurations:
            agent_name = agent.get_agent_id()
            agents_registry[agent_name] = agent
        
        agent_names = list(agents_registry.keys())
        return agents_registry, agent_names
    
    async def select_agent_for_task(
        self, 
        current_task: Task, 
        context: List[Any], 
        agents_registry: Dict[str, Any], 
        agent_names: List[str], 
        agent_configurations: List[Any]
    ) -> Optional[str]:
        """
        Select the most appropriate agent for a given task.
        
        Args:
            current_task: The task that needs an agent
            context: Context for agent selection
            agents_registry: Dictionary of available agents
            agent_names: List of agent names
            agent_configurations: Original agent configurations
            
        Returns:
            Selected agent name or None if selection fails
        """
        if current_task.agent is not None:
            for agent_name, agent_instance in agents_registry.items():
                if agent_instance == current_task.agent:
                    return agent_name
            
            predefined_agent_id = getattr(current_task.agent, 'get_agent_id', lambda: None)()
            if predefined_agent_id and predefined_agent_id in agents_registry:
                return predefined_agent_id
        
        class SelectedAgent(BaseModel):
            selected_agent: str
        
        max_attempts = 3
        attempts = 0
        if not agent_configurations or not hasattr(agent_configurations[0], 'model'):
            raise ValueError("Cannot perform agent selection: The first agent in the team must have a valid model.")
        
        selection_model = agent_configurations[0].model

        while attempts < max_attempts:
            selecting_task = Task(
                description=f"Select the most appropriate agent from the available agents to handle the current task. Consider all tasks in the workflow and previous results to make the best choice. Return only the exact agent name from the list.",
                attachments=current_task.attachments, 
                response_format=SelectedAgent, 
                context=context
            )
            
            await Agent(model=selection_model).do_async(selecting_task)

            if not isinstance(selecting_task.response, SelectedAgent):
                attempts += 1
                continue

            selected_name = selecting_task.response.selected_agent
            
            if selected_name in agents_registry:
                return selected_name
            
            for agent_name in agent_names:
                if (agent_name.lower() in selected_name.lower() or 
                    selected_name.lower() in agent_name.lower()):
                    return agent_name
            
            attempts += 1
        
        return agent_names[0] if agent_names else None 