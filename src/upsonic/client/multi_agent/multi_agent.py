from ..agent_configuration.agent_configuration import get_or_create_client, register_tools
from ..tasks.tasks import Task
from ..agent_configuration.agent_configuration import AgentConfiguration

from typing import Any

class MultiAgent:
    """A callable class for making direct LLM calls using the Upsonic client."""
    
    @staticmethod
    def do(agents: list[Any], tasks: list[Task], llm_model: str | None = None):
        """
        Execute a direct LLM call with the given task and model.
        
        Args:
            task: The task to execute
            llm_model: The LLM model to use (default: "openai/gpt-4")
            
        Returns:
            The response from the LLM
        """
        global latest_upsonic_client
        from ..latest_upsonic_client import latest_upsonic_client

        # Get or create client
        the_client = get_or_create_client()
        
        # Register tools if needed
        for task in tasks:
            the_client = register_tools(the_client, task.tools)
        
        # Update the global client reference
        if latest_upsonic_client is None:
            latest_upsonic_client = the_client

        # Execute the direct call
        return the_client.multi_agent(agents, tasks, llm_model)
    

    @staticmethod
    def print_do(agents: list[Any], tasks: list[Task], llm_model: str | None = None):
        result = MultiAgent.do(agents, tasks, llm_model)
        print(result)
        return result
