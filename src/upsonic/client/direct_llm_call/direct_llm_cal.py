from ..agent_configuration.agent_configuration import get_or_create_client, register_tools
from ..tasks.tasks import Task
from typing import Any


class Direct:
    """A callable class for making direct LLM calls using the Upsonic client."""
    
    @staticmethod
    def do(task: Task, model: str | None = None, client: Any = None):
        """
        Execute a direct LLM call with the given task and model.
        
        Args:
            task: The task to execute
            model: The LLM model to use (default: "openai/gpt-4")
            client: Optional custom client to use instead of creating a new one
            
        Returns:
            The response from the LLM
        """
        global latest_upsonic_client
        from ..latest_upsonic_client import latest_upsonic_client

        # Use provided client or get/create one
        if client is not None:
            the_client = client
        else:
            the_client = get_or_create_client()
        
        # Register tools if needed
        the_client = register_tools(the_client, task.tools)

        # Execute the direct call
        return the_client.call(task, model)
    

    @staticmethod
    def print_do(task: Task, model: str | None = None, client: Any = None):
        result = Direct.do(task, model, client)
        print(result)
        return result
