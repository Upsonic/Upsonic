from ..agent_configuration.agent_configuration import get_or_create_client, register_tools
from ..tasks.tasks import Task


class Direct:
    """A callable class for making direct LLM calls using the Upsonic client."""
    
    @staticmethod
    def do(task: Task, model: str | None = None):
        """
        Execute a direct LLM call with the given task and model.
        
        Args:
            task: The task to execute
            model: The LLM model to use (default: "openai/gpt-4")
            
        Returns:
            The response from the LLM
        """
        global latest_upsonic_client
        from ..latest_upsonic_client import latest_upsonic_client

        # Get or create client
        the_client = get_or_create_client()
        
        # Register tools if needed
        the_client = register_tools(the_client, task.tools)
        
        # Update the global client reference
        if latest_upsonic_client is None:
            latest_upsonic_client = the_client

        # Execute the direct call
        return the_client.call(task, model)
    

    @staticmethod
    def print_do(task: Task, model: str | None = None):
        result = Direct.do(task, model)
        print(result)
        return result
