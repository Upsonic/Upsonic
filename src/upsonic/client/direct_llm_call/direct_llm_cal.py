from ..agent_configuration.agent_configuration import get_or_create_client, register_tools
from ..tasks.tasks import Task
from typing import Any, Callable, TypeVar, cast

T = TypeVar('T')


class DirectStatic:
    """Static methods for making direct LLM calls using the Upsonic client."""
    
    @staticmethod
    def do(task: Task, model: str | None = None, client: Any = None, debug: bool = False):
        """
        Execute a direct LLM call with the given task and model.
        
        Args:
            task: The task to execute
            model: The LLM model to use (default: "openai/gpt-4")
            client: Optional custom client to use instead of creating a new one
            debug: Whether to enable debug mode
            
        Returns:
            The response from the LLM
        """
        global latest_upsonic_client
        from ..latest_upsonic_client import latest_upsonic_client

        # Use provided client or get/create one
        if client is not None:
            the_client = client
        else:
            the_client = get_or_create_client(debug=debug)
        
        # Register tools if needed
        the_client = register_tools(the_client, task.tools)

        # Execute the direct call
        return the_client.call(task, model)

    @staticmethod
    def print_do(task: Task, model: str | None = None, client: Any = None, debug: bool = False):
        """
        Execute a direct LLM call and print the result.
        
        Args:
            task: The task to execute
            model: The LLM model to use (default: "openai/gpt-4")
            client: Optional custom client to use instead of creating a new one
            debug: Whether to enable debug mode
            
        Returns:
            The response from the LLM
        """
        result = DirectStatic.do(task, model, client, debug)
        print(result)
        return result


class DirectInstance:
    """Instance-based class for making direct LLM calls using the Upsonic client."""
    
    def __init__(self, model: str | None = None, client: Any = None, debug: bool = False):
        """
        Initialize a DirectInstance with specific model and client settings.
        
        Args:
            model: The LLM model to use (default: None)
            client: Optional custom client to use instead of creating a new one
            debug: Whether to enable debug mode
        """
        self.model = model
        self.client = client
        self.debug = debug
    
    def do(self, task: Task, model: str | None = None, client: Any = None, debug: bool = False):
        """
        Execute a direct LLM call using instance defaults or overrides.
        
        Args:
            task: The task to execute
            model: The LLM model to use (overrides instance default if provided)
            client: Optional custom client (overrides instance default if provided)
            debug: Whether to enable debug mode (overrides instance default if provided)
            
        Returns:
            The response from the LLM
        """
        # Use provided parameters or instance defaults
        actual_model = model if model is not None else self.model
        actual_client = client if client is not None else self.client
        actual_debug = debug if debug is not False else self.debug
        
        # Call the static method with the resolved parameters
        return DirectStatic.do(task, actual_model, actual_client, actual_debug)
        
    def print_do(self, task: Task, model: str | None = None, client: Any = None, debug: bool = False):
        """
        Execute a direct LLM call and print the result.
        
        Args:
            task: The task to execute
            model: The LLM model to use (overrides instance default if provided)
            client: Optional custom client (overrides instance default if provided)
            debug: Whether to enable debug mode (overrides instance default if provided)
            
        Returns:
            The response from the LLM
        """
        result = self.do(task, model, client, debug)
        print(result)
        return result


class Direct:
    """
    Router class that provides both static and instance-based approaches for direct LLM calls.
    
    When used without initialization, it provides static methods.
    When initialized with parameters, it returns an instance-based object.
    """
    
    # Static methods that delegate to DirectStatic
    @staticmethod
    def do(task: Task, model: str | None = None, client: Any = None, debug: bool = False):
        return DirectStatic.do(task, model, client, debug)
    
    @staticmethod
    def print_do(task: Task, model: str | None = None, client: Any = None, debug: bool = False):
        return DirectStatic.print_do(task, model, client, debug)
    
    def __new__(cls, model: str | None = None, client: Any = None, debug: bool = False):
        """
        Factory method that returns a DirectInstance object when initialized.
        
        Args:
            model: The LLM model to use (default: None)
            client: Optional custom client to use instead of creating a new one
            debug: Whether to enable debug mode
            
        Returns:
            A DirectInstance object
        """
        return DirectInstance(model, client, debug)
