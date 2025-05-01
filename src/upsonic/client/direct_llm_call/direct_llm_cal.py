from ..agent_configuration.agent_configuration import get_or_create_client, register_tools
from ..tasks.tasks import Task
from typing import Any, Callable, TypeVar, cast

T = TypeVar('T')

from ...model_registry import ModelNames
from ..printing import print_price_id_summary

class DirectStatic:
    """Static methods for making direct LLM calls using the Upsonic client."""
    
    @staticmethod
    def do(task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int = 3):
        """
        Execute a direct LLM call with the given task and model.
        
        Args:
            task: The task to execute
            model: The LLM model to use (default: "openai/gpt-4")
            client: Optional custom client to use instead of creating a new one
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls (default: 3)
            
        Returns:
            The response from the LLM
        """
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    DirectStatic.do_async(task, model, client, debug, retry), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(DirectStatic.do_async(task, model, client, debug, retry))

    @staticmethod
    async def do_async(task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int = 3):
        """
        Execute a direct LLM call with the given task and model asynchronously.
        
        Args:
            task: The task to execute
            model: The LLM model to use (default: "openai/gpt-4")
            client: Optional custom client to use instead of creating a new one
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls (default: 3)
            
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

        # Execute the direct call asynchronously with retry parameter
        await the_client.call_async(task, model, retry=retry)
        
        # Print the price ID summary if the task has a price ID
        if not task.not_main_task:
            print_price_id_summary(task.price_id, task)
            
        return task.response

    @staticmethod
    def print_do(task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int = 3):
        """
        Execute a direct LLM call and print the result.
        
        Args:
            task: The task to execute
            model: The LLM model to use (default: "openai/gpt-4")
            client: Optional custom client to use instead of creating a new one
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls (default: 3)
            
        Returns:
            The response from the LLM
        """
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    DirectStatic.print_do_async(task, model, client, debug, retry), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(DirectStatic.print_do_async(task, model, client, debug, retry))

    @staticmethod
    async def print_do_async(task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int = 3):
        """
        Execute a direct LLM call and print the result asynchronously.
        
        Args:
            task: The task to execute
            model: The LLM model to use (default: "openai/gpt-4")
            client: Optional custom client to use instead of creating a new one
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls (default: 3)
            
        Returns:
            The response from the LLM
        """
        result = await DirectStatic.do_async(task, model, client, debug, retry)
        print(result)
        return result


class DirectInstance:
    """Instance-based class for making direct LLM calls using the Upsonic client."""
    
    def __init__(self, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int = 3):
        """
        Initialize a DirectInstance with specific model and client settings.
        
        Args:
            model: The LLM model to use (default: None)
            client: Optional custom client to use instead of creating a new one
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls (default: 3)
        """
        self.model = model
        self.client = client
        self.debug = debug
        self.retry = retry
    
    def do(self, task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int | None = None):
        """
        Execute a direct LLM call using instance defaults or overrides.
        
        Args:
            task: The task to execute
            model: The LLM model to use (overrides instance default if provided)
            client: Optional custom client (overrides instance default if provided)
            debug: Whether to enable debug mode (overrides instance default if provided)
            retry: Number of retries for failed calls (overrides instance default if provided)
            
        Returns:
            The response from the LLM
        """
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.do_async(task, model, client, debug, retry), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.do_async(task, model, client, debug, retry))

    async def do_async(self, task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int | None = None):
        """
        Execute a direct LLM call using instance defaults or overrides asynchronously.
        
        Args:
            task: The task to execute
            model: The LLM model to use (overrides instance default if provided)
            client: Optional custom client (overrides instance default if provided)
            debug: Whether to enable debug mode (overrides instance default if provided)
            retry: Number of retries for failed calls (overrides instance default if provided)
            
        Returns:
            The response from the LLM
        """
        # Use provided parameters or instance defaults
        actual_model = model if model is not None else self.model
        actual_client = client if client is not None else self.client
        actual_debug = debug if debug is not False else self.debug
        actual_retry = retry if retry is not None else self.retry
        
        # Call the static method with the resolved parameters
        result = await DirectStatic.do_async(task, actual_model, actual_client, actual_debug, actual_retry)
        
        # No need to print price_id summary here since DirectStatic.do_async already does it
        return result
        
    def print_do(self, task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int | None = None):
        """
        Execute a direct LLM call and print the result.
        
        Args:
            task: The task to execute
            model: The LLM model to use (overrides instance default if provided)
            client: Optional custom client (overrides instance default if provided)
            debug: Whether to enable debug mode (overrides instance default if provided)
            retry: Number of retries for failed calls (overrides instance default if provided)
            
        Returns:
            The response from the LLM
        """
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.print_do_async(task, model, client, debug, retry), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.print_do_async(task, model, client, debug, retry))

    async def print_do_async(self, task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int | None = None):
        """
        Execute a direct LLM call and print the result asynchronously.
        
        Args:
            task: The task to execute
            model: The LLM model to use (overrides instance default if provided)
            client: Optional custom client (overrides instance default if provided)
            debug: Whether to enable debug mode (overrides instance default if provided)
            retry: Number of retries for failed calls (overrides instance default if provided)
            
        Returns:
            The response from the LLM
        """
        result = await self.do_async(task, model, client, debug, retry)
        print(result)
        return result


class Direct:
    """
    Router class that provides both static and instance-based approaches for direct LLM calls.
    
    When used without initialization, it provides static methods.
    When initialized with parameters, it returns an instance-based object.
    
    Example:
        # Correct usage with named model parameter:
        direct = Direct(model="openai/gpt-4o")
        direct = Direct(model="claude/claude-3-5-sonnet")
        
        # Incorrect usage:
        # direct = Direct("openai/gpt-4o")  # Wrong! Must use model=
        # direct = Direct("Researcher Direct")  # Wrong!
        
        # For agent-based operations, use Agent instead:
        # agent = Agent("Researcher Agent")  # Correct!
    """
    
    # Static methods that delegate to DirectStatic
    @staticmethod
    def do(task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int = 3):
        return DirectStatic.do(task, model, client, debug, retry)
    
    @staticmethod
    def print_do(task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int = 3):
        return DirectStatic.print_do(task, model, client, debug, retry)

    @staticmethod
    async def do_async(task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int = 3):
        return await DirectStatic.do_async(task, model, client, debug, retry)
    
    @staticmethod
    async def print_do_async(task: Task, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int = 3):
        return await DirectStatic.print_do_async(task, model, client, debug, retry)
    
    def __new__(cls, *args, model: ModelNames | None = None, client: Any = None, debug: bool = False, retry: int = 3):
        """
        Factory method that returns a DirectInstance object when initialized.
        
        Args:
            model: The LLM model to use (default: None)
            client: Optional custom client to use instead of creating a new one
            debug: Whether to enable debug mode
            retry: Number of retries for failed calls (default: 3)
            
        Returns:
            A DirectInstance object
            
        Raises:
            ValueError: If positional arguments are provided instead of using the named parameter 'model'
        """
        if args:
            raise ValueError(
                "Direct() does not accept positional arguments. Use named parameter 'model' instead.\n"
                "Example: Direct(model='openai/gpt-4o') instead of Direct('openai/gpt-4o')"
            )
            
        return DirectInstance(model, client, debug, retry)
