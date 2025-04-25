from dataclasses import Field
import uuid
from pydantic import BaseModel
import asyncio
import subprocess
import sys

from typing import Any, List, Dict, Optional, Type, Union

from ..knowledge_base.knowledge_base import KnowledgeBase
from ..tasks.tasks import Task
from ..printing import mcp_tool_operation, tool_operation, error_message

from ..latest_upsonic_client import latest_upsonic_client
from ...model_registry import ModelNames


def register_tools(client, tools):
    """Register tools with the client."""
    if tools is not None:
        for tool in tools:
            # Handle special tool classes from upsonic.client.tools
            if tool.__module__ == 'upsonic.client.tools':
                client.tool()(tool)
                continue
                
            # If tool is a class (not an instance)
            if isinstance(tool, type):
                if hasattr(tool, 'command'):
                    # Check if command is UVX and UV is not installed
                    if hasattr(tool, 'command') and tool.command == 'uvx':
                        try:
                            # Try to run uv --version to check if it's installed
                            subprocess.run(['uv', '--version'], capture_output=True, check=True)
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            error_message(
                                "UV Installation Error",
                                "UV is not installed. Please install UV",
                                error_code=500
                            )
                            sys.exit(1)
                    
                    # Check if command is NPX and Node.js is not installed
                    if hasattr(tool, 'command') and tool.command == 'npx':
                        try:
                            # Try to run node --version to check if Node.js is installed
                            subprocess.run(['node', '--version'], capture_output=True, check=True)
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            error_message(
                                "Node.js Installation Error",
                                "Node.js is not installed. Please install Node.js to use tools with NPX command.",
                                error_code=500
                            )
                            sys.exit(1)

                    client.mcp()(tool)
                elif hasattr(tool, 'url'):
                    client.sse_mcp()(tool)
                else:
                    client.tool()(tool)
            else:
                # Get all attributes of the tool instance/object
                tool_attrs = dir(tool)
                
                # Filter out special methods and get only callable attributes
                functions = [attr for attr in tool_attrs 
                           if not attr.startswith('__') and callable(getattr(tool, attr))]
                
                if functions:
                    # If the tool has functions, use the tool() decorator
                    tool_operation(f"Tool: {tool.__class__.__name__}", "Successfully Registered")

                    if not isinstance(tool, object):
                        client.tool()(tool.__class__)
                    else:
                        client.tool()(tool)
                else:
                    # If the tool has no functions, use mcp()
                    mcp_tool_operation(f"MCP Tool: {tool.__class__.__name__}", "Successfully Registered")
                    client.mcp()(tool.__class__)
    return client


def get_or_create_client(debug: bool = False):
    """Get existing client or create a new one."""
    
    global latest_upsonic_client
    
    if latest_upsonic_client is not None:
        # Check if the existing client's status is False
        if not latest_upsonic_client.status():
            from ..base import UpsonicClient
            new_client = UpsonicClient("localserver", debug=debug)
            latest_upsonic_client = new_client
        return latest_upsonic_client
    
    from ..base import UpsonicClient
    the_client = UpsonicClient("localserver", debug=debug)
    latest_upsonic_client = the_client
    return the_client


def execute_task(agent_config, task: Task, debug: bool = False):
    """Execute a task with the given agent configuration."""
    import asyncio
    
    try:
        # Check if there's a running event loop
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If there's a running loop, run the coroutine in that loop
            return asyncio.run_coroutine_threadsafe(
                execute_task_async(agent_config, task, debug), 
                loop
            ).result()
    except RuntimeError:
        # No running event loop
        pass
    
    # If no running loop or exception occurred, create a new one
    return asyncio.run(execute_task_async(agent_config, task, debug))

async def execute_task_async(agent_config, task: Task, debug: bool = False):
    """Execute a task with the given agent configuration asynchronously using true async methods."""
    global latest_upsonic_client
    
    # If agent has a custom client, use it
    if hasattr(agent_config, 'client') and agent_config.client is not None:
        the_client = agent_config.client
    else:
        # Get or create client using existing process
        the_client = get_or_create_client(debug=debug)
    
    # If task has no tools defined but agent has tools, use the agent's tools
    if not task.tools and hasattr(agent_config, 'tools') and agent_config.tools:
        task.tools = agent_config.tools
    
    # Register tools if needed
    the_client = register_tools(the_client, task.tools)
    
    # Use the async run method directly
    await the_client.run_async(agent_config, task)
    
    return task.response


class AgentConfiguration(BaseModel):


    agent_id_: Optional[str] = None
    job_title: str
    company_url: Optional[str] = None
    company_objective: Optional[str] = None
    name: str = ""
    contact: str = ""
    model: str = "openai/gpt-4o"
    client: Any = None  # Add client parameter
    debug: bool = False
    reliability_layer: Any = None  # Changed to Any to accept any class or instance
    system_prompt: Optional[str] = None
    tools: List[Any] = []
    retry: int = 3


    sub_task: bool = True
    reflection: bool = False
    memory: bool = False
    caching: bool = True
    cache_expiry: int = 60 * 60
    knowledge_base: Optional[KnowledgeBase] = None
    context_compress: bool = False

    def __init__(
        self, 
        job_title: str, 
        company_url: Optional[str] = None, 
        company_objective: Optional[str] = None,
        name: str = "",
        contact: str = "",
        model: ModelNames = "openai/gpt-4o",
        client: Any = None,
        debug: bool = False,
        reliability_layer: Any = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        sub_task: bool = True,
        reflection: bool = False,
        memory: bool = False,
        caching: bool = True,
        cache_expiry: int = 60 * 60,
        knowledge_base: Optional[KnowledgeBase] = None,
        context_compress: bool = False,
        agent_id_: Optional[str] = None,
        retry: int = 3,
        **data
    ):
        if job_title is not None:
            data["job_title"] = job_title
        if client is not None:
            data["client"] = client
        
        if tools is None:
            tools = []
            
        data.update({
            "agent_id_": agent_id_,
            "company_url": company_url,
            "company_objective": company_objective,
            "name": name,
            "contact": contact,
            "model": model,
            "debug": debug,
            "reliability_layer": reliability_layer,
            "system_prompt": system_prompt,
            "tools": tools,
            "sub_task": sub_task,
            "retry": retry,
            "reflection": reflection,
            "memory": memory,
            "caching": caching,
            "cache_expiry": cache_expiry,
            "knowledge_base": knowledge_base,
            "context_compress": context_compress
        })

        super().__init__(**data)
        self.validate_tools()

    def validate_tools(self):
        """
        Validates each tool in the tools list.
        If a tool is a class and has a __control__ method, runs that method to verify it returns True.
        Raises an exception if the __control__ method returns False or raises an exception.
        """
        if not self.tools:
            return
            
        for tool in self.tools:
            # Check if the tool is a class
            if isinstance(tool, type) or hasattr(tool, '__class__'):
                # Check if the class has a __control__ method
                if hasattr(tool, '__control__') and callable(getattr(tool, '__control__')):
                    try:
                        # Run the __control__ method
                        control_result = tool.__control__()
                        if not control_result:
                            raise ValueError(f"Tool {tool} __control__ method returned False")
                    except Exception as e:
                        # Re-raise any exceptions from the __control__ method
                        raise ValueError(f"Error validating tool {tool}: {str(e)}")


    @property
    def agent_id(self):
        if self.agent_id_ is None:
            self.agent_id_ = str(uuid.uuid4())
        return self.agent_id_
    
    def do(self, task: Task):
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.do_async(task), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.do_async(task))
    
    async def do_async(self, task: Task):
        """Asynchronous version of the do method."""
        return await execute_task_async(self, task, self.debug)
    
    def print_do(self, task: Task):
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                result = asyncio.run_coroutine_threadsafe(
                    self.print_do_async(task), 
                    loop
                ).result()
                return result
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.print_do_async(task))
        
    async def print_do_async(self, task: Task):
        """Asynchronous version of the print_do method."""
        result = await self.do_async(task)
        print(result)
        return result
    
    def parallel_do(self, tasks: List[Task]):
        """Execute multiple tasks in parallel and return their results.
        
        Args:
            tasks: A list of Task objects to execute in parallel
            
        Returns:
            A list of task responses in the same order as the input tasks
        """
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.parallel_do_async(tasks), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.parallel_do_async(tasks))
    
    async def parallel_do_async(self, tasks: List[Task]):
        """Asynchronous version of the parallel_do method.
        
        Args:
            tasks: A list of Task objects to execute in parallel
            
        Returns:
            A list of task responses in the same order as the input tasks
        """
        # Create a list of coroutines for each task
        coroutines = [self.do_async(task) for task in tasks]
        
        # Execute all tasks in parallel and return their results
        return await asyncio.gather(*coroutines)
    
    def parallel_print_do(self, tasks: List[Task]):
        """Execute multiple tasks in parallel, print their results, and return them.
        
        Args:
            tasks: A list of Task objects to execute in parallel
            
        Returns:
            A list of task responses in the same order as the input tasks
        """
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.parallel_print_do_async(tasks), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.parallel_print_do_async(tasks))
    
    async def parallel_print_do_async(self, tasks: List[Task]):
        """Asynchronous version of the parallel_print_do method.
        
        Args:
            tasks: A list of Task objects to execute in parallel
            
        Returns:
            A list of task responses in the same order as the input tasks
        """
        # Execute all tasks in parallel
        results = await self.parallel_do_async(tasks)
        
        # Print each result
        for result in results:
            print(result)
        
        return results
