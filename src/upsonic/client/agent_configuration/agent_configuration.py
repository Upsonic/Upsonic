from dataclasses import Field
import uuid
from pydantic import BaseModel


from typing import Any, List, Dict, Optional, Type, Union


from ..knowledge_base.knowledge_base import KnowledgeBase
from ..tasks.tasks import Task


def register_tools(client, tools):
    """Register tools with the client."""
    if tools is not None:
        for tool in tools:
            # Handle special tool classes from upsonic.client.tools
            if tool.__module__ == 'upsonic.client.tools':
                print("SPECIAL TOOL REGISTERED:", tool.__name__)
                client.tool()(tool)
                continue
                
            # If tool is a class (not an instance)
            if isinstance(tool, type):
                if hasattr(tool, 'command') and hasattr(tool, 'args'):
                    print("MCP TOOL REGISTERED:", tool.__name__)
                    client.mcp()(tool)
                else:
                    print("CLASS TOOL REGISTERED:", tool.__name__)
                    client.tool()(tool)
            else:
                # Get all attributes of the tool instance/object
                tool_attrs = dir(tool)
                
                # Filter out special methods and get only callable attributes
                functions = [attr for attr in tool_attrs 
                           if not attr.startswith('__') and callable(getattr(tool, attr))]
                
                if functions:
                    # If the tool has functions, use the tool() decorator
                    print("INSTANCE TOOL REGISTERED:", tool.__class__.__name__)
                    client.tool()(tool.__class__)
                else:
                    # If the tool has no functions, use mcp()
                    print("INSTANCE MCP REGISTERED:", tool.__class__.__name__)
                    client.mcp()(tool.__class__)
    return client


def get_or_create_client():
    """Get existing client or create a new one."""
    from ..latest_upsonic_client import latest_upsonic_client
    
    if latest_upsonic_client is not None:
        return latest_upsonic_client
    
    from ..base import UpsonicClient
    return UpsonicClient("localserver")


def execute_task(agent_config, task: Task):
    """Execute a task with the given agent configuration."""
    global latest_upsonic_client
    from ..latest_upsonic_client import latest_upsonic_client

    # Get or create client
    the_client = get_or_create_client()
    
    # Register tools if needed
    the_client = register_tools(the_client, task.tools)
    
    # Update the global client reference
    if latest_upsonic_client is None:
        latest_upsonic_client = the_client

    the_client.run(agent_config, task)
    return task.response


class AgentConfiguration(BaseModel):
    agent_id_: str = None
    job_title: str
    company_url: str = None
    company_objective: str = None
    name: str = ""
    contact: str = ""
    model: str = "openai/gpt-4o"

    def __init__(self, job_title: str = None, **data):
        if job_title is not None:
            data["job_title"] = job_title
        super().__init__(**data)

    sub_task: bool = True
    reflection: bool = False
    memory: bool = False
    caching: bool = True
    cache_expiry: int = 60 * 60
    knowledge_base: KnowledgeBase = None
    tools: List[Any] = []
    context_compress: bool = True

    @property
    def retries(self):
        if self.reflection:
            return 5
        else:
            return 2

    @property
    def agent_id(self):
        if self.agent_id_ is None:
            self.agent_id_ = str(uuid.uuid4())
        return self.agent_id_
    
    def do(self, task: Task):
        return execute_task(self, task)
    
    def print_do(self, task: Task):
        result = self.do(task)
        print(result)
        return result
