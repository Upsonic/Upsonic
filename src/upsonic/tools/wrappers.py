"""Tool wrapper implementations."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from upsonic.tools.base import ToolBase, ToolSchema, ToolMetadata
from upsonic.tools.config import ToolConfig
from upsonic.tools.schema import FunctionSchema

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task


class FunctionTool(ToolBase):
    """Wrapper for function-based tools."""
    
    def __init__(
        self,
        function: Callable,
        schema: FunctionSchema,
        config: Optional[ToolConfig] = None
    ):
        self.function = function
        self.function_schema = schema
        self.config = config or ToolConfig()
        
        # Convert function schema to tool schema
        tool_schema = ToolSchema(
            parameters=schema.parameters_schema,
            return_type=schema.return_schema,
            strict=config.strict if config.strict is not None else False
        )
        
        # Create metadata
        metadata = ToolMetadata(
            name=schema.name,
            description=schema.description
        )
        
        super().__init__(
            name=schema.name,
            description=schema.description,
            schema=tool_schema,
            metadata=metadata
        )
        
        self.is_async = schema.is_async
        self.takes_ctx = schema.takes_ctx
    
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool function."""
        if self.is_async:
            return await self.function(*args, **kwargs)
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.function(*args, **kwargs)
            )


class AgentTool(ToolBase):
    """Wrapper for agent-based tools."""
    
    def __init__(self, agent: Any):
        self.agent = agent
        
        # Generate tool name and description
        agent_name = getattr(agent, 'name', None) or f"Agent_{id(agent)}"
        agent_role = getattr(agent, 'role', None)
        agent_goal = getattr(agent, 'goal', None)
        system_prompt = getattr(agent, 'system_prompt', None)
        
        # Create method name
        method_name = f"ask_{self._sanitize_name(agent_name)}"
        
        # Create description
        description_parts = [f"Delegate tasks to {agent_name}"]
        if agent_role:
            description_parts.append(f"Role: {agent_role}")
        if agent_goal:
            description_parts.append(f"Goal: {agent_goal}")
        if system_prompt:
            description_parts.append(f"Specialty: {system_prompt[:100]}...")
        
        description = ". ".join(description_parts)
        
        # Create schema
        schema = ToolSchema(
            parameters={
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The task or question to delegate to the agent"
                    }
                },
                "required": ["request"]
            }
        )
        
        super().__init__(
            name=method_name,
            description=description,
            schema=schema
        )
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use as method name."""
        import re
        # Remove non-alphanumeric characters and convert to snake_case
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name)
        return name.lower().strip('_')
    
    async def execute(self, request: str, **kwargs: Any) -> Any:
        """Execute the agent with the given request."""
        # Import here to avoid circular imports
        from upsonic.tasks.tasks import Task
        
        # Create task for the agent
        task = Task(description=request)
        
        # Execute based on agent capabilities
        if hasattr(self.agent, 'do_async'):
            result = await self.agent.do_async(task)
        elif hasattr(self.agent, 'do'):
            # Run sync method in executor
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.agent.do(task)
            )
        else:
            raise AttributeError(f"Agent {self.agent} has no do or do_async method")
        
        # Convert result to string if needed
        return str(result) if result is not None else "No response from agent"


class MethodTool(ToolBase):
    """Wrapper for class method tools."""
    
    def __init__(
        self,
        instance: Any,
        method: Callable,
        schema: FunctionSchema,
        config: Optional[ToolConfig] = None
    ):
        self.instance = instance
        self.method = method
        self.function_schema = schema
        self.config = config or ToolConfig()
        
        # Convert function schema to tool schema
        tool_schema = ToolSchema(
            parameters=schema.parameters_schema,
            return_type=schema.return_schema,
            strict=config.strict if config.strict is not None else False
        )
        
        super().__init__(
            name=schema.name,
            description=schema.description,
            schema=tool_schema
        )
        
        self.is_async = schema.is_async
    
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the method."""
        if self.is_async:
            return await self.method(*args, **kwargs)
        else:
            # Run sync method in executor
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.method(*args, **kwargs)
            )
