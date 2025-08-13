from __future__ import annotations
from typing import TYPE_CHECKING, List, Callable, Optional, Dict, Any, Union, Type
from pydantic import BaseModel

if TYPE_CHECKING:
    from upsonic.agent.agent import Direct as Agent
    from upsonic.storage.memory.memory import Memory
    
from upsonic.tasks.tasks import Task

class DelegationManager:
    """
    Manages the core mechanics of task delegation from a leader to a member.

    This class is responsible for generating the 'delegate_task' tool and handling
    the stateful execution of sub-tasks within a shared memory context.
    """
    def __init__(self, members: List[Agent], tool_mapping: Dict[str, Callable]):
        """
        Initializes the DelegationManager.

        Args:
            members (List[Agent]): The list of member agents available for delegation.
            tool_mapping (Dict[str, Callable]): A mapping from tool names to their callable function objects.
        """
        self.members = members
        self.tool_mapping = tool_mapping

    def get_delegation_tool(self, session_memory: Memory) -> Callable:
        """
        Dynamically generates the 'delegate_task' tool for the Team Leader.
        
        This tool allows the leader to delegate a specific task to a member agent.
        It ensures that the conversational context is maintained by temporarily
        assigning a shared memory object to the member agent for the duration
        of the sub-task.
        
        Args:
            session_memory: The shared Memory object for the coordination session.

        Returns:
            An asynchronous callable function that serves as the delegation tool.
        """
        async def delegate_task(
            member_id: str, 
            description: str, 
            tools: Optional[List[str]] = None, 
            context: Any = None, 
            attachments: Optional[List[str]] = None,
            expected_output: Union[Type[BaseModel], type[str], None] = str
        ) -> str:
            """
            Delegates a task to a specific team member using detailed parameters.

            Args:
                member_id (str): The unique ID of the team member to delegate the task to.
                description (str): A clear and concise description of the task for the member to execute.
                tools (Optional[List[str]]): A list of tool names the member might need.
                context (Any): Optional context or data needed for the task.
                attachments (Optional[List[str]]): Optional list of file paths for the task.
                expected_output (Union[Type[BaseModel], type[str], None]): The expected output type for the task.

            Returns:
                str: The result from the team member's execution of the task.
            """
            member_agent = None
            for agent in self.members:
                if agent.get_agent_id() == member_id:
                    member_agent = agent
                    break
            
            if not member_agent:
                return f"Error: Team member with ID '{member_id}' not found. Please use a valid member ID."
            
            sub_task_tools = []
            if tools:
                for tool_name in tools:
                    if callable_tool := self.tool_mapping.get(tool_name):
                        sub_task_tools.append(callable_tool)

            sub_task = Task(
                description=description,
                tools=sub_task_tools,
                context=context,
                attachments=attachments,
                response_format=expected_output
            )
            original_memory = member_agent.memory
            try:
                member_agent.memory = session_memory
                await member_agent.do_async(sub_task)
                return sub_task.response or "The team member did not return a result."
            except Exception as e:
                return f"An error occurred while delegating task to {member_id}: {e}"
            finally:
                member_agent.memory = original_memory

        return delegate_task