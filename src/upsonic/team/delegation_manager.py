from __future__ import annotations
from typing import TYPE_CHECKING, List, Callable, Optional, Dict, Any, Union, Type
from pydantic import BaseModel

if TYPE_CHECKING:
    from upsonic.agent.agent import Agent
    from upsonic.team.team import Team

from upsonic.tasks.tasks import Task
from upsonic.tools.config import ToolConfig, tool

class DelegationManager:
    """
    Manages the core mechanics of task delegation from a leader to a member.

    This class is responsible for generating the 'delegate_task' tool and handling
    the execution of sub-tasks. Task-level tools are forwarded to the sub-task so
    the member agent has them available alongside its own pre-configured tools.
    The agent autonomously decides which tools to use.
    """
    def __init__(self, members: List[Union[Agent, Team]], tool_mapping: Dict[str, Callable], debug: bool = False):
        """
        Initializes the DelegationManager.

        Args:
            members: The list of member entities (Agent or Team) available for delegation.
            tool_mapping: A mapping from tool names to their callable objects,
                          derived from the Task-level tools defined by the user.
            debug: Whether to emit info_log messages for delegation actions.
        """
        self.members: List[Union[Agent, Team]] = members
        self.tool_mapping: Dict[str, Callable] = tool_mapping
        self.debug: bool = debug
        self.routed_entity: Optional[Union[Agent, Team]] = None

    def get_delegation_tool(self) -> Callable:
        """
        Dynamically generates the 'delegate_task' tool for the Team Leader.
        
        This tool allows the leader to delegate a specific task to a member entity.
        Task-level tools (referenced by name) are attached to the sub-task so the
        member agent can use them. Only the leader holds memory for coordination.

        Returns:
            An asynchronous callable function that serves as the delegation tool.
        """
        async def delegate_task(
            member_id: str,
            description: str,
            tools: Optional[List[str]] = None,
            context: Any = None,
            attachments: Optional[List[str]] = None,
            expected_output: Union[Type[BaseModel], type[str], None] = None,
        ) -> str:
            """
            Delegates a task to a specific team member using detailed parameters.

            Args:
                member_id (str): The unique ID of the team member to delegate the task to.
                description (str): A clear description of the task objective and the expected output.
                tools (Optional[List[str]]): A list of task-level tool names to make available for this sub-task. The agent will autonomously decide which tools to use.
                context (Any): Optional context or data needed for the task, such as a result from a previous delegation step.
                attachments (Optional[List[str]]): Optional list of file paths for the task.
                expected_output (Union[Type[BaseModel], type[str], None]): The expected output type for the task. None means plain str.

            Returns:
                str: The result from the team member's execution of the task.
            """
            response_format: Union[Type[BaseModel], type[str], None] = str if expected_output is None else expected_output
            member_entity: Optional[Union[Agent, Team]] = None
            for entity in self.members:
                if entity.get_entity_id() == member_id:
                    member_entity = entity
                    break
            
            if not member_entity:
                return f"Error: Team member with ID '{member_id}' not found. Please use a valid member ID."

            sub_task_tools: List[Callable] = []
            if tools:
                for tool_name in tools:
                    if callable_tool := self.tool_mapping.get(tool_name):
                        sub_task_tools.append(callable_tool)

            sub_task = Task(
                description=description,
                tools=sub_task_tools if sub_task_tools else None,
                context=context,
                attachments=attachments,
                response_format=response_format,
            )

            if self.debug:
                _etype: str = "Team" if hasattr(member_entity, "entities") else "Agent"
                from upsonic.utils.printing import info_log
                info_log(
                    f"Coordinate mode — Delegating to {_etype} '{member_id}' | "
                    f"Description: {description[:200]} | "
                    f"Tools: {tools} | "
                    f"Context: {str(context)[:200] if context else None} | "
                    f"Attachments: {attachments}",
                    context="Team",
                )

            try:
                await member_entity.do_async(sub_task)
                return sub_task.response or "The team member did not return a result."
            except Exception as e:
                return f"An error occurred while delegating task to {member_id}: {e}"

        # Disable timeout — delegate_task runs a full sub-agent pipeline which
        # can take far longer than the default 30-second tool timeout.
        delegate_task._upsonic_tool_config = ToolConfig(timeout=None, max_retries=0)
        delegate_task._upsonic_is_tool = True
        return delegate_task
    
    def get_routing_tool(self) -> Callable:
        """
        Generates the 'route_request_to_member' tool for the 'route' mode.
        This tool selects an entity (Agent or Team) to handle the request.
        """
        async def route_request_to_member(member_id: str) -> str:
            """
            Selects the single best member to handle the user's entire request and ends the routing process.

            Args:
                member_id (str): The unique ID of the team member you have chosen to handle the request.

            Returns:
                str: A confirmation message indicating the request has been routed.
            """
            chosen_entity: Optional[Union[Agent, Team]] = None
            for entity in self.members:
                if entity.get_entity_id() == member_id:
                    chosen_entity = entity
                    break
            
            if not chosen_entity:
                return f"Error: Could not route to member with ID '{member_id}'. The ID is invalid. Please choose a valid ID from your team roster."
            self.routed_entity = chosen_entity
            return f"Request successfully routed to member '{member_id}'. Handoff complete."
        return route_request_to_member
