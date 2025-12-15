from upsonic.tasks.tasks import Task
from typing import Any, List, Dict, Optional, Type, Union, Literal

from upsonic.agent.agent import Agent
from upsonic.context.task import turn_task_to_string


try:
    from upsonic.storage import Memory, InMemoryStorage
except ImportError:
    Memory = None
    InMemoryStorage = None

from .coordinator_setup import CoordinatorSetup
from .delegation_manager import DelegationManager
from .context_sharing import ContextSharing
from .task_assignment import TaskAssignment
from .result_combiner import ResultCombiner


class Team:
    """A callable class for multi-agent operations using the Upsonic client."""
    
    def __init__(self, 
                 agents: list[Any], 
                 tasks: list[Task] | None = None, 
                 model: Optional[Any] = None,
                 response_format: Any = str,  
                 ask_other_team_members: bool = False,
                 mode: Literal["sequential", "coordinate", "route"] = "sequential",
                 memory: Optional[Memory] = None
                 ):
        """
        Initialize the Team with agents and optionally tasks.
        
        Args:
            agents: List of Agent instances to use as team members.
            tasks: List of tasks to execute (optional).
            response_format: The response format for the end task (optional).
            model: The model provider instance for any internal agents (leader, router).
            ask_other_team_members: A flag to automatically add other agents as tools.
            mode: The operational mode for the team ('sequential', 'coordinate', or 'route').
        """
        self.agents = agents
        self.tasks = tasks if isinstance(tasks, list) else [tasks] if tasks is not None else []
        self.model = model
        self.response_format = response_format
        self.ask_other_team_members = ask_other_team_members
        self.mode = mode
        self.memory = memory
        
        # The leader_agent is an internal construct, not passed by the user.
        self.leader_agent: Optional[Agent] = None

        if self.ask_other_team_members:
            self.add_tool()

    def complete(self, tasks: list[Task] | Task | None = None):
        return self.do(tasks)
    
    def print_complete(self, tasks: list[Task] | Task | None = None):
        return self.print_do(tasks)

    def do(self, tasks: list[Task] | Task | None = None):
        """
        Execute multi-agent operations with the predefined agents and tasks.
        
        Args:
            tasks: Optional list of tasks or single task to execute. If not provided, uses tasks from initialization.
        
        Returns:
            The response from the multi-agent operation
        """
        # Use provided tasks or fall back to initialized tasks
        tasks_to_execute = tasks if tasks is not None else self.tasks
        if not isinstance(tasks_to_execute, list):
            tasks_to_execute = [tasks_to_execute]
        
        # Execute the multi-agent call
        return self.multi_agent(self.agents, tasks_to_execute)
    
    def multi_agent(self, agent_configurations: List[Agent], tasks: Any):
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.multi_agent_async(agent_configurations, tasks), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.multi_agent_async(agent_configurations, tasks))

    async def multi_agent_async(self, agent_configurations: List[Agent], tasks: Any):
        """
        Asynchronous version of the multi_agent method.
        """
        if self.mode == "sequential":
            # Set shared memory on all agents if memory is provided
            if self.memory:
                for agent in agent_configurations:
                    if agent.memory is None:
                        agent.memory = self.memory

            context_sharing = ContextSharing()
            task_assignment = TaskAssignment()
            combiner_model = self.model
            if not combiner_model and self.agents:
                combiner_model = self.agents[0].model

            result_combiner = ResultCombiner(model=combiner_model, debug=self.agents[-1].debug if self.agents else False)
            if not isinstance(tasks, list):
                tasks = [tasks]
            agents_registry, agent_names = task_assignment.prepare_agents_registry(agent_configurations)
            all_results = []
            for task_index, current_task in enumerate(tasks):
                selection_context = context_sharing.build_selection_context(
                    current_task, tasks, task_index, agent_configurations, all_results
                )
                selected_agent_name = await task_assignment.select_agent_for_task(
                    current_task, selection_context, agents_registry, agent_names, agent_configurations
                )
                if selected_agent_name:
                    context_sharing.enhance_task_context(
                        current_task, tasks, task_index, agent_configurations, all_results
                    )
                    result = await agents_registry[selected_agent_name].do_async(current_task)
                    all_results.append(current_task)
            if not result_combiner.should_combine_results(all_results):
                return result_combiner.get_single_result(all_results)
            return await result_combiner.combine_results(
                all_results, self.response_format, self.agents
            )

        elif self.mode == "coordinate":
            if not self.model:
                raise ValueError(f"A `model` must be set on the Team for '{self.mode}' mode.")
            tool_mapping = {}
            for member in self.tasks:
                if member.tools:
                    for tool in member.tools:
                        if callable(tool):
                            tool_mapping[tool.__name__] = tool

            setup_manager = CoordinatorSetup(self.agents, tasks, mode="coordinate")
            delegation_manager = DelegationManager(self.agents, tool_mapping)

            if self.memory is None:
                self.memory = Memory(storage=InMemoryStorage(),
                                        full_session_memory=True,
                                        session_id="team_coordinator_session",
                                        )
            
            self.leader_agent = Agent(
                model=self.model, 
                memory=self.memory
            )
            
            leader_system_prompt = setup_manager.create_leader_prompt()
            self.leader_agent.system_prompt = leader_system_prompt

            master_description = (
                "Begin your mission. Review your system prompt for the full list of tasks and your team roster. "
                "Formulate your plan and start delegating tasks now."
            )

            all_attachments = []
            for task in tasks:
                if task.attachments:
                    all_attachments.extend(task.attachments)

            delegation_tool = delegation_manager.get_delegation_tool(self.memory)

            master_task = Task(
                description=master_description,
                attachments=all_attachments if all_attachments else None,
                tools=[delegation_tool],
                response_format=self.response_format,
            )

            final_response = await self.leader_agent.do_async(master_task)
            
            return final_response
        elif self.mode == "route":
            if not self.model:
                raise ValueError(f"A `model` must be set on the Team for '{self.mode}' mode.")
            
            setup_manager = CoordinatorSetup(self.agents, tasks, mode="route")
            delegation_manager = DelegationManager(self.agents, {})

            self.leader_agent = Agent(model=self.model)

            leader_system_prompt = setup_manager.create_leader_prompt()
            self.leader_agent.system_prompt = leader_system_prompt
            routing_tool = delegation_manager.get_routing_tool()

            router_task_description = "Analyze the MISSION OBJECTIVES in your system prompt and route the request to the best specialist."
            router_task = Task(description=router_task_description, tools=[routing_tool])

            await self.leader_agent.do_async(router_task)

            chosen_agent = delegation_manager.routed_agent

            if not chosen_agent:
                raise ValueError("Routing failed: The router agent did not select a team member.")
            
            consolidated_description = " ".join([task.description for task in tasks])
            all_attachments = [attachment for task in tasks if task.attachments for attachment in task.attachments]
            all_tools = [tool for task in tasks if task.tools for tool in task.tools]

            final_task = Task(
                description=consolidated_description,
                attachments=all_attachments or None,
                tools=list(set(all_tools)) if all_tools else None,
                response_format=self.response_format
            )

            await chosen_agent.do_async(final_task)
            return final_task.response

    def print_do(self, tasks: list[Task] | Task | None = None):
        """
        Execute the multi-agent operation and print the result.
        
        Returns:
            The response from the multi-agent operation
        """
        result = self.do(tasks)
        print(str(result))
        return result
    
    def add_tool(self):
        """
        Add agents as a tool to each Task object.
        """
        for task in self.tasks:
            if not hasattr(task, 'tools'):
                task.tools = []
            if isinstance(task.tools, list):
                task.tools.extend(self.agents)
            else:
                task.tools = self.agents