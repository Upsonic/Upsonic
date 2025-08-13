from upsonic.tasks.tasks import Task
from upsonic.agent.agent import Direct
from typing import Any, List, Dict, Optional, Type, Union, Literal
from upsonic.models.model_registry import ModelNames

from upsonic.agent.agent import Direct as Agent
from upsonic.context.task import turn_task_to_string
from upsonic.storage import Memory, InMemoryStorage

# --- New Imports for Modular Architecture ---
from .coordinator_setup import CoordinatorSetup
from .delegation_manager import DelegationManager
# --- Existing Sequential Mode Imports ---
from .context_sharing import ContextSharing
from .task_assignment import TaskAssignment
from .result_combiner import ResultCombiner


class Team:
    """A callable class for multi-agent operations using the Upsonic client."""
    
    def __init__(self, 
                 agents: list[Any], 
                 tasks: list[Task] | None = None, 
                 llm_model: str | None = None, 
                 response_format: Any = str, 
                 model: ModelNames | None = None, 
                 ask_other_team_members: bool = False,
                 mode: Literal["sequential", "coordinate"] = "sequential"
                 ):
        """
        Initialize the Team with agents and optionally tasks.
        
        Args:
            agents: List of agent configurations to use as team members.
            tasks: List of tasks to execute (optional).
            llm_model: The LLM model to use (optional).
            response_format: The response format for the end task (optional).
            model: The default model for the agents in the team and the leader.
            ask_other_team_members: A flag to automatically add other agents as tools.
            mode: The operational mode for the team ('sequential' or 'coordinate').
        """
        self.agents = agents # These are now considered the "member" agents
        self.tasks = tasks if isinstance(tasks, list) else [tasks] if tasks is not None else []
        self.llm_model = llm_model
        self.response_format = response_format
        self.model = model
        self.ask_other_team_members = ask_other_team_members
        self.mode = mode
        
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
        return self.multi_agent(self.agents, tasks_to_execute, self.llm_model)
    
    def multi_agent(self, agent_configurations: List[Agent], tasks: Any, llm_model: str = None):
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.multi_agent_async(agent_configurations, tasks, llm_model), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.multi_agent_async(agent_configurations, tasks, llm_model))

    async def multi_agent_async(self, agent_configurations: List[Agent], tasks: Any, llm_model: str = None):
        """
        Asynchronous version of the multi_agent method.
        """
        if self.mode == "sequential":
            context_sharing = ContextSharing()
            task_assignment = TaskAssignment()
            result_combiner = ResultCombiner(model=self.model, debug=self.agents[-1].debug if self.agents else False)
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
                    result = await agents_registry[selected_agent_name].do_async(current_task, llm_model)
                    all_results.append(current_task)
            if not result_combiner.should_combine_results(all_results):
                return result_combiner.get_single_result(all_results)
            return await result_combiner.combine_results(
                all_results, self.response_format, self.agents
            )

        elif self.mode == "coordinate":
            tool_mapping = {}
            for member in self.tasks:
                if member.tools:
                    for tool in member.tools:
                        if callable(tool):
                            tool_mapping[tool.__name__] = tool

            setup_manager = CoordinatorSetup(self.agents, tasks)
            delegation_manager = DelegationManager(self.agents, tool_mapping)

            storage = InMemoryStorage()
            session_memory = Memory(storage=storage,
                                    full_session_memory=True,
                                    session_id="team_coordinator_session",
                                    )
            
            self.leader_agent = Direct(
                model=self.model or llm_model, 
                memory=session_memory
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

            delegation_tool = delegation_manager.get_delegation_tool(session_memory)

            master_task = Task(
                description=master_description,
                attachments=all_attachments if all_attachments else None,
                tools=[delegation_tool],
                response_format=self.response_format,
            )

            final_response = await self.leader_agent.do_async(master_task, model=llm_model)
            
            return final_response

    def print_do(self, tasks: list[Task] | Task | None = None):
        """
        Execute the multi-agent operation and print the result.
        
        Returns:
            The response from the multi-agent operation
        """
        result = self.do(tasks)
        print(result)
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