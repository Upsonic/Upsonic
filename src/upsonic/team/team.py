from ..tasks.tasks import Task
from ..direct.direct_llm_cal import Direct
from typing import Any, List, Dict, Optional, Type, Union, Literal
from ..models.model_registry import ModelNames

from ..direct.direct_llm_cal import Direct as Agent
from ..context.task import turn_task_to_string

from .context_sharing import ContextSharing
from .task_assignment import TaskAssignment
from .result_combiner import ResultCombiner

class Team:
    """A callable class for multi-agent operations using the Upsonic client."""
    
    def __init__(self, agents: list[Any], tasks: list[Task] | None = None, llm_model: str | None = None, response_format: Any = str, model: ModelNames | None = None):
        """
        Initialize the Team with agents and optionally tasks.
        
        Args:
            agents: List of agent configurations to use
            tasks: List of tasks to execute (optional)
            llm_model: The LLM model to use (optional)
            response_format: The response format for the end task (optional)
        """
        self.agents = agents
        self.tasks = tasks if isinstance(tasks, list) else [tasks] if tasks is not None else []
        self.llm_model = llm_model
        self.response_format = response_format
        self.model = model

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
        # Initialize the specialized modules
        context_sharing = ContextSharing()
        task_assignment = TaskAssignment()
        result_combiner = ResultCombiner(model=self.model, debug=self.agents[-1].debug if self.agents else False)
        
        # Prepare tasks list
        if not isinstance(tasks, list):
            tasks = [tasks]
        
        # Set up agents registry
        agents_registry, agent_names = task_assignment.prepare_agents_registry(agent_configurations)
        
        # Process each task with full context awareness
        all_results = []
        
        for task_index, current_task in enumerate(tasks):
            # Build context for agent selection
            selection_context = context_sharing.build_selection_context(
                current_task, tasks, task_index, agent_configurations, all_results
            )
            
            # Select appropriate agent for the task
            selected_agent_name = await task_assignment.select_agent_for_task(
                current_task, selection_context, agents_registry, agent_names, agent_configurations
            )
            
            if selected_agent_name:
                # Enhance the current task's context with comprehensive information
                context_sharing.enhance_task_context(
                    current_task, tasks, task_index, agent_configurations, all_results
                )
                
                # Execute the task with the selected agent
                result = await agents_registry[selected_agent_name].do_async(current_task, llm_model)
                all_results.append(current_task)

        # Handle result combination
        if not result_combiner.should_combine_results(all_results):
            return result_combiner.get_single_result(all_results)
        
        # Combine multiple results
        return await result_combiner.combine_results(
            all_results, self.response_format, self.agents
        )

    def print_do(self, tasks: list[Task] | Task | None = None):
        """
        Execute the multi-agent operation and print the result.
        
        Returns:
            The response from the multi-agent operation
        """
        result = self.do(tasks)
        print(result)
        return result