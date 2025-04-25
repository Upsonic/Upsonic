from ..agent_configuration.agent_configuration import get_or_create_client, register_tools
from ..tasks.tasks import Task
from ..agent_configuration.agent_configuration import AgentConfiguration
from ..tasks.task_response import ObjectResponse
from ..direct_llm_call.direct_llm_cal import Direct
from typing import Any, List, Dict, Optional, Type, Union, Literal
from ...model_registry import ModelNames

from ..agent_configuration.agent_configuration import AgentConfiguration as Agent

class Team:
    """A callable class for multi-agent operations using the Upsonic client."""
    
    def __init__(self, agents: list[Any], tasks: list[Task] | None = None, llm_model: str | None = None, response_format: Any = None, model: ModelNames | None = None):
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
    def do(self, tasks: list[Task] | Task | None = None):
        """
        Execute multi-agent operations with the predefined agents and tasks.
        
        Args:
            tasks: Optional list of tasks or single task to execute. If not provided, uses tasks from initialization.
        
        Returns:
            The response from the multi-agent operation
        """
        global latest_upsonic_client
        from ..latest_upsonic_client import latest_upsonic_client

        # Get or create client for agents without custom clients
        the_client = get_or_create_client()
        
        # Use provided tasks or fall back to initialized tasks
        tasks_to_execute = tasks if tasks is not None else self.tasks
        if not isinstance(tasks_to_execute, list):
            tasks_to_execute = [tasks_to_execute]
        
        # Register tools for all tasks regardless of client
        for task in tasks_to_execute:
            the_client = register_tools(the_client, task.tools)
            # Also register tools for agents with custom clients
            for agent in self.agents:
                if agent.client is not None:
                    agent.client = register_tools(agent.client, task.tools)
        
        # Update the global client reference
        if latest_upsonic_client is None:
            latest_upsonic_client = the_client

        # Execute the multi-agent call
        return self.multi_agent(the_client, self.agents, tasks_to_execute, self.llm_model)
    
    def multi_agent(self, the_client: Any, agent_configurations: List[AgentConfiguration], tasks: Any, llm_model: str = None):
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.multi_agent_async(the_client, agent_configurations, tasks, llm_model), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.multi_agent_async(the_client, agent_configurations, tasks, llm_model))

    async def multi_agent_async(self, the_client: Any, agent_configurations: List[AgentConfiguration], tasks: Any, llm_model: str = None):
        """
        Asynchronous version of the multi_agent method.
        """
        agent_tasks = []
        all_results = []

        the_agents = {}

        for each in agent_configurations:
            agent_key = each.agent_id[:5] + "_" + each.job_title
            the_agents[agent_key] = each

        the_agents_keys = list(the_agents.keys())

        class TheAgents_(ObjectResponse):
            agents: List[str]

        the_agents_ = TheAgents_(agents=the_agents_keys)

        class SelectedAgent(ObjectResponse):
            selected_agent: str

        if isinstance(tasks, list) != True:
            tasks = [tasks]
        
        for each in tasks:
            is_end = False
            selected_agent = None
            while not is_end:
                selecting_task = Task(description="Select an agent for this task", images=each.images, response_format=SelectedAgent, context=[the_agents_, each])
                the_call_llm_model = agent_configurations[0].model
                await Direct.do_async(selecting_task, the_call_llm_model, retry=agent_configurations[0].retry)
                if selecting_task.response.selected_agent in the_agents:
                    is_end = True
                    selected_agent = selecting_task.response.selected_agent
            
            if selected_agent:
                agent_tasks.append({
                    "agent": the_agents[selected_agent],
                    "task": each
                })
                    
        # Store original client
        original_client = the_client

        # Process tasks asynchronously
        for each in agent_tasks:
            # Check if agent has a custom client
            if each["agent"].client is not None:
                # Use agent's custom client for this task with async method
                result = await each["agent"].client.agent_async(each["agent"], each["task"], llm_model)
                all_results.append({
                    "task": each["task"].description,
                    "result": result
                })
            else:
                # Use the default/automatic client with async method
                result = await original_client.agent_async(each["agent"], each["task"], llm_model)
                all_results.append({
                    "task": each["task"].description,
                    "result": result
                })

        # If there's only one task, return its result directly
        if len(all_results) == 1:
            return all_results[0]["result"]

        # Create an end task that combines all results
        class OtherTask(ObjectResponse):
            task: str
            result: Any

        # Create OtherTask objects for the context
        other_tasks = [
            OtherTask(task=result["task"], result=result["result"])
            for result in all_results
        ]

        end_task = Task(
            description="Combined results from all previous tasks that in your context. You Need to prepare an final answer to your users. Dont talk about yourself or tasks directly. Just catch everything from prevously things and prepare an final return. But please try to give answer to user questions. If there is an just one question, just return that answer. If there is an multiple questions, just return all of them. but with an summary of all of them.",
            context=other_tasks,
            response_format=self.response_format
        )

        end_agent = Direct(model=self.model, client=self.agents[-1].client, debug=self.agents[-1].debug)
        final_response = await end_agent.do_async(end_task)
        return final_response

    def print_do(self):
        """
        Execute the multi-agent operation and print the result.
        
        Returns:
            The response from the multi-agent operation
        """
        result = self.do()
        print(result)
        return result
