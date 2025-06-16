from ..tasks.tasks import Task

from ..tasks.task_response import ObjectResponse
from ..direct.direct_llm_cal import Direct
from typing import Any, List, Dict, Optional, Type, Union, Literal
from ..models.model_registry import ModelNames

from ..direct.direct_llm_cal import Direct as Agent
from ..context.task import turn_task_to_string

class MultiAgent:
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
        agent_tasks = []
        all_results = []

        the_agents = {}

        # Use agent names as keys instead of complex composite keys
        for each in agent_configurations:
            agent_name = each.get_agent_id()
            the_agents[agent_name] = each

        the_agents_keys = list(the_agents.keys())



  

        class SelectedAgent(ObjectResponse):
            selected_agent: str

        if isinstance(tasks, list) != True:
            tasks = [tasks]
        
        for each in tasks:
            is_end = False
            selected_agent = None
            max_attempts = 3  # Prevent infinite loops
            attempts = 0
            
            while not is_end and attempts < max_attempts:
                context = [each]
                context += agent_configurations
                selecting_task = Task(
                    description=f"Select the most appropriate agent from the available agents to handle the task. Return only the exact agent name from the list.",
                    images=each.images, 
                    response_format=SelectedAgent, 
                    context=context
                )
                await Direct(model=agent_configurations[0].model).do_async(selecting_task)
                
                selected_name = selecting_task.response.selected_agent
                if selected_name in the_agents:
                    is_end = True
                    selected_agent = selected_name
                else:
                    # Try to find partial matches if exact match fails
                    for agent_name in the_agents_keys:
                        if agent_name.lower() in selected_name.lower() or selected_name.lower() in agent_name.lower():
                            is_end = True
                            selected_agent = agent_name
                            break
                
                attempts += 1
            
            # If no agent selected after attempts, use the first agent as fallback
            if not selected_agent and the_agents_keys:
                selected_agent = the_agents_keys[0]
            
            if selected_agent:
                agent_tasks.append({
                    "agent": the_agents[selected_agent],
                    "task": each
                })

        # Process tasks asynchronously - Fixed to work with Direct agents
        for each in agent_tasks:
            # Use the agent directly (Direct class) to execute the task
            result = await each["agent"].do_async(each["task"], llm_model)
            all_results.append(each["task"])


        # If there's only one task, return its result directly
        if len(all_results) == 1:
            return all_results[0].response



        end_task = Task(
            description="Combined results from all previous tasks that in your context. You Need to prepare an final answer to your users. Dont talk about yourself or tasks directly. Just catch everything from prevously things and prepare an final return. But please try to give answer to user questions. If there is an just one question, just return that answer. If there is an multiple questions, just return all of them. but with an summary of all of them.",
            context=all_results,
            response_format=self.response_format
        )

        end_agent = Direct(model=self.model, debug=self.agents[-1].debug)
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