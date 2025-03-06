import copy
import time
import cloudpickle

from ..knowledge_base.knowledge_base import KnowledgeBase
cloudpickle.DEFAULT_PROTOCOL = 2
import dill
import base64
import httpx
import hashlib
from typing import Any, List, Dict, Optional, Type, Union, Literal
from pydantic import BaseModel
import uuid

from ..tasks.tasks import Task

from ..printing import agent_end, agent_total_cost, agent_retry, print_price_id_summary



from ..tasks.task_response import ObjectResponse

from ..agent_configuration.agent_configuration import AgentConfiguration

from ..level_utilized.utility import context_serializer


from ..level_utilized.utility import context_serializer, response_format_serializer, tools_serializer, response_format_deserializer, error_handler



from ...storage.caching import save_to_cache_with_expiry, get_from_cache_with_expiry

from ..tools.tools import Search

from ...reliability_processor import ReliabilityProcessor


class SubTask(ObjectResponse):
    description: str
    sources_can_be_used: List[str]
    required_output: str
    tools: List[str]
class SubTaskList(ObjectResponse):
    sub_tasks: List[SubTask]

class AgentMode(ObjectResponse):
    """Mode selection for task decomposition"""
    selected_mode: Literal["level_no_step", "level_one"]




class SearchResult(ObjectResponse):
    any_customers: bool
    products: List[str]
    services: List[str]
    potential_competitors: List[str]
class CompanyObjective(ObjectResponse):
    objective: str
    goals: List[str]
    state: str
class HumanObjective(ObjectResponse):
    job_title: str
    job_description: str
    job_goals: List[str]
    


class Characterization(ObjectResponse):
    website_content: Union[SearchResult, None]
    company_objective: Union[CompanyObjective, None]
    human_objective: Union[HumanObjective, None]
    name_of_the_human_of_tasks: str = None
    contact_of_the_human_of_tasks: str = None


class OtherTask(ObjectResponse):
    task: str
    result: Any



class Agent:


    def agent_(
        self,
        agent_configuration: AgentConfiguration,
        task: Task,
        llm_model: str = None,
    ) -> Any:
        

        
        start_time = time.time()


        results = []

        try:
            if isinstance(task, list):
                for each in task:
                    the_result = self.send_agent_request(agent_configuration, each, llm_model)
                    the_result["time"] = time.time() - start_time
                    results.append(the_result)
                    agent_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_count"], the_result["context_count"], self.debug, each.price_id)
            else:
                the_result = self.send_agent_request(agent_configuration, task, llm_model)
                the_result["time"] = time.time() - start_time
                results.append(the_result)
                agent_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_count"], the_result["context_count"], self.debug, task.price_id)
        except Exception as outer_e:
            try:
                from ...server import stop_dev_server, stop_main_server, is_tools_server_running, is_main_server_running

                if is_tools_server_running() or is_main_server_running():
                    stop_dev_server()

            except Exception:
                pass

            raise outer_e

        end_time = time.time()

        

        return results
        


    def send_agent_request(
        self,
        agent_configuration: AgentConfiguration,
        task: Task,
        llm_model: str = None,
    ) -> Any:
        from ..trace import sentry_sdk
        from ..level_utilized.utility import CallErrorException
        """
        Call GPT-4 with optional tools and MCP servers.

        Args:
            prompt: The input prompt for GPT-4
            response_format: The expected response format (can be a type or Pydantic model)
            tools: Optional list of tool names to use


        Returns:
            The response in the specified format
        """

        if llm_model is None:
            llm_model = self.default_llm_model

        retry_count = 0
        last_error = None


        while retry_count <= agent_configuration.retries:
            try:
                if retry_count > 0:
                    agent_retry(retry_count, agent_configuration.retries)

                tools = tools_serializer(task.tools)

                response_format = task.response_format
                with sentry_sdk.start_transaction(op="task", name="Agent.send_agent_request") as transaction:
                    with sentry_sdk.start_span(op="serialize"):
                        # Serialize the response format if it's a type or BaseModel
                        response_format_str = response_format_serializer(task.response_format)


                    new_context = []
                    if task.context:
                        
                        for each in task.context:
                            if isinstance(each, KnowledgeBase):
                                if not each.rag:
                                    new_context.append(each.markdown(self))
                            else:
                                new_context.append(each)

                        context = context_serializer(new_context, self)



                    with sentry_sdk.start_span(op="prepare_request"):
                        # Prepare the request data
                        data = {
                            "agent_id": agent_configuration.agent_id,
                            "prompt": task.description + task.additional_description(self), 
                            "images": task.images_base_64,
                            "response_format": response_format_str,
                            "tools": tools or [],
                            "context": context,
                            "llm_model": llm_model,
                            "system_prompt": None,
                            "retries": agent_configuration.retries,
                            "context_compress": agent_configuration.context_compress,
                            "memory": agent_configuration.memory
                        }

                    with sentry_sdk.start_span(op="send_request"):
    
                        result = self.send_request("/level_two/agent", data)

                        result = result["result"]

                        error_handler(result)

                    with sentry_sdk.start_span(op="deserialize"):
                        deserialized_result = response_format_deserializer(response_format_str, result)

                # Process result through reliability layer
                processed_result = ReliabilityProcessor.process_result(
                    deserialized_result["result"], 
                    agent_configuration.reliability_layer,
                    task,
                    llm_model
                )
                task._response = processed_result

                response_format_req = None
                if response_format_str == "str":
                    response_format_req = response_format_str
                else:
                    # Class name
                    response_format_req = response_format.__name__
                
                if context is None:
                    context = []

                len_of_context = len(task.context) if task.context is not None else 0

                return {"result": processed_result, "llm_model": llm_model, "response_format": response_format_req, "usage": deserialized_result["usage"], "tool_count": len(tools), "context_count": len_of_context}

            except CallErrorException as e:
                last_error = e
                retry_count += 1
                if retry_count > agent_configuration.retries:
                    raise last_error
                continue



    def create_characterization(self, agent_configuration: AgentConfiguration, llm_model: str = None, price_id: str = None):
        tools = [Search]
        
        search_result = None
        company_objective_result = None
        human_objective_result = None
        
        search_task = None
        company_objective_task = None

        # Handle website search if URL is provided
        if agent_configuration.company_url:
            search_task = Task(description=f"Make a search for {agent_configuration.company_url}", tools=tools, response_format=SearchResult, price_id_=price_id, not_main_task=True)
            self.call(search_task, llm_model=llm_model)
            search_result = search_task.response

        # Handle company objective if provided
        if agent_configuration.company_objective:
            context = [search_task] if search_task else None
            company_objective_task = Task(description=f"Generate the company objective for {agent_configuration.company_objective}", 
                                        tools=tools, 
                                        response_format=CompanyObjective,
                                        context=context,
                                        price_id_=price_id,
                                        not_main_task=True)
            self.call(company_objective_task, llm_model=llm_model)
            company_objective_result = company_objective_task.response

        # Handle human objective if job title is provided
        if agent_configuration.job_title:
            context = []
            if search_task:
                context.append(search_task)
            if company_objective_task:
                context.append(company_objective_task)
            
            context = context if context else None
            human_objective_task = Task(description=f"Generate the human objective for {agent_configuration.job_title}", 
                                      tools=tools, 
                                      response_format=HumanObjective,
                                      context=context,
                                      price_id_=price_id,
                                      not_main_task=True)
            self.call(human_objective_task, llm_model=llm_model)
            human_objective_result = human_objective_task.response

        total_character = Characterization(
            website_content=search_result,
            company_objective=company_objective_result,
            human_objective=human_objective_result,
            name_of_the_human_of_tasks=agent_configuration.name,
            contact_of_the_human_of_tasks=agent_configuration.contact
        )

        return total_character







    def agent(self, agent_configuration: AgentConfiguration, task: Task,  llm_model: str = None):
        original_task = task
        original_task.start_time = time.time()
        

        if llm_model is None:
            llm_model = agent_configuration.model






        copy_agent_configuration = copy.deepcopy(agent_configuration)
        copy_agent_configuration_json = copy_agent_configuration.model_dump_json(include={"job_title", "company_url", "company_objective", "name", "contact"})


        
        the_characterization_cache_key = f"characterization_{hashlib.sha256(copy_agent_configuration_json.encode()).hexdigest()}"

        if agent_configuration.system_prompt:
            the_characterization = agent_configuration.system_prompt
        elif agent_configuration.caching:
            the_characterization = get_from_cache_with_expiry(the_characterization_cache_key)
            if the_characterization is None:
                the_characterization = self.create_characterization(agent_configuration, llm_model, task.price_id)
                save_to_cache_with_expiry(the_characterization, the_characterization_cache_key, agent_configuration.cache_expiry)
        else:
            the_characterization = self.create_characterization(agent_configuration, llm_model, task.price_id)



        knowledge_base = None

        if agent_configuration.knowledge_base:
            knowledge_base = self.knowledge_base(agent_configuration, llm_model)
            



        
        the_task = task

        is_it_sub_task = False
        shared_context = []

        if agent_configuration.sub_task:
            # Create a new agent configuration for sub-tasks with memory enabled
            sub_task_agent_config = copy.deepcopy(agent_configuration)
            sub_task_agent_config.agent_id_ = str(uuid.uuid4())  # Generate new agent ID for sub-tasks
            sub_task_agent_config.memory = True  # Enable memory for sub-tasks
            
            sub_tasks = self.multiple(task, llm_model)
            is_it_sub_task = True
            the_task = sub_tasks
    



        if not isinstance(the_task, list):
            the_task = [the_task]


        for each in the_task:
            if not isinstance(each.context, list):
                each.context = [each.context]


        last_task = []
        for each in the_task:
            if isinstance(each.context, list):
                last_task.append(each)
        the_task = last_task


        for each in the_task:
            each.context.append(the_characterization)

        # Add knowledge base to the context for each task
        if knowledge_base:
            if isinstance(the_task, list):
                for each in the_task:
                    if each.context:
                        each.context.append(knowledge_base)
                    else:
                        each.context = [knowledge_base]

        if task.context:
            for each in the_task:
                each.context += task.context

        # Create copies of agent_configuration for all tasks except the last one
        task_specific_configs = []
        for i in range(len(the_task)):
            if i < len(the_task) - 1:
                # Create a copy and set reliability_layer to None for all except last task
                task_config = copy.deepcopy(sub_task_agent_config if agent_configuration.sub_task else agent_configuration)
                task_config.reliability_layer = None
                task_specific_configs.append(task_config)
            else:
                # Use original config for the last task
                task_specific_configs.append(agent_configuration)

        if agent_configuration.tools:
            if isinstance(the_task, list):
                for each in the_task:
                    each.tools = agent_configuration.tools

        


        results = []    
        if isinstance(the_task, list):
            for i, each in enumerate(the_task):
                if is_it_sub_task:
                    if shared_context:
                        each.context += shared_context

                result = self.agent_(task_specific_configs[i], each, llm_model=llm_model)
                results += result

                if is_it_sub_task:
                    shared_context.append(OtherTask(task=each.description, result=each.response))






        original_task._response = the_task[-1].response


        
        total_time = 0
        for each in results:
            total_time += each["time"]

        total_input_tokens = 0
        total_output_tokens = 0
        for each in results:

            total_input_tokens += each["usage"]["input_tokens"]
            total_output_tokens += each["usage"]["output_tokens"]

        the_llm_model = llm_model
        if the_llm_model is None:
            the_llm_model = self.default_llm_model

        agent_total_cost(total_input_tokens, total_output_tokens, total_time, the_llm_model)

        if not original_task.not_main_task:
            print_price_id_summary(original_task.price_id, original_task)



        original_task.end_time = time.time()
        return original_task.response




    def multiple(self, task: Task, llm_model: str = None):
        # First, determine the mode of operation
        mode_selection_prompt = f"""
You are a Task Analysis AI that helps determine the best mode of task decomposition.

Given task: "{task.description}"

Analyze the task characteristics:

Level No Step (Direct Execution) is suitable for:
- Tasks that can be completed in a single, atomic operation
- Tasks where the output format is simple and well-defined
- Tasks that don't require setup or configuration
- Tasks where AI can directly generate the complete result
- Tasks without dependencies or external integrations
Examples:
- Simple data transformations
- Direct text generation
- Single API call operations
- Basic calculations or conversions

Level One (Basic Decomposition) is suitable for:
- Tasks requiring multiple steps or verifications
- Tasks with clear, linear steps
- Tasks needing external information or resources
- Tasks requiring setup or configuration
- Tasks involving API integrations or data processing
- Tasks that need error handling
- Information retrieval and verification tasks
Examples of Level One Tasks:
- Finding and verifying documentation
- Implementation tasks with clear steps
- Multi-step data processing
- Tasks requiring setup and configuration
- Tasks involving API usage
- Tasks needing error handling
- Tasks that follow a linear sequence of steps

Select the mode based on these characteristics.
Prefer level_no_step when the task can be completed directly without any decomposition.
Use Level One for any task requiring multiple steps or verification.
"""
        mode_selector = Task(
            description=mode_selection_prompt,
            images=task.images,
            response_format=AgentMode,
            context=[task],
            price_id_=task.price_id,
            not_main_task=True
        )
        
        self.call(mode_selector, llm_model)
        
        # If level_no_step is selected, return just the end task
        if mode_selector.response.selected_mode == "level_no_step":
            return [Task(description=task.description, images=task.images, response_format=task.response_format, tools=task.tools, price_id_=task.price_id, not_main_task=True)]

        # Generate a list of sub tasks
        prompt = f"""
You are a Task Decomposition AI that helps break down large tasks into smaller, manageable subtasks.

Given task: "{task.description}"
Available tools: {task.tools if task.tools else "No tools available"}

Tool Dependency Guidelines:
- File Operations: Tasks involving file reading, writing, or manipulation require file system tools
- Terminal Operations: Tasks requiring command execution need terminal access tools
- Web Operations: Tasks involving web searches or API calls need web access tools
- System Operations: Tasks involving system configuration or environment setup need system tools

Task Decomposition Rules:
1. Only create subtasks that can be completed with the available tools
2. Skip any operations that would require unavailable tools
3. Each subtask must be achievable with the given tool set
4. If a critical operation cannot be performed due to missing tools, note it in the task description
5. Adapt the approach based on available tools rather than assuming tool availability

General Task Rules:
1. Each subtask should be clear, specific, and actionable
2. Subtasks should be ordered in a logical sequence
3. Each subtask should be necessary for completing the main task
4. Avoid overly broad or vague subtasks
5. Keep subtasks at a similar level of granularity

Tool Availability Impact:
- Without file system tools: Skip file operations
- Without terminal tools: Avoid command execution tasks
- Without web tools: Skip online searches, API calls
- Without system tools: Avoid system configuration tasks
"""
        sub_tasker_context = [task, task.response_format]
        if task.context:
            sub_tasker_context = task.context
        sub_tasker = Task(description=prompt, images=task.images, response_format=SubTaskList, context=sub_tasker_context, tools=task.tools, price_id_=task.price_id, not_main_task=True)

        self.call(sub_tasker, llm_model)

        sub_tasks = []

        # Create tasks from subtasks
        for each in sub_tasker.response.sub_tasks:
            new_task = Task(description=each.description + " " + each.required_output + " " + str(each.sources_can_be_used) + " " + str(each.tools) + "Focus to complete the task with right result, Dont ask to human directly do it and give the result.", images=task.images, price_id_=task.price_id, not_main_task=True)
            new_task.tools = task.tools
            sub_tasks.append(new_task)

        # Add the final task that will produce the original desired response format
        end_task = Task(description=task.description, images=task.images, response_format=task.response_format, price_id_=task.price_id, not_main_task=True)
        sub_tasks.append(end_task)

        return sub_tasks





    def multi_agent(self, agent_configurations: List[AgentConfiguration], tasks: Any, llm_model: str = None):
        agent_tasks = []

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
            while is_end == False:
                selecting_task  = Task(description="Select an agent for this task", images=each.images, response_format=SelectedAgent, context=[the_agents_, each])

                the_call_llm_model = agent_configurations[0].model
                self.call(selecting_task, the_call_llm_model)

                if selecting_task.response.selected_agent in the_agents:
                    is_end = True



                agent_tasks.append({
                    "agent": the_agents[selecting_task.response.selected_agent],
                    "task": each
                })
                    


        # Store original client
        original_client = self

        for each in agent_tasks:
            # Check if agent has a custom client
            if each["agent"].client is not None:
                # Use agent's custom client for this task
                each["agent"].client.agent(each["agent"], each["task"], llm_model)
            else:
                # Use the default/automatic client
                original_client.agent(each["agent"], each["task"], llm_model)


        return the_agents


