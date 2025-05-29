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
from ..direct_llm_call.direct_llm_cal import Direct

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
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.agent_async_(agent_configuration, task, llm_model), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.agent_async_(agent_configuration, task, llm_model))

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
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.send_agent_request_async(agent_configuration, task, llm_model), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.send_agent_request_async(agent_configuration, task, llm_model))

    def create_characterization(self, agent_configuration: AgentConfiguration, llm_model: str = None, price_id: str = None):
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.create_characterization_async(agent_configuration, llm_model, price_id), 
                    loop
                ).result()
        except RuntimeError:
            # No running loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.create_characterization_async(agent_configuration, llm_model, price_id))

    def agent(self, agent_configuration: AgentConfiguration, task: Task,  llm_model: str = None):
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.agent_async(agent_configuration, task, llm_model), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.agent_async(agent_configuration, task, llm_model))

    def multiple(self, agent_configuration: AgentConfiguration, task: Task, llm_model: str = None):
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.multiple_async(agent_configuration, task, llm_model), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.multiple_async(agent_configuration, task, llm_model))



    async def agent_async(self, agent_configuration: AgentConfiguration, task: Task, llm_model: str = None):
        """
        Asynchronous version of the agent method.
        """
        original_task = task
        original_task.start_time = time.time()
        
        if llm_model is None:
            llm_model = agent_configuration.model

        copy_agent_configuration = copy.deepcopy(agent_configuration)
        copy_agent_configuration_json = copy_agent_configuration.model_dump_json(include={"job_title", "company_url", "company_objective", "name", "contact"})
        
        the_characterization_cache_key = f"characterization_{hashlib.sha256(copy_agent_configuration_json.encode()).hexdigest()}"

        if agent_configuration.system_prompt:
            the_characterization = agent_configuration.system_prompt
        elif llm_model and llm_model.startswith("ollama"):
            the_characterization = agent_configuration.system_prompt if agent_configuration.system_prompt else agent_configuration.name
        elif agent_configuration.caching:
            the_characterization = get_from_cache_with_expiry(the_characterization_cache_key)
            if the_characterization is None:
                the_characterization = await self.create_characterization_async(agent_configuration, llm_model, task.price_id)
                save_to_cache_with_expiry(the_characterization, the_characterization_cache_key, agent_configuration.cache_expiry)
        else:
            the_characterization = await self.create_characterization_async(agent_configuration, llm_model, task.price_id)

        knowledge_base = None
        if agent_configuration.knowledge_base:
            knowledge_base = agent_configuration.knowledge_base
        
        the_task = task
        is_it_sub_task = False
        shared_context = []

        if agent_configuration.sub_task:
            # Create a new agent configuration for sub-tasks with memory enabled and same retry setting
            sub_task_agent_config = copy.deepcopy(agent_configuration)
            sub_task_agent_config.agent_id_ = str(uuid.uuid4())  # Generate new agent ID for sub-tasks
            sub_task_agent_config.memory = True  # Enable memory for sub-tasks
            
            # Use the async version of multiple
            sub_tasks = await self.multiple_async(sub_task_agent_config, task, llm_model)
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

                # Use the async version of agent_
                result = await self.agent_async_(task_specific_configs[i], each, llm_model=llm_model)
                results += result

                # Collect tool calls from each subtask
                for tool_call in each.tool_calls:
                    original_task.add_tool_call(tool_call)

                if is_it_sub_task:
                    shared_context.append(OtherTask(task=each.description, result=each.response))

        original_task._response = the_task[-1].response
        
        total_time = 0
        for each in results:
            total_time += each["time"]

        total_input_tokens = 0
        total_output_tokens = 0
        for each in results:
            if "usage" in each and each["usage"] is not None:
                total_input_tokens += each["usage"].get("input_tokens", 0)
                total_output_tokens += each["usage"].get("output_tokens", 0)

        the_llm_model = llm_model
        if the_llm_model is None:
            the_llm_model = self.default_llm_model

        agent_total_cost(total_input_tokens, total_output_tokens, total_time, the_llm_model)

        if not original_task.not_main_task:
            print_price_id_summary(original_task.price_id, original_task)

        original_task.end_time = time.time()
        return original_task.response

    async def agent_async_(
        self,
        agent_configuration: AgentConfiguration,
        task: Task,
        llm_model: str = None,
    ) -> Any:
        """
        Asynchronous version of agent_ method.
        """
        start_time = time.time()
        results = []

        try:
            if isinstance(task, list):
                for each in task:
                    the_result = await self.send_agent_request_async(agent_configuration, each, llm_model)
                    the_result["time"] = time.time() - start_time
                    results.append(the_result)
                    agent_end(the_result["result"], the_result["llm_model"], the_result["response_format"], 
                             start_time, time.time(), the_result["usage"], the_result["tool_usage"], the_result["tool_count"], 
                             the_result["context_count"], self.debug, each.price_id)
            else:
                the_result = await self.send_agent_request_async(agent_configuration, task, llm_model)
                the_result["time"] = time.time() - start_time
                results.append(the_result)
                agent_end(the_result["result"], the_result["llm_model"], the_result["response_format"], 
                         start_time, time.time(), the_result["usage"], the_result["tool_usage"], the_result["tool_count"], 
                         the_result["context_count"], self.debug, task.price_id)
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

    async def send_agent_request_async(
        self,
        agent_configuration: AgentConfiguration,
        task: Task,
        llm_model: str = None,
    ) -> Any:
        """
        Asynchronous version of send_agent_request method.
        """
        from ..trace import sentry_sdk
        from ..level_utilized.utility import CallErrorException

        if llm_model is None:
            llm_model = self.default_llm_model

        tools = tools_serializer(task.tools)
        response_format = task.response_format
        
        with sentry_sdk.start_transaction(op="task", name="Agent.send_agent_request_async") as transaction:
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
            else:
                context = None

            with sentry_sdk.start_span(op="prepare_request"):
                # Prepare the request data
                data = {
                    "agent_id": agent_configuration.agent_id,
                    "prompt": task.description + await task.additional_description(self), 
                    "images": task.images_base_64,
                    "response_format": response_format_str,
                    "tools": tools or [],
                    "context": context,
                    "llm_model": llm_model,
                    "system_prompt": None,
                    "context_compress": agent_configuration.context_compress,
                    "memory": agent_configuration.memory
                }

            retry_count = 0
            while True:
                try:
                    with sentry_sdk.start_span(op="send_request"):
                        # Send the request asynchronously
                        result = await self.send_request_async("/level_two/agent", data)
                        result = result["result"]
                        
                        # Store tool calls in the task if available
                        if isinstance(result, dict) and 'tool_usage' in result:
                            for tool_call in result['tool_usage']:
                                task.add_tool_call(tool_call)
                        
                        if error_handler(result):  # If it's a retriable error
                            if agent_configuration.retry > 0 and retry_count < agent_configuration.retry:  # Check if retries are enabled and we can retry
                                retry_count += 1
                                from ..printing import agent_retry
                                agent_retry(retry_count, agent_configuration.retry)
                                continue  # Try again
                            else:
                                raise CallErrorException(result)  # No more retries, raise the error
                        
                        break  # If no error or non-retriable error, break the loop

                except Exception as e:
                    if agent_configuration.retry > 0 and retry_count < agent_configuration.retry:  # Check if retries are enabled and we can retry
                        retry_count += 1
                        from ..printing import agent_retry
                        agent_retry(retry_count, agent_configuration.retry)
                        continue  # Try again
                    raise e  # No more retries, raise the error

            with sentry_sdk.start_span(op="deserialize"):
                deserialized_result = response_format_deserializer(response_format_str, result)

            # Process result through reliability layer
            processed_result = await ReliabilityProcessor.process_result(
                deserialized_result["result"], 
                agent_configuration.reliability_layer,
                task,
                llm_model
            )
            task._response = processed_result

            if task.response_lang:
                language = Language(task.response_lang, task, llm_model)
                processed_result = await language.transform()
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

            return {
                "result": processed_result, 
                "llm_model": llm_model, 
                "response_format": response_format_req, 
                "usage": deserialized_result["usage"],
                "tool_usage": deserialized_result["tool_usage"],
                "tool_count": len(tools), 
                "context_count": len_of_context
            }

    async def create_characterization_async(self, agent_configuration: AgentConfiguration, llm_model: str = None, price_id: str = None):
        tools = [Search]

        search_task = None
        search_result = None
        if agent_configuration.company_url:
            search_task = Task(description=f"Make a search for {agent_configuration.company_url}", tools=tools, response_format=SearchResult, price_id_=price_id, not_main_task=True)
            await Direct.do_async(search_task, llm_model, retry=agent_configuration.retry, client=agent_configuration.client)
            search_result = search_task.response

        company_objective_task = None
        company_objective_result = None
        if agent_configuration.company_objective:
            context = [search_task] if search_task else None
            company_objective_task = Task(description=f"Generate the company objective for {agent_configuration.company_objective}", 
                                        tools=tools, 
                                        response_format=CompanyObjective,
                                        context=context,
                                        price_id_=price_id,
                                        not_main_task=True)
            await Direct.do_async(company_objective_task, llm_model, retry=agent_configuration.retry, client=agent_configuration.client)
            company_objective_result = company_objective_task.response

        human_objective_result = None
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
            await Direct.do_async(human_objective_task, llm_model, retry=agent_configuration.retry, client=agent_configuration.client)
            human_objective_result = human_objective_task.response

        total_character = Characterization(
            website_content=search_result,
            company_objective=company_objective_result,
            human_objective=human_objective_result,
            name_of_the_human_of_tasks=agent_configuration.name,
            contact_of_the_human_of_tasks=agent_configuration.contact
        )

        return total_character

    async def call_async(self, task: Task, llm_model: str = None):
        """
        Asynchronous version of the call method.
        """
        if llm_model is None:
            llm_model = self.default_llm_model


        result = await self.send_agent_request_async(AgentConfiguration(), task, llm_model)
        task._response = result["result"]
        return task.response

    async def multiple_async(self, agent_configuration: AgentConfiguration, task: Task, llm_model: str = None):
        """
        Asynchronous version of the multiple method.
        """
        if agent_configuration.system_prompt:
            system_prompt = "System prompt: " + agent_configuration.system_prompt
        else:
            system_prompt = None
        # First, determine the mode of operation
        mode_selection_prompt = f"""
You are a Task Analysis AI that helps determine the best mode of task decomposition.

Task Agent name: {agent_configuration.job_title}
{system_prompt}

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
        
        # Use Direct.do_async with the agent's retry setting
        await Direct.do_async(mode_selector, llm_model, retry=agent_configuration.retry, client=agent_configuration.client)
        
        # If level_no_step is selected, return just the end task
        if mode_selector.response.selected_mode == "level_no_step":
            return [Task(description=task.description, images=task.images, response_format=task.response_format, response_lang=task.response_lang, tools=task.tools, price_id_=task.price_id, not_main_task=True)]

        # Generate a list of sub tasks
        prompt = f"""
You are a Task Decomposition AI that helps break down large tasks into smaller, manageable subtasks.

Task Agent name: {agent_configuration.job_title}
{system_prompt}

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

        # Use Direct.do_async with the agent's retry setting
        await Direct.do_async(sub_tasker, llm_model, retry=agent_configuration.retry, client=agent_configuration.client)

        sub_tasks = []

        # Create tasks from subtasks
        for each in sub_tasker.response.sub_tasks:
            new_task = Task(description=each.description + " " + each.required_output + " " + str(each.sources_can_be_used) + " " + str(each.tools) + "Focus to complete the task with right result, Dont ask to human directly do it and give the result.", images=task.images, price_id_=task.price_id, not_main_task=True)
            new_task.tools = task.tools
            sub_tasks.append(new_task)

        # Add the final task that will produce the original desired response format
        end_task = Task(description=task.description, images=task.images, response_format=task.response_format, response_lang=task.response_lang, price_id_=task.price_id, not_main_task=True)
        sub_tasks.append(end_task)

        return sub_tasks

    def call(self, task: Task, llm_model: str = None):
        """
        Synchronous version of the call method that uses the async version internally.
        """
        import asyncio
        
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # If there's a running loop, run the coroutine in that loop
                return asyncio.run_coroutine_threadsafe(
                    self.call_async(task, llm_model), 
                    loop
                ).result()
        except RuntimeError:
            # No running event loop
            pass
        
        # If no running loop or exception occurred, create a new one
        return asyncio.run(self.call_async(task, llm_model))


