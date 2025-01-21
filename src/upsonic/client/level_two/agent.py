import copy
import time
import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
import dill
import base64
import httpx
import hashlib
from typing import Any, List, Dict, Optional, Type, Union
from pydantic import BaseModel

from ..tasks.tasks import Task

from ..printing import agent_end, agent_total_cost



from ..tasks.task_response import ObjectResponse

from ..agent_configuration.agent_configuration import AgentConfiguration

from ..level_utilized.utility import context_serializer


from ..level_utilized.utility import context_serializer, response_format_serializer, tools_serializer, response_format_deserializer, error_handler



from ...storage.caching import save_to_cache_with_expiry, get_from_cache_with_expiry


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
                    agent_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_count"], the_result["context_count"], self.debug)
            else:
                the_result = self.send_agent_request(agent_configuration, task, llm_model)
                the_result["time"] = time.time() - start_time
                results.append(the_result)
                agent_end(the_result["result"], the_result["llm_model"], the_result["response_format"], start_time, time.time(), the_result["usage"], the_result["tool_count"], the_result["context_count"], self.debug)
        except Exception as e:

            try:
                from ...server import stop_dev_server, stop_main_server, is_tools_server_running, is_main_server_running

                if is_tools_server_running() or is_main_server_running():
                    stop_dev_server()

            except Exception as e:
                pass

            raise e

        end_time = time.time()

        

        return results
        


    def send_agent_request(
        self,
        agent_configuration: AgentConfiguration,
        task: Task,

        llm_model: str = None,
    ) -> Any:
        from ..trace import sentry_sdk
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



        tools = tools_serializer(task.tools)

        response_format = task.response_format
        with sentry_sdk.start_transaction(op="task", name="Agent.send_agent_request") as transaction:
            with sentry_sdk.start_span(op="serialize"):
                # Serialize the response format if it's a type or BaseModel
                response_format_str = response_format_serializer(task.response_format)


                context = context_serializer(task.context, self)



            with sentry_sdk.start_span(op="prepare_request"):
                # Prepare the request data
                data = {
                    "agent_id": agent_configuration.agent_id,
                    "prompt": task.description,
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




        task._response = deserialized_result["result"]



        response_format_req = None
        if response_format_str == "str":
            response_format_req = response_format_str
        else:
            # Class name
            response_format_req = response_format.__name__
        
        if context is None:
            context = []

        len_of_context = len(task.context) if task.context is not None else 0

        return {"result": deserialized_result["result"], "llm_model": llm_model, "response_format": response_format_req, "usage": deserialized_result["usage"], "tool_count": len(tools), "context_count": len_of_context}








    def create_characterization(self, agent_configuration: AgentConfiguration, llm_model: str = None):
        tools = ["google", "read_website"]




        search_task = Task(description=f"Make a search for {agent_configuration.company_url}", tools=tools, response_format=SearchResult)

        self.call(search_task, llm_model=llm_model)





        company_objective_task = Task(description=f"Generate the company objective for {agent_configuration.company_url}", tools=tools, response_format=CompanyObjective, context=search_task)

        self.call(company_objective_task, llm_model=llm_model)




        human_objective_task = Task(description=f"Generate the human objective for {agent_configuration.job_title}", tools=tools, response_format=HumanObjective, context=[search_task, company_objective_task])

        self.call(human_objective_task, llm_model=llm_model)





        total_character = Characterization(website_content=search_task.response, company_objective=company_objective_task.response, human_objective=human_objective_task.response, name_of_the_human_of_tasks=agent_configuration.name, contact_of_the_human_of_tasks=agent_configuration.contact)

        return total_character







    def agent(self, agent_configuration: AgentConfiguration, task: Task,  llm_model: str = None):

        original_task = task




        copy_agent_configuration = copy.deepcopy(agent_configuration)
        copy_agent_configuration_json = copy_agent_configuration.model_dump_json(include={"job_title", "company_url", "company_objective", "name", "contact"})


        
        the_characterization_cache_key = f"characterization_{hashlib.sha256(copy_agent_configuration_json.encode()).hexdigest()}"

        if agent_configuration.caching:
            the_characterization = get_from_cache_with_expiry(the_characterization_cache_key)
            if the_characterization is None:
                the_characterization = self.create_characterization(agent_configuration, llm_model)
                save_to_cache_with_expiry(the_characterization, the_characterization_cache_key, agent_configuration.cache_expiry)
        else:
            the_characterization = self.create_characterization(agent_configuration, llm_model)



        knowledge_base = None

        if agent_configuration.knowledge_base:
            knowledge_base = self.knowledge_base(agent_configuration, llm_model)
            



        
        the_task = task

        is_it_sub_task = False
        shared_context = []

        if agent_configuration.sub_task:
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


        if agent_configuration.tools:
            if isinstance(the_task, list):
                for each in the_task:
                    each.tools = agent_configuration.tools

        


        results = []    
        if isinstance(the_task, list):
            for each in the_task:
                if is_it_sub_task:
                    if shared_context:
                        each.context += shared_context


                result = self.agent_(agent_configuration, each, llm_model=llm_model)
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




    def multiple(self, task: Task, llm_model: str = None):
        # Generate a list of sub tasks


        
        class SubTask(ObjectResponse):
            description: str
            sources_can_be_used: List[str]
            required_output: str
            tools: List[str]
        class SubTaskList(ObjectResponse):
            sub_tasks: List[SubTask]




        prompt = "You are a helpful assistant. User have an general task. You need to generate a list of sub tasks. Each sub task should be a Actionable step of main task. You need to return a list of sub tasks. You should say to agent to make this job not making plan again and again. We need actions. If  If you have tools that can help you for the task specify them in the task. If there is an context its the user want to see so create tasks to fill them all. Create rich tasks for every user requested field. Only do user requested things. Dont make any assumptions."

        sub_tasker = Task(description=prompt, response_format=SubTaskList, context=[task, task.response_format], tools=task.tools)

        self.call(sub_tasker, llm_model)

        sub_tasks = []

        for each in sub_tasker.response.sub_tasks:

            new_task = Task(description=each.description+ " " + each.required_output + " " + str(each.sources_can_be_used) + " " + str(each.tools))
            new_task.tools = task.tools
            sub_tasks.append(new_task)




        end_task = Task(description=task.description, response_format=task.response_format)
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
                selecting_task  = Task(description="Select an agent for this task", response_format=SelectedAgent, context=[the_agents_, each])

                self.call(selecting_task, llm_model)

                if selecting_task.response.selected_agent in the_agents:
                    is_end = True



                agent_tasks.append({
                    "agent": the_agents[selecting_task.response.selected_agent],
                    "task": each
                })
                    



        for each in agent_tasks:
            self.agent(each["agent"], each["task"], llm_model)


        return the_agents


