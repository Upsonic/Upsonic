import json
from ..direct.direct_llm_cal import Direct as Agent

def turn_agent_to_string(agent: Agent):
    the_dict = {}
    the_dict["id"] = agent.agent_id
    the_dict["name"] = agent.name
    the_dict["company_url"] = agent.company_url
    the_dict["company_objective"] = agent.company_objective
    the_dict["company_description"] = agent.company_description
    the_dict["system_prompt"] = agent.system_prompt

    # Turn the dict to string
    string_of_dict = json.dumps(the_dict)
    return string_of_dict
