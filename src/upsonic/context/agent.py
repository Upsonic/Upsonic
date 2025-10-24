from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.agent.agent import Agent

def turn_agent_to_string(agent: Agent):
    the_dict = {}
    the_dict["id"] = agent.agent_id
    the_dict["name"] = agent.name
    the_dict["company_url"] = agent.company_url
    the_dict["company_objective"] = agent.company_objective
    the_dict["company_description"] = agent.company_description
    the_dict["company_name"] = agent.company_name
    the_dict["system_prompt"] = agent.system_prompt

    # Turn the dict to string
    string_of_dict = json.dumps(the_dict)
    return string_of_dict
