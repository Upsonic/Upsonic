import json
from typing import Any, Optional

from ..direct.direct_llm_cal import Direct as Agent
from ..direct.direct_llm_cal import Direct
from .base_strategy import ContextProcessingStrategy


class AgentContextStrategy(ContextProcessingStrategy):
    """Strategy for processing Agent context items."""

    def can_process(self, context_item: Any) -> bool:
        """Check if this strategy can process the given context item."""
        return isinstance(context_item, (Agent, Direct))

    def process(self, context_item: Any) -> str:
        """Process the agent context item."""
        if not self.can_process(context_item):
            raise ValueError(
                f"AgentContextStrategy cannot process {type(context_item)}"
            )

        agent = context_item
        return f"Agent ID ({agent.get_agent_id()}): {turn_agent_to_string(agent)}\n"

    def get_section_name(self) -> str:
        """Get the XML section name for agents."""
        return "Agents"

    def validate(self, context_item: Any) -> Optional[str]:
        """Validate the agent context item."""
        if not isinstance(context_item, (Agent, Direct)):
            return f"Expected Agent or Direct, got {type(context_item)}"

        if not hasattr(context_item, "get_agent_id"):
            return "Agent missing get_agent_id method"

        return None


def turn_agent_to_string(agent: Agent):
    """Convert agent to JSON string representation."""
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
