from __future__ import annotations
from typing import List, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from ..tasks.tasks import Task
    from ..direct.direct_llm_cal import Direct

class SystemPromptManager:
    """
    Orchestrates the processing of context sources that belong in the
    system prompt, such as agent persona and default instructions.
    """

    def __init__(self):
        pass


    def _get_default_prompt_text(self) -> str:
        """Contains the hardcoded default prompt for the framework."""
        return """<DefaultSystemPrompt>
You are a helpful assistant that can answer questions and help with tasks.
Please be logical, concise, and to the point.
Your provider is Upsonic.
Think in your backend and dont waste time to write to the answer. Write only what the user want.
</DefaultSystemPrompt>
"""

    def _build_agent_self_context(self, agent: "Direct") -> str:
        """
        Serializes the executing agent's own core information.
        """
        the_dict = {
            "id": agent.get_agent_id(),
            "name": agent.name,
            "company_objective": agent.company_objective,
            "company_url": agent.company_url,
            "company_description": agent.company_description,
            "system_prompt": agent.system_prompt,
        }
        agent_string = json.dumps({k: v for k, v in the_dict.items() if v is not None})
        return f"<AgentInfo id='{agent.get_agent_id()}'>\n{agent_string}\n</AgentInfo>"

    async def build_system_prompt(self, agent: "Direct", task: "Task") -> str:
        """
        Builds the final, static system prompt string.
        """
        processed_parts: List[str] = []
        if not agent.override_default_prompt:
            processed_parts.append(self._get_default_prompt_text().strip())
            
        agent_self_context = self._build_agent_self_context(agent)
        processed_parts.append(agent_self_context)


        return "\n\n".join(processed_parts)