from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from upsonic.context.agent import turn_agent_to_string
from upsonic.context.default_prompt import default_prompt

if TYPE_CHECKING:
    from upsonic.agent.agent import Direct
    from upsonic.tasks.tasks import Task


class SystemPromptManager:
    """
    A context manager responsible for constructing the final system prompt.

    This manager isolates the logic for defining the AI's core behavior,
    persona, and high-level instructions, separating it from the dynamic,
    task-specific context.
    """

    def __init__(self, agent: Direct, task: Task):
        """
        Initializes the SystemPromptManager.

        Args:
            agent: The parent `Direct` agent instance executing the call.
                   This is used to access agent-level configurations like a
                   custom system prompt.
            task: The `Task` object for the current operation. This is used
                  to scan for any agent identity information provided in the
                  task's context.
        """
        self.agent = agent
        self.task = task
        self.system_prompt: str = ""

    def _build_system_prompt(self) -> str:
        """
        Builds the complete system prompt string by assembling its components.

        This method implements the core logic:
        1.  Selects the base behavioral prompt, giving precedence to a custom
            prompt on the `Direct` agent over the framework's default.
        2.  Scans the task's context for other `Direct` agent instances
            and injects their serialized information as an <Agents> block.

        Returns:
            The fully constructed system prompt string.
        """
        prompt_parts = []

        if self.agent.system_prompt:
            base_prompt = self.agent.system_prompt
        else:
            base_prompt = default_prompt().prompt
        
        prompt_parts.append(base_prompt.strip())

        agent_context_str = "<YourCharacter>"
        found_agent_context = False

        if self.task.context:
            for item in self.task.context:
                if isinstance(item, type(self.agent)):
                    agent_context_str += f"\nAgent ID ({item.get_agent_id()}): {turn_agent_to_string(item)}"
                    found_agent_context = True
        
        if found_agent_context:
            agent_context_str += "\n</YourCharacter>"
            prompt_parts.append(agent_context_str)
            
        return "\n\n".join(prompt_parts)
    
    def get_system_prompt(self) -> str:
        """
        Public getter to retrieve the constructed system prompt.

        This method is called from within the `do_async` pipeline after this
        manager has been entered.

        Returns:
            The final system prompt string.
        """
        return self.system_prompt

    @asynccontextmanager
    async def manage_system_prompt(self):
        """
        The asynchronous context manager for building the system prompt.

        Upon entering the `async with` block, this manager builds the
        system prompt and makes it available via the `get_system_prompt` method.
        """
        self.system_prompt = self._build_system_prompt()
        try:
            yield self
        finally:
            pass