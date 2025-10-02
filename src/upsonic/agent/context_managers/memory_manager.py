from __future__ import annotations
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import asyncio # This import is crucial for the 'finally' block


if TYPE_CHECKING:
    from upsonic.storage.memory.memory import Memory


class MemoryManager:
    """
    A context manager that integrates the Memory orchestrator into the agent's
    execution pipeline.

    This manager is responsible for two critical phases:
    1.  On entry (`async with`): It calls the memory module to prepare all
        necessary inputs (history, summaries, profiles) before the LLM call.
    2.  On exit (`finally`): It calls the memory module to process the LLM
        response and update all relevant memories in the storage backend.
    """

    def __init__(self, memory: Optional["Memory"]):
        """
        Initializes the MemoryManager.

        Args:
            memory: The configured Memory object from the parent agent.
        """
        self.memory = memory
        self._prepared_inputs: Dict[str, Any] = {
            "message_history": [],
            "context_injection": "",
            "system_prompt_injection": ""
        }
        self._model_response: Optional[Any] = None

    def get_message_history(self) -> List[Any]:
        """
        Provides the prepared message history (full session memory) to the
        agent's core run method.
        """
        return self._prepared_inputs.get("message_history", [])

    def get_context_injection(self) -> str:
        """
        Provides the prepared context string (e.g., session summary) to the
        ContextManager.
        """
        return self._prepared_inputs.get("context_injection", "")

    def get_system_prompt_injection(self) -> str:
        """
        Provides the prepared system prompt string (e.g., user profile) to
        the SystemPromptManager.
        """
        return self._prepared_inputs.get("system_prompt_injection", "")

    def process_response(self, model_response) -> Any:
        """
        Captures the final model response from the LLM call, making it
        available for the memory update process on exit.
        """
        self._model_response = model_response
        return model_response

    @asynccontextmanager
    async def manage_memory(self):
        """
        The asynchronous context manager for orchestrating memory operations
        throughout a task's lifecycle.
        """

        if self.memory:
            self._prepared_inputs = await self.memory.prepare_inputs_for_task()
        
        try:
            yield self
        finally:
            if self.memory and self._model_response:
                task = asyncio.create_task(self.memory.update_memories_after_task(self._model_response))
                await task