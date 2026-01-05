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
        necessary inputs (history, summaries, profiles, metadata) before the LLM call.
    2.  On exit (`finally`): It calls the memory module to process the LLM
        response and update all relevant memories in the storage backend.
    """

    def __init__(self, memory: Optional["Memory"], agent_metadata: Optional[Dict[str, Any]] = None):
        """
        Initializes the MemoryManager.

        Args:
            memory: The configured Memory object from the parent agent.
            agent_metadata: Optional metadata dict from the Agent to inject into prompts.
        """
        self.memory = memory
        self.agent_metadata = agent_metadata or {}
        self._prepared_inputs: Dict[str, Any] = {
            "message_history": [],
            "context_injection": "",
            "system_prompt_injection": "",
            "metadata_injection": ""
        }
        self._model_response: Optional[Any] = None
        self._agent_run_output: Optional[Any] = None

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
        injection = self._prepared_inputs.get("system_prompt_injection", "")
        return injection
    
    def get_metadata_injection(self) -> str:
        """
        Provides the prepared metadata string to inject into the user prompt.
        This includes both agent-level metadata and session-level metadata.
        """
        return self._prepared_inputs.get("metadata_injection", "")

    def process_response(self, response: Any, agent_run_output: Any = None) -> Any:
        """
        Captures the response for memory update on exit.
        
        Args:
            response: The model response OR AgentRunOutput (for backward compat)
            agent_run_output: The AgentRunOutput if response is the raw model response
        """
        # If agent_run_output is provided, use it directly
        if agent_run_output is not None:
            self._model_response = response
            self._agent_run_output = agent_run_output
        else:
            # Check if response is an AgentRunOutput (backward compat)
            from upsonic.run.agent.output import AgentRunOutput
            if isinstance(response, AgentRunOutput):
                self._agent_run_output = response
                self._model_response = None
            else:
                self._model_response = response
        return response
    
    def set_run_output(self, run_output: Any) -> None:
        """Explicitly set the AgentRunOutput for memory update."""
        self._agent_run_output = run_output

    async def aprepare(self) -> None:
        """
        Prepare memory inputs before the LLM call.
        
        This method prepares message history, context injection, system prompt 
        injection, and metadata injection from the memory module.
        """
        if self.memory:
            self._prepared_inputs = await self.memory.prepare_inputs_for_task(
                agent_metadata=self.agent_metadata
            )
        else:
            # Even without memory, inject agent metadata if available
            if self.agent_metadata:
                metadata_parts = []
                for key, value in self.agent_metadata.items():
                    metadata_parts.append(f"  {key}: {value}")
                if metadata_parts:
                    self._prepared_inputs["metadata_injection"] = (
                        "<AgentMetadata>\n" + "\n".join(metadata_parts) + "\n</AgentMetadata>"
                    )
    
    async def afinalize(self) -> None:
        """
        Finalize and update memories after the LLM call.
        
        This method processes the model response and updates all relevant
        memories in the storage backend.
        """
        if self.memory and (self._model_response or self._agent_run_output):
            await self.memory.update_memories_after_task(
                model_response=self._model_response,
                agent_run_output=self._agent_run_output
            )
    
    def prepare(self) -> None:
        """Synchronous version of aprepare."""
        asyncio.get_event_loop().run_until_complete(self.aprepare())
    
    def finalize(self) -> None:
        """Synchronous version of afinalize."""
        asyncio.get_event_loop().run_until_complete(self.afinalize())

    @asynccontextmanager
    async def manage_memory(self):
        """
        The asynchronous context manager for orchestrating memory operations
        throughout a task's lifecycle.
        
        Note: This context manager is kept for backward compatibility.
        For step-based architecture, use aprepare() and afinalize() directly.
        """
        await self.aprepare()
        
        try:
            yield self
        finally:
            await self.afinalize()