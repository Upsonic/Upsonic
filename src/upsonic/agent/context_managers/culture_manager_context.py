"""
CultureContextManager - Context manager for culture lifecycle in agent pipeline.

This module provides the CultureContextManager class that integrates
cultural knowledge into the agent's execution pipeline.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from upsonic.culture.manager import CultureManager


class CultureContextManager:
    """
    A context manager that integrates CultureManager into the agent's
    execution pipeline.
    
    This manager is responsible for:
    1. On entry: Preparing cultural knowledge context for system prompt injection
    2. Tracking conversation messages for potential culture extraction
    
    Note: Actual culture extraction is handled by CultureUpdateStep in the pipeline,
    NOT in this context manager's finally block. This ensures proper async handling
    and separation of concerns.
    
    Notice: Culture is an experimental feature and is subject to change.
    """

    def __init__(
        self,
        culture_manager: Optional["CultureManager"],
        update_cultural_knowledge: bool = False,
    ):
        """
        Initialize the CultureContextManager.
        
        Args:
            culture_manager: The CultureManager instance for culture operations
            update_cultural_knowledge: If True, track messages for culture extraction
        """
        self.culture_manager = culture_manager
        self.update_cultural_knowledge = update_cultural_knowledge
        
        self._culture_context: str = ""
        self._conversation_messages: List[Any] = []
        self._model_response: Optional[Any] = None

    def get_culture_context(self) -> str:
        """
        Get the prepared cultural knowledge context for system prompt injection.
        
        Returns:
            Formatted cultural knowledge string
        """
        return self._culture_context

    def add_conversation_message(self, message: Any) -> None:
        """
        Add a message to the conversation history for culture extraction.
        
        Args:
            message: Message to add (user input or assistant response)
        """
        self._conversation_messages.append(message)

    def get_conversation_messages(self) -> List[Any]:
        """
        Get all tracked conversation messages.
        
        Returns:
            List of conversation messages
        """
        return self._conversation_messages

    def process_response(self, model_response: Any) -> Any:
        """
        Capture the final model response for culture extraction.
        
        Args:
            model_response: The model's response
            
        Returns:
            The same model response (passthrough)
        """
        self._model_response = model_response
        return model_response

    def get_model_response(self) -> Optional[Any]:
        """
        Get the captured model response.
        
        Returns:
            The model response if captured, None otherwise
        """
        return self._model_response

    async def aprepare(self) -> None:
        """
        Prepare cultural knowledge context before the LLM call.
        
        This prepares cultural knowledge context for system prompt injection.
        """
        from upsonic.utils.printing import culture_debug, culture_warning
        
        if self.culture_manager:
            try:
                culture_debug(
                    "Preparing culture context for system prompt",
                    debug=self.culture_manager.debug
                )
                self._culture_context = await self.culture_manager.aget_culture_context()
                
                if self._culture_context:
                    culture_debug(
                        f"Culture context prepared ({len(self._culture_context)} chars)",
                        debug=self.culture_manager.debug
                    )
                else:
                    culture_debug(
                        "No cultural knowledge available for context",
                        debug=self.culture_manager.debug
                    )
            except Exception as e:
                # Don't fail the agent if culture preparation fails
                culture_warning(
                    f"Failed to prepare culture context: {e}",
                    debug=self.culture_manager.debug if self.culture_manager else False
                )
                self._culture_context = ""
    
    async def afinalize(self) -> None:
        """
        Finalize culture context after the LLM call.
        
        Note: Culture extraction is handled by CultureUpdateStep in the pipeline,
        NOT here. This ensures proper async handling and that it runs AFTER
        memory tracking completes.
        """
        pass
    
    def prepare(self) -> None:
        """Synchronous version of aprepare."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.aprepare())
    
    def finalize(self) -> None:
        """Synchronous version of afinalize."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.afinalize())

    @asynccontextmanager
    async def manage_culture(self):
        """
        Async context manager for orchestrating culture operations
        throughout a task's lifecycle.
        
        Note: This context manager is kept for backward compatibility.
        For step-based architecture, use aprepare() and afinalize() directly.
        
        Yields:
            self for accessing culture context and methods
        """
        await self.aprepare()
        
        try:
            yield self
        finally:
            await self.afinalize()
