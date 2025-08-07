from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Optional

from upsonic.tasks.tasks import Task
from upsonic.knowledge_base.knowledge_base import KnowledgeBase
from upsonic.context.task import turn_task_to_string
from upsonic.context.sources import TaskOutputSource

if TYPE_CHECKING:
    from upsonic.agent.agent import Direct
    from upsonic.graph.graph import State
    from upsonic.agent.context_managers.memory_manager import MemoryManager


class ContextManager:
    """
    A context manager for building the dynamic, task-specific context prompt.

    This manager is responsible for aggregating all situational data relevant
    to the current task. This version is enhanced to be fully "graph-aware,"
    capable of resolving `TaskOutputSource` objects using a `State` object
    from a graph execution.
    """

    def __init__(self, agent: Direct, task: Task, state: Optional[State] = None):
        """
        Initializes the ContextManager.

        Args:
            agent: The parent `Direct` agent instance.
            task: The `Task` object for the current operation.
            state: An optional `State` object from a `Graph` execution.
                   This is essential for resolving `TaskOutputSource` context.
        """
        self.agent = agent
        self.task = task
        self.state = state
        self.context_prompt: str = ""

    async def _build_context_prompt(self, memory_handler: Optional[MemoryManager]) -> str:
        """
        Asynchronously builds the complete contextual prompt string.

        This method now fully supports graph-specific context by resolving
        `TaskOutputSource` objects into concrete data from the provided
        `state`, in addition to its existing responsibilities.

        Returns:
            A formatted string containing all relevant situational context.
        """
        final_context_parts = []

        if memory_handler:
            context_injection = memory_handler.get_context_injection()
            if context_injection:
                final_context_parts.append(context_injection)

        if self.task.context:

            knowledge_base_parts = []
            task_parts = []
            previous_task_output_parts = []
            additional_parts = []

            for item in self.task.context:
                if isinstance(item, Task):
                    task_parts.append(f"Task ID ({item.get_task_id()}): " + turn_task_to_string(item))
                
                elif isinstance(item, KnowledgeBase):
                    if item.rag:
                        await item.setup_rag(self.agent)
                        rag_results = await item.query(self.task.description)
                        if rag_results:
                            knowledge_base_parts.append(f"<rag>{' '.join(rag_results)}</rag>")
                    else:
                        knowledge_base_parts.append(item.markdown())

                elif isinstance(item, str):
                    additional_parts.append(item)

                elif isinstance(item, TaskOutputSource) and self.state:

                    source_output = self.state.get_task_output(item.task_description_or_id)
                    
                    if source_output is not None:
                        output_str = ""
                        if hasattr(source_output, 'model_dump_json'):
                            output_str = source_output.model_dump_json(indent=2)
                        elif hasattr(source_output, 'model_dump'):
                            output_str = json.dumps(source_output.model_dump(), default=str, indent=2)
                        else:
                            output_str = str(source_output)

                        previous_task_output_parts.append(
                            f"<PreviousTaskOutput id='{item.task_description_or_id}'>\n{output_str}\n</PreviousTaskOutput>"
                        )

            if task_parts:
                final_context_parts.append("<Tasks>\n" + "\n".join(task_parts) + "\n</Tasks>")
            if knowledge_base_parts:
                final_context_parts.append("<Knowledge Base>\n" + "\n".join(knowledge_base_parts) + "\n</Knowledge Base>")
            if previous_task_output_parts:
                final_context_parts.extend(previous_task_output_parts)
            if additional_parts:
                final_context_parts.append("<Additional Context>\n" + "\n".join(additional_parts) + "\n</Additional Context>")

        if not final_context_parts:
            return ""
        
        return "<Context>\n" + "\n\n".join(final_context_parts) + "\n</Context>"

    def get_context_prompt(self) -> str:
        """Public getter to retrieve the constructed context prompt."""
        return self.context_prompt

    @asynccontextmanager
    async def manage_context(self, memory_handler: Optional[MemoryManager] = None):
        """The asynchronous context manager for building the task-specific context."""
        self.context_prompt = await self._build_context_prompt(memory_handler)
        self.task.context_formatted = self.context_prompt
            
        try:
            yield self
        finally:
            pass