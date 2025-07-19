from __future__ import annotations
from typing import Dict, Type, List, TYPE_CHECKING

from .sources import KnowledgeBaseSource
from .processors import PROCESSOR_REGISTRY, ContextProcessor

if TYPE_CHECKING:
    from ..tasks.tasks import Task
    from ..graph.graph import State

class DataInjector:
    """
    Handles the injection of dynamic, data-centric context (like RAG results
    and previous task outputs) into the user-facing prompt.
    """
    def __init__(self):
        """
        Initializes the injector by registering only data-level processors.
        """
        self.processors: Dict[Type, ContextProcessor] = PROCESSOR_REGISTRY
        # Filtering out non-data processors
        self.processors = {k: v for k, v in self.processors.items() if v}
        print(f"DataInjector initialized with {len(self.processors)} processors.")

    async def prepare_rag_systems(self, task: "Task"):
        """
        Initializes all RAG systems defined in the task's context.
        """
        rag_prepared = False
        for source in task.context:
            if isinstance(source, KnowledgeBaseSource) and source.enabled:

                await source.knowledge_base.setup_rag(client=None) # passing placeholder right now
                rag_prepared = True
        if not rag_prepared:
            print("No RAG systems to prepare for this task.")


    async def get_data_context_string(self, task: "Task", state: "State" = None) -> str:
        """
        Builds a single string containing all processed data-level context.
        """
        data_parts: List[str] = []
        if task.context:
            for source_block in task.context:
                if type(source_block) in self.processors:
                    if source_block.enabled:
                        processor = self.processors[type(source_block)]
                        processed_string = await processor.process(source_block, task, state)
                        if processed_string:
                            data_parts.append(processed_string)
        
        if not data_parts:
            return ""

        return f"<CONTEXT_DATA>\n" + "\n\n".join(data_parts) + "\n</CONTEXT_DATA>\n"