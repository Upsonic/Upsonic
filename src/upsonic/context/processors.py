from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Type, Dict
import json
from enum import Enum

from .sources import (
    ContextSource,
    StaticTextSource,
    TaskOutputSource,
    KnowledgeBaseSource
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..tasks.tasks import Task
    from ..graph.graph import State



class SourceType(Enum):
    STATIC = StaticTextSource
    TASK_OUTPUT = TaskOutputSource
    KNOWLEDGE_BASE = KnowledgeBaseSource


class ContextProcessor(ABC):
    """
    Abstract base class for processing a ContextSource into a string for an LLM prompt.
    Each concrete implementation of this class will be responsible for handling one
    specific type of ContextSource.
    """

    @property
    @abstractmethod
    def source_type(self) -> Type["ContextSource"]:
        """
        A property that must be implemented by subclasses. It returns the
        specific ContextSource class that this processor is designed to handle. 
        This allows for automatic registration.
        """
        pass

    @abstractmethod
    async def process(self, source: "ContextSource", task: "Task", state: "State" = None) -> str:
        """
        The core logic for processing the context source. It takes the source data,
        the current task, and optionally the graph state, and returns a formatted
        string ready to be injected into the system prompt.

        Args:
            source: The specific ContextSource object to process (e.g., an instance of
                    KnowledgeBaseSource).
            task: The current task being executed. This is useful for context-aware
                  processing, like using the task's description for a RAG query.
            state: The graph's state object. This is essential for processors that
                   depend on the output of other nodes in a graph, like the
                   TaskOutputProcessor. It is None when not running in a graph.

        Returns:
            A formatted string to be injected into the system prompt. If no context
            can be generated, it should return an empty string.
        """
        pass



class StaticTextProcessor(ContextProcessor):
    """Processes a StaticTextSource to include arbitrary, user-provided text."""

    @property
    def source_type(self) -> Type[ContextSource]:
        return StaticTextSource

    async def process(self, source: StaticTextSource, task: "Task", state: "State" = None) -> str:
        if not source.text or not source.text.strip():
            return ""
        
        return f"<StaticContext>\n{source.text.strip()}\n</StaticContext>"



class TaskOutputProcessor(ContextProcessor):
    """Processes a TaskOutputSource to include the result of a previous task."""

    @property
    def source_type(self) -> Type[ContextSource]:
        return TaskOutputSource

    def _turn_task_to_string(self, task_output: any) -> str:
        """
        This logic is inspired by the old `task.py` file.
        Serializes a task's output to a string.
        """
        if isinstance(task_output, (str, int, float, bool)):
            return str(task_output)
        
        try:
            return json.dumps(task_output, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
        except (TypeError, AttributeError):
            return str(task_output)

    async def process(self, source: TaskOutputSource, task: "Task", state: "State" = None) -> str:
        if not state:
            return ""

        # We might want to imporve this
        previous_output = state.get_task_output(source.task_description_or_id)

        if previous_output is None:
            return ""

        if source.retrieval_mode == "summary":
            # We might want to summarize the previous task's output.
            pass

        else:
            output_content = self._turn_task_to_string(previous_output)

        return f"<PreviousTaskOutput id='{source.task_description_or_id}'>\n{output_content}\n</PreviousTaskOutput>"



class KnowledgeBaseProcessor(ContextProcessor):
    """
    Processes a KnowledgeBaseSource, performing a RAG query to retrieve relevant context.
    This processor assumes that the KnowledgeBase's RAG system has already been
    initialized by the DataInjector.
    """
    @property
    def source_type(self) -> Type[ContextSource]:
        return KnowledgeBaseSource

    async def process(self, source: KnowledgeBaseSource, task: "Task", state: "State" = None) -> str:
        kb = source.knowledge_base
        
        if not kb or not kb.sources:
            return ""
        

        query = task.description

        rag_results = await kb.query(query)

        if not rag_results:
            return ""
            
        formatted_results = "\n\n".join(f"<Result>{chunk}</Result>" for chunk in rag_results)
        return f"<KnowledgeBaseResults query='{query}'>\n{formatted_results}\n</KnowledgeBaseResults>"
    
PROCESSOR_REGISTRY: Dict[Type, ContextProcessor] = {
    SourceType.STATIC.value: StaticTextProcessor(),
    SourceType.TASK_OUTPUT.value: TaskOutputProcessor(),
    SourceType.KNOWLEDGE_BASE.value: KnowledgeBaseProcessor(),
}