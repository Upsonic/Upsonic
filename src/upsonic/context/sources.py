from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..knowledge_base.knowledge_base import KnowledgeBase


class ContextSource(BaseModel):
    """
    Abstract base model for all context sources. Serves as a common interface
    and type hint for any object that can be injected into a Task's context.
    """
    enabled: bool = True # might be useful
    source_id: Optional[str] = None


class KnowledgeBaseSource(ContextSource):
    """
    Specifies a KnowledgeBase as a context source for Retrieval-Augmented Generation (RAG).
    This model holds the configuration for how to query the knowledge base.
    """
    knowledge_base: "KnowledgeBase"
    top_k: int = 3

    class Config:
        arbitrary_types_allowed = True


class TaskOutputSource(ContextSource):
    """
    Specifies the output of a previously executed task as a context source.
    This is primarily for use within a Graph to pass state between nodes.
    """
    task_description_or_id: str
    retrieval_mode: str = "full"  # Options: "full", "summary".



class StaticTextSource(ContextSource):
    """
    Specifies a simple, user-provided static string as a context source.
    This is the direct replacement for the old method of passing a raw string.
    """
    text: str

