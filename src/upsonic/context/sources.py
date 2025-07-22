from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, TYPE_CHECKING



class ContextSource(BaseModel):
    """
    Abstract base model for all context sources. Serves as a common interface
    and type hint for any object that can be injected into a Task's context.
    """
    enabled: bool = True # might be useful
    source_id: Optional[str] = None



class TaskOutputSource(ContextSource):
    """
    Specifies the output of a previously executed task as a context source.
    This is primarily for use within a Graph to pass state between nodes.
    """
    task_description_or_id: str
    retrieval_mode: str = "full"  # Options: "full", "summary". Might be useful


