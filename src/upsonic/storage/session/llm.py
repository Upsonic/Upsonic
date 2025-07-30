from __future__ import annotations
import time
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field



class Artifact(BaseModel):
    """
    Represents the metadata for a non-textual artifact (e.g., an image,
    document, or audio file) associated with an LLMConversation.

    The binary data itself is stored in a separate blob store (like S3 or a
    local file system), and this model holds the pointer and metadata.
    """
    artifact_id: str = Field(default_factory=lambda: str(uuid4()))

    conversation_id: str

    turn_id: Optional[str] = None

    mime_type: str

    storage_uri: str

    metadata: Dict[str, Any] = Field(default_factory=dict)
