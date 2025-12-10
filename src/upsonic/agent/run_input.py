"""
Run Input Module

This module provides the RunInput dataclass that captures the raw input
data passed to Agent.do() or Agent.stream() methods.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task
    from upsonic.messages.messages import BinaryContent


@dataclass
class RunInput:
    """Container for the raw input data passed to Agent.run().

    This captures the original input exactly as provided by the user,
    separate from the processed messages that go to the model.

    Attributes:
        input_content: The literal input message/content passed to run() - can be a string or Task
        images: Images directly passed to run()
        videos: Videos directly passed to run()
        audios: Audio files directly passed to run()
        files: Files directly passed to run()
        created_at: Timestamp when the input was created
    """

    input_content: Union[str, "Task", Dict[str, Any], List[Any]]
    """The literal input message/content passed to run()."""

    images: Optional[Sequence["BinaryContent"]] = None
    """Images directly passed to run()."""

    videos: Optional[Sequence["BinaryContent"]] = None
    """Videos directly passed to run()."""

    audios: Optional[Sequence["BinaryContent"]] = None
    """Audio files directly passed to run()."""

    files: Optional[Sequence["BinaryContent"]] = None
    """Files directly passed to run()."""

    created_at: datetime = field(default_factory=datetime.utcnow)
    """Timestamp when the input was created."""

    def input_content_string(self) -> str:
        """Convert input content to a string representation.
        
        Returns:
            str: String representation of the input content
        """
        import json
        from pydantic import BaseModel

        if isinstance(self.input_content, str):
            return self.input_content
        elif isinstance(self.input_content, BaseModel):
            return self.input_content.model_dump_json(exclude_none=True)
        elif hasattr(self.input_content, 'description'):
            # Task object - return its description
            return str(self.input_content.description)
        elif isinstance(self.input_content, dict):
            return json.dumps(self.input_content)
        elif isinstance(self.input_content, list):
            return json.dumps(self.input_content)
        else:
            return str(self.input_content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the RunInput
        """
        result: Dict[str, Any] = {}

        if self.input_content is not None:
            if isinstance(self.input_content, str):
                result["input_content"] = self.input_content
            elif hasattr(self.input_content, 'model_dump'):
                # Pydantic BaseModel (including Task)
                result["input_content"] = self.input_content.model_dump(exclude_none=True)
            elif hasattr(self.input_content, 'description'):
                # Task object without model_dump
                result["input_content"] = {
                    "description": self.input_content.description,
                    "type": "Task"
                }
            elif isinstance(self.input_content, (dict, list)):
                result["input_content"] = self.input_content
            else:
                result["input_content"] = str(self.input_content)

        if self.images:
            result["images"] = [
                img.to_dict() if hasattr(img, 'to_dict') else {"data": str(img)}
                for img in self.images
            ]
        if self.videos:
            result["videos"] = [
                vid.to_dict() if hasattr(vid, 'to_dict') else {"data": str(vid)}
                for vid in self.videos
            ]
        if self.audios:
            result["audios"] = [
                aud.to_dict() if hasattr(aud, 'to_dict') else {"data": str(aud)}
                for aud in self.audios
            ]
        if self.files:
            result["files"] = [
                file.to_dict() if hasattr(file, 'to_dict') else {"data": str(file)}
                for file in self.files
            ]

        if self.created_at:
            result["created_at"] = self.created_at.isoformat()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunInput":
        """Create RunInput from dictionary.
        
        Args:
            data: Dictionary representation of RunInput
            
        Returns:
            RunInput: Reconstructed RunInput instance
        """
        from datetime import datetime as dt
        
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = dt.fromisoformat(created_at)
        elif created_at is None:
            created_at = dt.utcnow()

        return cls(
            input_content=data.get("input_content", ""),
            images=data.get("images"),
            videos=data.get("videos"),
            audios=data.get("audios"),
            files=data.get("files"),
            created_at=created_at
        )

    @classmethod
    def from_task(cls, task: "Task") -> "RunInput":
        """Create RunInput from a Task object.
        
        This is a convenience method to create RunInput from a Task,
        extracting attachments as appropriate media types.
        
        Args:
            task: The Task object to create RunInput from
            
        Returns:
            RunInput: A new RunInput instance
        """
        return cls(
            input_content=task,
            images=None,  # Attachments are handled by Task itself
            videos=None,
            audios=None,
            files=None,
            created_at=datetime.utcnow()
        )

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        content_preview = self.input_content_string()[:50]
        if len(self.input_content_string()) > 50:
            content_preview += "..."
        return f"RunInput(content={content_preview!r})"
