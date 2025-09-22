"""
This module defines the individual, atomic "parts" of a message.
Each class represents a specific type of content, such as text, an image URL,
or a tool call, that can be assembled into a complete ModelRequest or ModelResponse.
"""

import mimetypes
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .types import (
    AudioMediaType,
    DocumentMediaType,
    ImageMediaType,
    VideoMediaType,
)

# --- Abstract Base Classes ---

class FileUrl(BaseModel, ABC):
    """Abstract base class for any URL-based file content."""
    url: str = Field(..., description="The URL to the file")
    force_download: bool = Field(default=False, description="Hint to the model to download the content instead of just using the URL.")
    identifier: Optional[str] = Field(default=None, description="A unique identifier for the file.")

    @property
    @abstractmethod
    def media_type(self) -> str:
        """The inferred media type of the file based on its URL."""
        raise NotImplementedError
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v or not isinstance(v, str):
            raise ValueError("URL must be a non-empty string")
        if not v.startswith(('http://', 'https://', 'file://', 'data:')):
            raise ValueError("URL must start with http://, https://, file://, or data:")
        return v

# --- Multi-Modal Content Parts ---

class ImageUrl(FileUrl):
    """Represents a URL to an image."""
    @property
    def media_type(self) -> ImageMediaType:
        url_str = str(self.url)
        
        # Handle data URIs
        if url_str.startswith('data:'):
            if 'image/jpeg' in url_str or 'image/jpg' in url_str:
                return 'image/jpeg'
            elif 'image/png' in url_str:
                return 'image/png'
            elif 'image/gif' in url_str:
                return 'image/gif'
            elif 'image/webp' in url_str:
                return 'image/webp'
            else:
                # Default to PNG for data URIs if no specific type found
                return 'image/png'
        
        # Handle regular URLs
        if url_str.endswith(('.jpg', '.jpeg')):
            return 'image/jpeg'
        elif url_str.endswith('.png'):
            return 'image/png'
        elif url_str.endswith('.gif'):
            return 'image/gif'
        elif url_str.endswith('.webp'):
            return 'image/webp'
        raise ValueError(f"Could not infer image media type from URL: {url_str}")

class AudioUrl(FileUrl):
    """Represents a URL to an audio file."""
    @property
    def media_type(self) -> AudioMediaType:
        url_str = str(self.url)
        
        # Handle data URIs
        if url_str.startswith('data:'):
            if 'audio/mpeg' in url_str or 'audio/mp3' in url_str:
                return 'audio/mpeg'
            elif 'audio/wav' in url_str:
                return 'audio/wav'
            elif 'audio/ogg' in url_str:
                return 'audio/ogg'
            elif 'audio/flac' in url_str:
                return 'audio/flac'
            elif 'audio/aiff' in url_str:
                return 'audio/aiff'
            elif 'audio/aac' in url_str:
                return 'audio/aac'
            else:
                # Default to WAV for data URIs if no specific type found
                return 'audio/wav'
        
        # Handle regular URLs
        if url_str.endswith('.mp3'):
            return 'audio/mpeg'
        elif url_str.endswith('.wav'):
            return 'audio/wav'
        elif url_str.endswith('.ogg'):
            return 'audio/ogg'
        elif url_str.endswith('.flac'):
            return 'audio/flac'
        elif url_str.endswith('.aiff'):
            return 'audio/aiff'
        elif url_str.endswith('.aac'):
            return 'audio/aac'
        # Add other audio formats as needed
        raise ValueError(f"Could not infer audio media type from URL: {url_str}")

class VideoUrl(FileUrl):
    """Represents a URL to a video file."""
    @property
    def media_type(self) -> VideoMediaType:
        url_str = str(self.url)
        if url_str.endswith('.mp4'):
            return 'video/mp4'
        elif url_str.endswith('.mov'):
            return 'video/quicktime'
        # Add other video formats as needed
        raise ValueError(f"Could not infer video media type from URL: {url_str}")

class DocumentUrl(FileUrl):
    """Represents a URL to a document."""
    @property
    def media_type(self) -> DocumentMediaType:
        """Infers media type, using the mimetypes library as a fallback."""
        url_str = str(self.url)
        if url_str.endswith('.pdf'):
            return 'application/pdf'
        elif url_str.endswith('.txt'):
            return 'text/plain'
        elif url_str.endswith('.md'):
            return 'text/markdown'
        
        type_, _ = mimetypes.guess_type(url_str)
        if type_ is None:
            raise ValueError(f"Could not infer document media type from URL: {url_str}")
        # Validate that the type is in our allowed types
        allowed_types = [
            'application/pdf', 'text/plain', 'text/csv',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'text/html', 'text/markdown', 'application/vnd.ms-excel'
        ]
        if type_ not in allowed_types:
            raise ValueError(f"Unsupported document media type: {type_}")
        return type_  # type: ignore

class BinaryContent(BaseModel):
    """Represents raw binary content, such as an image or audio file."""
    data: bytes
    media_type: str
    identifier: Optional[str] = Field(default=None, description="A unique identifier for the content.")

# Union type for any multi-modal content part
MultiModalContent = Union[ImageUrl, AudioUrl, VideoUrl, DocumentUrl, BinaryContent]

# --- User and System Prompt Parts ---

class SystemPromptPart(BaseModel):
    """A system prompt, providing context and guidance to the model."""
    content: str
    part_kind: Literal["system_prompt"] = "system_prompt"

class UserPromptPart(BaseModel):
    """A user prompt, which can contain text and/or multi-modal content."""
    content: Union[str, List[Union[str, MultiModalContent]]]
    part_kind: Literal["user_prompt"] = "user_prompt"

# --- Model Response Parts ---

class TextPart(BaseModel):
    """A plain text response from a model."""
    content: str
    part_kind: Literal["text"] = "text"

class ThinkingPart(BaseModel):
    """A 'thinking' or internal monologue response from a model."""
    content: str
    part_kind: Literal["thinking"] = "thinking"

# --- Tool-Related Parts ---

class ToolCallPart(BaseModel):
    """A request from the model to call a tool."""
    tool_name: str
    args: Dict[str, Any]
    tool_call_id: str
    part_kind: Literal["tool_call"] = "tool_call"

class BuiltinToolCallPart(BaseModel):
    """A tool call to a built-in, framework-managed tool."""
    tool_name: str
    args: Dict[str, Any]
    tool_call_id: str
    provider_name: str
    part_kind: Literal["builtin_tool_call"] = "builtin_tool_call"

class ToolReturnPart(BaseModel):
    """The result of a successful tool execution, to be sent back to the model."""
    tool_name: str
    content: Any
    tool_call_id: str
    part_kind: Literal["tool_return"] = "tool_return"

class BuiltinToolReturnPart(BaseModel):
    """The result from a built-in, framework-managed tool."""
    tool_name: str
    content: Any
    tool_call_id: str
    provider_name: str
    part_kind: Literal["builtin_tool_return"] = "builtin_tool_return"

class RetryPromptPart(BaseModel):
    """A message asking the model to retry due to a validation error or other issue."""
    content: Union[str, List[Dict]]
    tool_name: Optional[str]
    tool_call_id: str
    part_kind: Literal["retry_prompt"] = "retry_prompt"
