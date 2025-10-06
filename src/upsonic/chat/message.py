import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.messages.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart, TextPart, ToolCallPart, BuiltinToolCallPart, ThinkingPart


@dataclass
class ChatMessage:
    """
    Represents a single message in the chat history.
    
    This class provides a clean interface for accessing message content
    and metadata without exposing the internal ModelMessage complexity.
    """
    content: str
    role: Literal["user", "assistant"]
    timestamp: float
    attachments: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_model_message(cls, message: "ModelMessage") -> "ChatMessage":
        """Create a ChatMessage from a ModelMessage."""
        # Import here to avoid circular imports
        from upsonic.messages.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart, ToolCallPart, BuiltinToolCallPart, ThinkingPart
        
        if isinstance(message, ModelRequest):
            # Extract user content from ModelRequest
            content_parts = []
            attachments = []
            
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    # Handle UserPromptPart.content which is str | Sequence[UserContent]
                    if isinstance(part.content, str):
                        content_parts.append(part.content)
                    else:
                        # Handle Sequence[UserContent] where UserContent = str | MultiModalContent
                        for item in part.content:
                            if isinstance(item, str):
                                content_parts.append(item)
                            else:
                                # Handle MultiModalContent (ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent)
                                if hasattr(item, 'url'):
                                    # Handle URL-based content (ImageUrl, AudioUrl, DocumentUrl, VideoUrl)
                                    attachments.append(f"{type(item).__name__}: {item.url}")
                                elif hasattr(item, 'data'):
                                    # Handle BinaryContent
                                    attachments.append(f"{type(item).__name__}: {item.identifier or 'binary_data'}")
                                else:
                                    # Fallback for unknown content types
                                    attachments.append(f"{type(item).__name__}: {str(item)}")
            
            return cls(
                content=" ".join(content_parts),
                role="user",
                timestamp=time.time(),
                attachments=attachments if attachments else None
            )
        elif isinstance(message, ModelResponse):
            # Extract assistant content from ModelResponse
            content_parts = []
            tool_calls = []
            
            for part in message.parts:
                if isinstance(part, TextPart):
                    content_parts.append(part.content)
                elif isinstance(part, (ToolCallPart, BuiltinToolCallPart)):
                    # Handle tool calls
                    tool_calls.append({
                        "tool_name": part.tool_name,
                        "tool_call_id": part.tool_call_id,
                        "args": part.args_as_dict() if part.args else {}
                    })
                elif isinstance(part, ThinkingPart):
                    # Handle thinking parts (for models that support reasoning)
                    content_parts.append(f"[Thinking: {part.content}]")
            
            # Clean up content - remove extra whitespace and ensure proper formatting
            content = " ".join(content_parts).strip()
            if not content and tool_calls:
                # If no text content but has tool calls, provide a default message
                content = f"Used {len(tool_calls)} tool(s)"
            
            return cls(
                content=content,
                role="assistant",
                timestamp=time.time(),
                tool_calls=tool_calls if tool_calls else None,
                metadata={
                    "model_name": getattr(message, 'model_name', None),
                    "provider_name": getattr(message, 'provider_name', None),
                    "usage": message.usage.__dict__ if hasattr(message, 'usage') and message.usage else None,
                    "finish_reason": getattr(message, 'finish_reason', None),
                    "provider_response_id": getattr(message, 'provider_response_id', None),
                    "provider_details": getattr(message, 'provider_details', None),
                    "timestamp": getattr(message, 'timestamp', None),
                    "kind": getattr(message, 'kind', None)
                }
            )
        else:
            # Fallback for unknown message types (including mock objects)
            content = str(message)
            # Try to extract meaningful content from mock objects
            if hasattr(message, 'parts') and message.parts:
                content_parts = []
                for part in message.parts:
                    if hasattr(part, 'content'):
                        content_parts.append(str(part.content))
                if content_parts:
                    content = " ".join(content_parts)
            
            return cls(
                content=content,
                role="assistant",
                timestamp=time.time()
            )
