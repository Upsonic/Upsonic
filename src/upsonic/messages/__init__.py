# my_agent_framework/messages/__init__.py

from .base import ModelRequest, ModelResponse, TokenUsage
from .parts import (
    TextPart,
    UserPromptPart,
    SystemPromptPart,
    ImageUrl,
    DocumentUrl,
    BinaryContent,
    ToolCallPart,
    ToolReturnPart,
    ThinkingPart,
    MultiModalContent,
    AudioUrl,
    VideoUrl,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    RetryPromptPart,
)
from .tool_defs import ToolReturn

from .types import FinishReason, MessageKind, ProviderDetails
from .utils import (
    serialize_message,
    deserialize_request,
    deserialize_response,
    to_json,
    from_json_request,
    from_json_response,
    extract_text_content,
    has_tool_calls,
    get_tool_calls,
    create_simple_request,
    create_simple_response,
    add_tool_return,
    validate_message_structure,
    merge_responses,
    filter_stream_events,
)
from .validators import (
    validate_url_format,
    validate_media_type,
    validate_tool_call_id,
    validate_message_consistency,
    validate_multimodal_content,
    validate_token_usage,
    validate_provider_details,
    comprehensive_validation,
    is_message_valid,
    get_validation_summary,
)

# The __all__ list defines the public API for the 'messages' module.
# When a user writes 'from my_agent_framework.messages import *', only these
# names will be imported, preventing clutter and exposing a clean interface.
__all__ = [
    # Core message containers
    "ModelRequest",
    "ModelResponse", 
    "TokenUsage",
    
    # Message parts
    "TextPart",
    "UserPromptPart",
    "SystemPromptPart",
    "ToolCallPart",
    "ToolReturnPart",
    "ThinkingPart",
    "BuiltinToolCallPart",
    "BuiltinToolReturnPart",
    "RetryPromptPart",
    
    # Multi-modal content
    "ImageUrl",
    "AudioUrl", 
    "VideoUrl",
    "DocumentUrl",
    "BinaryContent",
    "MultiModalContent",
    
    # Streaming (commented out due to import issues)
    
    # Tool definitions
    "ToolReturn",
    
    # Types and enums
    "FinishReason",
    "MessageKind",
    "ProviderDetails",
    
    # Utility functions
    "serialize_message",
    "deserialize_request",
    "deserialize_response",
    "to_json",
    "from_json_request",
    "from_json_response",
    "extract_text_content",
    "has_tool_calls",
    "get_tool_calls",
    "create_simple_request",
    "create_simple_response",
    "add_tool_return",
    "validate_message_structure",
    "merge_responses",
    "filter_stream_events",
    
    # Validation functions
    "validate_url_format",
    "validate_media_type",
    "validate_tool_call_id",
    "validate_message_consistency",
    "validate_multimodal_content",
    "validate_token_usage",
    "validate_provider_details",
    "comprehensive_validation",
    "is_message_valid",
    "get_validation_summary",
]