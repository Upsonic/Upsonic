"""
This module defines the foundational, non-structural types and enumerations
used throughout the agent framework's message models.
"""
from enum import Enum
from typing import Any, Dict, Literal, TypeAlias

# A TypeAlias for a flexible dictionary to hold provider-specific metadata.
ProviderDetails: TypeAlias = Dict[str, Any]


class MessageKind(str, Enum):
    """Discriminator for top-level message containers."""
    REQUEST = "request"
    RESPONSE = "response"


class FinishReason(str, Enum):
    """Normalized reasons for why a model stopped generating tokens."""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    
    @classmethod
    def from_provider_reason(cls, provider_reason: str, provider: str) -> "FinishReason":
        """
        Convert a provider-specific finish reason to our normalized enum.
        
        Args:
            provider_reason: The provider's finish reason string
            provider: The provider name (e.g., 'openai', 'anthropic')
            
        Returns:
            Normalized FinishReason
        """
        provider_reason = provider_reason.lower()
        
        # OpenAI mappings
        if provider.lower() == "openai":
            if provider_reason in ["stop", "end"]:
                return cls.STOP
            elif provider_reason in ["length", "max_tokens"]:
                return cls.LENGTH
            elif provider_reason in ["content_filter", "safety"]:
                return cls.CONTENT_FILTER
            elif provider_reason in ["tool_calls", "function_call"]:
                return cls.TOOL_CALL
        
        # Anthropic mappings
        elif provider.lower() == "anthropic":
            if provider_reason in ["end_turn", "stop"]:
                return cls.STOP
            elif provider_reason in ["max_tokens"]:
                return cls.LENGTH
            elif provider_reason in ["content_filter"]:
                return cls.CONTENT_FILTER
            elif provider_reason in ["tool_use"]:
                return cls.TOOL_CALL
        
        # Default fallback
        if "stop" in provider_reason or "end" in provider_reason:
            return cls.STOP
        elif "length" in provider_reason or "token" in provider_reason:
            return cls.LENGTH
        elif "filter" in provider_reason or "safety" in provider_reason:
            return cls.CONTENT_FILTER
        elif "tool" in provider_reason or "function" in provider_reason:
            return cls.TOOL_CALL
        else:
            return cls.ERROR


# Type aliases for media types to ensure consistency and provide type hinting.
AudioMediaType: TypeAlias = Literal['audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/aiff', 'audio/aac']
ImageMediaType: TypeAlias = Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']
VideoMediaType: TypeAlias = Literal[
    'video/x-matroska',
    'video/quicktime',
    'video/mp4',
    'video/webm',
    'video/x-flv',
    'video/mpeg',
    'video/x-ms-wmv',
    'video/3gpp',
]
DocumentMediaType: TypeAlias = Literal[
    'application/pdf',
    'text/plain',
    'text/csv',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'text/html',
    'text/markdown',
    'application/vnd.ms-excel',
]

# Type aliases for simple file formats, useful for APIs that require a format string.
AudioFormat: TypeAlias = Literal['wav', 'mp3', 'oga', 'flac', 'aiff', 'aac']
ImageFormat: TypeAlias = Literal['jpeg', 'png', 'gif', 'webp']
VideoFormat: TypeAlias = Literal['mkv', 'mov', 'mp4', 'webm', 'flv', 'mpeg', 'mpg', 'wmv', 'three_gp']
DocumentFormat: TypeAlias = Literal['csv', 'doc', 'docx', 'html', 'md', 'pdf', 'txt', 'xls', 'xlsx']
