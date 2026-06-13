"""
Utility functions for working with message objects.

This module provides helper functions for extracting and analyzing
parts from ModelRequest and ModelResponse objects, as well as
serialization utilities for message classes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pydantic

if TYPE_CHECKING:
    from upsonic.messages.messages import (
        BaseToolReturnPart,
        BuiltinToolCallPart,
        ModelMessage,
        ModelRequest,
        ModelResponse,
        TextPart,
        ThinkingPart,
        ToolCallPart,
    )


def get_text_content(response: "ModelResponse") -> str | None:
    """
    Extract text content from a ModelResponse.
    
    Returns the content of the last TextPart in the response, or None if no TextPart exists.
    This is the most common way to get the model's text output.
    
    Args:
        response: The ModelResponse to extract text from
        
    Returns:
        The text content as a string, or None if no TextPart exists
        
    Example:
        ```python
        from upsonic.utils.messages import get_text_content
        
        result = chain.invoke({"topic": "AI"})
        text = get_text_content(result)
        print(text)
        ```
    """
    from upsonic.messages.messages import TextPart
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    if text_parts:
        return text_parts[-1].content
    return None


def get_text_parts(response: "ModelResponse") -> list["TextPart"]:
    """
    Extract all TextPart instances from a ModelResponse.
    
    Args:
        response: The ModelResponse to extract TextParts from
        
    Returns:
        A list of all TextPart instances in the response
    """
    from upsonic.messages.messages import TextPart
    return [part for part in response.parts if isinstance(part, TextPart)]


def get_tool_calls(response: "ModelResponse") -> list["ToolCallPart"]:
    """
    Extract all ToolCallPart instances from a ModelResponse.
    
    Args:
        response: The ModelResponse to extract tool calls from
        
    Returns:
        A list of all ToolCallPart instances in the response
    """
    return response.tool_calls


def get_builtin_tool_calls(response: "ModelResponse") -> list["BuiltinToolCallPart"]:
    """
    Extract all BuiltinToolCallPart instances from a ModelResponse.
    
    Args:
        response: The ModelResponse to extract builtin tool calls from
        
    Returns:
        A list of all BuiltinToolCallPart instances in the response
    """
    from upsonic.messages.messages import BuiltinToolCallPart
    return [part for part in response.parts if isinstance(part, BuiltinToolCallPart)]


def get_thinking_content(response: "ModelResponse") -> str | None:
    """
    Extract thinking content from a ModelResponse.
    
    Returns the content of the last ThinkingPart in the response, or None if no ThinkingPart exists.
    This is useful for models that support reasoning/thinking steps.
    
    Args:
        response: The ModelResponse to extract thinking content from
        
    Returns:
        The thinking content as a string, or None if no ThinkingPart exists
    """
    from upsonic.messages.messages import ThinkingPart
    thinking_parts = [part for part in response.parts if isinstance(part, ThinkingPart)]
    if thinking_parts:
        return thinking_parts[-1].content
    return None


def get_thinking_parts(response: "ModelResponse") -> list["ThinkingPart"]:
    """
    Extract all ThinkingPart instances from a ModelResponse.
    
    Args:
        response: The ModelResponse to extract thinking parts from
        
    Returns:
        A list of all ThinkingPart instances in the response
    """
    from upsonic.messages.messages import ThinkingPart
    return [part for part in response.parts if isinstance(part, ThinkingPart)]


def get_tool_returns(response: "ModelResponse") -> list["BaseToolReturnPart"]:
    """
    Extract all tool return parts from a ModelResponse.
    
    Args:
        response: The ModelResponse to extract tool returns from
        
    Returns:
        A list of all BaseToolReturnPart instances (including BuiltinToolReturnPart) in the response
    """
    from upsonic.messages.messages import BaseToolReturnPart
    return [part for part in response.parts if isinstance(part, BaseToolReturnPart)]


def analyze_model_request_messages(messages: Sequence["ModelMessage"]) -> tuple[list[dict[str, Any]], int]:
    """
    Analyze ModelRequest messages in a sequence of messages and extract details about their parts.
    
    This function iterates through messages, identifies ModelRequest instances, and collects
    information about their parts including part counts and whether they contain system prompts.
    
    Args:
        messages: A sequence of ModelMessage instances (can include both ModelRequest and ModelResponse)
        
    Returns:
        A tuple containing:
        - message_details: List of dictionaries with 'parts_count' and 'has_system' for each ModelRequest
        - total_parts: Total count of parts across all ModelRequest messages
    """
    from upsonic.messages.messages import ModelRequest, SystemPromptPart
    
    message_details: list[dict[str, Any]] = []
    total_parts: int = 0
    
    for msg in messages:
        if isinstance(msg, ModelRequest):
            parts_count = len(msg.parts) if msg.parts else 0
            total_parts += parts_count
            has_system_part = any(isinstance(p, SystemPromptPart) for p in (msg.parts or []))
            message_details.append({
                'parts_count': parts_count,
                'has_system': has_system_part
            })
    
    return message_details, total_parts


# Serialization utilities

def _get_model_messages_type_adapter() -> pydantic.TypeAdapter:
    """Lazy import and return ModelMessagesTypeAdapter."""
    from upsonic.messages.messages import ModelMessagesTypeAdapter
    return ModelMessagesTypeAdapter


def _get_model_request_type_adapter() -> pydantic.TypeAdapter:
    """Lazy create and return ModelRequest TypeAdapter."""
    from upsonic.messages.messages import ModelRequest
    return pydantic.TypeAdapter(
        ModelRequest,
        config=pydantic.ConfigDict(defer_build=True, ser_json_bytes='base64', val_json_bytes='base64')
    )


def _get_model_response_type_adapter() -> pydantic.TypeAdapter:
    """Lazy create and return ModelResponse TypeAdapter."""
    from upsonic.messages.messages import ModelResponse
    return pydantic.TypeAdapter(
        ModelResponse,
        config=pydantic.ConfigDict(defer_build=True, ser_json_bytes='base64', val_json_bytes='base64')
    )


def serialize_model_request(request: "ModelRequest") -> bytes:
    """Serialize a ModelRequest to bytes."""
    ta = _get_model_request_type_adapter()
    return ta.dump_json(request)


def deserialize_model_request(data: bytes) -> "ModelRequest":
    """Deserialize bytes to a ModelRequest."""
    ta = _get_model_request_type_adapter()
    return ta.validate_json(data)


def serialize_model_response(response: "ModelResponse") -> bytes:
    """Serialize a ModelResponse to bytes."""
    ta = _get_model_response_type_adapter()
    return ta.dump_json(response)


def deserialize_model_response(data: bytes) -> "ModelResponse":
    """Deserialize bytes to a ModelResponse."""
    ta = _get_model_response_type_adapter()
    return ta.validate_json(data)


def serialize_messages(messages: list["ModelMessage"]) -> bytes:
    """Serialize a list of ModelMessages to bytes."""
    ta = _get_model_messages_type_adapter()
    return ta.dump_json(messages)


def deserialize_messages(data: bytes) -> list["ModelMessage"]:
    """Deserialize bytes to a list of ModelMessages."""
    ta = _get_model_messages_type_adapter()
    return ta.validate_json(data)

