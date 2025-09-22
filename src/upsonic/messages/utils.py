"""
Utility functions for working with message models in the agent framework.
This module provides helper functions for serialization, validation, and common operations.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from .base import ModelRequest, ModelResponse, ModelRequestPart, ModelResponsePart
from .parts import (
    TextPart,
    UserPromptPart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart
)
# from .streaming import ModelResponseStreamEvent  # Commented out due to import issues
from .types import FinishReason


def serialize_message(message: Union[ModelRequest, ModelResponse]) -> Dict[str, Any]:
    """
    Serialize a message to a dictionary for storage or transmission.
    
    Args:
        message: The message to serialize
        
    Returns:
        A dictionary representation of the message
    """
    return message.model_dump()


def deserialize_request(data: Dict[str, Any]) -> ModelRequest:
    """
    Deserialize a dictionary back to a ModelRequest.
    
    Args:
        data: Dictionary representation of the request
        
    Returns:
        A ModelRequest instance
        
    Raises:
        ValidationError: If the data is invalid
    """
    return ModelRequest.model_validate(data)


def deserialize_response(data: Dict[str, Any]) -> ModelResponse:
    """
    Deserialize a dictionary back to a ModelResponse.
    
    Args:
        data: Dictionary representation of the response
        
    Returns:
        A ModelResponse instance
        
    Raises:
        ValidationError: If the data is invalid
    """
    return ModelResponse.model_validate(data)


def to_json(message: Union[ModelRequest, ModelResponse], indent: Optional[int] = None) -> str:
    """
    Convert a message to JSON string.
    
    Args:
        message: The message to convert
        indent: JSON indentation level (None for compact)
        
    Returns:
        JSON string representation
    """
    return message.model_dump_json(indent=indent)


def from_json_request(json_str: str) -> ModelRequest:
    """
    Create a ModelRequest from JSON string.
    
    Args:
        json_str: JSON string representation
        
    Returns:
        A ModelRequest instance
        
    Raises:
        ValidationError: If the JSON is invalid
    """
    return ModelRequest.model_validate_json(json_str)


def from_json_response(json_str: str) -> ModelResponse:
    """
    Create a ModelResponse from JSON string.
    
    Args:
        json_str: JSON string representation
        
    Returns:
        A ModelResponse instance
        
    Raises:
        ValidationError: If the JSON is invalid
    """
    return ModelResponse.model_validate_json(json_str)


def extract_text_content(message: Union[ModelRequest, ModelResponse]) -> str:
    """
    Extract all text content from a message, combining text from all relevant parts.
    
    Args:
        message: The message to extract text from
        
    Returns:
        Combined text content
    """
    if isinstance(message, ModelRequest):
        request_text_parts: List[str] = []
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                if isinstance(part.content, str):
                    request_text_parts.append(part.content)
                elif isinstance(part.content, list):
                    for item in part.content:
                        if isinstance(item, str):
                            request_text_parts.append(item)
        return "\n".join(request_text_parts)
    
    elif isinstance(message, ModelResponse):
        response_text_parts: List[str] = []
        for part in message.parts:  # type: ignore
            if hasattr(part, 'content') and hasattr(part, 'part_kind'):
                if part.part_kind == 'text':
                    response_text_parts.append(part.content)  # type: ignore
                elif part.part_kind == 'thinking':
                    response_text_parts.append(part.content)  # type: ignore
        return "\n".join(response_text_parts)
    
    return ""


def has_tool_calls(message: ModelResponse) -> bool:
    """
    Check if a response contains any tool calls.
    
    Args:
        message: The response to check
        
    Returns:
        True if the response contains tool calls
    """
    return message.has_tool_calls


def get_tool_calls(message: ModelResponse) -> List[ToolCallPart]:
    """
    Get all tool calls from a response.
    
    Args:
        message: The response to extract tool calls from
        
    Returns:
        List of tool call parts
    """
    return message.tool_calls


def create_simple_request(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    instructions: Optional[str] = None
) -> ModelRequest:
    """
    Create a simple ModelRequest with text content.
    
    Args:
        user_prompt: The user's prompt text
        system_prompt: Optional system prompt
        instructions: Optional instructions
        
    Returns:
        A ModelRequest instance
    """
    parts: List[ModelRequestPart] = []
    
    if system_prompt:
        parts.append(SystemPromptPart(content=system_prompt))
    
    parts.append(UserPromptPart(content=user_prompt))
    
    return ModelRequest(
        parts=parts,
        instructions=instructions
    )


def create_simple_response(
    text_content: str,
    model_name: Optional[str] = None,
    finish_reason: Optional[FinishReason] = None
) -> ModelResponse:
    """
    Create a simple ModelResponse with text content.
    
    Args:
        text_content: The response text
        model_name: Optional model name
        finish_reason: Optional finish reason
        
    Returns:
        A ModelResponse instance
    """
    parts: List[ModelResponsePart] = [TextPart(content=text_content)]
    
    return ModelResponse(
        parts=parts,
        model_name=model_name,
        finish_reason=finish_reason
    )


def add_tool_return(
    request: ModelRequest,
    tool_name: str,
    content: Any,
    tool_call_id: str
) -> ModelRequest:
    """
    Add a tool return part to a request.
    
    Args:
        request: The request to modify
        tool_name: Name of the tool
        content: Tool return content
        tool_call_id: ID of the tool call
        
    Returns:
        A new ModelRequest with the tool return added
    """
    tool_return = ToolReturnPart(
        tool_name=tool_name,
        content=content,
        tool_call_id=tool_call_id
    )
    
    new_parts = request.parts.copy()
    new_parts.append(tool_return)
    
    return ModelRequest(
        parts=new_parts,
        instructions=request.instructions
    )


def validate_message_structure(message: Union[ModelRequest, ModelResponse]) -> bool:
    """
    Validate that a message has proper structure and required fields.
    
    Args:
        message: The message to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if isinstance(message, ModelRequest):
            # Validate that request has at least one part
            if not message.parts:
                return False
            
            # Validate each part - use hasattr to check for required attributes
            for part in message.parts:
                if not hasattr(part, 'part_kind'):
                    return False
                    
        elif isinstance(message, ModelResponse):
            # Validate that response has at least one part
            if not message.parts:
                return False
            
            # Validate each part - use hasattr to check for required attributes
            for part in message.parts:
                if not hasattr(part, 'part_kind'):
                    return False
        
        return True
        
    except (ValidationError, AttributeError):
        return False


def merge_responses(responses: List[ModelResponse]) -> ModelResponse:
    """
    Merge multiple responses into a single response.
    
    Args:
        responses: List of responses to merge
        
    Returns:
        A merged ModelResponse
        
    Raises:
        ValueError: If no responses provided
    """
    if not responses:
        raise ValueError("Cannot merge empty list of responses")
    
    if len(responses) == 1:
        return responses[0]
    
    # Combine all parts
    all_parts = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    for response in responses:
        all_parts.extend(response.parts)
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
    
    # Use metadata from the first response
    first_response = responses[0]
    
    return ModelResponse(
        parts=all_parts,
        usage=first_response.usage.model_copy(update={
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens
        }),
        model_name=first_response.model_name,
        provider_name=first_response.provider_name,
        provider_details=first_response.provider_details,
        provider_response_id=first_response.provider_response_id,
        finish_reason=responses[-1].finish_reason  # Use finish reason from last response
    )


def filter_stream_events(
    events: List[Any],  # List[ModelResponseStreamEvent] when streaming is fixed
    event_types: Optional[List[str]] = None
) -> List[Any]:  # List[ModelResponseStreamEvent] when streaming is fixed
    """
    Filter streaming events by event type.
    
    Args:
        events: List of streaming events
        event_types: Optional list of event types to include
        
    Returns:
        Filtered list of events
    """
    if not event_types:
        return events
    
    return [event for event in events if event.event_kind in event_types]
