"""
Validation utilities for message models.
This module provides additional validation functions beyond Pydantic's built-in validation.
"""

import re
from typing import Any, Dict, List, Union
from urllib.parse import urlparse

from .base import ModelRequest, ModelResponse
from .parts import FileUrl, MultiModalContent
from .types import FinishReason


def validate_url_format(url: str) -> bool:
    """
    Validate that a URL has proper format.
    
    Args:
        url: The URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        result = urlparse(url)
        # Only allow http, https, file, and data URLs
        allowed_schemes = {'http', 'https', 'file', 'data'}
        return bool(result.scheme in allowed_schemes and 
                   (result.netloc or result.scheme == 'file')) or url.startswith('data:')
    except (ValueError, TypeError):
        return False


def validate_media_type(media_type: str, allowed_types: List[str]) -> bool:
    """
    Validate that a media type is in the allowed list.
    
    Args:
        media_type: The media type to validate
        allowed_types: List of allowed media types
        
    Returns:
        True if valid, False otherwise
    """
    return media_type in allowed_types


def validate_tool_call_id(tool_call_id: str) -> bool:
    """
    Validate that a tool call ID has proper format.
    
    Args:
        tool_call_id: The tool call ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not tool_call_id or not isinstance(tool_call_id, str):
        return False
    
    # Tool call IDs should be alphanumeric with optional underscores/hyphens
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, tool_call_id))


def validate_message_consistency(message: Union[ModelRequest, ModelResponse]) -> List[str]:
    """
    Validate message consistency and return list of issues.
    
    Args:
        message: The message to validate
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    if isinstance(message, ModelRequest):
        # Check for empty parts
        if not message.parts:
            issues.append("Request must have at least one part")
        
        # Check for duplicate tool call IDs in tool returns
        tool_call_ids = set()
        for part in message.parts:  # type: ignore
            if hasattr(part, 'tool_call_id'):
                tool_call_id = getattr(part, 'tool_call_id')
                if tool_call_id in tool_call_ids:
                    issues.append(f"Duplicate tool call ID: {tool_call_id}")
                tool_call_ids.add(tool_call_id)  # type: ignore
    
    elif isinstance(message, ModelResponse):
        # Check for empty parts
        if not message.parts:
            issues.append("Response must have at least one part")
        
        # Check for duplicate tool call IDs in tool calls
        tool_call_ids = set()
        for part in message.parts:  # type: ignore
            if hasattr(part, 'tool_call_id'):
                tool_call_id = getattr(part, 'tool_call_id')
                if tool_call_id in tool_call_ids:
                    issues.append(f"Duplicate tool call ID: {tool_call_id}")
                tool_call_ids.add(tool_call_id)  # type: ignore
        
        # Validate finish reason consistency
        if message.finish_reason == FinishReason.TOOL_CALL and not message.has_tool_calls:
            issues.append("Finish reason is TOOL_CALL but no tool calls found")
        
        if message.finish_reason == FinishReason.ERROR and not message.text_content:
            issues.append("Finish reason is ERROR but no error message found")
    
    return issues


def validate_multimodal_content(content: MultiModalContent) -> List[str]:
    """
    Validate multimodal content and return list of issues.
    
    Args:
        content: The multimodal content to validate
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    if isinstance(content, FileUrl):
        # Validate URL format
        if not validate_url_format(content.url):
            issues.append(f"Invalid URL format: {content.url}")
        
        # Validate media type
        try:
            media_type = content.media_type
            if not media_type:
                issues.append("Could not determine media type from URL")
        except ValueError as e:
            issues.append(str(e))
    
    elif hasattr(content, 'data') and hasattr(content, 'media_type'):
        # Binary content validation
        if not content.data:
            issues.append("Binary content data is empty")
        
        if not content.media_type:
            issues.append("Binary content missing media type")
    
    return issues


def validate_token_usage(usage: Any) -> List[str]:
    """
    Validate token usage data.
    
    Args:
        usage: The token usage object to validate
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    if not hasattr(usage, 'input_tokens') or not hasattr(usage, 'output_tokens'):
        issues.append("Token usage missing required fields")
        return issues
    
    if usage.input_tokens < 0:
        issues.append("Input tokens cannot be negative")
    
    if usage.output_tokens < 0:
        issues.append("Output tokens cannot be negative")
    
    if usage.input_tokens == 0 and usage.output_tokens == 0:
        issues.append("Token usage is zero - this might indicate an issue")
    
    return issues


def validate_provider_details(details: Dict[str, Any]) -> List[str]:
    """
    Validate provider-specific details.
    
    Args:
        details: The provider details dictionary to validate
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    if not isinstance(details, dict):
        issues.append("Provider details must be a dictionary")
        return issues
    
    # Check for common required fields
    if 'model' in details and not details['model']:
        issues.append("Provider model field is empty")
    
    if 'temperature' in details:
        temp = details['temperature']
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            issues.append("Temperature must be between 0 and 2")
    
    if 'max_tokens' in details:
        max_tokens = details['max_tokens']
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            issues.append("Max tokens must be a positive integer")
    
    return issues


def comprehensive_validation(message: Union[ModelRequest, ModelResponse]) -> Dict[str, List[str]]:
    """
    Perform comprehensive validation on a message and return all issues.
    
    Args:
        message: The message to validate
        
    Returns:
        Dictionary mapping validation categories to lists of issues
    """
    issues: Dict[str, List[str]] = {
        'consistency': [],
        'multimodal': [],
        'token_usage': [],
        'provider_details': [],
        'general': []
    }
    
    # Consistency validation
    issues['consistency'] = validate_message_consistency(message)
    
    # Multimodal content validation
    for part in message.parts:  # type: ignore
        if hasattr(part, 'content'):
            content = getattr(part, 'content')
            if isinstance(content, list):  # type: ignore
                for item in content:
                    if hasattr(item, 'media_type'):  # Check if it's a multimodal content
                        try:
                            issues['multimodal'].extend(validate_multimodal_content(item))  # type: ignore
                        except (TypeError, AttributeError):
                            pass  # Skip invalid multimodal content
            elif hasattr(content, 'media_type'):  # Check if it's a multimodal content
                try:
                    issues['multimodal'].extend(validate_multimodal_content(content))  # type: ignore
                except (TypeError, AttributeError):
                    pass  # Skip invalid multimodal content
    
    # Token usage validation (for responses)
    if isinstance(message, ModelResponse):
        issues['token_usage'] = validate_token_usage(message.usage)
        issues['provider_details'] = validate_provider_details(message.provider_details)
    
    # Remove empty categories
    return {k: v for k, v in issues.items() if v}


def is_message_valid(message: Union[ModelRequest, ModelResponse]) -> bool:
    """
    Quick check if a message is valid (no issues found).
    
    Args:
        message: The message to validate
        
    Returns:
        True if valid, False otherwise
    """
    issues = comprehensive_validation(message)
    return not any(issues.values())


def get_validation_summary(message: Union[ModelRequest, ModelResponse]) -> str:
    """
    Get a human-readable validation summary.
    
    Args:
        message: The message to validate
        
    Returns:
        Validation summary string
    """
    issues = comprehensive_validation(message)
    
    if not any(issues.values()):
        return "Message is valid"
    
    summary_parts = []
    for category, category_issues in issues.items():
        if category_issues:
            summary_parts.append(f"{category.title()}: {len(category_issues)} issues")
    
    return f"Message has validation issues - {', '.join(summary_parts)}"
