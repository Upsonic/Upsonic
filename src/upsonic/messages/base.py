"""
This module defines the top-level message containers, `ModelRequest` and `ModelResponse`.

These classes assemble the individual parts from `parts.py` into the complete, coherent
data structures that are sent to and received from language models. They also include
helper properties to make accessing common data more convenient.
"""

import datetime
from typing import List, Optional, Union

from pydantic import BaseModel, Field

# Import all defined parts to create comprehensive Union types for request and response.
# Every import here is used in the `ModelRequestPart` or `ModelResponsePart` Unions.
from .parts import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from .types import FinishReason, MessageKind, ProviderDetails

# --- Type Unions for Message Parts ---

# A comprehensive Union of all parts that can be included in a request TO a model.
# This represents everything the agent framework can "say" to the language model.
ModelRequestPart = Union[
    SystemPromptPart, UserPromptPart, ToolReturnPart, BuiltinToolReturnPart, RetryPromptPart
]

# A comprehensive Union of all parts that can be included in a response FROM a model.
# This represents everything the language model can "say" back to the agent framework.
ModelResponsePart = Union[
    TextPart, ThinkingPart, ToolCallPart, BuiltinToolCallPart, BuiltinToolReturnPart
]

# --- Core Data Models ---

class TokenUsage(BaseModel):
    """A dedicated structure to represent token usage for a model interaction."""

    input_tokens: int = Field(0, description="Number of tokens in the input/prompt.")
    output_tokens: int = Field(0, description="Number of tokens in the output/completion.")

    @property
    def total_tokens(self) -> int:
        """A convenience property to get the sum of input and output tokens."""
        return self.input_tokens + self.output_tokens


class ModelRequest(BaseModel):
    """Represents a complete request to be sent to a language model."""

    parts: List[ModelRequestPart] = Field(
        default_factory=list,
        description="A list of message parts comprising the request.",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Optional set of system-level instructions for the model.",
    )
    kind: MessageKind = Field(
        default=MessageKind.REQUEST,
        description="The type of the message, always 'request'.",
        frozen=True,
    )

    @property
    def user_text_prompt(self) -> str:
        """
        A convenience property to extract and combine text from all UserPromptParts.
        This is useful for logging or for models that only accept a single text prompt.
        """
        text_parts = []
        for part in self.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                text_parts.append(part.content)
        return "\n".join(text_parts)


class ModelResponse(BaseModel):
    """Represents a complete response received from a language model."""

    parts: List[ModelResponsePart] = Field(
        default_factory=list,
        description="A list of message parts comprising the response.",
    )
    usage: TokenUsage = Field(
        default_factory=lambda: TokenUsage(input_tokens=0, output_tokens=0),
        description="Information about token usage for the interaction.",
    )
    model_name: Optional[str] = Field(
        default=None, description="The name of the model that generated the response."
    )
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        description="The UTC timestamp when the response was generated.",
    )
    kind: MessageKind = Field(
        default=MessageKind.RESPONSE,
        description="The type of the message, always 'response'.",
        frozen=True,
    )
    provider_name: Optional[str] = Field(
        default=None, description="The name of the model provider (e.g., 'openai', 'anthropic')."
    )
    provider_details: ProviderDetails = Field(
        default_factory=dict,
        description="A dictionary for any additional, provider-specific details.",
    )
    provider_response_id: Optional[str] = Field(
        default=None, description="A unique ID from the provider for tracing and debugging."
    )
    finish_reason: Optional[FinishReason] = Field(
        default=None, description="The reason the model stopped generating tokens."
    )

    @property
    def text_content(self) -> Optional[str]:
        """
        A convenience property to extract and combine text from all TextParts.
        Returns None if no text parts are present.
        """
        text_parts = [
            part.content for part in self.parts if isinstance(part, TextPart)
        ]
        return "\n".join(text_parts) if text_parts else None

    @property
    def tool_calls(self) -> List[ToolCallPart]:
        """A convenience property to get a list of all tool calls in the response."""
        return [part for part in self.parts if isinstance(part, ToolCallPart)]

    @property
    def has_tool_calls(self) -> bool:
        """A simple boolean check to see if the response contains any tool calls."""
        return bool(self.tool_calls)
    
    @property
    def thinking_content(self) -> Optional[str]:
        """
        A convenience property to extract and combine thinking content from all ThinkingParts.
        Returns None if no thinking parts are present.
        """
        thinking_parts = [
            part.content for part in self.parts if isinstance(part, ThinkingPart)
        ]
        return "\n".join(thinking_parts) if thinking_parts else None
    
    @property
    def is_complete(self) -> bool:
        """Check if the response is complete (has a finish reason)."""
        return self.finish_reason is not None
    
    @property
    def is_streaming(self) -> bool:
        """Check if the response is still streaming (no finish reason)."""
        return self.finish_reason is None

