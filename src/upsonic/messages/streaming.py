"""
This module defines the models for handling streaming responses from a language model.
It models the deltas (incremental changes) to message parts and wraps them
in event containers, allowing for real-time processing of incoming data.
"""

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from .base import ModelResponsePart

# --- Delta Models ---

class TextPartDelta(BaseModel):
    """A partial update for a TextPart, representing an incremental addition to the content."""
    content_delta: str
    part_delta_kind: Literal["text"] = "text"

class ThinkingPartDelta(BaseModel):
    """A partial update for a ThinkingPart."""
    content_delta: str
    part_delta_kind: Literal["thinking"] = "thinking"

class ToolCallPartDelta(BaseModel):
    """
    A partial update for a ToolCallPart.
    This is complex as the tool name, arguments (as a JSON string), and ID can arrive in chunks.
    """
    tool_name_delta: Optional[str] = None
    args_delta: Optional[str] = None
    tool_call_id: Optional[str] = None
    part_delta_kind: Literal["tool_call"] = "tool_call"

# A Union of all possible delta types.
ModelResponsePartDelta = Union[TextPartDelta, ThinkingPartDelta, ToolCallPartDelta]


# --- Stream Event Models ---

class PartStartEvent(BaseModel):
    """An event indicating that a new part has started streaming."""
    index: int = Field(description="The index of the part within the response's `parts` list.")
    part: ModelResponsePart = Field(description="The newly started ModelResponsePart.")
    event_kind: Literal["part_start"] = "part_start"

class PartDeltaEvent(BaseModel):
    """An event indicating a delta update for an existing part."""
    index: int = Field(description="The index of the part to which the delta should be applied.")
    delta: ModelResponsePartDelta = Field(description="The delta containing the incremental update.")
    event_kind: Literal["part_delta"] = "part_delta"

class FinalResultEvent(BaseModel):
    """A special event signaling that the full, structured response is valid and complete."""
    event_kind: Literal["final_result"] = "final_result"

# A comprehensive Union of all possible stream events.
# This will be the type yielded by an agent's streaming run method.
ModelResponseStreamEvent = Union[PartStartEvent, PartDeltaEvent, FinalResultEvent]