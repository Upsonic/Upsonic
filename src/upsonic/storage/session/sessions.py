from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

# Import the new, distinct semantic ID types from our types module
from upsonic.storage.types import SessionId, UserId


class UserProfile(BaseModel):
    """
    Represents all persistent, user-specific data that is not tied to any
    single conversation session.

    This model is the data structure for the "User Analysis Memory," holding
    long-term information about a user's preferences, expertise, tone, and
    other traits discovered over multiple interactions. It is uniquely
    identified and queried by its `user_id`, which is of the semantic
    type `UserId`.
    """
    user_id: UserId = Field(
        ...,
        description="The unique identifier for the user, of type UserId. This is the primary key.",
        index=True
    )

    profile_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="A flexible key-value store for persistent user metadata."
    )

    created_at: float = Field(
        default_factory=time.time,
        description="The Unix timestamp when the user profile was first created."
    )
    updated_at: float = Field(
        default_factory=time.time,
        description="The Unix timestamp when the user profile was last updated."
    )

    model_config = ConfigDict(from_attributes=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the user profile model to a dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UserProfile:
        """Creates a UserProfile instance from a dictionary."""
        return cls.model_validate(data)


class InteractionSession(BaseModel):
    """
    Represents all data related to a single, continuous interaction or
    conversation, uniquely identified by a `session_id`.

    This model serves as the data structure for both "Full Session Memory"
    (via `chat_history`) and "Summary Memory" (via `summary`). Its identifier
    is of the semantic type `SessionId`.
    """
    session_id: SessionId = Field(
        # Note: We cast the uuid to our new SessionId type.
        default_factory=lambda: SessionId(str(uuid.uuid4())),
        description="The unique identifier for the session, of type SessionId. This is the primary key."
    )
    user_id: Optional[UserId] = Field(
        None,
        description="The ID of the user associated with this session, linking it to a UserProfile.",
        index=True
    )
    agent_id: Optional[str] = Field(
        None,
        description="The unique identifier of the agent entity involved in the session.",
        index=True
    )
    team_session_id: Optional[str] = Field(
        None,
        description="An optional foreign key linking this session to a parent team session.",
        index=True
    )

    chat_history: List[Any] = Field(
        default_factory=list,
        description="The complete, ordered list of messages for the session ('Full Session Memory')."
    )
    summary: Optional[str] = Field(
        None,
        description="An evolving, high-level summary of the session's content ('Summary Memory')."
    )
    
    session_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="A flexible key-value store for structured data relevant to the session's immediate state."
    )
    extra_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="A flexible key-value store for any other custom metadata, tags, or IDs."
    )

    created_at: float = Field(
        default_factory=time.time,
        description="The Unix timestamp when the session was created."
    )
    updated_at: float = Field(
        default_factory=time.time,
        description="The Unix timestamp when the session was last updated."
    )

    class Config:
        """Pydantic model configuration."""
        from_attributes = True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the session model to a dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InteractionSession:
        """Creates an InteractionSession instance from a dictionary."""
        return cls.model_validate(data)