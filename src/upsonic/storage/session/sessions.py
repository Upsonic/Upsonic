from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BaseSession(BaseModel):
    """
    The base model for all session types, containing common fields.

    This class defines the core attributes that are shared across agent,
    team, and workflow sessions, establishing a consistent foundation.
    """
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The unique identifier for the session."
    )
    user_id: Optional[str] = Field(
        None,
        description="The ID of the user associated with this session.",
        index=True
    )

    memory: List[List[Any]] = Field(
        default_factory=list,
        description="A list of objects representing the message history for the session."
    )
    session_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="A flexible key-value store for structured data relevant to the session's state (e.g., tool calls)."
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
        # Allows Pydantic models to be created from objects that aren't dicts,
        # such as SQLAlchemy's RowProxy objects, by accessing attributes.
        from_attributes = True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the session model to a dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaseSession:
        """Creates a session model instance from a dictionary."""
        return cls.model_validate(data)


class AgentSession(BaseSession):
    """
    Represents the complete state of a single agent's interaction session.

    This model inherits all common fields from BaseSession and adds fields
    specific to an agent's context. It maps directly to the 'agent' mode
    schema in the storage providers.
    """
    agent_id: Optional[str] = Field(
        None,
        description="The unique identifier of the agent entity.",
        index=True
    )
    agent_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="A flexible key-value store for agent-specific configuration or state that needs to be persisted."
    )
    team_session_id: Optional[str] = Field(
        None,
        description="An optional foreign key linking this agent session to a parent team session.",
        index=True
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentSession:
        """
        Creates an AgentSession instance from a dictionary.

        This method is particularly useful for deserializing records fetched
        from a database into a validated Pydantic model.

        Args:
            data: A dictionary containing the session data.

        Returns:
            A validated instance of AgentSession.
        """
        return cls.model_validate(data)