"""
CulturalKnowledge data model for storing shared knowledge, principles,
and best practices across agents.

Notice: Culture is an experimental feature and is subject to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


def _now_epoch_s() -> int:
    """Get current time as epoch seconds (UTC)."""
    return int(datetime.now(timezone.utc).timestamp())


def _to_epoch_s(value: Union[int, float, str, datetime]) -> int:
    """Normalize various datetime representations to epoch seconds (UTC)."""
    if isinstance(value, (int, float)):
        # Assume value is already in seconds
        return int(value)

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    if isinstance(value, str):
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError as e:
            raise ValueError(f"Unsupported datetime string: {value!r}") from e
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    raise TypeError(f"Unsupported datetime value: {type(value)}")


def _epoch_to_rfc3339_z(ts: Union[int, float]) -> str:
    """Convert epoch seconds to RFC3339 format with Z suffix."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class CulturalKnowledge:
    """
    Model for Cultural Knowledge.
    
    Cultural knowledge represents shared knowledge, insights, and practices
    that can improve performance across agents. Unlike user-specific Memory,
    Culture stores universal principles that benefit everyone.
    
    Notice: Culture is an experimental feature and is subject to change.
    
    Attributes:
        id: Unique identifier, auto-generated if not provided
        name: Short, specific title for the knowledge (required for meaningful entries)
        content: The main principle, rule, or guideline content
        summary: One-line purpose or takeaway
        categories: List of tags (e.g., ['guardrails', 'rules', 'principles', 'practices'])
        notes: List of contextual notes, rationale, or examples
        metadata: Arbitrary structured info (source, author, version, etc.)
        input: Original input that generated this knowledge
        created_at: Timestamp when created (epoch seconds, UTC)
        updated_at: Timestamp when last updated (epoch seconds, UTC)
        agent_id: ID of the agent that created this knowledge
        team_id: ID of the team associated with this knowledge
    """

    # The id of the cultural knowledge, auto-generated if not provided
    id: Optional[str] = None
    name: Optional[str] = None

    content: Optional[str] = None
    categories: Optional[List[str]] = None
    notes: Optional[List[str]] = None

    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    input: Optional[str] = None

    created_at: Optional[int] = field(default=None)
    updated_at: Optional[int] = field(default=None)

    agent_id: Optional[str] = None
    team_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and initialize timestamps after dataclass initialization."""
        if self.name is not None and not self.name.strip():
            raise ValueError("name must be a non-empty string")
        
        self.created_at = _now_epoch_s() if self.created_at is None else _to_epoch_s(self.created_at)
        self.updated_at = self.created_at if self.updated_at is None else _to_epoch_s(self.updated_at)

    def bump_updated_at(self) -> None:
        """Bump updated_at to now (UTC)."""
        self.updated_at = _now_epoch_s()

    def preview(self) -> Dict[str, Any]:
        """
        Return a preview of the cultural knowledge.
        
        Used for sending to LLM context - truncates long fields to reduce token usage.
        
        Returns:
            Dictionary with key fields, truncated if necessary.
        """
        _preview: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
        }
        
        if self.categories is not None:
            _preview["categories"] = self.categories
        
        if self.summary is not None:
            _preview["summary"] = self.summary[:100] + "..." if len(self.summary) > 100 else self.summary
        
        if self.content is not None:
            _preview["content"] = self.content[:100] + "..." if len(self.content) > 100 else self.content
        
        if self.notes is not None:
            _preview["notes"] = [
                note[:100] + "..." if len(note) > 100 else note 
                for note in self.notes
            ]
        
        return _preview

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the cultural knowledge to a dictionary.
        
        Converts timestamps to RFC3339 format for JSON compatibility.
        Excludes None values for cleaner output.
        
        Returns:
            Dictionary representation of the cultural knowledge.
        """
        _dict = {
            "id": self.id,
            "name": self.name,
            "summary": self.summary,
            "content": self.content,
            "categories": self.categories,
            "metadata": self.metadata,
            "notes": self.notes,
            "input": self.input,
            "created_at": (_epoch_to_rfc3339_z(self.created_at) if self.created_at is not None else None),
            "updated_at": (_epoch_to_rfc3339_z(self.updated_at) if self.updated_at is not None else None),
            "agent_id": self.agent_id,
            "team_id": self.team_id,
        }
        return {k: v for k, v in _dict.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CulturalKnowledge":
        """
        Create a CulturalKnowledge instance from a dictionary.
        
        Handles various timestamp formats (epoch, RFC3339, datetime).
        
        Args:
            data: Dictionary with cultural knowledge fields.
            
        Returns:
            New CulturalKnowledge instance.
        """
        d = dict(data)

        # Preserve 0 and None explicitly; only process if key exists and has value
        if "created_at" in d and d["created_at"] is not None:
            d["created_at"] = _to_epoch_s(d["created_at"])
        if "updated_at" in d and d["updated_at"] is not None:
            d["updated_at"] = _to_epoch_s(d["updated_at"])

        return cls(**d)

    def __repr__(self) -> str:
        """Return a readable representation of the cultural knowledge."""
        return (
            f"CulturalKnowledge(id={self.id!r}, name={self.name!r}, "
            f"categories={self.categories!r})"
        )
