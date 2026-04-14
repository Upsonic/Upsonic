from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from upsonic.utils.dttm import now_epoch_s, to_epoch_s

@dataclass
class UserMemory:
    """Model for User Memories"""

    user_memory: Dict[str, Any]
    user_id: Optional[str] = None
    created_at: Optional[int] = None
    updated_at: Optional[int] = None
    agent_id: Optional[str] = None
    team_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Automatically set/normalize created_at and updated_at."""
        self.created_at = now_epoch_s() if self.created_at is None else to_epoch_s(self.created_at)
        if self.updated_at is not None:
            self.updated_at = to_epoch_s(self.updated_at)

    def to_dict(self) -> Dict[str, Any]:
        created_at = datetime.fromtimestamp(self.created_at).isoformat() if self.created_at is not None else None
        updated_at = datetime.fromtimestamp(self.updated_at).isoformat() if self.updated_at is not None else created_at
        _dict = {
            "user_memory": self.user_memory,
            "created_at": created_at,
            "updated_at": updated_at,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "team_id": self.team_id,
        }
        return {k: v for k, v in _dict.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserMemory":
        data = dict(data)

        # Preserve 0 and None explicitly; only process if key exists
        if "created_at" in data and data["created_at"] is not None:
            data["created_at"] = to_epoch_s(data["created_at"])
        if "updated_at" in data and data["updated_at"] is not None:
            data["updated_at"] = to_epoch_s(data["updated_at"])

        return cls(**data)


@dataclass
class KnowledgeRow:
    """Model for Knowledge Base document registry entries.

    Tracks document metadata for knowledge bases -- what documents exist,
    their processing status, sizes, content hashes, and access patterns.
    Acts as the relational companion to the VectorDB's chunk storage.
    """

    id: str
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    type: Optional[str] = None
    size: Optional[int] = None
    knowledge_base_id: Optional[str] = None
    content_hash: Optional[str] = None
    chunk_count: Optional[int] = None
    source: Optional[str] = None
    status: Optional[str] = None
    status_message: Optional[str] = None
    access_count: Optional[int] = field(default=0)
    created_at: Optional[int] = None
    updated_at: Optional[int] = None

    def __post_init__(self) -> None:
        self.created_at = now_epoch_s() if self.created_at is None else to_epoch_s(self.created_at)
        if self.updated_at is not None:
            self.updated_at = to_epoch_s(self.updated_at)

    def to_dict(self) -> Dict[str, Any]:
        _dict: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "type": self.type,
            "size": self.size,
            "knowledge_base_id": self.knowledge_base_id,
            "content_hash": self.content_hash,
            "chunk_count": self.chunk_count,
            "source": self.source,
            "status": self.status,
            "status_message": self.status_message,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        return {k: v for k, v in _dict.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeRow":
        d = dict(data)

        if "created_at" in d and d["created_at"] is not None:
            d["created_at"] = to_epoch_s(d["created_at"])
        if "updated_at" in d and d["updated_at"] is not None:
            d["updated_at"] = to_epoch_s(d["updated_at"])

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        d = {k: v for k, v in d.items() if k in valid_fields}

        return cls(**d)