"""Utility functions for Mem0 storage implementations."""
from __future__ import annotations

import gzip
import json
import time
import base64
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from upsonic.utils.logging_config import get_logger

if TYPE_CHECKING:
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.session.base import Session, SessionType
    from upsonic.storage.schemas import KnowledgeRow

_logger = get_logger("upsonic.storage.mem0.utils")

# Mem0 metadata size limit (2000 characters)
MEM0_METADATA_SIZE_LIMIT = 2000
# We compress if data exceeds this size (leave room for other metadata fields)
MEM0_COMPRESSION_THRESHOLD = 1500


def _compress_data(data: str) -> str:
    """Compress a string using gzip and base64 encoding."""
    compressed = gzip.compress(data.encode('utf-8'))
    return base64.b64encode(compressed).decode('utf-8')


def _decompress_data(compressed_data: str) -> str:
    """Decompress a base64-encoded gzip string."""
    compressed_bytes = base64.b64decode(compressed_data.encode('utf-8'))
    return gzip.decompress(compressed_bytes).decode('utf-8')


def _epoch_to_iso(epoch: int) -> str:
    """Convert epoch seconds to ISO 8601 datetime string for Mem0 compatibility."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _iso_to_epoch(iso_str: str) -> int:
    """Convert ISO 8601 datetime string back to epoch seconds."""
    from datetime import datetime
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp())


# ======================== Session Schema Definition ========================

SESSION_SCHEMA: Dict[str, Dict[str, Any]] = {
    "session_id": {"type": "string", "primary_key": True},
    "session_type": {"type": "string"},
    "agent_id": {"type": "string"},
    "team_id": {"type": "string"},
    "workflow_id": {"type": "string"},
    "user_id": {"type": "string"},
    "session_data": {"type": "json"},
    "agent_data": {"type": "json"},
    "team_data": {"type": "json"},
    "workflow_data": {"type": "json"},
    "metadata": {"type": "json"},
    "runs": {"type": "json"},
    "summary": {"type": "string"},
    "created_at": {"type": "integer"},
    "updated_at": {"type": "integer"},
}

USER_MEMORY_SCHEMA: Dict[str, Dict[str, Any]] = {
    "user_id": {"type": "string", "primary_key": True},
    "memory": {"type": "json"},
    "agent_id": {"type": "string"},
    "team_id": {"type": "string"},
    "created_at": {"type": "integer"},
    "updated_at": {"type": "integer"},
}

CULTURAL_KNOWLEDGE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "id": {"type": "string", "primary_key": True},
    "name": {"type": "string"},
    "summary": {"type": "string"},
    "content": {"type": "string"},
    "metadata": {"type": "json"},
    "notes": {"type": "json"},
    "categories": {"type": "json"},
    "input": {"type": "string"},
    "created_at": {"type": "integer"},
    "updated_at": {"type": "integer"},
    "agent_id": {"type": "string"},
    "team_id": {"type": "string"},
}


# ======================== Memory ID Generation ========================

def generate_session_memory_id(session_id: str, table_name: str) -> str:
    """
    Generate a unique memory ID for a session record.
    
    Mem0 uses memory IDs to identify records. We combine table name and
    session_id to create a unique identifier that allows filtering.
    
    Note: Uses underscores instead of colons for URL compatibility with Mem0 Platform.
    
    Args:
        session_id: The session ID
        table_name: The table name for namespacing
    
    Returns:
        A unique memory ID string (URL-safe format)
    """
    return f"session__{table_name}__{session_id}"


def generate_user_memory_id(user_id: str, table_name: str) -> str:
    """
    Generate a unique memory ID for a user memory record.
    
    Note: Uses underscores instead of colons for URL compatibility with Mem0 Platform.
    
    Args:
        user_id: The user ID
        table_name: The table name for namespacing
    
    Returns:
        A unique memory ID string (URL-safe format)
    """
    return f"user_memory__{table_name}__{user_id}"


def extract_session_id_from_memory_id(memory_id: str) -> Optional[str]:
    """
    Extract session_id from a memory ID.
    
    Args:
        memory_id: The memory ID (format: "session__{table}__{session_id}")
    
    Returns:
        The session_id or None if format is invalid
    """
    if not memory_id or not memory_id.startswith("session__"):
        return None
    
    parts = memory_id.split("__", 2)
    if len(parts) >= 3:
        return parts[2]
    return None


def extract_user_id_from_memory_id(memory_id: str) -> Optional[str]:
    """
    Extract user_id from a memory ID.
    
    Args:
        memory_id: The memory ID (format: "user_memory__{table}__{user_id}")
    
    Returns:
        The user_id or None if format is invalid
    """
    if not memory_id or not memory_id.startswith("user_memory__"):
        return None
    
    parts = memory_id.split("__", 2)
    if len(parts) >= 3:
        return parts[2]
    return None


def generate_cultural_knowledge_memory_id(cultural_knowledge_id: str, table_name: str) -> str:
    """
    Generate a unique memory ID for a cultural knowledge record.
    
    Note: Uses underscores instead of colons for URL compatibility with Mem0 Platform.
    
    Args:
        cultural_knowledge_id: The cultural knowledge ID
        table_name: The table name for namespacing
    
    Returns:
        A unique memory ID string (URL-safe format)
    """
    return f"cultural_knowledge__{table_name}__{cultural_knowledge_id}"


def extract_cultural_knowledge_id_from_memory_id(memory_id: str) -> Optional[str]:
    """
    Extract cultural_knowledge_id from a memory ID.
    
    Args:
        memory_id: The memory ID (format: "cultural_knowledge__{table}__{id}")
    
    Returns:
        The cultural knowledge id or None if format is invalid
    """
    if not memory_id or not memory_id.startswith("cultural_knowledge__"):
        return None
    
    parts = memory_id.split("__", 2)
    if len(parts) >= 3:
        return parts[2]
    return None


# ======================== Serialization ========================

def serialize_session_to_mem0(
    session: "Session",
    table_name: str,
) -> Dict[str, Any]:
    """
    Serialize a session object for Mem0 storage.
    
    For Mem0 Platform, the message content is processed by LLM and transformed.
    We store the actual session data in metadata._data as a JSON string,
    and use a simple identifier as the message content.
    
    Args:
        session: The session object to serialize
        table_name: The table name for namespacing
    
    Returns:
        Dictionary with 'content', 'metadata', and 'memory_id'
    """
    from upsonic.session.agent import AgentSession
    from upsonic.session.base import SessionType

    # Get session dict
    if hasattr(session, "to_dict"):
        session_dict = session.to_dict(serialize_flag=True)
    else:
        session_dict = dict(session)
    
    session_id = session_dict.get("session_id")
    if not session_id:
        raise ValueError("Session must have session_id for serialization")
    
    current_time = int(time.time())
    
    # Prepare the full data payload based on session type
    if isinstance(session, AgentSession):
        session_type_value = SessionType.AGENT.value
        data_payload = {
            "session_id": session_id,
            "session_type": session_type_value,
            "agent_id": session_dict.get("agent_id"),
            "team_id": session_dict.get("team_id"),
            "workflow_id": session_dict.get("workflow_id"),
            "user_id": session_dict.get("user_id"),
            "session_data": session_dict.get("session_data"),
            "agent_data": session_dict.get("agent_data"),
            "metadata": session_dict.get("metadata"),
            "runs": session_dict.get("runs"),
            "messages": session_dict.get("messages"),
            "summary": session_dict.get("summary"),
            "created_at": session_dict.get("created_at") or current_time,
            "updated_at": current_time,
        }
    else:
        # Fallback for other session types
        session_type_value = "agent"
        if hasattr(session, "session_type"):
            st = session.session_type
            if hasattr(st, "value"):
                session_type_value = st.value
            elif isinstance(st, str):
                session_type_value = st
        
        data_payload = {
            "session_id": session_id,
            "session_type": session_type_value,
            "agent_id": session_dict.get("agent_id"),
            "team_id": session_dict.get("team_id"),
            "workflow_id": session_dict.get("workflow_id"),
            "user_id": session_dict.get("user_id"),
            "session_data": session_dict.get("session_data"),
            "agent_data": session_dict.get("agent_data"),
            "team_data": session_dict.get("team_data"),
            "workflow_data": session_dict.get("workflow_data"),
            "metadata": session_dict.get("metadata"),
            "runs": session_dict.get("runs"),
            "messages": session_dict.get("messages"),
            "summary": session_dict.get("summary"),
            "created_at": session_dict.get("created_at") or current_time,
            "updated_at": current_time,
        }
    
    memory_id = generate_session_memory_id(session_id, table_name)
    
    # Serialize data payload to JSON
    data_json = json.dumps(data_payload)
    
    # Always compress for sessions (they can be large with cloudpickle serialized tools)
    compressed_data = _compress_data(data_json)
    
    # Calculate what metadata would look like
    base_metadata = {
        "_type": "session",
        "_table": table_name,
        "_upsonic_memory_id": memory_id,
        "session_id": session_id,
        "session_type": session_type_value,
        "agent_id": session_dict.get("agent_id"),
        "team_id": session_dict.get("team_id"),
        "workflow_id": session_dict.get("workflow_id"),
        "user_id": session_dict.get("user_id"),
        "created_at": _epoch_to_iso(data_payload["created_at"]),
        "updated_at": _epoch_to_iso(data_payload["updated_at"]),
    }
    
    # Remove None values from metadata
    base_metadata = {k: v for k, v in base_metadata.items() if v is not None}
    
    # Check if compressed data fits in metadata (leave room for other fields)
    base_metadata_size = len(json.dumps(base_metadata))
    available_for_data = MEM0_METADATA_SIZE_LIMIT - base_metadata_size - 100  # 100 chars buffer
    
    if len(compressed_data) <= available_for_data:
        # Data fits in metadata - store it there
        metadata = base_metadata.copy()
        metadata["_data"] = compressed_data
        metadata["_compressed"] = True
        metadata["_chunked"] = False
        content = f"Upsonic session {session_id}"
    else:
        # Data too large for single metadata - split into chunks
        # Mem0 Platform API processes content through LLM, so we can't use content for raw data
        # Instead, we store chunk references in the main memory and actual data in chunk memories
        chunk_size = available_for_data - 50  # Leave room for chunk metadata
        chunks = [compressed_data[i:i+chunk_size] for i in range(0, len(compressed_data), chunk_size)]
        
        metadata = base_metadata.copy()
        metadata["_compressed"] = True
        metadata["_chunked"] = True
        metadata["_chunk_count"] = len(chunks)
        metadata["_chunk_ids"] = [f"{memory_id}__chunk_{i}" for i in range(len(chunks))]
        # Store chunks inline as _chunk_0, _chunk_1, etc. within size limit
        # We'll store them separately if needed
        content = f"Upsonic session {session_id} (chunked: {len(chunks)} parts)"
    
    return {
        "content": content,
        "metadata": metadata,
        "memory_id": memory_id,
        "_chunks": chunks if len(compressed_data) > available_for_data else None,
    }


def deserialize_session_from_mem0(
    mem0_record: Dict[str, Any],
    session_type: Optional["SessionType"] = None,
) -> Dict[str, Any]:
    """
    Deserialize a Mem0 record back to session dictionary.
    
    The actual session data is stored in metadata._data as a compressed JSON string.
    For large sessions, data may be chunked across multiple memories.
    
    Args:
        mem0_record: The Mem0 record containing metadata with _data field
        session_type: Optional session type hint
    
    Returns:
        Session dictionary
    """
    metadata = mem0_record.get("metadata", {})
    is_compressed = metadata.get("_compressed", False)
    
    # Primary: Get data from metadata._data (where we store the actual data)
    # Note: For chunked sessions, the _data field is reassembled before this call
    data_str = metadata.get("_data")
    
    if data_str:
        try:
            # Decompress if needed
            if is_compressed:
                data_str = _decompress_data(data_str)
            return json.loads(data_str)
        except (json.JSONDecodeError, Exception) as e:
            _logger.warning(f"Failed to parse _data from metadata: {str(e)}")
    
    # Fallback: Try to get from content (for backwards compatibility)
    content = mem0_record.get("memory") or mem0_record.get("content", "")
    
    if isinstance(content, str):
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError:
            pass
    elif isinstance(content, dict):
        return content
    
    # Last resort: Reconstruct from metadata fields
    if metadata:
        return {
            "session_id": metadata.get("session_id"),
            "session_type": metadata.get("session_type"),
            "agent_id": metadata.get("agent_id"),
            "team_id": metadata.get("team_id"),
            "workflow_id": metadata.get("workflow_id"),
            "user_id": metadata.get("user_id"),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
        }
    
    _logger.warning("Could not deserialize session from Mem0 record")
    return {}


def serialize_user_memory_to_mem0(
    user_id: str,
    user_memory: Dict[str, Any],
    table_name: str,
    agent_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Serialize user memory data for Mem0 storage.
    
    For Mem0 Platform, the message content is processed by LLM and transformed.
    We store the actual data in metadata._data as a JSON string.
    
    Args:
        user_id: The user ID
        user_memory: The memory data to store
        table_name: The table name for namespacing
        agent_id: Optional agent ID
        team_id: Optional team ID
    
    Returns:
        Dictionary with 'content', 'metadata', and 'memory_id'
    """
    current_time = int(time.time())
    
    # Prepare the full data payload
    data_payload = {
        "user_id": user_id,
        "user_memory": user_memory,
        "agent_id": agent_id,
        "team_id": team_id,
        "created_at": current_time,
        "updated_at": current_time,
    }
    
    memory_id = generate_user_memory_id(user_id, table_name)
    
    # Metadata contains:
    # 1. Filtering fields for lookups
    # 2. _data: Full user memory data as JSON string (not processed by LLM)
    metadata = {
        "_type": "user_memory",
        "_table": table_name,
        "_upsonic_memory_id": memory_id,
        "_data": json.dumps(data_payload),
        "user_id": user_id,
        "agent_id": agent_id,
        "team_id": team_id,
        "created_at": _epoch_to_iso(current_time),
        "updated_at": _epoch_to_iso(current_time),
    }
    
    # Remove None values from metadata
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    # Content is just a simple identifier (LLM will process this but we ignore it)
    content = f"Upsonic user memory for {user_id}"
    
    return {
        "content": content,
        "metadata": metadata,
        "memory_id": memory_id,
    }


def deserialize_user_memory_from_mem0(
    mem0_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deserialize a Mem0 record back to user memory dictionary.
    
    The actual user memory data is stored in metadata._data as a JSON string
    (not in the memory/content field which gets processed by LLM).
    
    Args:
        mem0_record: The Mem0 record containing metadata with _data field
    
    Returns:
        User memory dictionary
    """
    # Primary: Get data from metadata._data (where we store the actual data)
    metadata = mem0_record.get("metadata", {})
    data_str = metadata.get("_data")
    
    if data_str:
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            _logger.warning(f"Failed to parse _data from metadata: {data_str[:100]}")
    
    # Fallback: Try to get from content (for backwards compatibility)
    content = mem0_record.get("memory") or mem0_record.get("content", "")
    
    if isinstance(content, str):
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError:
            pass
    elif isinstance(content, dict):
        return content
    
    # Last resort: Reconstruct from metadata fields
    if metadata:
        return {
            "user_id": metadata.get("user_id"),
            "agent_id": metadata.get("agent_id"),
            "team_id": metadata.get("team_id"),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
        }
    
    _logger.warning("Could not deserialize user memory from Mem0 record")
    return {}


def serialize_cultural_knowledge_to_mem0(
    cultural_knowledge: "CulturalKnowledge",
    table_name: str,
) -> Dict[str, Any]:
    """
    Serialize cultural knowledge for Mem0 storage.
    
    For Mem0 Platform, the message content is processed by LLM and transformed.
    We store the actual data in metadata._data as a JSON string.
    
    Args:
        cultural_knowledge: The CulturalKnowledge instance to serialize
        table_name: The table name for namespacing
    
    Returns:
        Dictionary with 'content', 'metadata', and 'memory_id'
    """
    if not cultural_knowledge.id:
        import uuid
        cultural_knowledge.id = str(uuid.uuid4())
    
    current_time = int(time.time())
    
    # Use to_dict for serialization
    data_payload = cultural_knowledge.to_dict()
    
    # Convert RFC3339 timestamps to epoch seconds for storage
    if "created_at" in data_payload and data_payload["created_at"] is not None:
        if isinstance(data_payload["created_at"], str):
            # Parse RFC3339 to epoch
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(data_payload["created_at"].replace("Z", "+00:00"))
                data_payload["created_at"] = int(dt.timestamp())
            except (ValueError, AttributeError):
                data_payload["created_at"] = current_time
    else:
        data_payload["created_at"] = current_time
    
    # Always update updated_at
    data_payload["updated_at"] = current_time
    
    memory_id = generate_cultural_knowledge_memory_id(cultural_knowledge.id, table_name)
    
    # Metadata contains:
    # 1. Filtering fields for lookups
    # 2. _data: Full data as JSON string (not processed by LLM)
    metadata = {
        "_type": "cultural_knowledge",
        "_table": table_name,
        "_upsonic_memory_id": memory_id,
        "_data": json.dumps(data_payload),
        "id": cultural_knowledge.id,
        "name": cultural_knowledge.name,
        "agent_id": cultural_knowledge.agent_id,
        "team_id": cultural_knowledge.team_id,
        "created_at": _epoch_to_iso(data_payload["created_at"]),
        "updated_at": _epoch_to_iso(data_payload["updated_at"]),
    }
    
    # Remove None values from metadata
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    # Content is just a simple identifier (LLM will process this but we ignore it)
    content = f"Upsonic cultural knowledge: {cultural_knowledge.name or cultural_knowledge.id}"
    
    return {
        "content": content,
        "metadata": metadata,
        "memory_id": memory_id,
    }


def deserialize_cultural_knowledge_from_mem0(
    mem0_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deserialize a Mem0 record back to cultural knowledge dictionary.
    
    The actual data is stored in metadata._data as a JSON string.
    
    Args:
        mem0_record: The Mem0 record containing metadata with _data field
    
    Returns:
        Cultural knowledge dictionary
    """
    # Primary: Get data from metadata._data (where we store the actual data)
    metadata = mem0_record.get("metadata", {})
    data_str = metadata.get("_data")
    
    if data_str:
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            _logger.warning(f"Failed to parse _data from metadata: {data_str[:100]}")
    
    # Fallback: Try to get from content (for backwards compatibility)
    content = mem0_record.get("memory") or mem0_record.get("content", "")
    
    if isinstance(content, str):
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError:
            pass
    elif isinstance(content, dict):
        return content
    
    # Last resort: Reconstruct from metadata fields
    if metadata:
        return {
            "id": metadata.get("id"),
            "name": metadata.get("name"),
            "agent_id": metadata.get("agent_id"),
            "team_id": metadata.get("team_id"),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
        }
    
    _logger.warning("Could not deserialize cultural knowledge from Mem0 record")
    return {}


# ======================== Session Deserialization ========================

def deserialize_session_to_object(
    session_dict: Dict[str, Any],
    session_type: Optional["SessionType"] = None,
) -> "Session":
    """
    Deserialize a session dictionary to the appropriate session type object.
    
    Args:
        session_dict: The session data dictionary
        session_type: Optional session type hint
    
    Returns:
        Deserialized session object (AgentSession, TeamSession, or WorkflowSession)
    
    Raises:
        ValueError: If session type is invalid
    """
    from upsonic.session.agent import AgentSession
    from upsonic.session.base import SessionType as ST
    
    # Determine session type from dict if not provided
    if session_type is None:
        type_value = session_dict.get("session_type")
        if isinstance(type_value, str):
            session_type = ST(type_value)
        elif isinstance(type_value, dict):
            session_type = ST(type_value.get("session_type", "agent"))
        else:
            session_type = ST.AGENT
    
    if session_type == ST.AGENT:
        return AgentSession.from_dict(session_dict)
    elif session_type == ST.TEAM:
        # TeamSession not yet implemented - fallback to AgentSession
        return AgentSession.from_dict(session_dict)
    elif session_type == ST.WORKFLOW:
        # WorkflowSession not yet implemented - fallback to AgentSession
        return AgentSession.from_dict(session_dict)
    else:
        raise ValueError(f"Invalid session type: {session_type}")


# ======================== Metadata Filters ========================

def build_session_filters(
    table_name: str,
    session_id: Optional[str] = None,
    session_type: Optional["SessionType"] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build metadata filters for Mem0 session queries.
    
    Args:
        table_name: The table name to filter by
        session_id: Optional session ID filter
        session_type: Optional session type filter
        user_id: Optional user ID filter
        agent_id: Optional agent ID filter
    
    Returns:
        Dictionary of metadata filters for Mem0 search
    """
    filters: Dict[str, Any] = {
        "_type": "session",
        "_table": table_name,
    }
    
    if session_id is not None:
        filters["session_id"] = session_id
    if session_type is not None:
        filters["session_type"] = session_type.value if hasattr(session_type, "value") else str(session_type)
    if user_id is not None:
        filters["user_id"] = user_id
    if agent_id is not None:
        filters["agent_id"] = agent_id
    
    return filters


def build_user_memory_filters(
    table_name: str,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build metadata filters for Mem0 user memory queries.
    
    Args:
        table_name: The table name to filter by
        user_id: Optional user ID filter
        agent_id: Optional agent ID filter
        team_id: Optional team ID filter
    
    Returns:
        Dictionary of metadata filters for Mem0 search
    """
    filters: Dict[str, Any] = {
        "_type": "user_memory",
        "_table": table_name,
    }
    
    if user_id is not None:
        filters["user_id"] = user_id
    if agent_id is not None:
        filters["agent_id"] = agent_id
    if team_id is not None:
        filters["team_id"] = team_id
    
    return filters


def build_cultural_knowledge_filters(
    table_name: str,
    name: Optional[str] = None,
    agent_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build metadata filters for Mem0 cultural knowledge queries.
    
    Args:
        table_name: The table name to filter by
        name: Optional name filter
        agent_id: Optional agent ID filter
        team_id: Optional team ID filter
    
    Returns:
        Dictionary of metadata filters for Mem0 search
    """
    filters: Dict[str, Any] = {
        "_type": "cultural_knowledge",
        "_table": table_name,
    }
    
    if name is not None:
        filters["name"] = name
    if agent_id is not None:
        filters["agent_id"] = agent_id
    if team_id is not None:
        filters["team_id"] = team_id
    
    return filters


# ======================== Knowledge Helpers ========================


def generate_knowledge_memory_id(document_id: str, table_name: str) -> str:
    """Generate a unique memory ID for a knowledge document record."""
    return f"knowledge__{table_name}__{document_id}"


def serialize_knowledge_to_mem0(
    knowledge_row: "KnowledgeRow",
    table_name: str,
) -> Dict[str, Any]:
    """Serialize a KnowledgeRow for Mem0 storage.

    Stores the full KnowledgeRow data as a JSON string in metadata._data,
    with key fields duplicated in metadata for filtering.
    """
    current_time = int(time.time())

    data_payload = knowledge_row.to_dict()
    data_payload.setdefault("created_at", current_time)
    data_payload["updated_at"] = current_time

    memory_id = generate_knowledge_memory_id(data_payload["id"], table_name)

    metadata: Dict[str, Any] = {
        "_type": "knowledge",
        "_table": table_name,
        "_upsonic_memory_id": memory_id,
        "_data": json.dumps(data_payload, default=str),
        "knowledge_base_id": data_payload.get("knowledge_base_id") or "",
        "document_id": data_payload["id"],
        "status": data_payload.get("status") or "",
        "created_at": _epoch_to_iso(data_payload["created_at"])
        if isinstance(data_payload.get("created_at"), int)
        else data_payload.get("created_at", ""),
        "updated_at": _epoch_to_iso(data_payload["updated_at"])
        if isinstance(data_payload.get("updated_at"), int)
        else data_payload.get("updated_at", ""),
    }

    content = f"Knowledge document: {data_payload.get('name', data_payload['id'])}"

    return {
        "content": content,
        "metadata": metadata,
        "memory_id": memory_id,
    }


def deserialize_knowledge_from_mem0(
    mem0_record: Dict[str, Any],
) -> Dict[str, Any]:
    """Deserialize a Mem0 record back to a knowledge dictionary."""
    metadata = mem0_record.get("metadata", {})
    data_str = metadata.get("_data")

    if data_str:
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            _logger.warning(f"Failed to parse knowledge _data from metadata: {data_str[:100]}")

    return {}


def build_knowledge_filters(
    table_name: str,
    knowledge_base_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build metadata filters for Mem0 knowledge queries."""
    filters: Dict[str, Any] = {
        "_type": "knowledge",
        "_table": table_name,
    }

    if knowledge_base_id is not None:
        filters["knowledge_base_id"] = knowledge_base_id

    return filters


# ======================== Sorting and Pagination ========================

def sort_records_by_field(
    records: List[Dict[str, Any]],
    sort_by: Optional[str] = "created_at",
    sort_order: Optional[str] = "desc",
) -> List[Dict[str, Any]]:
    """
    Sort records by a given field.
    
    Handles both ISO 8601 datetime strings and integer epoch timestamps.
    
    Args:
        records: List of record dictionaries
        sort_by: Field to sort by (default: created_at)
        sort_order: Sort order ('asc' or 'desc', default: desc)
    
    Returns:
        Sorted list of records
    """
    if not records or not sort_by:
        return records
    
    reverse = sort_order != "asc"
    
    def get_sort_key(record: Dict[str, Any]) -> Any:
        metadata = record.get("metadata", {})
        value = metadata.get(sort_by) or record.get(sort_by, 0)
        if value is None:
            return 0
        if isinstance(value, str):
            try:
                return _iso_to_epoch(value)
            except (ValueError, TypeError):
                return 0
        return value
    
    return sorted(records, key=get_sort_key, reverse=reverse)


def apply_pagination(
    records: List[Dict[str, Any]],
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Apply pagination to a list of records.
    
    Args:
        records: List of record dictionaries
        limit: Maximum number of records to return
        offset: Number of records to skip
    
    Returns:
        Paginated list of records
    """
    if not records:
        return records
    
    start = offset or 0
    if limit is not None:
        end = start + limit
        return records[start:end]
    
    return records[start:]

