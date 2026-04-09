"""Schema definitions for Redis storage tables.

This module defines the structure of tables stored in Redis.
Each schema describes the fields, their types, and constraints.

SESSION_SCHEMA: Schema for storing agent/team/workflow sessions.
USER_MEMORY_SCHEMA: Schema for storing user memory/profile data.
"""
from typing import Any, Dict, Final

# Session table schema - stores AgentSession, TeamSession, WorkflowSession
SESSION_SCHEMA: Final[Dict[str, Dict[str, Any]]] = {
    "session_id": {"type": "string", "primary_key": True},
    "session_type": {"type": "string"},  # "agent", "team", or "workflow"
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
    "messages": {"type": "json"},
    "summary": {"type": "string"},
    "usage": {"type": "json"},
    "created_at": {"type": "integer"},
    "updated_at": {"type": "integer"},
}

# User memory table schema - stores user profile/memory data
USER_MEMORY_SCHEMA: Final[Dict[str, Dict[str, Any]]] = {
    "user_id": {"type": "string", "primary_key": True},
    "user_memory": {"type": "json"},
    "agent_id": {"type": "string"},
    "team_id": {"type": "string"},
    "created_at": {"type": "integer"},
    "updated_at": {"type": "integer"},
}

# Cultural knowledge table schema - stores CulturalKnowledge data
CULTURAL_KNOWLEDGE_SCHEMA: Final[Dict[str, Dict[str, Any]]] = {
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

# Index fields for faster lookups
SESSION_INDEX_FIELDS: Final[list[str]] = [
    "user_id",
    "agent_id", 
    "team_id",
    "workflow_id",
    "session_type",
]

USER_MEMORY_INDEX_FIELDS: Final[list[str]] = [
    "user_id",
    "agent_id",
    "team_id",
]

CULTURAL_KNOWLEDGE_INDEX_FIELDS: Final[list[str]] = [
    "name",
    "agent_id",
    "team_id",
]

# Knowledge table schema - stores KnowledgeRow data
KNOWLEDGE_SCHEMA: Final[Dict[str, Dict[str, Any]]] = {
    "id": {"type": "string", "primary_key": True},
    "name": {"type": "string"},
    "description": {"type": "string"},
    "metadata": {"type": "json"},
    "type": {"type": "string"},
    "size": {"type": "integer"},
    "knowledge_base_id": {"type": "string"},
    "content_hash": {"type": "string"},
    "chunk_count": {"type": "integer"},
    "source": {"type": "string"},
    "status": {"type": "string"},
    "status_message": {"type": "string"},
    "access_count": {"type": "integer"},
    "created_at": {"type": "integer"},
    "updated_at": {"type": "integer"},
}

KNOWLEDGE_INDEX_FIELDS: Final[list[str]] = [
    "knowledge_base_id",
    "type",
    "content_hash",
    "status",
]
