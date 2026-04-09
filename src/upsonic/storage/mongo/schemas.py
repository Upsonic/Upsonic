"""MongoDB collection schema definitions for Upsonic storage.

This module defines the MongoDB collection schemas as index specifications
and validation rules for sessions and user memories.
"""
from typing import Any, Dict, List


# Session collection indexes
SESSION_COLLECTION_INDEXES: List[Dict[str, Any]] = [
    {"keys": [("session_id", 1)], "unique": True},
    {"keys": [("session_type", 1)]},
    {"keys": [("agent_id", 1)]},
    {"keys": [("team_id", 1)]},
    {"keys": [("workflow_id", 1)]},
    {"keys": [("user_id", 1)]},
    {"keys": [("created_at", -1)]},
    {"keys": [("updated_at", -1)]},
]

# User memory collection indexes
USER_MEMORY_COLLECTION_INDEXES: List[Dict[str, Any]] = [
    {"keys": [("user_id", 1)], "unique": True},
    {"keys": [("agent_id", 1)]},
    {"keys": [("team_id", 1)]},
    {"keys": [("created_at", -1)]},
    {"keys": [("updated_at", -1)]},
]


# Cultural knowledge collection indexes
CULTURAL_KNOWLEDGE_COLLECTION_INDEXES: List[Dict[str, Any]] = [
    {"keys": [("id", 1)], "unique": True},
    {"keys": [("name", 1)]},
    {"keys": [("agent_id", 1)]},
    {"keys": [("team_id", 1)]},
    {"keys": [("created_at", -1)]},
    {"keys": [("updated_at", -1)]},
]


# Knowledge collection indexes
KNOWLEDGE_COLLECTION_INDEXES: List[Dict[str, Any]] = [
    {"keys": [("id", 1)], "unique": True},
    {"keys": [("knowledge_base_id", 1)]},
    {"keys": [("type", 1)]},
    {"keys": [("content_hash", 1)]},
    {"keys": [("status", 1)]},
    {"keys": [("created_at", -1)]},
    {"keys": [("updated_at", -1)]},
]


# Session document structure (for reference/validation)
SESSION_DOCUMENT_STRUCTURE: Dict[str, Dict[str, Any]] = {
    "session_id": {"type": "string", "required": True, "unique": True},
    "session_type": {"type": "string", "required": True},
    "agent_id": {"type": "string", "required": False},
    "team_id": {"type": "string", "required": False},
    "workflow_id": {"type": "string", "required": False},
    "user_id": {"type": "string", "required": False},
    "session_data": {"type": "object", "required": False},
    "agent_data": {"type": "object", "required": False},
    "team_data": {"type": "object", "required": False},
    "workflow_data": {"type": "object", "required": False},
    "metadata": {"type": "object", "required": False},
    "runs": {"type": "object", "required": False},
    "messages": {"type": "object", "required": False},
    "summary": {"type": "string", "required": False},
    "usage": {"type": "object", "required": False},
    "created_at": {"type": "int", "required": True},
    "updated_at": {"type": "int", "required": False},
}


# User memory document structure (for reference/validation)
USER_MEMORY_DOCUMENT_STRUCTURE: Dict[str, Dict[str, Any]] = {
    "user_id": {"type": "string", "required": True, "unique": True},
    "user_memory": {"type": "object", "required": True},
    "agent_id": {"type": "string", "required": False},
    "team_id": {"type": "string", "required": False},
    "created_at": {"type": "int", "required": True},
    "updated_at": {"type": "int", "required": False},
}


# Cultural knowledge document structure (for reference/validation)
CULTURAL_KNOWLEDGE_DOCUMENT_STRUCTURE: Dict[str, Dict[str, Any]] = {
    "id": {"type": "string", "required": True, "unique": True},
    "name": {"type": "string", "required": True},
    "summary": {"type": "string", "required": False},
    "content": {"type": "string", "required": False},
    "metadata": {"type": "object", "required": False},
    "notes": {"type": "array", "required": False},
    "categories": {"type": "array", "required": False},
    "input": {"type": "string", "required": False},
    "created_at": {"type": "int", "required": False},
    "updated_at": {"type": "int", "required": False},
    "agent_id": {"type": "string", "required": False},
    "team_id": {"type": "string", "required": False},
}


# Knowledge document structure (for reference/validation)
KNOWLEDGE_DOCUMENT_STRUCTURE: Dict[str, Dict[str, Any]] = {
    "id": {"type": "string", "required": True, "unique": True},
    "name": {"type": "string", "required": True},
    "description": {"type": "string", "required": False},
    "metadata": {"type": "object", "required": False},
    "type": {"type": "string", "required": False},
    "size": {"type": "int", "required": False},
    "knowledge_base_id": {"type": "string", "required": False},
    "content_hash": {"type": "string", "required": False},
    "chunk_count": {"type": "int", "required": False},
    "source": {"type": "string", "required": False},
    "status": {"type": "string", "required": False},
    "status_message": {"type": "string", "required": False},
    "access_count": {"type": "int", "required": False},
    "created_at": {"type": "int", "required": False},
    "updated_at": {"type": "int", "required": False},
}


def get_collection_indexes(collection_type: str) -> List[Dict[str, Any]]:
    """
    Get the index specifications for a given collection type.
    
    Args:
        collection_type: Type of collection ('sessions', 'user_memories', or 'cultural_knowledge')
    
    Returns:
        List of index specification dictionaries
    
    Raises:
        ValueError: If collection_type is not recognized
    """
    indexes: Dict[str, List[Dict[str, Any]]] = {
        "sessions": SESSION_COLLECTION_INDEXES,
        "user_memories": USER_MEMORY_COLLECTION_INDEXES,
        "cultural_knowledge": CULTURAL_KNOWLEDGE_COLLECTION_INDEXES,
        "knowledge": KNOWLEDGE_COLLECTION_INDEXES,
    }
    
    if collection_type not in indexes:
        raise ValueError(
            f"Unknown collection type: '{collection_type}'. "
            f"Available types: {list(indexes.keys())}"
        )
    
    return indexes[collection_type].copy()


def get_document_structure(collection_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get the document structure for a given collection type.
    
    Args:
        collection_type: Type of collection ('sessions', 'user_memories', or 'cultural_knowledge')
    
    Returns:
        Dictionary containing document field definitions
    
    Raises:
        ValueError: If collection_type is not recognized
    """
    structures: Dict[str, Dict[str, Dict[str, Any]]] = {
        "sessions": SESSION_DOCUMENT_STRUCTURE,
        "user_memories": USER_MEMORY_DOCUMENT_STRUCTURE,
        "cultural_knowledge": CULTURAL_KNOWLEDGE_DOCUMENT_STRUCTURE,
        "knowledge": KNOWLEDGE_DOCUMENT_STRUCTURE,
    }
    
    if collection_type not in structures:
        raise ValueError(
            f"Unknown collection type: '{collection_type}'. "
            f"Available types: {list(structures.keys())}"
        )
    
    return structures[collection_type].copy()

