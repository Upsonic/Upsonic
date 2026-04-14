"""SQLite table schema definitions for Upsonic storage."""
from typing import Any, Dict

try:
    from sqlalchemy import BigInteger, JSON, String
except ImportError:
    raise ImportError(
        "`sqlalchemy` not installed. Please install it using `pip install sqlalchemy aiosqlite`"
    )


SESSION_TABLE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "session_id": {"type": String, "primary_key": True, "nullable": False},
    "session_type": {"type": String, "nullable": False, "index": True},
    "agent_id": {"type": String, "nullable": True, "index": True},
    "team_id": {"type": String, "nullable": True, "index": True},
    "workflow_id": {"type": String, "nullable": True, "index": True},
    "user_id": {"type": String, "nullable": True, "index": True},
    "session_data": {"type": JSON, "nullable": True},
    "agent_data": {"type": JSON, "nullable": True},
    "team_data": {"type": JSON, "nullable": True},
    "workflow_data": {"type": JSON, "nullable": True},
    "metadata": {"type": JSON, "nullable": True},
    "runs": {"type": JSON, "nullable": True},
    "messages": {"type": JSON, "nullable": True},
    "usage": {"type": JSON, "nullable": True},  # Session-level aggregated RunUsage
    "summary": {"type": String, "nullable": True},
    "created_at": {"type": BigInteger, "nullable": False, "index": True},
    "updated_at": {"type": BigInteger, "nullable": True, "index": True},
}


USER_MEMORY_TABLE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "user_id": {"type": String, "primary_key": True, "nullable": False},
    "user_memory": {"type": JSON, "nullable": False},
    "agent_id": {"type": String, "nullable": True, "index": True},
    "team_id": {"type": String, "nullable": True, "index": True},
    "created_at": {"type": BigInteger, "nullable": False, "index": True},
    "updated_at": {"type": BigInteger, "nullable": True, "index": True},
}


CULTURAL_KNOWLEDGE_TABLE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "id": {"type": String, "primary_key": True, "nullable": False},
    "name": {"type": String, "nullable": False, "index": True},
    "summary": {"type": String, "nullable": True},
    "content": {"type": String, "nullable": True},
    "metadata": {"type": JSON, "nullable": True},
    "notes": {"type": JSON, "nullable": True},
    "categories": {"type": JSON, "nullable": True},
    "input": {"type": String, "nullable": True},
    "created_at": {"type": BigInteger, "nullable": True, "index": True},
    "updated_at": {"type": BigInteger, "nullable": True, "index": True},
    "agent_id": {"type": String, "nullable": True, "index": True},
    "team_id": {"type": String, "nullable": True, "index": True},
}


KNOWLEDGE_TABLE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "id": {"type": String, "primary_key": True, "nullable": False},
    "name": {"type": String, "nullable": False},
    "description": {"type": String, "nullable": True},
    "metadata": {"type": JSON, "nullable": True},
    "type": {"type": String, "nullable": True, "index": True},
    "size": {"type": BigInteger, "nullable": True},
    "knowledge_base_id": {"type": String, "nullable": True, "index": True},
    "content_hash": {"type": String, "nullable": True, "index": True},
    "chunk_count": {"type": BigInteger, "nullable": True},
    "source": {"type": String, "nullable": True},
    "status": {"type": String, "nullable": True, "index": True},
    "status_message": {"type": String, "nullable": True},
    "access_count": {"type": BigInteger, "nullable": True},
    "created_at": {"type": BigInteger, "nullable": True, "index": True},
    "updated_at": {"type": BigInteger, "nullable": True, "index": True},
}


def get_table_schema_definition(table_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get the schema definition for a given table type.
    
    Args:
        table_type: Type of table ('sessions', 'user_memories', 'cultural_knowledge', or 'knowledge')
    
    Returns:
        Dictionary containing schema definition for the table
    
    Raises:
        ValueError: If table_type is not recognized
    """
    schemas: Dict[str, Dict[str, Dict[str, Any]]] = {
        "sessions": SESSION_TABLE_SCHEMA,
        "user_memories": USER_MEMORY_TABLE_SCHEMA,
        "cultural_knowledge": CULTURAL_KNOWLEDGE_TABLE_SCHEMA,
        "knowledge": KNOWLEDGE_TABLE_SCHEMA,
    }
    
    if table_type not in schemas:
        raise ValueError(
            f"Unknown table type: '{table_type}'. "
            f"Available types: {list(schemas.keys())}"
        )
    
    return schemas[table_type].copy()

