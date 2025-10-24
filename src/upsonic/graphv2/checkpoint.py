"""
Checkpoint system for persisting graph state across executions.

This module provides the infrastructure for durable execution, allowing graphs
to save their state at specific points and resume from those checkpoints later.
"""

from __future__ import annotations

import json
import pickle
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Iterator

from upsonic._utils import now_utc


@dataclass
class Checkpoint:
    """A snapshot of graph state at a specific point in execution.
    
    Attributes:
        checkpoint_id: Unique identifier for this checkpoint
        thread_id: Thread identifier grouping related executions
        state: The actual state data at this checkpoint
        next_nodes: List of nodes to execute next
        parent_checkpoint_id: ID of the previous checkpoint (for history)
        timestamp: When this checkpoint was created
        metadata: Additional checkpoint metadata
    """
    
    checkpoint_id: str
    thread_id: str
    state: Dict[str, Any]
    next_nodes: List[str]
    parent_checkpoint_id: Optional[str] = None
    timestamp: datetime = field(default_factory=now_utc)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "thread_id": self.thread_id,
            "state": self.state,
            "next_nodes": self.next_nodes,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Deserialize checkpoint from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            from datetime import datetime
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class StateSnapshot:
    """A snapshot of graph state with execution context.
    
    This is returned when querying graph state and provides not just the state
    values but also information about where execution is and configuration.
    """
    
    values: Dict[str, Any]
    """Current state values."""
    
    next: List[str]
    """List of node names to execute next."""
    
    config: Dict[str, Any]
    """Configuration including thread_id and checkpoint_id."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about this snapshot."""
    
    parent_config: Optional[Dict[str, Any]] = None
    """Configuration of the parent checkpoint."""


class BaseCheckpointer(ABC):
    """Abstract base class for checkpoint persistence.
    
    Checkpointers are responsible for saving and retrieving graph state,
    enabling features like persistence, time travel, and interrupts.
    """
    
    @abstractmethod
    def put(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint.
        
        Args:
            checkpoint: The checkpoint to save
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        """Retrieve a checkpoint.
        
        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint ID, or None for latest
            
        Returns:
            The checkpoint if found, None otherwise
        """
        raise NotImplementedError()
    
    @abstractmethod
    def list(self, thread_id: str) -> Iterator[Checkpoint]:
        """List all checkpoints for a thread in reverse chronological order.
        
        Args:
            thread_id: Thread identifier
            
        Yields:
            Checkpoints from newest to oldest
        """
        raise NotImplementedError()
    
    def get_history(self, thread_id: str, limit: Optional[int] = None) -> List[Checkpoint]:
        """Get checkpoint history for a thread.
        
        Args:
            thread_id: Thread identifier
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoints from newest to oldest
        """
        checkpoints = []
        for checkpoint in self.list(thread_id):
            checkpoints.append(checkpoint)
            if limit and len(checkpoints) >= limit:
                break
        return checkpoints


class MemorySaver(BaseCheckpointer):
    """In-memory checkpoint storage.
    
    Stores checkpoints in memory. Data is lost when the program ends.
    Useful for development, testing, and single-session applications.
    """
    
    def __init__(self):
        """Initialize the memory saver."""
        self._storage: Dict[str, List[Checkpoint]] = {}
    
    def put(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint to memory.
        
        Args:
            checkpoint: The checkpoint to save
        """
        thread_id = checkpoint.thread_id
        
        if thread_id not in self._storage:
            self._storage[thread_id] = []
        
        # Add to the beginning (newest first)
        self._storage[thread_id].insert(0, checkpoint)
    
    def get(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        """Retrieve a checkpoint from memory.
        
        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint ID, or None for latest
            
        Returns:
            The checkpoint if found, None otherwise
        """
        if thread_id not in self._storage:
            return None
        
        checkpoints = self._storage[thread_id]
        
        if not checkpoints:
            return None
        
        if checkpoint_id is None:
            # Return the latest checkpoint
            return checkpoints[0]
        
        # Search for specific checkpoint
        for checkpoint in checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        
        return None
    
    def list(self, thread_id: str) -> Iterator[Checkpoint]:
        """List all checkpoints for a thread.
        
        Args:
            thread_id: Thread identifier
            
        Yields:
            Checkpoints from newest to oldest
        """
        if thread_id in self._storage:
            for checkpoint in self._storage[thread_id]:
                yield checkpoint


class SqliteCheckpointer(BaseCheckpointer):
    """SQLite-based checkpoint storage.
    
    Stores checkpoints in a SQLite database for persistence across
    program runs. Suitable for local development and single-machine deployments.
    """
    
    def __init__(self, connection):
        """Initialize the SQLite checkpointer.
        
        Args:
            connection: sqlite3 connection object
        """
        self.connection = connection
        self._create_tables()
    
    def _create_tables(self):
        """Create the checkpoints table if it doesn't exist."""
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                state BLOB NOT NULL,
                next_nodes TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_timestamp 
            ON checkpoints(thread_id, timestamp DESC)
        """)
        self.connection.commit()
    
    def put(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint to SQLite.
        
        Args:
            checkpoint: The checkpoint to save
        """
        cursor = self.connection.cursor()
        
        # Serialize state using pickle for complex objects
        state_blob = pickle.dumps(checkpoint.state)
        next_nodes_json = json.dumps(checkpoint.next_nodes)
        metadata_json = json.dumps(checkpoint.metadata)
        
        cursor.execute("""
            INSERT OR REPLACE INTO checkpoints 
            (checkpoint_id, thread_id, parent_checkpoint_id, state, next_nodes, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            checkpoint.checkpoint_id,
            checkpoint.thread_id,
            checkpoint.parent_checkpoint_id,
            state_blob,
            next_nodes_json,
            checkpoint.timestamp.isoformat(),
            metadata_json,
        ))
        
        self.connection.commit()
    
    def get(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        """Retrieve a checkpoint from SQLite.
        
        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint ID, or None for latest
            
        Returns:
            The checkpoint if found, None otherwise
        """
        cursor = self.connection.cursor()
        
        if checkpoint_id is None:
            # Get the latest checkpoint
            cursor.execute("""
                SELECT checkpoint_id, thread_id, parent_checkpoint_id, state, 
                       next_nodes, timestamp, metadata
                FROM checkpoints
                WHERE thread_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (thread_id,))
        else:
            # Get specific checkpoint
            cursor.execute("""
                SELECT checkpoint_id, thread_id, parent_checkpoint_id, state, 
                       next_nodes, timestamp, metadata
                FROM checkpoints
                WHERE checkpoint_id = ?
            """, (checkpoint_id,))
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return self._row_to_checkpoint(row)
    
    def list(self, thread_id: str) -> Iterator[Checkpoint]:
        """List all checkpoints for a thread.
        
        Args:
            thread_id: Thread identifier
            
        Yields:
            Checkpoints from newest to oldest
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT checkpoint_id, thread_id, parent_checkpoint_id, state, 
                   next_nodes, timestamp, metadata
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY timestamp DESC
        """, (thread_id,))
        
        for row in cursor.fetchall():
            yield self._row_to_checkpoint(row)
    
    def _row_to_checkpoint(self, row) -> Checkpoint:
        """Convert a database row to a Checkpoint object."""
        checkpoint_id, thread_id, parent_checkpoint_id, state_blob, next_nodes_json, timestamp_str, metadata_json = row
        
        state = pickle.loads(state_blob)
        next_nodes = json.loads(next_nodes_json)
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        from datetime import datetime
        timestamp = datetime.fromisoformat(timestamp_str)
        
        return Checkpoint(
            checkpoint_id=checkpoint_id,
            thread_id=thread_id,
            parent_checkpoint_id=parent_checkpoint_id,
            state=state,
            next_nodes=next_nodes,
            timestamp=timestamp,
            metadata=metadata,
        )


def generate_checkpoint_id() -> str:
    """Generate a unique checkpoint ID."""
    return str(uuid.uuid4())

