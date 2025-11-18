"""
Tests for checkpoint functionality in GraphV2.
"""

import pytest
import sqlite3
from upsonic.graphv2.checkpoint import (
    Checkpoint,
    StateSnapshot,
    MemorySaver,
    SqliteCheckpointer,
    generate_checkpoint_id,
)
from datetime import datetime


class TestMemorySaver:
    """Test MemorySaver checkpointer."""

    def test_memory_saver(self):
        """Test MemorySaver checkpointer."""
        checkpointer = MemorySaver()

        checkpoint = Checkpoint(
            checkpoint_id="test-1",
            thread_id="thread-1",
            state={"count": 1},
            next_nodes=["node1"],
        )

        checkpointer.put(checkpoint)

        retrieved = checkpointer.get("thread-1")
        assert retrieved is not None
        assert retrieved.checkpoint_id == "test-1"
        assert retrieved.state["count"] == 1

    def test_memory_saver_get_latest(self):
        """Test getting latest checkpoint."""
        checkpointer = MemorySaver()

        checkpoint1 = Checkpoint(
            checkpoint_id="test-1",
            thread_id="thread-1",
            state={"count": 1},
            next_nodes=["node1"],
        )

        checkpoint2 = Checkpoint(
            checkpoint_id="test-2",
            thread_id="thread-1",
            state={"count": 2},
            next_nodes=["node2"],
        )

        checkpointer.put(checkpoint1)
        checkpointer.put(checkpoint2)

        latest = checkpointer.get("thread-1")
        assert latest.checkpoint_id == "test-2"

    def test_memory_saver_get_specific(self):
        """Test getting specific checkpoint."""
        checkpointer = MemorySaver()

        checkpoint1 = Checkpoint(
            checkpoint_id="test-1",
            thread_id="thread-1",
            state={"count": 1},
            next_nodes=["node1"],
        )

        checkpoint2 = Checkpoint(
            checkpoint_id="test-2",
            thread_id="thread-1",
            state={"count": 2},
            next_nodes=["node2"],
        )

        checkpointer.put(checkpoint1)
        checkpointer.put(checkpoint2)

        specific = checkpointer.get("thread-1", "test-1")
        assert specific.checkpoint_id == "test-1"

    def test_memory_saver_list(self):
        """Test listing checkpoints."""
        checkpointer = MemorySaver()

        checkpoint1 = Checkpoint(
            checkpoint_id="test-1",
            thread_id="thread-1",
            state={"count": 1},
            next_nodes=["node1"],
        )

        checkpoint2 = Checkpoint(
            checkpoint_id="test-2",
            thread_id="thread-1",
            state={"count": 2},
            next_nodes=["node2"],
        )

        checkpointer.put(checkpoint1)
        checkpointer.put(checkpoint2)

        checkpoints = list(checkpointer.list("thread-1"))
        assert len(checkpoints) == 2
        assert checkpoints[0].checkpoint_id == "test-2"  # Newest first


class TestSqliteCheckpointer:
    """Test SqliteCheckpointer."""

    def test_sqlite_checkpointer(self):
        """Test SqliteCheckpointer."""
        conn = sqlite3.connect(":memory:")
        checkpointer = SqliteCheckpointer(conn)

        checkpoint = Checkpoint(
            checkpoint_id="test-1",
            thread_id="thread-1",
            state={"count": 1},
            next_nodes=["node1"],
        )

        checkpointer.put(checkpoint)

        retrieved = checkpointer.get("thread-1")
        assert retrieved is not None
        assert retrieved.checkpoint_id == "test-1"
        assert retrieved.state["count"] == 1

        conn.close()

    def test_sqlite_checkpointer_get_specific(self):
        """Test getting specific checkpoint from SQLite."""
        conn = sqlite3.connect(":memory:")
        checkpointer = SqliteCheckpointer(conn)

        checkpoint1 = Checkpoint(
            checkpoint_id="test-1",
            thread_id="thread-1",
            state={"count": 1},
            next_nodes=["node1"],
        )

        checkpoint2 = Checkpoint(
            checkpoint_id="test-2",
            thread_id="thread-1",
            state={"count": 2},
            next_nodes=["node2"],
        )

        checkpointer.put(checkpoint1)
        checkpointer.put(checkpoint2)

        specific = checkpointer.get("thread-1", "test-1")
        assert specific.checkpoint_id == "test-1"

        conn.close()

    def test_sqlite_checkpointer_list(self):
        """Test listing checkpoints from SQLite."""
        conn = sqlite3.connect(":memory:")
        checkpointer = SqliteCheckpointer(conn)

        checkpoint1 = Checkpoint(
            checkpoint_id="test-1",
            thread_id="thread-1",
            state={"count": 1},
            next_nodes=["node1"],
        )

        checkpoint2 = Checkpoint(
            checkpoint_id="test-2",
            thread_id="thread-1",
            state={"count": 2},
            next_nodes=["node2"],
        )

        checkpointer.put(checkpoint1)
        checkpointer.put(checkpoint2)

        checkpoints = list(checkpointer.list("thread-1"))
        assert len(checkpoints) == 2

        conn.close()


class TestCheckpointCreation:
    """Test checkpoint creation."""

    def test_checkpoint_creation(self):
        """Test checkpoint creation."""
        checkpoint = Checkpoint(
            checkpoint_id="test-1",
            thread_id="thread-1",
            state={"count": 1, "message": "test"},
            next_nodes=["node1", "node2"],
        )

        assert checkpoint.checkpoint_id == "test-1"
        assert checkpoint.thread_id == "thread-1"
        assert checkpoint.state["count"] == 1
        assert checkpoint.next_nodes == ["node1", "node2"]
        assert checkpoint.parent_checkpoint_id is None
        assert isinstance(checkpoint.timestamp, datetime)

    def test_checkpoint_with_parent(self):
        """Test checkpoint with parent."""
        checkpoint = Checkpoint(
            checkpoint_id="test-2",
            thread_id="thread-1",
            state={"count": 2},
            next_nodes=["node2"],
            parent_checkpoint_id="test-1",
        )

        assert checkpoint.parent_checkpoint_id == "test-1"

    def test_checkpoint_to_dict(self):
        """Test checkpoint serialization."""
        checkpoint = Checkpoint(
            checkpoint_id="test-1",
            thread_id="thread-1",
            state={"count": 1},
            next_nodes=["node1"],
        )

        data = checkpoint.to_dict()
        assert data["checkpoint_id"] == "test-1"
        assert data["thread_id"] == "thread-1"
        assert data["state"]["count"] == 1

    def test_checkpoint_from_dict(self):
        """Test checkpoint deserialization."""
        data = {
            "checkpoint_id": "test-1",
            "thread_id": "thread-1",
            "state": {"count": 1},
            "next_nodes": ["node1"],
            "parent_checkpoint_id": None,
            "timestamp": datetime.now().isoformat(),
            "metadata": {},
        }

        checkpoint = Checkpoint.from_dict(data)
        assert checkpoint.checkpoint_id == "test-1"
        assert checkpoint.state["count"] == 1


class TestCheckpointRetrieval:
    """Test checkpoint retrieval."""

    def test_checkpoint_retrieval(self):
        """Test checkpoint retrieval."""
        checkpointer = MemorySaver()

        checkpoint = Checkpoint(
            checkpoint_id="test-1",
            thread_id="thread-1",
            state={"count": 1},
            next_nodes=["node1"],
        )

        checkpointer.put(checkpoint)

        retrieved = checkpointer.get("thread-1")
        assert retrieved is not None
        assert retrieved.checkpoint_id == "test-1"

    def test_checkpoint_retrieval_nonexistent(self):
        """Test retrieving nonexistent checkpoint."""
        checkpointer = MemorySaver()

        retrieved = checkpointer.get("nonexistent")
        assert retrieved is None

    def test_checkpoint_get_history(self):
        """Test getting checkpoint history."""
        checkpointer = MemorySaver()

        checkpoint1 = Checkpoint(
            checkpoint_id="test-1",
            thread_id="thread-1",
            state={"count": 1},
            next_nodes=["node1"],
        )

        checkpoint2 = Checkpoint(
            checkpoint_id="test-2",
            thread_id="thread-1",
            state={"count": 2},
            next_nodes=["node2"],
        )

        checkpointer.put(checkpoint1)
        checkpointer.put(checkpoint2)

        history = checkpointer.get_history("thread-1")
        assert len(history) == 2
        assert history[0].checkpoint_id == "test-2"  # Newest first

    def test_checkpoint_get_history_limit(self):
        """Test getting checkpoint history with limit."""
        checkpointer = MemorySaver()

        for i in range(5):
            checkpoint = Checkpoint(
                checkpoint_id=f"test-{i}",
                thread_id="thread-1",
                state={"count": i},
                next_nodes=[f"node{i}"],
            )
            checkpointer.put(checkpoint)

        history = checkpointer.get_history("thread-1", limit=3)
        assert len(history) == 3


class TestStateSnapshot:
    """Test state snapshot functionality."""

    def test_state_snapshot(self):
        """Test state snapshot functionality."""
        checkpoint = Checkpoint(
            checkpoint_id="test-1",
            thread_id="thread-1",
            state={"count": 1, "message": "test"},
            next_nodes=["node1"],
            metadata={"key": "value"},
        )

        snapshot = StateSnapshot(
            values=checkpoint.state,
            next=checkpoint.next_nodes,
            config={
                "configurable": {
                    "thread_id": checkpoint.thread_id,
                    "checkpoint_id": checkpoint.checkpoint_id,
                }
            },
            metadata=checkpoint.metadata,
        )

        assert snapshot.values["count"] == 1
        assert snapshot.next == ["node1"]
        assert snapshot.config["configurable"]["thread_id"] == "thread-1"
        assert snapshot.metadata["key"] == "value"

    def test_state_snapshot_with_parent(self):
        """Test state snapshot with parent config."""
        checkpoint = Checkpoint(
            checkpoint_id="test-2",
            thread_id="thread-1",
            state={"count": 2},
            next_nodes=["node2"],
            parent_checkpoint_id="test-1",
        )

        snapshot = StateSnapshot(
            values=checkpoint.state,
            next=checkpoint.next_nodes,
            config={
                "configurable": {
                    "thread_id": checkpoint.thread_id,
                    "checkpoint_id": checkpoint.checkpoint_id,
                }
            },
            metadata={},
            parent_config={
                "configurable": {
                    "thread_id": checkpoint.thread_id,
                    "checkpoint_id": checkpoint.parent_checkpoint_id,
                }
            },
        )

        assert snapshot.parent_config is not None
        assert snapshot.parent_config["configurable"]["checkpoint_id"] == "test-1"


class TestGenerateCheckpointId:
    """Test checkpoint ID generation."""

    def test_generate_checkpoint_id(self):
        """Test checkpoint ID generation."""
        id1 = generate_checkpoint_id()
        id2 = generate_checkpoint_id()

        assert id1 != id2
        assert len(id1) > 0
        assert len(id2) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
