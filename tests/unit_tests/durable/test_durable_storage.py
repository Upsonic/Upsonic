import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from upsonic.durable.storage import ExecutionState
from upsonic.durable.storages.memory import InMemoryDurableStorage
from upsonic.durable.storages.file import FileDurableStorage
from upsonic.durable.storages.sqlite import SQLiteDurableStorage


def test_in_memory_durable_storage():
    """Test InMemoryDurableStorage."""
    storage = InMemoryDurableStorage()

    state = ExecutionState(
        {
            "execution_id": "test-1",
            "status": "running",
            "step_index": 0,
            "step_name": "test_step",
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )

    async def run_test():
        # Test save
        await storage.save_state_async("test-1", state)

        # Test load
        loaded = await storage.load_state_async("test-1")
        assert loaded is not None
        assert loaded["execution_id"] == "test-1"

        # Test list
        executions = await storage.list_executions_async()
        assert len(executions) == 1

        # Test delete
        deleted = await storage.delete_state_async("test-1")
        assert deleted is True

        # Verify deleted
        loaded_after = await storage.load_state_async("test-1")
        assert loaded_after is None

    asyncio.run(run_test())


def test_file_durable_storage():
    """Test FileDurableStorage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = FileDurableStorage(path=tmpdir)

        state = ExecutionState(
            {
                "execution_id": "test-2",
                "status": "running",
                "step_index": 0,
                "step_name": "test_step",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        )

        async def run_test():
            # Test save
            await storage.save_state_async("test-2", state)

            # Verify file exists
            file_path = Path(tmpdir) / "test-2.json"
            assert file_path.exists()

            # Test load
            loaded = await storage.load_state_async("test-2")
            assert loaded is not None
            assert loaded["execution_id"] == "test-2"

            # Test list
            executions = await storage.list_executions_async()
            assert len(executions) == 1

            # Test delete
            deleted = await storage.delete_state_async("test-2")
            assert deleted is True
            assert not file_path.exists()

        asyncio.run(run_test())


def test_sqlite_durable_storage():
    """Test SQLiteDurableStorage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = SQLiteDurableStorage(db_path=db_path)

        state = ExecutionState(
            {
                "execution_id": "test-3",
                "status": "running",
                "step_index": 0,
                "step_name": "test_step",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        )

        async def run_test():
            # Test save
            await storage.save_state_async("test-3", state)

            # Test load
            loaded = await storage.load_state_async("test-3")
            assert loaded is not None
            assert loaded["execution_id"] == "test-3"

            # Test list
            executions = await storage.list_executions_async()
            assert len(executions) == 1

            # Test delete
            deleted = await storage.delete_state_async("test-3")
            assert deleted is True

            # Verify deleted
            loaded_after = await storage.load_state_async("test-3")
            assert loaded_after is None

        asyncio.run(run_test())


@pytest.mark.skip(reason="Requires Redis server running")
def test_redis_durable_storage():
    """Test RedisDurableStorage."""
    try:
        from upsonic.durable.storages.redis import RedisDurableStorage

        storage = RedisDurableStorage(host="localhost", port=6379, db=0)

        state = ExecutionState(
            {
                "execution_id": "test-4",
                "status": "running",
                "step_index": 0,
                "step_name": "test_step",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        )

        async def run_test():
            # Test save
            await storage.save_state_async("test-4", state)

            # Test load
            loaded = await storage.load_state_async("test-4")
            assert loaded is not None
            assert loaded["execution_id"] == "test-4"

            # Test list
            executions = await storage.list_executions_async()
            assert len(executions) >= 1

            # Test delete
            deleted = await storage.delete_state_async("test-4")
            assert deleted is True

        asyncio.run(run_test())
    except (ImportError, ConnectionError):
        pytest.skip("Redis not available")
