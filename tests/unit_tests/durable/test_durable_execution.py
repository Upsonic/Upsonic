import asyncio
from unittest.mock import Mock
from upsonic.durable.execution import DurableExecution
from upsonic.durable.storages.memory import InMemoryDurableStorage
from upsonic.tasks.tasks import Task


def test_durable_execution_initialization():
    """Test DurableExecution init."""
    storage = InMemoryDurableStorage()
    durable = DurableExecution(
        storage=storage, execution_id="test-exec-1", auto_cleanup=True, debug=False
    )

    assert durable.storage == storage
    assert durable.execution_id == "test-exec-1"
    assert durable.auto_cleanup is True
    assert durable.debug is False


def test_durable_execution_execute():
    """Test execution."""
    storage = InMemoryDurableStorage()
    durable = DurableExecution(storage=storage)

    # Create real task and mock context
    real_task = Task(description="Test task description")
    mock_context = Mock()
    mock_context.messages = []  # Set messages to empty list to avoid serialization error

    # Test save checkpoint
    async def run_test():
        await durable.save_checkpoint_async(
            task=real_task,
            context=mock_context,
            step_index=0,
            step_name="test_step",
            status="running",
        )

        # Test load checkpoint
        loaded = await durable.load_checkpoint_async()
        assert loaded is not None
        assert loaded["step_index"] == 0
        assert loaded["step_name"] == "test_step"

    asyncio.run(run_test())


def test_durable_execution_resume():
    """Test resuming execution."""
    storage = InMemoryDurableStorage()
    durable = DurableExecution(storage=storage, execution_id="test-exec-2")

    # Create real task and mock context
    real_task = Task(description="Test task for resume")
    mock_context = Mock()
    mock_context.messages = []  # Set messages to empty list to avoid serialization error

    async def run_test():
        # Save initial checkpoint
        await durable.save_checkpoint_async(
            task=real_task,
            context=mock_context,
            step_index=2,
            step_name="step_2",
            status="running",
        )

        # Load by ID
        loaded_durable = await DurableExecution.load_by_id_async("test-exec-2", storage)
        assert loaded_durable is not None
        assert loaded_durable.execution_id == "test-exec-2"

        # Load checkpoint
        checkpoint = await loaded_durable.load_checkpoint_async()
        assert checkpoint["step_index"] == 2

    asyncio.run(run_test())


def test_durable_execution_serialization():
    """Test serialization."""
    storage = InMemoryDurableStorage()
    durable = DurableExecution(storage=storage)

    # Create real task and mock context
    real_task = Task(description="Test task for serialization")
    mock_context = Mock()
    mock_context.messages = []  # Set messages to empty list to avoid serialization error

    async def run_test():
        # Save checkpoint with serialized state
        await durable.save_checkpoint_async(
            task=real_task,
            context=mock_context,
            step_index=1,
            step_name="serialize_test",
            status="running",
            agent_state={"key": "value"},
        )

        # Load and verify serialization
        loaded = await durable.load_checkpoint_async()
        assert loaded is not None
        assert loaded["step_index"] == 1
        assert loaded["agent_state"] == {"key": "value"}

        # Test mark completed (with auto_cleanup, state will be deleted)
        await durable.mark_completed_async()

        # Test get execution info (will be None if auto_cleanup deleted it)
        info = await durable._get_execution_info_async()
        # Info may be None if auto_cleanup deleted the state, which is expected behavior
        # Just verify the method doesn't raise an error

    asyncio.run(run_test())
