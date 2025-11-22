from upsonic.context.sources import ContextSource, TaskOutputSource


def test_context_sources_initialization():
    """Test context sources."""
    source = ContextSource(enabled=True, source_id="source-1")

    assert source.enabled is True
    assert source.source_id == "source-1"

    # Test with defaults
    default_source = ContextSource()
    assert default_source.enabled is True
    assert default_source.source_id is None


def test_context_sources_load():
    """Test source loading."""
    # Test TaskOutputSource
    task_source = TaskOutputSource(
        task_description_or_id="task-123",
        retrieval_mode="full",
        enabled=True,
        source_id="task-output-1",
    )

    assert task_source.task_description_or_id == "task-123"
    assert task_source.retrieval_mode == "full"
    assert task_source.enabled is True
    assert task_source.source_id == "task-output-1"

    # Test with default retrieval mode
    task_source_default = TaskOutputSource(task_description_or_id="task-456")

    assert task_source_default.retrieval_mode == "full"
    assert task_source_default.enabled is True
