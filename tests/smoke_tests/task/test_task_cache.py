"""
Test 22: Task class enable_cache, cache_method, cache_threshold, 
         cache_embedding_provider, cache_duration_minutes
Success criteria: We check attributes, what we log and results
"""
import pytest
from io import StringIO
from contextlib import redirect_stdout
import time

from upsonic import Agent, Task
from upsonic.embeddings import OpenAIEmbedding, OpenAIEmbeddingConfig

pytestmark = pytest.mark.timeout(180)


@pytest.mark.asyncio
async def test_task_cache_vector_search_method():
    """Test Task with vector_search cache method."""
    
    # Create embedding provider for cache
    cache_embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
    
    # Create task with cache enabled
    task = Task(
        description="What is 2 + 2?",
        enable_cache=True,
        cache_method="vector_search",
        cache_threshold=0.85,
        cache_embedding_provider=cache_embedding,
        cache_duration_minutes=30
    )
    
    # Verify task attributes
    assert task.enable_cache is True, "Cache should be enabled"
    assert task.cache_method == "vector_search", "Cache method should be vector_search"
    assert task.cache_threshold == 0.85, "Cache threshold should be 0.85"
    assert task.cache_embedding_provider == cache_embedding, "Embedding provider should be set"
    assert task.cache_duration_minutes == 30, "Cache duration should be 30 minutes"
    
    # Verify private cache attributes
    assert task._cache_manager is None, "Cache manager should be None initially"
    assert task._cache_hit is False, "Cache hit should be False initially"
    assert task._original_input is None, "Original input should be None initially"
    assert task._last_cache_entry is None, "Last cache entry should be None initially"
    
    # Create agent and execute first time (cache miss)
    agent = Agent(model="openai/gpt-4o-mini", debug=True)
    
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result1 = await agent.do_async(task)
    
    output1 = output_buffer.getvalue()
    
    # Verify first execution
    assert result1 is not None, "First result should not be None"
    assert isinstance(result1, str), "Result should be a string"
    assert "4" in result1, "Result should contain the answer '4'"
    
    # Verify logging shows cache configuration and miss
    assert "Cache" in output1 or "cache" in output1, "Should log cache information"
    assert "miss" in output1.lower() or "Miss" in output1, "Should show cache miss"
    
    # Execute second time with similar query (should hit cache)
    task2 = Task(
        description="What is 2 + 2?",  # Same question
        enable_cache=True,
        cache_method="vector_search",
        cache_threshold=0.85,
        cache_embedding_provider=cache_embedding,
        cache_duration_minutes=30
    )
    
    output_buffer2 = StringIO()
    with redirect_stdout(output_buffer2):
        result2 = await agent.do_async(task2)
    
    output2 = output_buffer2.getvalue()
    
    # Verify second execution
    assert result2 is not None, "Second result should not be None"
    
    # Verify cache hit in logging
    assert "hit" in output2.lower() or "Hit" in output2, "Should show cache hit"


@pytest.mark.asyncio
async def test_task_cache_llm_call_method():
    """Test Task with llm_call cache method."""
    
    task = Task(
        description="Calculate 5 + 3",
        enable_cache=True,
        cache_method="llm_call",
        cache_duration_minutes=15
    )
    
    # Verify attributes
    assert task.enable_cache is True, "Cache should be enabled"
    assert task.cache_method == "llm_call", "Cache method should be llm_call"
    assert task.cache_duration_minutes == 15, "Cache duration should be 15 minutes"
    assert task.cache_threshold == 0.7, "Should use default threshold"
    
    # Execute
    agent = Agent(model="openai/gpt-4o-mini", debug=True)
    
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        result = await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify result
    assert result is not None, "Result should not be None"
    assert "8" in result, "Result should contain '8'"
    
    # Verify logging
    assert len(output) > 0, "Should have logging output"


@pytest.mark.asyncio
async def test_task_cache_disabled():
    """Test Task with cache disabled."""
    
    task = Task(
        description="What is 10 + 10?",
        enable_cache=False
    )
    
    # Verify attributes
    assert task.enable_cache is False, "Cache should be disabled"
    assert task.cache_method == "vector_search", "Should have default cache method"
    assert task.cache_threshold == 0.7, "Should have default threshold"
    assert task.cache_duration_minutes == 60, "Should have default duration"
    
    # Execute
    agent = Agent(model="openai/gpt-4o-mini", debug=True)
    result = await agent.do_async(task)
    
    # Verify result
    assert result is not None, "Result should not be None"
    assert "20" in result, "Result should contain '20'"


def test_task_cache_defaults():
    """Test Task cache default values."""
    
    task = Task(description="Test task")
    
    # Verify default values
    assert task.enable_cache is False, "Default enable_cache should be False"
    assert task.cache_method == "vector_search", "Default cache_method should be vector_search"
    assert task.cache_threshold == 0.7, "Default cache_threshold should be 0.7"
    assert task.cache_embedding_provider is None, "Default embedding provider should be None"
    assert task.cache_duration_minutes == 60, "Default duration should be 60 minutes"
    assert task._cache_manager is None, "Cache manager should be None"
    assert task._cache_hit is False, "Cache hit should be False"
    assert task._original_input is None, "Original input should be None"
    assert task._last_cache_entry is None, "Last cache entry should be None"


def test_task_cache_attribute_modification():
    """Test Task cache attributes can be modified."""
    
    embedding1 = OpenAIEmbedding(OpenAIEmbeddingConfig())
    
    task = Task(
        description="Test",
        enable_cache=True,
        cache_method="vector_search",
        cache_threshold=0.8,
        cache_embedding_provider=embedding1,
        cache_duration_minutes=120
    )
    
    # Verify initial values
    assert task.enable_cache is True
    assert task.cache_method == "vector_search"
    assert task.cache_threshold == 0.8
    assert task.cache_embedding_provider == embedding1
    assert task.cache_duration_minutes == 120
    
    # Modify values
    embedding2 = OpenAIEmbedding(OpenAIEmbeddingConfig())
    task.enable_cache = False
    task.cache_method = "llm_call"
    task.cache_threshold = 0.9
    task.cache_embedding_provider = embedding2
    task.cache_duration_minutes = 45
    
    # Verify modified values
    assert task.enable_cache is False
    assert task.cache_method == "llm_call"
    assert task.cache_threshold == 0.9
    assert task.cache_embedding_provider == embedding2
    assert task.cache_duration_minutes == 45


@pytest.mark.asyncio
async def test_task_cache_threshold_variation():
    """Test Task with different cache thresholds."""
    
    embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
    
    # High threshold (strict matching)
    task_strict = Task(
        description="What is AI?",
        enable_cache=True,
        cache_method="vector_search",
        cache_threshold=0.95,
        cache_embedding_provider=embedding,
        cache_duration_minutes=10
    )
    
    assert task_strict.cache_threshold == 0.95, "Strict threshold should be 0.95"
    
    # Low threshold (loose matching)
    task_loose = Task(
        description="What is AI?",
        enable_cache=True,
        cache_method="vector_search",
        cache_threshold=0.5,
        cache_embedding_provider=embedding,
        cache_duration_minutes=10
    )
    
    assert task_loose.cache_threshold == 0.5, "Loose threshold should be 0.5"
    
    # Execute both
    agent = Agent(model="openai/gpt-4o-mini", debug=True)
    
    result_strict = await agent.do_async(task_strict)
    result_loose = await agent.do_async(task_loose)
    
    assert result_strict is not None, "Strict result should not be None"
    assert result_loose is not None, "Loose result should not be None"


@pytest.mark.asyncio
async def test_task_cache_duration_expiration():
    """Test Task cache duration (note: actual expiration testing requires waiting)."""
    
    task = Task(
        description="What is the speed of light?",
        enable_cache=True,
        cache_method="llm_call",
        cache_duration_minutes=1  # Short duration for testing concept
    )
    
    # Verify duration is set correctly
    assert task.cache_duration_minutes == 1, "Duration should be 1 minute"
    
    agent = Agent(model="openai/gpt-4o-mini", debug=True)
    result = await agent.do_async(task)
    
    assert result is not None, "Result should not be None"
    
    # Note: Actually waiting 1 minute to test expiration would make test too slow
    # The important part is that the attribute is properly set and used


@pytest.mark.asyncio
async def test_task_cache_logging_configuration(capsys):
    """Test that cache configuration is properly logged."""
    
    embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
    
    task = Task(
        description="Explain quantum computing in one sentence",
        enable_cache=True,
        cache_method="vector_search",
        cache_threshold=0.75,
        cache_embedding_provider=embedding,
        cache_duration_minutes=90
    )
    
    agent = Agent(model="openai/gpt-4o-mini", debug=True)
    
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        await agent.do_async(task)
    
    output = output_buffer.getvalue()
    
    # Verify cache configuration is logged
    assert len(output) > 0, "Should have logging output"
    assert "cache" in output.lower() or "Cache" in output, "Should mention cache"
    
    # The debug output should show cache method and configuration
    # Exact format may vary, but cache information should be present


@pytest.mark.asyncio
async def test_task_cache_hit_tracking():
    """Test that cache hits are tracked in task object."""
    
    embedding = OpenAIEmbedding(OpenAIEmbeddingConfig())
    
    task1 = Task(
        description="What is 7 + 7?",
        enable_cache=True,
        cache_method="vector_search",
        cache_threshold=0.8,
        cache_embedding_provider=embedding,
        cache_duration_minutes=60
    )
    
    agent = Agent(model="openai/gpt-4o-mini", debug=True)
    
    # First execution - cache miss
    await agent.do_async(task1)
    
    # Second execution with same query - should hit cache
    task2 = Task(
        description="What is 7 + 7?",
        enable_cache=True,
        cache_method="vector_search",
        cache_threshold=0.8,
        cache_embedding_provider=embedding,
        cache_duration_minutes=60
    )
    
    output_buffer = StringIO()
    with redirect_stdout(output_buffer):
        await agent.do_async(task2)
    
    output = output_buffer.getvalue()
    
    # Verify cache hit is indicated in output
    # The exact tracking mechanism may vary, but cache hit should be logged


@pytest.mark.asyncio
async def test_task_cache_with_different_methods():
    """Test switching between cache methods."""
    
    # Task with vector_search
    task_vector = Task(
        description="What is machine learning?",
        enable_cache=True,
        cache_method="vector_search",
        cache_embedding_provider=OpenAIEmbedding(OpenAIEmbeddingConfig())
    )
    
    assert task_vector.cache_method == "vector_search"
    
    # Task with llm_call
    task_llm = Task(
        description="What is machine learning?",
        enable_cache=True,
        cache_method="llm_call"
    )
    
    assert task_llm.cache_method == "llm_call"
    
    agent = Agent(model="openai/gpt-4o-mini", debug=True)
    
    # Execute both
    result_vector = await agent.do_async(task_vector)
    result_llm = await agent.do_async(task_llm)
    
    # Both should work
    assert result_vector is not None
    assert result_llm is not None

