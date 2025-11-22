import asyncio
from upsonic.cache.cache_manager import CacheManager


def test_cache_manager_initialization():
    """Test CacheManager init."""
    manager = CacheManager(session_id="test-session-1")

    assert manager.session_id == "test-session-1"
    assert manager.get_cache_size() == 0
    assert manager._cache_hits == 0
    assert manager._cache_misses == 0


def test_cache_manager_get():
    """Test cache get."""
    manager = CacheManager(session_id="test-session-2")

    async def run_test():
        # Test get with no cache
        result = await manager.get_cached_response(
            input_text="test query",
            cache_method="llm_call",
            cache_threshold=0.8,
            duration_minutes=60,
        )

        assert result is None
        assert manager._cache_misses == 1

    asyncio.run(run_test())


def test_cache_manager_set():
    """Test cache set."""
    manager = CacheManager(session_id="test-session-3")

    async def run_test():
        # Store cache entry
        await manager.store_cache_entry(
            input_text="test query", output="test response", cache_method="llm_call"
        )

        assert manager.get_cache_size() == 1

        # Try to get cached response
        result = await manager.get_cached_response(
            input_text="test query",
            cache_method="llm_call",
            cache_threshold=0.8,
            duration_minutes=60,
        )

        assert result == "test response"
        assert manager._cache_hits == 1

    asyncio.run(run_test())


def test_cache_manager_delete():
    """Test cache delete."""
    manager = CacheManager(session_id="test-session-4")

    async def run_test():
        # Store entry
        await manager.store_cache_entry(
            input_text="test query", output="test response", cache_method="llm_call"
        )

        assert manager.get_cache_size() == 1

        # Clear cache
        manager.clear_cache()

        assert manager.get_cache_size() == 0
        assert manager._cache_hits == 0
        assert manager._cache_misses == 0

    asyncio.run(run_test())


def test_cache_manager_clear():
    """Test cache clear."""
    manager = CacheManager(session_id="test-session-5")

    async def run_test():
        # Store multiple entries
        await manager.store_cache_entry(
            input_text="query1", output="response1", cache_method="llm_call"
        )
        await manager.store_cache_entry(
            input_text="query2", output="response2", cache_method="llm_call"
        )

        assert manager.get_cache_size() == 2

        # Clear all
        manager.clear_cache()

        assert manager.get_cache_size() == 0

    asyncio.run(run_test())


def test_cache_manager_ttl():
    """Test TTL functionality."""
    manager = CacheManager(session_id="test-session-6")

    async def run_test():
        # Store entry
        await manager.store_cache_entry(
            input_text="test query", output="test response", cache_method="llm_call"
        )

        # Try to get with very short TTL (should expire)
        import time

        time.sleep(0.1)  # Small delay

        # With 0 duration, should expire immediately
        result = await manager.get_cached_response(
            input_text="test query",
            cache_method="llm_call",
            cache_threshold=0.8,
            duration_minutes=0,  # Expired immediately
        )

        # Should be None due to expiration
        assert result is None

    asyncio.run(run_test())
