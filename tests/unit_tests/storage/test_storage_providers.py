import pytest
import json
import time
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
from upsonic.storage.providers.in_memory import InMemoryStorage
from upsonic.storage.providers.json import JSONStorage
from upsonic.storage.providers.sqlite import SqliteStorage
from upsonic.storage.providers.redis import RedisStorage
from upsonic.storage.providers.postgres import PostgresStorage
from upsonic.storage.providers.mongo import MongoStorage
from upsonic.storage.providers.mem0 import Mem0Storage
from upsonic.storage.session.sessions import InteractionSession, UserProfile
from upsonic.storage.types import SessionId, UserId


class TestInMemoryStorage:
    """Test suite for InMemoryStorage provider."""

    @pytest.fixture
    def storage(self):
        """Create InMemoryStorage instance."""
        return InMemoryStorage()

    @pytest.mark.asyncio
    async def test_in_memory_storage_connect(self, storage):
        """Test InMemoryStorage connection."""
        assert not await storage.is_connected_async()
        await storage.connect_async()
        assert await storage.is_connected_async()

    @pytest.mark.asyncio
    async def test_in_memory_storage_disconnect(self, storage):
        """Test InMemoryStorage disconnection."""
        await storage.connect_async()
        await storage.disconnect_async()
        assert not await storage.is_connected_async()

    @pytest.mark.asyncio
    async def test_in_memory_storage_create(self, storage):
        """Test InMemoryStorage create operation."""
        await storage.create_async()
        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_in_memory_storage_upsert_read_session(self, storage):
        """Test InMemoryStorage upsert and read for InteractionSession."""
        await storage.connect_async()

        session = InteractionSession(
            session_id=SessionId("test-session"),
            user_id=UserId("test-user"),
            chat_history=[{"role": "user", "content": "Hello"}],
        )

        await storage.upsert_async(session)
        result = await storage.read_async("test-session", InteractionSession)

        assert result is not None
        assert result.session_id == SessionId("test-session")
        assert result.user_id == UserId("test-user")

    @pytest.mark.asyncio
    async def test_in_memory_storage_upsert_read_profile(self, storage):
        """Test InMemoryStorage upsert and read for UserProfile."""
        await storage.connect_async()

        profile = UserProfile(
            user_id=UserId("test-user"), profile_data={"name": "John", "age": 30}
        )

        await storage.upsert_async(profile)
        result = await storage.read_async("test-user", UserProfile)

        assert result is not None
        assert result.user_id == UserId("test-user")
        assert result.profile_data["name"] == "John"

    @pytest.mark.asyncio
    async def test_in_memory_storage_delete(self, storage):
        """Test InMemoryStorage delete operation."""
        await storage.connect_async()

        session = InteractionSession(session_id=SessionId("test-session"))
        await storage.upsert_async(session)

        await storage.delete_async("test-session", InteractionSession)
        result = await storage.read_async("test-session", InteractionSession)

        assert result is None

    @pytest.mark.asyncio
    async def test_in_memory_storage_drop(self, storage):
        """Test InMemoryStorage drop operation."""
        await storage.connect_async()

        session = InteractionSession(session_id=SessionId("test-session"))
        profile = UserProfile(user_id=UserId("test-user"))

        await storage.upsert_async(session)
        await storage.upsert_async(profile)

        await storage.drop_async()

        assert await storage.read_async("test-session", InteractionSession) is None
        assert await storage.read_async("test-user", UserProfile) is None

    @pytest.mark.asyncio
    async def test_in_memory_storage_lru_cache(self):
        """Test InMemoryStorage LRU cache functionality."""
        storage = InMemoryStorage(max_sessions=2)
        await storage.connect_async()

        # Add 3 sessions, should only keep last 2
        for i in range(3):
            session = InteractionSession(session_id=SessionId(f"session-{i}"))
            await storage.upsert_async(session)

        # First session should be evicted
        assert await storage.read_async("session-0", InteractionSession) is None
        assert await storage.read_async("session-1", InteractionSession) is not None
        assert await storage.read_async("session-2", InteractionSession) is not None


class TestJSONStorage:
    """Test suite for JSONStorage provider."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create JSONStorage instance."""
        return JSONStorage(directory_path=temp_dir)

    @pytest.mark.asyncio
    async def test_json_storage_connect(self, storage):
        """Test JSONStorage connection."""
        assert await storage.is_connected_async()
        await storage.connect_async()
        assert await storage.is_connected_async()

    @pytest.mark.asyncio
    async def test_json_storage_disconnect(self, storage):
        """Test JSONStorage disconnection."""
        await storage.disconnect_async()
        assert not await storage.is_connected_async()

    @pytest.mark.asyncio
    async def test_json_storage_create(self, storage):
        """Test JSONStorage create operation."""
        await storage.create_async()
        # Should create directories

    @pytest.mark.asyncio
    async def test_json_storage_upsert_read_session(self, storage):
        """Test JSONStorage upsert and read for InteractionSession."""
        session = InteractionSession(
            session_id=SessionId("test-session"),
            user_id=UserId("test-user"),
            chat_history=[{"role": "user", "content": "Hello"}],
        )

        await storage.upsert_async(session)
        result = await storage.read_async("test-session", InteractionSession)

        assert result is not None
        assert result.session_id == SessionId("test-session")
        assert result.user_id == UserId("test-user")

    @pytest.mark.asyncio
    async def test_json_storage_upsert_read_profile(self, storage):
        """Test JSONStorage upsert and read for UserProfile."""
        profile = UserProfile(
            user_id=UserId("test-user"), profile_data={"name": "John", "age": 30}
        )

        await storage.upsert_async(profile)
        result = await storage.read_async("test-user", UserProfile)

        assert result is not None
        assert result.user_id == UserId("test-user")
        assert result.profile_data["name"] == "John"

    @pytest.mark.asyncio
    async def test_json_storage_delete(self, storage):
        """Test JSONStorage delete operation."""
        session = InteractionSession(session_id=SessionId("test-session"))
        await storage.upsert_async(session)

        await storage.delete_async("test-session", InteractionSession)
        result = await storage.read_async("test-session", InteractionSession)

        assert result is None

    @pytest.mark.asyncio
    async def test_json_storage_drop(self, storage):
        """Test JSONStorage drop operation."""
        session = InteractionSession(session_id=SessionId("test-session"))
        profile = UserProfile(user_id=UserId("test-user"))

        await storage.upsert_async(session)
        await storage.upsert_async(profile)

        await storage.drop_async()

        assert await storage.read_async("test-session", InteractionSession) is None
        assert await storage.read_async("test-user", UserProfile) is None

    @pytest.mark.asyncio
    async def test_json_storage_read_nonexistent(self, storage):
        """Test JSONStorage read for nonexistent object."""
        result = await storage.read_async("nonexistent", InteractionSession)
        assert result is None


class TestSqliteStorage:
    """Test suite for SqliteStorage provider."""

    @pytest.fixture
    def storage(self):
        """Create SqliteStorage instance with in-memory database."""
        return SqliteStorage(
            sessions_table_name="test_sessions",
            profiles_table_name="test_profiles",
            db_file=None,  # Use in-memory
        )

    @pytest.mark.asyncio
    async def test_sqlite_storage_connect(self, storage):
        """Test SqliteStorage connection."""
        assert not await storage.is_connected_async()
        await storage.connect_async()
        assert await storage.is_connected_async()

    @pytest.mark.asyncio
    async def test_sqlite_storage_disconnect(self, storage):
        """Test SqliteStorage disconnection."""
        await storage.connect_async()
        await storage.disconnect_async()
        assert not await storage.is_connected_async()

    @pytest.mark.asyncio
    async def test_sqlite_storage_create(self, storage):
        """Test SqliteStorage create operation."""
        await storage.connect_async()
        await storage.create_async()
        # Should create tables

    @pytest.mark.asyncio
    async def test_sqlite_storage_upsert_read_session(self, storage):
        """Test SqliteStorage upsert and read for InteractionSession."""
        await storage.connect_async()

        session = InteractionSession(
            session_id=SessionId("test-session"),
            user_id=UserId("test-user"),
            chat_history=[{"role": "user", "content": "Hello"}],
        )

        await storage.upsert_async(session)
        result = await storage.read_async("test-session", InteractionSession)

        assert result is not None
        assert result.session_id == SessionId("test-session")
        assert result.user_id == UserId("test-user")

    @pytest.mark.asyncio
    async def test_sqlite_storage_upsert_read_profile(self, storage):
        """Test SqliteStorage upsert and read for UserProfile."""
        await storage.connect_async()

        profile = UserProfile(
            user_id=UserId("test-user"), profile_data={"name": "John", "age": 30}
        )

        await storage.upsert_async(profile)
        result = await storage.read_async("test-user", UserProfile)

        assert result is not None
        assert result.user_id == UserId("test-user")
        assert result.profile_data["name"] == "John"

    @pytest.mark.asyncio
    async def test_sqlite_storage_delete(self, storage):
        """Test SqliteStorage delete operation."""
        await storage.connect_async()

        session = InteractionSession(session_id=SessionId("test-session"))
        await storage.upsert_async(session)

        await storage.delete_async("test-session", InteractionSession)
        result = await storage.read_async("test-session", InteractionSession)

        assert result is None

    @pytest.mark.asyncio
    async def test_sqlite_storage_drop(self, storage):
        """Test SqliteStorage drop operation."""
        await storage.connect_async()

        session = InteractionSession(session_id=SessionId("test-session"))
        profile = UserProfile(user_id=UserId("test-user"))

        await storage.upsert_async(session)
        await storage.upsert_async(profile)

        # Verify data exists before drop
        assert await storage.read_async("test-session", InteractionSession) is not None
        assert await storage.read_async("test-user", UserProfile) is not None

        await storage.drop_async()

        # After drop, tables are deleted, so reads will fail with OperationalError
        # or return None depending on implementation
        import sqlite3

        try:
            result_session = await storage.read_async(
                "test-session", InteractionSession
            )
            result_profile = await storage.read_async("test-user", UserProfile)
            # If no exception, both should be None
            assert result_session is None
            assert result_profile is None
        except sqlite3.OperationalError:
            # Expected: tables don't exist after drop
            pass


class TestRedisStorage:
    """Test suite for RedisStorage provider."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create mock Redis client."""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.get = AsyncMock(return_value=None)
        mock_client.set = AsyncMock(return_value=True)
        mock_client.delete = AsyncMock(return_value=1)
        mock_client.scan_iter = AsyncMock()
        mock_client.close = AsyncMock()
        return mock_client

    @pytest.fixture
    def storage(self, mock_redis_client):
        """Create RedisStorage instance with mocked client."""
        with patch(
            "upsonic.storage.providers.redis.Redis", return_value=mock_redis_client
        ):
            return RedisStorage(prefix="test")

    @pytest.mark.asyncio
    async def test_redis_storage_connect(self, storage, mock_redis_client):
        """Test RedisStorage connection."""
        assert not await storage.is_connected_async()
        await storage.connect_async()
        assert await storage.is_connected_async()
        mock_redis_client.ping.assert_called()

    @pytest.mark.asyncio
    async def test_redis_storage_disconnect(self, storage, mock_redis_client):
        """Test RedisStorage disconnection."""
        await storage.connect_async()
        await storage.disconnect_async()
        assert not await storage.is_connected_async()
        mock_redis_client.close.assert_called()

    @pytest.mark.asyncio
    async def test_redis_storage_create(self, storage):
        """Test RedisStorage create operation."""
        await storage.create_async()
        # Should connect

    @pytest.mark.asyncio
    async def test_redis_storage_upsert_read_session(self, storage, mock_redis_client):
        """Test RedisStorage upsert and read for InteractionSession."""
        session = InteractionSession(
            session_id=SessionId("test-session"),
            user_id=UserId("test-user"),
            chat_history=[{"role": "user", "content": "Hello"}],
        )

        # Mock successful read
        session_dict = session.model_dump(mode="json")
        mock_redis_client.get.return_value = json.dumps(session_dict)

        await storage.upsert_async(session)
        result = await storage.read_async("test-session", InteractionSession)

        assert result is not None
        assert result.session_id == SessionId("test-session")
        mock_redis_client.set.assert_called()
        mock_redis_client.get.assert_called()

    @pytest.mark.asyncio
    async def test_redis_storage_upsert_read_profile(self, storage, mock_redis_client):
        """Test RedisStorage upsert and read for UserProfile."""
        profile = UserProfile(
            user_id=UserId("test-user"), profile_data={"name": "John", "age": 30}
        )

        # Mock successful read
        profile_dict = profile.model_dump(mode="json")
        mock_redis_client.get.return_value = json.dumps(profile_dict)

        await storage.upsert_async(profile)
        result = await storage.read_async("test-user", UserProfile)

        assert result is not None
        assert result.user_id == UserId("test-user")
        mock_redis_client.set.assert_called()
        mock_redis_client.get.assert_called()

    @pytest.mark.asyncio
    async def test_redis_storage_delete(self, storage, mock_redis_client):
        """Test RedisStorage delete operation."""
        await storage.delete_async("test-session", InteractionSession)
        mock_redis_client.delete.assert_called()

    @pytest.mark.asyncio
    async def test_redis_storage_drop(self, storage, mock_redis_client):
        """Test RedisStorage drop operation."""

        # Mock scan_iter to return some keys
        async def mock_scan_iter(match):
            yield "test:session:session1"
            yield "test:profile:user1"

        mock_redis_client.scan_iter = mock_scan_iter

        await storage.drop_async()
        mock_redis_client.delete.assert_called()


class TestPostgresStorage:
    """Test suite for PostgresStorage provider."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        # Make acquire return the connection directly (not awaitable)
        mock_pool.acquire = Mock(return_value=mock_conn)
        mock_pool._closing = False
        return mock_pool, mock_conn

    @pytest.fixture
    def storage(self, mock_pool):
        """Create PostgresStorage instance with mocked pool."""
        mock_pool_obj, _ = mock_pool

        # Make create_pool return an awaitable that returns the mock pool
        async def create_pool_mock(*args, **kwargs):
            return mock_pool_obj

        with patch(
            "upsonic.storage.providers.postgres.asyncpg.create_pool",
            side_effect=create_pool_mock,
        ):
            storage = PostgresStorage(
                sessions_table_name="test_sessions",
                profiles_table_name="test_profiles",
                db_url="postgresql://user:pass@localhost/db",
            )
            return storage

    @pytest.mark.asyncio
    async def test_postgres_storage_connect(self, storage, mock_pool):
        """Test PostgresStorage connection."""
        mock_pool_obj, _ = mock_pool

        async def create_pool_mock(*args, **kwargs):
            return mock_pool_obj

        with patch(
            "upsonic.storage.providers.postgres.asyncpg.create_pool",
            side_effect=create_pool_mock,
        ):
            assert not await storage.is_connected_async()
            await storage.connect_async()
            assert await storage.is_connected_async()

    @pytest.mark.asyncio
    async def test_postgres_storage_disconnect(self, storage, mock_pool):
        """Test PostgresStorage disconnection."""
        mock_pool_obj, _ = mock_pool

        async def create_pool_mock(*args, **kwargs):
            return mock_pool_obj

        with patch(
            "upsonic.storage.providers.postgres.asyncpg.create_pool",
            side_effect=create_pool_mock,
        ):
            await storage.connect_async()
            await storage.disconnect_async()
            assert not await storage.is_connected_async()
            mock_pool_obj.close.assert_called()

    @pytest.mark.asyncio
    async def test_postgres_storage_create(self, storage, mock_pool):
        """Test PostgresStorage create operation."""
        mock_pool_obj, mock_conn = mock_pool

        async def create_pool_mock(*args, **kwargs):
            return mock_pool_obj

        with patch(
            "upsonic.storage.providers.postgres.asyncpg.create_pool",
            side_effect=create_pool_mock,
        ):
            await storage.connect_async()
            await storage.create_async()
            assert mock_conn.execute.call_count >= 2  # Schema and tables

    @pytest.mark.asyncio
    async def test_postgres_storage_upsert_read_session(self, storage, mock_pool):
        """Test PostgresStorage upsert and read for InteractionSession."""
        mock_pool_obj, mock_conn = mock_pool

        # Mock fetchrow to return session data
        mock_row = {
            "session_id": "test-session",
            "user_id": "test-user",
            "agent_id": None,
            "team_session_id": None,
            "chat_history": json.dumps([{"role": "user", "content": "Hello"}]),
            "summary": None,
            "session_data": json.dumps({}),
            "extra_data": json.dumps({}),
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        mock_conn.fetchrow.return_value = mock_row

        async def create_pool_mock(*args, **kwargs):
            return mock_pool_obj

        with patch(
            "upsonic.storage.providers.postgres.asyncpg.create_pool",
            side_effect=create_pool_mock,
        ):
            await storage.connect_async()

            session = InteractionSession(
                session_id=SessionId("test-session"),
                user_id=UserId("test-user"),
                chat_history=[{"role": "user", "content": "Hello"}],
            )

            await storage.upsert_async(session)
            result = await storage.read_async("test-session", InteractionSession)

            assert result is not None
            assert result.session_id == SessionId("test-session")
            mock_conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_postgres_storage_delete(self, storage, mock_pool):
        """Test PostgresStorage delete operation."""
        mock_pool_obj, mock_conn = mock_pool

        async def create_pool_mock(*args, **kwargs):
            return mock_pool_obj

        with patch(
            "upsonic.storage.providers.postgres.asyncpg.create_pool",
            side_effect=create_pool_mock,
        ):
            await storage.connect_async()

            await storage.delete_async("test-session", InteractionSession)
            mock_conn.execute.assert_called()


class TestStorageSerialization:
    """Test suite for data serialization across storage providers."""

    @pytest.mark.asyncio
    async def test_storage_serialization_complex_session(self):
        """Test serialization of complex InteractionSession."""
        storage = InMemoryStorage()
        await storage.connect_async()

        session = InteractionSession(
            session_id=SessionId("test-session"),
            user_id=UserId("test-user"),
            agent_id="agent-123",
            team_session_id="team-456",
            chat_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            summary="Test conversation",
            session_data={"key1": "value1", "key2": 123},
            extra_data={"metadata": {"source": "test"}},
        )

        await storage.upsert_async(session)
        result = await storage.read_async("test-session", InteractionSession)

        assert result is not None
        assert result.session_id == SessionId("test-session")
        assert result.user_id == UserId("test-user")
        assert result.agent_id == "agent-123"
        assert len(result.chat_history) == 2
        assert result.summary == "Test conversation"
        assert result.session_data["key1"] == "value1"
        assert result.extra_data["metadata"]["source"] == "test"

    @pytest.mark.asyncio
    async def test_storage_serialization_complex_profile(self):
        """Test serialization of complex UserProfile."""
        storage = InMemoryStorage()
        await storage.connect_async()

        profile = UserProfile(
            user_id=UserId("test-user"),
            profile_data={
                "name": "John Doe",
                "age": 30,
                "preferences": {"theme": "dark", "language": "en"},
                "tags": ["developer", "python"],
            },
        )

        await storage.upsert_async(profile)
        result = await storage.read_async("test-user", UserProfile)

        assert result is not None
        assert result.user_id == UserId("test-user")
        assert result.profile_data["name"] == "John Doe"
        assert result.profile_data["age"] == 30
        assert result.profile_data["preferences"]["theme"] == "dark"
        assert len(result.profile_data["tags"]) == 2

    @pytest.mark.asyncio
    async def test_storage_serialization_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        temp_dir = tempfile.mkdtemp()
        try:
            storage = JSONStorage(directory_path=temp_dir)

            session = InteractionSession(
                session_id=SessionId("test-session"),
                user_id=UserId("test-user"),
                chat_history=[{"role": "user", "content": "Hello"}],
            )

            await storage.upsert_async(session)
            result = await storage.read_async("test-session", InteractionSession)

            assert result is not None
            assert result.session_id == SessionId("test-session")
            assert result.user_id == UserId("test-user")
            assert len(result.chat_history) == 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_storage_serialization_unicode_data(self):
        """Test serialization of unicode data."""
        storage = InMemoryStorage()
        await storage.connect_async()

        session = InteractionSession(
            session_id=SessionId("test-session"),
            chat_history=[{"role": "user", "content": "Hello 世界 🌍"}],
            summary="Test with unicode: 测试",
        )

        await storage.upsert_async(session)
        result = await storage.read_async("test-session", InteractionSession)

        assert result is not None
