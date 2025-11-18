from unittest.mock import Mock, patch
from upsonic.db.database import (
    SqliteDatabase,
    PostgresDatabase,
    MongoDatabase,
    RedisDatabase,
    InMemoryDatabase,
    JSONDatabase,
)


def test_database_initialization():
    """Test Database initialization."""
    # Test SqliteDatabase
    with (
        patch("upsonic.db.database.SqliteStorage") as mock_storage_class,
        patch("upsonic.db.database.Memory") as mock_memory_class,
    ):
        mock_storage = Mock()
        mock_memory = Mock()
        mock_storage_class.return_value = mock_storage
        mock_memory_class.return_value = mock_memory

        db = SqliteDatabase(
            sessions_table_name="sessions",
            profiles_table_name="profiles",
            db_file=":memory:",
            session_id="test-session",
        )

        assert db.storage == mock_storage
        assert db.memory == mock_memory


def test_database_operations():
    """Test database operations."""
    # Test InMemoryDatabase
    with (
        patch("upsonic.db.database.InMemoryStorage") as mock_storage_class,
        patch("upsonic.db.database.Memory") as mock_memory_class,
    ):
        mock_storage = Mock()
        mock_memory = Mock()
        mock_storage_class.return_value = mock_storage
        mock_memory_class.return_value = mock_memory

        db = InMemoryDatabase(
            max_sessions=100,
            max_profiles=50,
            session_id="test-session",
            user_id="test-user",
        )

        assert db.storage == mock_storage
        assert db.memory == mock_memory

        # Test string representation
        repr_str = repr(db)
        assert "InMemoryDatabase" in repr_str

    # Test JSONDatabase
    with (
        patch("upsonic.db.database.JSONStorage") as mock_storage_class,
        patch("upsonic.db.database.Memory") as mock_memory_class,
    ):
        mock_storage = Mock()
        mock_memory = Mock()
        mock_storage_class.return_value = mock_storage
        mock_memory_class.return_value = mock_memory

        db = JSONDatabase(
            directory_path="/tmp/test", pretty_print=True, session_id="test-session"
        )

        assert db.storage == mock_storage
        assert db.memory == mock_memory
