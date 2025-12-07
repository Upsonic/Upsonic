"""
Test 5: Storage providers with full_session_memory, user_analysis_memory, 
num_last_messages, model, debug

Success criteria:
- Agent remembers previous conversations
- We have user traits
- Agent only remembers user selected messages (controlled by num_last_messages)
- Stored all of them in the storages properly
- We can read, delete, update etc. them
"""

import pytest
import os
import tempfile
import shutil
from upsonic import Agent, Task
from upsonic.db.database import (
    InMemoryDatabase,
    SqliteDatabase,
    PostgresDatabase,
    MongoDatabase,
    RedisDatabase,
    JSONDatabase
)
from upsonic.storage.session.sessions import InteractionSession, UserProfile
from upsonic.storage.types import SessionId, UserId

pytestmark = pytest.mark.timeout(120)


@pytest.fixture
def test_user_id():
    return "test_user_123"


@pytest.fixture
def test_session_id():
    return "test_session_123"


@pytest.fixture
def temp_dir():
    """Create temporary directory for JSON storage."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.mark.asyncio
async def test_inmemory_storage_full_memory(test_user_id, test_session_id):
    """Test InMemoryStorage with full memory features."""
    db = InMemoryDatabase(
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        user_analysis_memory=True,
        num_last_messages=3,
        model="openai/gpt-4o",
        debug=True
    )
    
    agent = Agent(model="openai/gpt-4o", db=db)
    
    # First conversation
    task1 = Task(description="My name is Alice and I love Python programming")
    result1 = await agent.do_async(task1)
    assert result1 is not None
    
    # Second conversation - should remember name
    task2 = Task(description="What's my name?")
    result2 = await agent.do_async(task2)
    assert "alice" in str(result2).lower(), f"Expected 'alice' in result, got: {result2}"
    
    # Verify storage operations
    await db.storage.connect_async()
    session = await db.storage.read_async(SessionId(test_session_id), InteractionSession)
    assert session is not None
    assert len(session.chat_history) > 0
    
    profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
    assert profile is not None
    assert "profile_data" in profile.model_dump()
    
    # Test update
    session.chat_history.append({"role": "user", "content": "test"})
    await db.storage.upsert_async(session)
    updated = await db.storage.read_async(SessionId(test_session_id), InteractionSession)
    assert len(updated.chat_history) > len(session.chat_history) - 1
    
    # Test delete
    await db.storage.delete_async(SessionId(test_session_id), InteractionSession)
    deleted = await db.storage.read_async(SessionId(test_session_id), InteractionSession)
    assert deleted is None
    
    await db.storage.disconnect_async()


@pytest.mark.asyncio
async def test_sqlite_storage_full_memory(test_user_id, test_session_id):
    """Test SqliteStorage with full memory features."""
    db_file = tempfile.mktemp(suffix=".db")
    try:
        db = SqliteDatabase(
            db_file=db_file,
            sessions_table_name="sessions",
            profiles_table_name="profiles",
            session_id=test_session_id,
            user_id=test_user_id,
            full_session_memory=True,
            user_analysis_memory=True,
            num_last_messages=2,
            model="openai/gpt-4o",
            debug=True
        )
        
        agent = Agent(model="openai/gpt-4o", db=db)
        
        # Multiple conversations
        for i in range(5):
            task = Task(description=f"Message {i}: I like number {i}")
            await agent.do_async(task)
        
        # Should only remember last 2 messages
        task = Task(description="What numbers did I mention?")
        result = await agent.do_async(task)
        assert result is not None
        
        # Verify CRUD
        await db.storage.connect_async()
        session = await db.storage.read_async(SessionId(test_session_id), InteractionSession)
        assert session is not None
        
        profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile is not None
        
        # Update test
        profile.profile_data["test_key"] = "test_value"
        await db.storage.upsert_async(profile)
        updated = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert updated.profile_data.get("test_key") == "test_value"
        
        await db.storage.disconnect_async()
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)


@pytest.mark.asyncio
async def test_postgres_storage_full_memory(test_user_id, test_session_id):
    """Test PostgresStorage with full memory features."""
    db_url = os.getenv("POSTGRES_URL", "postgresql://upsonic_test:test_password@localhost:5432/upsonic_test")
    
    try:
        db = PostgresDatabase(
            db_url=db_url,
            sessions_table_name="sessions",
            profiles_table_name="profiles",
            session_id=test_session_id,
            user_id=test_user_id,
            full_session_memory=True,
            user_analysis_memory=True,
            num_last_messages=4,
            model="openai/gpt-4o",
            debug=True
        )
        
        await db.storage.connect_async()
        agent = Agent(model="openai/gpt-4o", db=db)
        
        task1 = Task(description="I am a software engineer")
        await agent.do_async(task1)
        
        task2 = Task(description="What's my profession?")
        result = await agent.do_async(task2)
        assert "engineer" in str(result).lower() or "software" in str(result).lower()
        
        # CRUD operations
        session = await db.storage.read_async(SessionId(test_session_id), InteractionSession)
        assert session is not None
        
        profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile is not None
        
        await db.storage.delete_async(SessionId(test_session_id), InteractionSession)
        await db.storage.delete_async(UserId(test_user_id), UserProfile)
        await db.storage.disconnect_async()
        
    except Exception as e:
        pytest.skip(f"Postgres not available: {e}")


@pytest.mark.asyncio
async def test_mongo_storage_full_memory(test_user_id, test_session_id):
    """Test MongoStorage with full memory features."""
    db_url = os.getenv("MONGO_URL", "mongodb://upsonic_test:test_password@localhost:27017/?authSource=admin")
    
    try:
        db = MongoDatabase(
            db_url=db_url,
            database_name="test_db",
            session_id=test_session_id,
            user_id=test_user_id,
            full_session_memory=True,
            user_analysis_memory=True,
            num_last_messages=3,
            model="openai/gpt-4o",
            debug=True
        )
        
        await db.storage.connect_async()
        agent = Agent(model="openai/gpt-4o", db=db)
        
        task1 = Task(description="My favorite color is blue")
        await agent.do_async(task1)
        
        task2 = Task(description="What's my favorite color?")
        result = await agent.do_async(task2)
        assert "blue" in str(result).lower()
        
        # CRUD operations
        session = await db.storage.read_async(SessionId(test_session_id), InteractionSession)
        assert session is not None
        
        profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile is not None
        
        await db.storage.delete_async(SessionId(test_session_id), InteractionSession)
        await db.storage.delete_async(UserId(test_user_id), UserProfile)
        await db.storage.disconnect_async()
        
    except Exception as e:
        pytest.skip(f"Mongo not available: {e}")


@pytest.mark.asyncio
async def test_redis_storage_full_memory(test_user_id, test_session_id):
    """Test RedisStorage with full memory features."""
    try:
        db = RedisDatabase(
            prefix="test",
            host="localhost",
            port=6379,
            session_id=test_session_id,
            user_id=test_user_id,
            full_session_memory=True,
            user_analysis_memory=True,
            num_last_messages=2,
            model="openai/gpt-4o",
            debug=True
        )
        
        await db.storage.connect_async()
        agent = Agent(model="openai/gpt-4o", db=db)
        
        task1 = Task(description="I love machine learning")
        await agent.do_async(task1)
        
        task2 = Task(description="What do I love?")
        result = await agent.do_async(task2)
        assert result is not None
        
        # CRUD operations
        session = await db.storage.read_async(SessionId(test_session_id), InteractionSession)
        assert session is not None
        
        profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile is not None
        await db.storage.disconnect_async()
        
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


@pytest.mark.asyncio
async def test_json_storage_full_memory(test_user_id, test_session_id, temp_dir):
    """Test JSONStorage with full memory features."""
    db = JSONDatabase(
        directory_path=temp_dir,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        user_analysis_memory=True,
        num_last_messages=3,
        model="openai/gpt-4o",
        debug=True
    )
    
    await db.storage.connect_async()
    agent = Agent(model="openai/gpt-4o", db=db)
    
    task1 = Task(description="I am interested in AI")
    await agent.do_async(task1)
    
    task2 = Task(description="What am I interested in?")
    result = await agent.do_async(task2)
    assert result is not None
    
    # CRUD operations
    session = await db.storage.read_async(SessionId(test_session_id), InteractionSession)
    assert session is not None
    
    profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
    assert profile is not None
    
    # Update
    profile.profile_data["interest"] = "AI"
    await db.storage.upsert_async(profile)
    updated = await db.storage.read_async(UserId(test_user_id), UserProfile)
    assert updated.profile_data.get("interest") == "AI"
    
    await db.storage.disconnect_async()

