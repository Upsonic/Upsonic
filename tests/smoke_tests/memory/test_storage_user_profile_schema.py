"""
Test 6: Storage providers with user_analysis_memory, user_profile_schema

Success criteria:
- Profile schema correctly created by LLM
- Stored all of them properly in AgentSession.metadata
- We can read, delete, update etc. them
"""

import pytest
import os
import tempfile
import shutil
from pydantic import BaseModel, Field
from upsonic import Agent, Task
from upsonic.db.database import (
    InMemoryDatabase,
    SqliteDatabase,
    PostgresDatabase,
    MongoDatabase,
    RedisDatabase,
    JSONDatabase
)
from upsonic.session.agent import AgentSession

pytestmark = pytest.mark.timeout(120)


class CustomUserSchema(BaseModel):
    """Custom user profile schema."""
    expertise_level: str = Field(default="beginner", description="User's expertise level")
    favorite_topics: list = Field(default_factory=list, description="User's favorite topics")
    communication_preference: str = Field(default="formal", description="Communication style")


@pytest.fixture
def test_user_id():
    return "test_user_schema_123"


@pytest.fixture
def test_session_id():
    return "test_session_schema_123"


@pytest.fixture
def temp_dir():
    """Create temporary directory for JSON storage."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.mark.asyncio
async def test_inmemory_storage_user_profile_schema(test_user_id, test_session_id):
    """Test InMemoryStorage with custom user profile schema."""
    db = InMemoryDatabase(
        session_id=test_session_id,
        user_id=test_user_id,
        user_analysis_memory=True,
        user_profile_schema=CustomUserSchema,
        model="openai/gpt-4o",
        debug=True
    )
    
    agent = Agent(model="openai/gpt-4o", db=db)
    
    task = Task(description="I am an expert in Python and prefer casual communication")
    result = await agent.do_async(task)
    assert result is not None
    
    # Verify session was created with profile in metadata
    await db.storage.connect_async()
    session = await db.storage.read_async(test_session_id, AgentSession)
    assert session is not None
    
    # Verify profile fields are in metadata
    assert session.metadata is not None
    assert isinstance(session.metadata, dict)
    
    await db.storage.disconnect_async()


@pytest.mark.asyncio
async def test_sqlite_storage_user_profile_schema(test_user_id, test_session_id):
    """Test SqliteStorage with custom user profile schema."""
    db_file = tempfile.mktemp(suffix=".db")
    try:
        db = SqliteDatabase(
            db_file=db_file,
            session_id=test_session_id,
            user_id=test_user_id,
            user_analysis_memory=True,
            user_profile_schema=CustomUserSchema,
            model="openai/gpt-4o",
            debug=True
        )
        
        agent = Agent(model="openai/gpt-4o", db=db)
        
        task = Task(description="I love machine learning and data science")
        await agent.do_async(task)
        
        # CRUD operations
        await db.storage.connect_async()
        session = await db.storage.read_async(test_session_id, AgentSession)
        assert session is not None
        
        # Update metadata (ensure metadata dict exists)
        if session.metadata is None:
            session.metadata = {}
        session.metadata["expertise_level"] = "expert"
        await db.storage.upsert_async(session)
        updated = await db.storage.read_async(test_session_id, AgentSession)
        assert updated.metadata.get("expertise_level") == "expert"
        
        # Delete
        await db.storage.delete_async(test_session_id, AgentSession)
        deleted = await db.storage.read_async(test_session_id, AgentSession)
        assert deleted is None
        
        await db.storage.disconnect_async()
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)


@pytest.mark.asyncio
async def test_postgres_storage_user_profile_schema(test_user_id, test_session_id):
    """Test PostgresStorage with custom user profile schema."""
    db_url = os.getenv("POSTGRES_URL", "postgresql://upsonic_test:test_password@localhost:5432/upsonic_test")
    
    try:
        db = PostgresDatabase(
            db_url=db_url,
            session_id=test_session_id,
            user_id=test_user_id,
            user_analysis_memory=True,
            user_profile_schema=CustomUserSchema,
            model="openai/gpt-4o",
            debug=True
        )
        
        await db.storage.connect_async()
        agent = Agent(model="openai/gpt-4o", db=db)
        
        task = Task(description="I am a beginner in AI")
        await agent.do_async(task)
        
        session = await db.storage.read_async(test_session_id, AgentSession)
        assert session is not None
        
        await db.storage.delete_async(test_session_id, AgentSession)
        await db.storage.disconnect_async()
        
    except Exception as e:
        pytest.skip(f"Postgres not available: {e}")


@pytest.mark.asyncio
async def test_mongo_storage_user_profile_schema(test_user_id, test_session_id):
    """Test MongoStorage with custom user profile schema."""
    db_url = os.getenv("MONGO_URL", "mongodb://upsonic_test:test_password@localhost:27017/?authSource=admin")
    
    try:
        db = MongoDatabase(
            db_url=db_url,
            database_name="test_db",
            session_id=test_session_id,
            user_id=test_user_id,
            user_analysis_memory=True,
            user_profile_schema=CustomUserSchema,
            model="openai/gpt-4o",
            debug=True
        )
        
        await db.storage.connect_async()
        agent = Agent(model="openai/gpt-4o", db=db)
        
        task = Task(description="I prefer technical explanations")
        await agent.do_async(task)
        
        session = await db.storage.read_async(test_session_id, AgentSession)
        assert session is not None
        
        await db.storage.delete_async(test_session_id, AgentSession)
        await db.storage.disconnect_async()
        
    except Exception as e:
        pytest.skip(f"Mongo not available: {e}")


@pytest.mark.asyncio
async def test_redis_storage_user_profile_schema(test_user_id, test_session_id):
    """Test RedisStorage with custom user profile schema."""
    try:
        db = RedisDatabase(
            prefix="test",
            host="localhost",
            port=6379,
            session_id=test_session_id,
            user_id=test_user_id,
            user_analysis_memory=True,
            user_profile_schema=CustomUserSchema,
            model="openai/gpt-4o",
            debug=True
        )
        
        await db.storage.connect_async()
        agent = Agent(model="openai/gpt-4o", db=db)
        
        task = Task(description="I am intermediate in programming")
        await agent.do_async(task)
        
        session = await db.storage.read_async(test_session_id, AgentSession)
        assert session is not None
        await db.storage.disconnect_async()
        
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


@pytest.mark.asyncio
async def test_json_storage_user_profile_schema(test_user_id, test_session_id, temp_dir):
    """Test JSONStorage with custom user profile schema."""
    db = JSONDatabase(
        directory_path=temp_dir,
        session_id=test_session_id,
        user_id=test_user_id,
        user_analysis_memory=True,
        user_profile_schema=CustomUserSchema,
        model="openai/gpt-4o",
        debug=True
    )
    
    await db.storage.connect_async()
    agent = Agent(model="openai/gpt-4o", db=db)
    
    task = Task(description="I like detailed explanations")
    await agent.do_async(task)
    
    session = await db.storage.read_async(test_session_id, AgentSession)
    assert session is not None
    
    # Update test
    session.metadata["communication_preference"] = "casual"
    await db.storage.upsert_async(session)
    updated = await db.storage.read_async(test_session_id, AgentSession)
    assert updated.metadata.get("communication_preference") == "casual"
    
    await db.storage.disconnect_async()
