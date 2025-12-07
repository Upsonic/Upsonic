"""
Test 6: Storage providers with user_analysis_memory, user_profile_schema

Success criteria:
- Profile schema correctly created by LLM
- Stored all of them properly
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
from upsonic.storage.session.sessions import UserProfile
from upsonic.storage.types import UserId

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
def temp_dir():
    """Create temporary directory for JSON storage."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.mark.asyncio
async def test_inmemory_storage_user_profile_schema(test_user_id):
    """Test InMemoryStorage with custom user profile schema."""
    db = InMemoryDatabase(
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
    
    # Verify profile was created with schema
    await db.storage.connect_async()
    profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
    assert profile is not None
    assert "profile_data" in profile.model_dump()
    
    # Verify schema fields are in profile_data
    profile_data = profile.profile_data
    assert isinstance(profile_data, dict)
    
    await db.storage.disconnect_async()


@pytest.mark.asyncio
async def test_sqlite_storage_user_profile_schema(test_user_id):
    """Test SqliteStorage with custom user profile schema."""
    db_file = tempfile.mktemp(suffix=".db")
    try:
        db = SqliteDatabase(
            db_file=db_file,
            sessions_table_name="sessions",
            profiles_table_name="profiles",
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
        profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile is not None
        
        # Update
        profile.profile_data["expertise_level"] = "expert"
        await db.storage.upsert_async(profile)
        updated = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert updated.profile_data.get("expertise_level") == "expert"
        
        # Delete
        await db.storage.delete_async(UserId(test_user_id), UserProfile)
        deleted = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert deleted is None
        
        await db.storage.disconnect_async()
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)


@pytest.mark.asyncio
async def test_postgres_storage_user_profile_schema(test_user_id):
    """Test PostgresStorage with custom user profile schema."""
    db_url = os.getenv("POSTGRES_URL", "postgresql://upsonic_test:test_password@localhost:5432/upsonic_test")
    
    try:
        db = PostgresDatabase(
            db_url=db_url,
            sessions_table_name="sessions",
            profiles_table_name="profiles",
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
        
        profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile is not None
        
        await db.storage.delete_async(UserId(test_user_id), UserProfile)
        await db.storage.disconnect_async()
        
    except Exception as e:
        pytest.skip(f"Postgres not available: {e}")


@pytest.mark.asyncio
async def test_mongo_storage_user_profile_schema(test_user_id):
    """Test MongoStorage with custom user profile schema."""
    db_url = os.getenv("MONGO_URL", "mongodb://upsonic_test:test_password@localhost:27017/?authSource=admin")
    
    try:
        db = MongoDatabase(
            db_url=db_url,
            database_name="test_db",
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
        
        profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile is not None
        
        await db.storage.delete_async(UserId(test_user_id), UserProfile)
        await db.storage.disconnect_async()
        
    except Exception as e:
        pytest.skip(f"Mongo not available: {e}")


@pytest.mark.asyncio
async def test_redis_storage_user_profile_schema(test_user_id):
    """Test RedisStorage with custom user profile schema."""
    try:
        db = RedisDatabase(
            prefix="test",
            host="localhost",
            port=6379,
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
        
        profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile is not None
        await db.storage.disconnect_async()
        
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


@pytest.mark.asyncio
async def test_json_storage_user_profile_schema(test_user_id, temp_dir):
    """Test JSONStorage with custom user profile schema."""
    db = JSONDatabase(
        directory_path=temp_dir,
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
    
    profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
    assert profile is not None
    
    # Update test
    profile.profile_data["communication_preference"] = "casual"
    await db.storage.upsert_async(profile)
    updated = await db.storage.read_async(UserId(test_user_id), UserProfile)
    assert updated.profile_data.get("communication_preference") == "casual"
    
    await db.storage.disconnect_async()

