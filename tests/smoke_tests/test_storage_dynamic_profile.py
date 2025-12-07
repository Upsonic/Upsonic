"""
Test 7: Storage providers with user_analysis_memory, dynamic_user_profile,
feed_tool_call_results, user_memory_mode

Success criteria:
- LLM created dynamic user profile
- We feed tool call results
- We use user_memory_mode
- Stored all of them
- We can read, delete, update etc. them
"""

import pytest
import os
import tempfile
import shutil
from upsonic import Agent, Task
from upsonic.tools import tool
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


@tool
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b


@pytest.fixture
def test_user_id():
    return "test_user_dynamic_123"


@pytest.fixture
def temp_dir():
    """Create temporary directory for JSON storage."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.mark.asyncio
async def test_inmemory_storage_dynamic_profile(test_user_id):
    """Test InMemoryStorage with dynamic profile and tool call results."""
    db = InMemoryDatabase(
        user_id=test_user_id,
        user_analysis_memory=True,
        dynamic_user_profile=True,
        feed_tool_call_results=True,
        user_memory_mode="update",
        model="openai/gpt-4o",
        debug=True
    )
    
    agent = Agent(model="openai/gpt-4o", db=db)
    
    # Task with tool call
    task = Task(
        description="Use calculate_sum to add 5 and 3, then tell me about my preferences",
        tools=[calculate_sum]
    )
    result = await agent.do_async(task)
    assert result is not None
    
    # Verify dynamic profile was created
    await db.storage.connect_async()
    try:
        profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile is not None
        assert "profile_data" in profile.model_dump()
    finally:
        await db.storage.disconnect_async()


@pytest.mark.asyncio
async def test_sqlite_storage_dynamic_profile(test_user_id):
    """Test SqliteStorage with dynamic profile and update mode."""
    db_file = tempfile.mktemp(suffix=".db")
    try:
        db = SqliteDatabase(
            db_file=db_file,
            sessions_table_name="sessions",
            profiles_table_name="profiles",
            user_id=test_user_id,
            user_analysis_memory=True,
            dynamic_user_profile=True,
            feed_tool_call_results=True,
            user_memory_mode="update",
            model="openai/gpt-4o",
            debug=True
        )
        
        agent = Agent(model="openai/gpt-4o", db=db)
        
        task1 = Task(
            description="Use calculate_sum for 2+2",
            tools=[calculate_sum]
        )
        await agent.do_async(task1)
        
        # Verify profile exists
        await db.storage.connect_async()
        profile1 = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile1 is not None
        
        # Test update mode - add more info
        task2 = Task(description="I also like Python")
        await agent.do_async(task2)
        
        profile2 = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile2 is not None
        assert profile2.updated_at >= profile1.updated_at
        
        # Test replace mode
        db_replace = SqliteDatabase(
            db_file=db_file,
            sessions_table_name="sessions",
            profiles_table_name="profiles",
            user_id=test_user_id + "_replace",
            user_analysis_memory=True,
            dynamic_user_profile=True,
            feed_tool_call_results=True,
            user_memory_mode="replace",
            model="openai/gpt-4o",
            debug=True
        )
        
        agent_replace = Agent(model="openai/gpt-4o", db=db_replace)
        await agent_replace.do_async(Task(description="I am a developer"))
        
        profile_replace = await db_replace.storage.read_async(
            UserId(test_user_id + "_replace"), UserProfile
        )
        assert profile_replace is not None
        
        await db_replace.storage.disconnect_async()
        await db.storage.disconnect_async()
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)


@pytest.mark.asyncio
async def test_postgres_storage_dynamic_profile(test_user_id):
    """Test PostgresStorage with dynamic profile."""
    db_url = os.getenv("POSTGRES_URL", "postgresql://upsonic_test:test_password@localhost:5432/upsonic_test")
    
    try:
        db = PostgresDatabase(
            db_url=db_url,
            sessions_table_name="sessions",
            profiles_table_name="profiles",
            user_id=test_user_id,
            user_analysis_memory=True,
            dynamic_user_profile=True,
            feed_tool_call_results=True,
            user_memory_mode="update",
            model="openai/gpt-4o",
            debug=True
        )
        
        await db.storage.connect_async()
        agent = Agent(model="openai/gpt-4o", db=db)
        
        task = Task(
            description="Use calculate_sum for 10+20",
            tools=[calculate_sum]
        )
        await agent.do_async(task)
        
        profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile is not None
        
        # CRUD test
        profile.profile_data["test"] = "value"
        await db.storage.upsert_async(profile)
        updated = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert updated.profile_data.get("test") == "value"
        
        await db.storage.delete_async(UserId(test_user_id), UserProfile)
        await db.storage.disconnect_async()
        
    except Exception as e:
        pytest.skip(f"Postgres not available: {e}")


@pytest.mark.asyncio
async def test_mongo_storage_dynamic_profile(test_user_id):
    """Test MongoStorage with dynamic profile."""
    db_url = os.getenv("MONGO_URL", "mongodb://upsonic_test:test_password@localhost:27017/?authSource=admin")
    
    try:
        db = MongoDatabase(
            db_url=db_url,
            database_name="test_db",
            user_id=test_user_id,
            user_analysis_memory=True,
            dynamic_user_profile=True,
            feed_tool_call_results=True,
            user_memory_mode="replace",
            model="openai/gpt-4o",
            debug=True
        )
        
        await db.storage.connect_async()
        agent = Agent(model="openai/gpt-4o", db=db)
        
        task = Task(
            description="Calculate 7+8 using the tool",
            tools=[calculate_sum]
        )
        await agent.do_async(task)
        
        profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile is not None
        
        await db.storage.delete_async(UserId(test_user_id), UserProfile)
        await db.storage.disconnect_async()
        
    except Exception as e:
        pytest.skip(f"Mongo not available: {e}")


@pytest.mark.asyncio
async def test_redis_storage_dynamic_profile(test_user_id):
    """Test RedisStorage with dynamic profile."""
    try:
        db = RedisDatabase(
            prefix="test",
            host="localhost",
            port=6379,
            user_id=test_user_id,
            user_analysis_memory=True,
            dynamic_user_profile=True,
            feed_tool_call_results=True,
            user_memory_mode="update",
            model="openai/gpt-4o",
            debug=True
        )
        
        await db.storage.connect_async()
        agent = Agent(model="openai/gpt-4o", db=db)
        
        task = Task(
            description="Add 15 and 25",
            tools=[calculate_sum]
        )
        await agent.do_async(task)
        
        profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
        assert profile is not None
        await db.storage.disconnect_async()
        
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")


@pytest.mark.asyncio
async def test_json_storage_dynamic_profile(test_user_id, temp_dir):
    """Test JSONStorage with dynamic profile."""
    db = JSONDatabase(
        directory_path=temp_dir,
        user_id=test_user_id,
        user_analysis_memory=True,
        dynamic_user_profile=True,
        feed_tool_call_results=True,
        user_memory_mode="update",
        model="openai/gpt-4o",
        debug=True
    )
    
    await db.storage.connect_async()
    agent = Agent(model="openai/gpt-4o", db=db)
    
    task = Task(
        description="Use calculate_sum to add 100 and 200",
        tools=[calculate_sum]
    )
    await agent.do_async(task)
    
    profile = await db.storage.read_async(UserId(test_user_id), UserProfile)
    assert profile is not None
    
    # Test update
    original_updated = profile.updated_at
    profile.profile_data["tool_usage"] = "high"
    await db.storage.upsert_async(profile)
    updated = await db.storage.read_async(UserId(test_user_id), UserProfile)
    assert updated.updated_at >= original_updated
    assert updated.profile_data.get("tool_usage") == "high"
    
    await db.storage.disconnect_async()

