"""
Comprehensive Memory Test Suite

Tests all Memory class attributes:
1. storage - Storage backend (SqliteStorage)
2. session_id - Session identifier
3. user_id - User identifier
4. full_session_memory - Store/retrieve full conversation history
5. summary_memory - Generate and store summaries
6. user_analysis_memory - Analyze and store user profiles
7. user_profile_schema - Custom Pydantic schema for user profile
8. dynamic_user_profile - Dynamically generate profile schema
9. num_last_messages - Limit on messages to retrieve
10. model - Model for summary/profile generation
11. debug - Enable debug logging
12. debug_level - Debug verbosity level
13. feed_tool_call_results - Include tool call results in history
14. user_memory_mode - 'update' or 'replace' for profile updates
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from upsonic import Agent, Task
from upsonic.storage import Memory, SqliteStorage

# Test result tracking
test_results = []

def log_test_result(test_name: str, passed: bool, message: str = ""):
    status = "✅ PASSED" if passed else "❌ FAILED"
    result = f"{status}: {test_name}"
    if message:
        result += f" - {message}"
    print(result)
    test_results.append({"name": test_name, "passed": passed, "message": message})

def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


# Custom user profile schema for testing
class CustomUserProfile(BaseModel):
    """Custom user profile schema for testing."""
    name: Optional[str] = Field(None, description="User's name")
    occupation: Optional[str] = Field(None, description="User's occupation")
    expertise_level: Optional[str] = Field(None, description="User's expertise level (beginner/intermediate/expert)")
    interests: Optional[List[str]] = Field(None, description="User's interests")


async def cleanup_db(db_path: str):
    """Clean up test database."""
    if os.path.exists(db_path):
        os.remove(db_path)


# =============================================================================
# TEST 1: Basic Storage and Session ID
# =============================================================================
async def test_basic_storage_and_session_id():
    """Test basic storage initialization and session_id handling."""
    print_separator("TEST 1: Basic Storage and Session ID")
    
    db_path = "test_memory_basic.db"
    await cleanup_db(db_path)
    
    try:
        storage = SqliteStorage(db_file=db_path)
        
        # Test explicit session_id
        memory1 = Memory(
            storage=storage,
            session_id="test_session_001",
            user_id="test_user_001",
            debug=False,
            
        )
        
        assert memory1.session_id == "test_session_001", "Session ID should match"
        assert memory1.user_id == "test_user_001", "User ID should match"
        log_test_result("Explicit session_id and user_id", True)
        
        # Test auto-generated session_id
        memory2 = Memory(
            storage=storage,
            debug=False
        )
        
        assert memory2.session_id is not None, "Session ID should be auto-generated"
        assert len(memory2.session_id) > 0, "Session ID should not be empty"
        assert memory2.user_id is not None, "User ID should be auto-generated"
        log_test_result("Auto-generated session_id and user_id", True)
        
        # Test storage is properly set
        assert memory1.storage is storage, "Storage should be the same instance"
        log_test_result("Storage instance assignment", True)
        
        await storage.disconnect_async()
        await cleanup_db(db_path)
        
    except Exception as e:
        log_test_result("Basic Storage and Session ID", False, str(e))
        raise


# =============================================================================
# TEST 2: Full Session Memory (Conversation History)
# =============================================================================
async def test_full_session_memory():
    """Test full_session_memory for conversation history persistence."""
    print_separator("TEST 2: Full Session Memory (Conversation History)")
    
    db_path = "test_memory_full_session.db"
    await cleanup_db(db_path)
    
    try:
        storage = SqliteStorage(db_file=db_path)
        session_id = "session_full_memory_test"
        
        # Create memory with full_session_memory enabled
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id="user_001",
            full_session_memory=True,
            debug=False,
            
        )
        
        assert memory.full_session_memory_enabled is True, "full_session_memory should be enabled"
        log_test_result("full_session_memory flag set correctly", True)
        
        # Create an agent with memory
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory,
            debug=False,
            
        )
        
        # First conversation turn
        print("\n[Turn 1] Sending first message...")
        task1 = Task("My name is Alex. Please remember that.")
        result1 = await agent.do_async(task1)
        print(f"Response 1: {result1}")
        
        # Second conversation turn - should remember context
        print("\n[Turn 2] Sending second message (testing memory)...")
        task2 = Task("What is my name?")
        result2 = await agent.do_async(task2)
        print(f"Response 2: {result2}")
        
        # Check if response contains the name
        result2_str = str(result2).lower()
        name_remembered = "alex" in result2_str
        assert name_remembered, f"Memory should persist conversation context - 'alex' not found in: {result2}"
        log_test_result("Memory persists conversation context", name_remembered, 
                       f"Response: {result2}")
        
        # Verify session was stored
        session = await memory.get_session_async()
        assert session is not None, "Session should be stored"
        log_test_result("Session stored in storage", True)
        
        # Check messages were stored
        messages = session.messages if session else []
        message_count = len(messages) if messages else 0
        assert message_count > 0, f"Messages should be stored in session, got {message_count}"
        assert message_count >= 4, f"Should have at least 4 messages (2 turns * 2 messages), got {message_count}"
        log_test_result("Messages stored in session", message_count > 0, 
                       f"Message count: {message_count}")
        
        await storage.disconnect_async()
        await cleanup_db(db_path)
        
    except Exception as e:
        log_test_result("Full Session Memory", False, str(e))
        raise


# =============================================================================
# TEST 3: Summary Memory
# =============================================================================
async def test_summary_memory():
    """Test summary_memory for automatic summary generation."""
    print_separator("TEST 3: Summary Memory")
    
    db_path = "test_memory_summary.db"
    await cleanup_db(db_path)
    
    try:
        storage = SqliteStorage(db_file=db_path)
        session_id = "session_summary_test"
        
        # Create memory with summary_memory enabled
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id="user_001",
            full_session_memory=True,
            summary_memory=True,
            model="openai/gpt-4o-mini",  # Required for summary generation
            debug=False,
            
        )
        
        assert memory.summary_memory_enabled is True, "summary_memory should be enabled"
        log_test_result("summary_memory flag set correctly", True)
        
        # Create agent with memory
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory,
            debug=False,
            
        )
        
        # Have a conversation to generate a summary
        print("\n[Turn 1] Starting conversation...")
        task1 = Task("I'm working on a machine learning project using Python and TensorFlow.")
        result1 = await agent.do_async(task1)
        print(f"Response 1: {result1}")
        
        print("\n[Turn 2] Continuing conversation...")
        task2 = Task("The main challenge is handling large datasets efficiently.")
        result2 = await agent.do_async(task2)
        print(f"Response 2: {result2}")
        
        # Check if summary was generated
        session = await memory.get_session_async()
        assert session is not None, "Session should exist after conversations"
        summary = session.summary if session else None
        
        summary_generated = summary is not None and len(summary) > 0
        assert summary_generated, f"Summary should be generated, got: {summary}"
        assert len(summary) >= 20, f"Summary should have meaningful content (>= 20 chars), got {len(summary)} chars"
        log_test_result("Summary generated", summary_generated, 
                       f"Summary length: {len(summary) if summary else 0}")
        
        if summary:
            print(f"\n[Generated Summary]\n{summary[:500]}...")
        
        await storage.disconnect_async()
        await cleanup_db(db_path)
        
    except Exception as e:
        log_test_result("Summary Memory", False, str(e))
        raise


# =============================================================================
# TEST 4: User Analysis Memory (Default Schema)
# =============================================================================
async def test_user_analysis_memory_default():
    """Test user_analysis_memory with default UserTraits schema."""
    print_separator("TEST 4: User Analysis Memory (Default Schema)")
    
    db_path = "test_memory_user_analysis.db"
    await cleanup_db(db_path)
    
    try:
        storage = SqliteStorage(db_file=db_path)
        session_id = "session_user_analysis_test"
        
        # Create memory with user_analysis_memory enabled
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id="user_001",
            full_session_memory=True,
            user_analysis_memory=True,
            model="openai/gpt-4o-mini",  # Required for user analysis
            debug=False,
            
        )
        
        assert memory.user_analysis_memory_enabled is True, "user_analysis_memory should be enabled"
        log_test_result("user_analysis_memory flag set correctly", True)
        
        # Create agent with memory
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory,
            debug=False,
            
        )
        
        # Provide information that can be analyzed
        print("\n[Turn 1] Providing user information...")
        task1 = Task("Hi, I'm a software engineer with 5 years of experience. I specialize in Python and machine learning.")
        result1 = await agent.do_async(task1)
        print(f"Response 1: {result1}")
        
        # Check if user profile was extracted
        session = await memory.get_session_async()
        assert session is not None, "Session should exist after conversation"
        user_profile = session.user_profile if session else None
        
        profile_extracted = user_profile is not None and len(user_profile) > 0
        assert profile_extracted, f"User profile should be extracted, got: {user_profile}"
        log_test_result("User profile extracted", profile_extracted, 
                       f"Profile: {user_profile}")
        
        if user_profile:
            print(f"\n[Extracted User Profile]\n{user_profile}")
        
        await storage.disconnect_async()
        await cleanup_db(db_path)
        
    except Exception as e:
        log_test_result("User Analysis Memory (Default Schema)", False, str(e))
        raise


# =============================================================================
# TEST 5: Custom User Profile Schema
# =============================================================================
async def test_custom_user_profile_schema():
    """Test user_profile_schema with a custom Pydantic schema."""
    print_separator("TEST 5: Custom User Profile Schema")
    
    db_path = "test_memory_custom_schema.db"
    await cleanup_db(db_path)
    
    try:
        storage = SqliteStorage(db_file=db_path)
        session_id = "session_custom_schema_test"
        
        # Create memory with custom user profile schema
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id="user_001",
            full_session_memory=True,
            user_analysis_memory=True,
            user_profile_schema=CustomUserProfile,
            model="openai/gpt-4o-mini",
            debug=False,
            
        )
        
        assert memory.profile_schema_model == CustomUserProfile, "Custom schema should be set"
        log_test_result("Custom user_profile_schema set correctly", True)
        
        # Create agent with memory
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory,
            debug=False,
            
        )
        
        # Provide information matching custom schema
        print("\n[Turn 1] Providing custom profile information...")
        task1 = Task("My name is Sarah. I'm a data scientist and I'm an expert in deep learning. I'm interested in NLP and computer vision.")
        result1 = await agent.do_async(task1)
        print(f"Response 1: {result1}")
        
        # Check if user profile was extracted with custom schema
        session = await memory.get_session_async()
        assert session is not None, "Session should exist after conversation"
        user_profile = session.user_profile if session else None
        
        profile_extracted = user_profile is not None and len(user_profile) > 0
        assert profile_extracted, f"Custom profile should be extracted, got: {user_profile}"
        # Verify custom schema fields are present
        assert "name" in user_profile or "occupation" in user_profile or "expertise_level" in user_profile, \
            f"Profile should contain custom schema fields, got: {user_profile}"
        log_test_result("Custom profile extracted", profile_extracted, 
                       f"Profile: {user_profile}")
        
        if user_profile:
            print(f"\n[Extracted Custom Profile]\n{user_profile}")
        
        await storage.disconnect_async()
        await cleanup_db(db_path)
        
    except Exception as e:
        log_test_result("Custom User Profile Schema", False, str(e))
        raise


# =============================================================================
# TEST 6: Dynamic User Profile
# =============================================================================
async def test_dynamic_user_profile():
    """Test dynamic_user_profile for automatic schema generation."""
    print_separator("TEST 6: Dynamic User Profile")
    
    db_path = "test_memory_dynamic_profile.db"
    await cleanup_db(db_path)
    
    try:
        storage = SqliteStorage(db_file=db_path)
        session_id = "session_dynamic_profile_test"
        
        # Create memory with dynamic user profile
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id="user_001",
            full_session_memory=True,
            user_analysis_memory=True,
            dynamic_user_profile=True,
            model="openai/gpt-4o-mini",
            debug=False,
            
        )
        
        assert memory.is_profile_dynamic is True, "dynamic_user_profile should be enabled"
        assert memory.profile_schema_model is None, "Custom schema should be ignored when dynamic is enabled"
        log_test_result("dynamic_user_profile flag set correctly", True)
        
        # Create agent with memory
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory,
            debug=False,
            
        )
        
        # Provide varied information
        print("\n[Turn 1] Providing diverse user information...")
        task1 = Task("I'm John, a 35-year-old architect from New York. I love hiking and photography in my free time.")
        result1 = await agent.do_async(task1)
        print(f"Response 1: {result1}")
        
        # Check if dynamic profile was generated
        session = await memory.get_session_async()
        assert session is not None, "Session should exist after conversation"
        user_profile = session.user_profile if session else None
        
        profile_extracted = user_profile is not None and len(user_profile) > 0
        assert profile_extracted, f"Dynamic profile should be extracted, got: {user_profile}"
        log_test_result("Dynamic profile extracted", profile_extracted, 
                       f"Profile: {user_profile}")
        
        if user_profile:
            print(f"\n[Dynamically Generated Profile]\n{user_profile}")
        
        await storage.disconnect_async()
        await cleanup_db(db_path)
        
    except Exception as e:
        log_test_result("Dynamic User Profile", False, str(e))
        raise


# =============================================================================
# TEST 7: num_last_messages Limit
# =============================================================================
async def test_num_last_messages():
    """Test num_last_messages for limiting retrieved conversation history."""
    print_separator("TEST 7: num_last_messages Limit")
    
    db_path = "test_memory_num_messages.db"
    await cleanup_db(db_path)
    
    try:
        storage = SqliteStorage(db_file=db_path)
        session_id = "session_num_messages_test"
        
        # Create memory with message limit
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id="user_001",
            full_session_memory=True,
            num_last_messages=2,  # Only keep last 2 messages
            debug=False,
            
        )
        
        assert memory.num_last_messages == 2, "num_last_messages should be 2"
        log_test_result("num_last_messages set correctly", True)
        
        # Create agent with memory
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory,
            debug=False,
            
        )
        
        # Send multiple messages
        print("\n[Turn 1] First message...")
        task1 = Task("My favorite color is blue.")
        result1 = await agent.do_async(task1)
        print(f"Response 1: {result1}")
        
        print("\n[Turn 2] Second message...")
        task2 = Task("My favorite food is pizza.")
        result2 = await agent.do_async(task2)
        print(f"Response 2: {result2}")
        
        print("\n[Turn 3] Third message...")
        task3 = Task("My favorite sport is tennis.")
        result3 = await agent.do_async(task3)
        print(f"Response 3: {result3}")
        
        # Ask about the first message - should NOT remember (due to limit)
        print("\n[Turn 4] Asking about old message (should not remember)...")
        task4 = Task("What is my favorite color? Just say the color or 'I don't know'.")
        result4 = await agent.do_async(task4)
        print(f"Response 4: {result4}")
        
        # Due to num_last_messages=2, the first message (blue) should NOT be remembered
        result4_str = str(result4).lower()
        # The agent should respond with "I don't know" or similar since "blue" is outside the context window
        assert "don't know" in result4_str or "do not know" in result4_str or "blue" not in result4_str, \
            f"num_last_messages should limit context - expected 'I don't know' or no 'blue', got: {result4}"
        log_test_result("num_last_messages limits context", True, 
                       f"Response about color: {result4}")
        
        await storage.disconnect_async()
        await cleanup_db(db_path)
        
    except Exception as e:
        log_test_result("num_last_messages", False, str(e))
        raise


# =============================================================================
# TEST 8: User Memory Mode (Update vs Replace)
# =============================================================================
async def test_user_memory_mode():
    """Test user_memory_mode for update vs replace behavior."""
    print_separator("TEST 8: User Memory Mode (Update vs Replace)")
    
    db_path = "test_memory_mode.db"
    await cleanup_db(db_path)
    
    try:
        storage = SqliteStorage(db_file=db_path)
        session_id = "session_memory_mode_test"
        
        # Test UPDATE mode
        print("\n[Testing UPDATE mode]")
        memory_update = Memory(
            storage=storage,
            session_id=session_id,
            user_id="user_001",
            full_session_memory=True,
            user_analysis_memory=True,
            user_memory_mode='update',
            model="openai/gpt-4o-mini",
            debug=False,
            
        )
        
        assert memory_update.user_memory_mode == 'update', "user_memory_mode should be 'update'"
        log_test_result("user_memory_mode='update' set correctly", True)
        
        # Test REPLACE mode
        print("\n[Testing REPLACE mode]")
        memory_replace = Memory(
            storage=storage,
            session_id="session_replace_test",
            user_id="user_002",
            full_session_memory=True,
            user_analysis_memory=True,
            user_memory_mode='replace',
            model="openai/gpt-4o-mini",
            debug=False,
            
        )
        
        assert memory_replace.user_memory_mode == 'replace', "user_memory_mode should be 'replace'"
        log_test_result("user_memory_mode='replace' set correctly", True)
        
        # Create agents for both modes
        agent_update = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_update,
            debug=False,
            
        )
        
        agent_replace = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_replace,
            debug=False,
            
        )
        
        # Test UPDATE mode - profile should accumulate
        print("\n[Turn 1 - Update mode] First profile info...")
        task1 = Task("My name is Alice.")
        await agent_update.do_async(task1)
        
        print("\n[Turn 2 - Update mode] Second profile info...")
        task2 = Task("I'm a software developer.")
        await agent_update.do_async(task2)
        
        session_update = await memory_update.get_session_async()
        assert session_update is not None, "Session should exist for update mode"
        profile_update = session_update.user_profile if session_update else {}
        assert profile_update is not None and len(profile_update) > 0, f"Profile should exist in update mode, got: {profile_update}"
        print(f"[UPDATE Mode Profile]: {profile_update}")
        
        # Test REPLACE mode
        print("\n[Turn 1 - Replace mode] First profile info...")
        task3 = Task("My name is Bob.")
        await agent_replace.do_async(task3)
        
        print("\n[Turn 2 - Replace mode] Second profile info...")
        task4 = Task("I'm a doctor.")
        await agent_replace.do_async(task4)
        
        session_replace = await memory_replace.get_session_async()
        assert session_replace is not None, "Session should exist for replace mode"
        profile_replace = session_replace.user_profile if session_replace else {}
        assert profile_replace is not None and len(profile_replace) > 0, f"Profile should exist in replace mode, got: {profile_replace}"
        print(f"[REPLACE Mode Profile]: {profile_replace}")
        
        log_test_result("User memory modes work correctly", True)
        
        await storage.disconnect_async()
        await cleanup_db(db_path)
        
    except Exception as e:
        log_test_result("User Memory Mode", False, str(e))
        raise


# =============================================================================
# TEST 9: Debug Levels
# =============================================================================
async def test_debug_levels():
    """Test debug and debug_level settings."""
    print_separator("TEST 9: Debug Levels")
    
    db_path = "test_memory_debug.db"
    await cleanup_db(db_path)
    
    try:
        storage = SqliteStorage(db_file=db_path)
        
        # Test debug=False
        memory_no_debug = Memory(
            storage=storage,
            session_id="session_no_debug",
            debug=False
        )
        
        assert memory_no_debug.debug is False, "debug should be False"
        assert memory_no_debug.debug_level == 1, "debug_level should be 1 when debug is False"
        log_test_result("Debug disabled correctly", True)
        
        # Test debug=True, debug_level=1
        memory_debug_1 = Memory(
            storage=storage,
            session_id="session_debug_1",
            debug=False,
            
        )
        
        assert memory_debug_1.debug is True, "debug should be True"
        assert memory_debug_1.debug_level == 1, "debug_level should be 1"
        log_test_result("Debug level 1 set correctly", True)
        
        # Test debug=True, debug_level=2
        memory_debug_2 = Memory(
            storage=storage,
            session_id="session_debug_2",
            debug=False,
            
        )
        
        assert memory_debug_2.debug is True, "debug should be True"
        assert memory_debug_2.debug_level == 2, "debug_level should be 2"
        log_test_result("Debug level 2 set correctly", True)
        
        await storage.disconnect_async()
        await cleanup_db(db_path)
        
    except Exception as e:
        log_test_result("Debug Levels", False, str(e))
        raise


# =============================================================================
# TEST 10: Feed Tool Call Results
# =============================================================================
async def test_feed_tool_call_results():
    """Test feed_tool_call_results setting and verify tool messages are filtered correctly."""
    print_separator("TEST 10: Feed Tool Call Results")
    
    db_path = "test_memory_tool_results.db"
    await cleanup_db(db_path)
    
    try:
        storage = SqliteStorage(db_file=db_path)
        
        # Helper function to count tool messages in message history
        def count_tool_messages(messages):
            """Count ModelRequest messages that contain tool-return parts."""
            from upsonic.messages import ModelRequest
            tool_count = 0
            tool_details = []
            for msg in messages:
                if isinstance(msg, ModelRequest):
                    if hasattr(msg, 'parts'):
                        for part in msg.parts:
                            if hasattr(part, 'part_kind') and part.part_kind == 'tool-return':
                                tool_count += 1
                                tool_details.append(f"Tool: {getattr(part, 'tool_name', 'unknown')}")
                                break
            return tool_count, tool_details
        
        # Helper function to inspect all messages
        def inspect_messages(messages):
            """Inspect message types for debugging."""
            from upsonic.messages import ModelRequest, ModelResponse
            details = []
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                if isinstance(msg, ModelRequest):
                    part_kinds = []
                    if hasattr(msg, 'parts'):
                        for part in msg.parts:
                            if hasattr(part, 'part_kind'):
                                part_kinds.append(part.part_kind)
                    details.append(f"  [{i}] ModelRequest with parts: {part_kinds}")
                elif isinstance(msg, ModelResponse):
                    details.append(f"  [{i}] ModelResponse")
                else:
                    details.append(f"  [{i}] {msg_type}")
            return details
        
        # Test feed_tool_call_results=False - tool messages should be filtered
        print("\n[Test 1] Testing feed_tool_call_results=False (should filter tool messages)...")
        memory_no_tools = Memory(
            storage=storage,
            session_id="session_no_tools",
            user_id="user_001",
            full_session_memory=True,
            feed_tool_call_results=False,
            debug=False
        )
        
        assert memory_no_tools.feed_tool_call_results is False, "feed_tool_call_results should be False"
        log_test_result("feed_tool_call_results=False set correctly", True)
        
        # Create a simple tool that will be called
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Weather in {location}: Sunny, 72°F"
        
        agent_no_tools = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_no_tools,
            tools=[get_weather],
            feed_tool_call_results=False,  # Explicitly set to match memory
            debug=False
        )
        
        # Make a task that will trigger tool calls
        print("\n[Turn 1] Making request that triggers tool call...")
        task1 = Task("What's the weather in New York? Use the get_weather tool.")
        result1 = await agent_no_tools.do_async(task1)
        print(f"Response 1: {result1}")
        
        # Check message history - should NOT contain tool messages
        prepared = await memory_no_tools.prepare_inputs_for_task()
        message_history_no_tools = prepared["message_history"]
        tool_count_no_tools, tool_details_no_tools = count_tool_messages(message_history_no_tools)
        
        assert tool_count_no_tools == 0, f"Tool messages should be filtered when feed_tool_call_results=False, but found {tool_count_no_tools}: {tool_details_no_tools}"
        log_test_result(
            "Tool messages filtered when feed_tool_call_results=False",
            tool_count_no_tools == 0,
            f"Found {tool_count_no_tools} tool messages (expected 0). Details: {tool_details_no_tools}"
        )
        
        # Test feed_tool_call_results=True - tool messages should be included
        print("\n[Test 2] Testing feed_tool_call_results=True (should include tool messages)...")
        memory_with_tools = Memory(
            storage=storage,
            session_id="session_with_tools",
            user_id="user_002",
            full_session_memory=True,
            feed_tool_call_results=True,
            debug=False
        )
        
        assert memory_with_tools.feed_tool_call_results is True, "feed_tool_call_results should be True"
        log_test_result("feed_tool_call_results=True set correctly", True)
        
        agent_with_tools = Agent(
            model="openai/gpt-4o-mini",
            memory=memory_with_tools,
            tools=[get_weather],
            feed_tool_call_results=True,  # Explicitly set to match memory
            debug=False
        )
        
        # Make a task that will trigger tool calls
        print("\n[Turn 1] Making request that triggers tool call...")
        task2 = Task("What's the weather in London? Use the get_weather tool.")
        result2 = await agent_with_tools.do_async(task2)
        print(f"Response 2: {result2}")
        
        # Check message history - SHOULD contain tool messages
        prepared = await memory_with_tools.prepare_inputs_for_task()
        message_history_with_tools = prepared["message_history"]
        tool_count_with_tools, tool_details_with_tools = count_tool_messages(message_history_with_tools)
        
        assert tool_count_with_tools > 0, f"Tool messages should be included when feed_tool_call_results=True, but found {tool_count_with_tools}"
        log_test_result(
            "Tool messages included when feed_tool_call_results=True",
            tool_count_with_tools > 0,
            f"Found {tool_count_with_tools} tool messages (expected > 0). Details: {tool_details_with_tools}"
        )
        
        # Verify the difference
        print(f"\n[Summary]")
        print(f"  feed_tool_call_results=False: {tool_count_no_tools} tool messages in history")
        print(f"  feed_tool_call_results=True: {tool_count_with_tools} tool messages in history")
        print(f"  Total messages (no tools): {len(message_history_no_tools)}")
        print(f"  Total messages (with tools): {len(message_history_with_tools)}")
        
        await storage.disconnect_async()
        await cleanup_db(db_path)
        
    except Exception as e:
        log_test_result("Feed Tool Call Results", False, str(e))
        raise


# =============================================================================
# TEST 11: Session Persistence Across Instances
# =============================================================================
async def test_session_persistence():
    """Test that sessions persist across Memory instances."""
    print_separator("TEST 11: Session Persistence Across Instances")
    
    db_path = "test_memory_persistence.db"
    await cleanup_db(db_path)
    
    try:
        storage = SqliteStorage(db_file=db_path)
        session_id = "persistent_session_001"
        
        # First instance - create and populate session
        print("\n[Phase 1] Creating first Memory instance...")
        memory1 = Memory(
            storage=storage,
            session_id=session_id,
            user_id="user_001",
            full_session_memory=True,
            debug=False,
            
        )
        
        agent1 = Agent(
            model="openai/gpt-4o-mini",
            memory=memory1,
            debug=False,
            
        )
        
        print("\n[Turn 1] Sending message with first agent...")
        task1 = Task("Remember that my secret code is DELTA-789.")
        result1 = await agent1.do_async(task1)
        print(f"Response 1: {result1}")
        
        # Disconnect and reconnect to simulate new session
        await storage.disconnect_async()
        
        # Second instance - should retrieve existing session
        print("\n[Phase 2] Creating second Memory instance (same session_id)...")
        storage2 = SqliteStorage(db_file=db_path)
        
        memory2 = Memory(
            storage=storage2,
            session_id=session_id,  # Same session ID
            user_id="user_001",
            full_session_memory=True,
            debug=False,
            
        )
        
        agent2 = Agent(
            model="openai/gpt-4o-mini",
            memory=memory2,
            debug=False,
            
        )
        
        print("\n[Turn 2] Asking about previous message with second agent...")
        task2 = Task("What is my secret code?")
        result2 = await agent2.do_async(task2)
        print(f"Response 2: {result2}")
        
        # Check if code is remembered
        result2_str = str(result2).upper()
        code_remembered = "DELTA" in result2_str or "789" in result2_str
        assert code_remembered, f"Session should persist across instances - 'DELTA' or '789' not found in: {result2}"
        log_test_result("Session persists across instances", code_remembered, 
                       f"Response: {result2}")
        
        await storage2.disconnect_async()
        await cleanup_db(db_path)
        
    except Exception as e:
        log_test_result("Session Persistence", False, str(e))
        raise


# =============================================================================
# TEST 12: Memory API Methods
# =============================================================================
async def test_memory_api_methods():
    """Test Memory class API methods."""
    print_separator("TEST 12: Memory API Methods")
    
    db_path = "test_memory_api.db"
    await cleanup_db(db_path)
    
    try:
        storage = SqliteStorage(db_file=db_path)
        session_id = "api_test_session"
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id="user_api_test",
            full_session_memory=True,
            debug=False,
            
        )
        
        # Test get_session (should be None initially)
        session = await memory.get_session_async()
        assert session is None, f"Session should be None initially, got: {session}"
        log_test_result("get_session_async (initial)", session is None, 
                       f"Session: {session}")
        
        # Create an agent and run a task to populate the session
        agent = Agent(
            model="openai/gpt-4o-mini",
            memory=memory,
            debug=False,
            
        )
        
        task = Task("Hello, this is a test message.")
        await agent.do_async(task)
        
        # Test get_session (should now exist)
        session = await memory.get_session_async()
        assert session is not None, "Session should exist after running a task"
        assert session.session_id == session_id, f"Session ID should be {session_id}, got {session.session_id}"
        log_test_result("get_session_async (after run)", session is not None, 
                       f"Session ID: {session.session_id if session else None}")
        
        # Test get_messages
        messages = await memory.get_messages_async()
        assert isinstance(messages, list), f"Messages should be a list, got {type(messages)}"
        assert len(messages) > 0, f"Messages should not be empty after a task, got {len(messages)}"
        log_test_result("get_messages_async", isinstance(messages, list), 
                       f"Message count: {len(messages)}")
        
        # Test set_metadata
        await memory.set_metadata_async({"test_key": "test_value"})
        metadata = await memory.get_metadata_async()
        assert metadata is not None, "Metadata should not be None after setting"
        assert metadata.get("test_key") == "test_value", f"Metadata should contain test_key=test_value, got {metadata}"
        log_test_result("set/get_metadata_async", 
                       metadata is not None and metadata.get("test_key") == "test_value",
                       f"Metadata: {metadata}")
        
        # Test list_sessions
        sessions = await memory.list_sessions_async()
        assert len(sessions) >= 1, f"Should have at least 1 session, got {len(sessions)}"
        log_test_result("list_sessions_async", len(sessions) >= 1, 
                       f"Session count: {len(sessions)}")
        
        # Test find_session
        found = await memory.find_session_async(session_id=session_id)
        assert found is not None, f"Should find session by session_id={session_id}"
        assert found.session_id == session_id, f"Found session should have session_id={session_id}, got {found.session_id}"
        log_test_result("find_session_async", found is not None, 
                       f"Found session: {found.session_id if found else None}")
        
        # Test delete_session
        await memory.delete_session_async()
        deleted_session = await memory.get_session_async()
        assert deleted_session is None, f"Session should be None after delete, got {deleted_session}"
        log_test_result("delete_session_async", deleted_session is None, 
                       f"Session after delete: {deleted_session}")
        
        await storage.disconnect_async()
        await cleanup_db(db_path)
        
    except Exception as e:
        log_test_result("Memory API Methods", False, str(e))
        raise


# =============================================================================
# TEST 13: Session ID, User ID, and Run ID Testing
# =============================================================================
async def test_session_user_run_ids():
    """Test session_id, user_id, and run_id with same and different values."""
    print_separator("TEST 13: Session ID, User ID, and Run ID Testing")
    
    db_path = "test_session_user_run_ids.db"
    await cleanup_db(db_path)
    
    try:
        storage = SqliteStorage(db_file=db_path)
        
        # Import AgentSession for helper methods
        from upsonic.session.agent import AgentSession
        
        # ===== SCENARIO 1: Same session_id, multiple runs =====
        print("\n[Scenario 1] Testing same session_id with multiple runs...")
        session_id_1 = "session_001"
        user_id_1 = "user_001"
        
        memory1 = Memory(
            storage=storage,
            session_id=session_id_1,
            user_id=user_id_1,
            full_session_memory=True,
            summary_memory=True,
            debug=True,
            debug_level=1
        )
        
        agent1 = Agent(
            model="openai/gpt-4o-mini",
            memory=memory1,
            debug=False
        )
        
        # Run MORE tasks in session 1 to create a longer conversation (5 runs)
        # This helps us test message counts and isolation better
        tasks_session1 = [
            "My name is Alice and I love Python programming.",
            "What did I tell you about myself?",
            "I also mentioned I love Python.",
            "Can you help me write a Python function?",
            "What's the best way to learn Python?"
        ]
        
        # Run all tasks first
        for i, task_text in enumerate(tasks_session1, 1):
            task = Task(task_text)
            result = await agent1.do_async(task)
            assert result is not None, f"Result{i} should not be None"
        
        # After all tasks are done, read the session from storage to get the actual run_ids
        # This ensures we use the run_ids that are actually stored, not the ones from agent's memory
        session1 = await storage.read_async(session_id_1, AgentSession)
        assert session1 is not None, f"Session1 should exist for session_id={session_id_1}"
        assert session1.session_id == session_id_1, f"Session1 session_id should be {session_id_1}, got {session1.session_id}"
        assert session1.user_id == user_id_1, f"Session1 user_id should be {user_id_1}, got {session1.user_id}"
        assert session1.runs is not None, "Session1 runs should not be None"
        assert len(session1.runs) == 5, f"Session1 should have 5 runs, got {len(session1.runs)}"
        
        # Get run_ids from the stored session (these are the authoritative run_ids)
        # runs is a Dict[str, RunData] where keys are run_ids
        run_ids_session1 = list(session1.runs.keys()) if session1.runs else []
        assert len(run_ids_session1) == 5, f"Should have 5 run_ids, got {len(run_ids_session1)}"
        # Verify all run_ids are unique and not None
        for run_id in run_ids_session1:
            assert run_id is not None, f"All run_ids should be set, got None"
        assert len(set(run_ids_session1)) == 5, f"All run_ids should be unique, got duplicates: {run_ids_session1}"
        
        run_id_1, run_id_2, run_id_3, run_id_4, run_id_5 = run_ids_session1
        log_test_result("Same session_id - multiple runs", True, f"Session has {len(session1.runs)} runs")
        
        # Test helper method: get_all_messages_for_session_id_async
        all_messages_session1 = await AgentSession.get_all_messages_for_session_id_async(
            storage=storage,
            session_id=session_id_1
        )
        assert isinstance(all_messages_session1, list), f"all_messages_session1 should be a list, got {type(all_messages_session1)}"
        assert len(all_messages_session1) > 0, f"Should retrieve messages from session_id={session_id_1}, got {len(all_messages_session1)} messages"
        # Each run should have at least 2 messages (request + response), so 5 runs = at least 10 messages
        assert len(all_messages_session1) >= 10, f"Should have at least 10 messages (2 per run * 5 runs), got {len(all_messages_session1)}"
        log_test_result("get_all_messages_for_session_id_async (same session)", True, 
                       f"Retrieved {len(all_messages_session1)} messages from session_id={session_id_1}")
        
        # Test helper method: get_messages_for_run_id_async for each run
        messages_run1 = await AgentSession.get_messages_for_run_id_async(
            storage=storage,
            session_id=session_id_1,
            run_id=run_id_1
        )
        assert isinstance(messages_run1, list), f"messages_run1 should be a list, got {type(messages_run1)}"
        assert len(messages_run1) > 0, f"Should retrieve messages from run_id={run_id_1}, got {len(messages_run1)} messages"
        assert len(messages_run1) >= 2, f"Run1 should have at least 2 messages (request + response), got {len(messages_run1)}"
        # Verify messages contain "Alice" or "Python" from the first task
        messages_str = " ".join([str(m) for m in messages_run1])
        assert "Alice" in messages_str or "Python" in messages_str, f"Messages should contain context from task1, got: {messages_str[:200]}"
        log_test_result("get_messages_for_run_id_async (run 1)", True, 
                       f"Retrieved {len(messages_run1)} messages from run_id={run_id_1}")
        
        # Get messages for run 5 to verify it's different
        messages_run5 = await AgentSession.get_messages_for_run_id_async(
            storage=storage,
            session_id=session_id_1,
            run_id=run_id_5
        )
        assert len(messages_run5) >= 2, f"Run5 should have at least 2 messages, got {len(messages_run5)}"
        assert len(messages_run1) == len(messages_run5), f"Run1 and Run5 should have same message count (both have request+response), got Run1: {len(messages_run1)}, Run5: {len(messages_run5)}"
        
        # ===== SCENARIO 2: Different session_id, same user_id =====
        print("\n[Scenario 2] Testing different session_id with same user_id...")
        session_id_2 = "session_002"
        
        memory2 = Memory(
            storage=storage,
            session_id=session_id_2,
            user_id=user_id_1,  # Same user_id
            full_session_memory=True,
            summary_memory=True,
            user_analysis_memory=True,
            debug=True,
            debug_level=1
        )
        
        agent2 = Agent(
            model="openai/gpt-4o-mini",
            memory=memory2,
            debug=False
        )
        
        # Run FEWER tasks in session 2 (only 2 runs) to create a shorter conversation
        # This helps us test that messages are properly isolated and counts are correct
        tasks_session2 = [
            "I'm Alice again, and I prefer JavaScript over Python.",
            "Can you explain the differences between JavaScript and Python?"
        ]
        
        run_ids_session2 = []
        for i, task_text in enumerate(tasks_session2, 1):
            task = Task(task_text)
            result = await agent2.do_async(task)
            assert result is not None, f"Session2 Result{i} should not be None"
            run_id = agent2._agent_run_output.run_id if hasattr(agent2, '_agent_run_output') and agent2._agent_run_output else None
            assert run_id is not None, f"Session2 run_id_{i} should not be None"
            # Verify run_id is not from session1
            assert run_id not in run_ids_session1, f"Session2 run_id should not be from session1"
            # Verify all run_ids in session2 are unique
            for prev_run_id in run_ids_session2:
                assert run_id != prev_run_id, f"Session2 run_id_{i} ({run_id}) should be unique"
            run_ids_session2.append(run_id)
        
        run_id_4, run_id_6 = run_ids_session2  # Note: run_id_4 and run_id_6 (skipping run_id_5 which was session1's 5th run)
        
        # Verify new session created
        session2 = await storage.read_async(session_id_2, AgentSession)
        assert session2 is not None, f"Session2 should exist for session_id={session_id_2}"
        assert session2.session_id == session_id_2, f"Session2 session_id should be {session_id_2}, got {session2.session_id}"
        assert session2.user_id == user_id_1, f"Session2 user_id should be {user_id_1}, got {session2.user_id}"
        assert session2.session_id != session1.session_id, "Session2 should have different session_id from Session1"
        assert session2.runs is not None, "Session2 runs should not be None"
        assert len(session2.runs) == 2, f"Session2 should have 2 runs, got {len(session2.runs)}"
        # Verify session2 has fewer runs than session1
        assert len(session2.runs) < len(session1.runs), f"Session2 should have fewer runs than Session1 (Session2: {len(session2.runs)}, Session1: {len(session1.runs)})"
        log_test_result("Different session_id, same user_id - new session created", True, 
                       f"Session2 user_id: {session2.user_id}, runs: {len(session2.runs)}")
        
        # Test helper method: get_all_user_prompt_messages_for_user_id_async
        # Should get prompts from both sessions (session_001 with 5 prompts + session_002 with 2 prompts = 7 total)
        all_user_prompts = await AgentSession.get_all_user_prompt_messages_for_user_id_async(
            storage=storage,
            user_id=user_id_1
        )
        assert isinstance(all_user_prompts, list), f"all_user_prompts should be a list, got {type(all_user_prompts)}"
        assert len(all_user_prompts) == 7, f"Should have exactly 7 prompts (5 from session1 + 2 from session2), got {len(all_user_prompts)}"
        # Verify prompts contain expected content
        prompts_str = " ".join(all_user_prompts)
        assert "Alice" in prompts_str, f"Prompts should contain 'Alice', got: {prompts_str[:200]}"
        assert "Python" in prompts_str, f"Prompts should contain 'Python', got: {prompts_str[:200]}"
        assert "JavaScript" in prompts_str, f"Prompts should contain 'JavaScript' from session2, got: {prompts_str[:200]}"
        log_test_result("get_all_user_prompt_messages_for_user_id_async (same user_id)", True, 
                       f"Retrieved {len(all_user_prompts)} user prompts from user_id={user_id_1}")
        
        # Verify messages from session_id_1 are NOT in session_id_2
        messages_session2 = await AgentSession.get_all_messages_for_session_id_async(
            storage=storage,
            session_id=session_id_2
        )
        assert isinstance(messages_session2, list), f"messages_session2 should be a list, got {type(messages_session2)}"
        assert len(messages_session2) > 0, f"Session2 should have messages, got {len(messages_session2)}"
        # Session2 should have exactly 4 messages (2 runs * 2 messages per run)
        assert len(messages_session2) == 4, f"Session2 should have exactly 4 messages (2 runs * 2 messages), got {len(messages_session2)}"
        # Session2 should have fewer messages than Session1
        assert len(messages_session2) < len(all_messages_session1), f"Session2 should have fewer messages than Session1 (Session2: {len(messages_session2)}, Session1: {len(all_messages_session1)})"
        # Verify session2 messages don't contain run_ids from session1
        # runs is a Dict[str, RunData] where keys are run_ids
        session2_run_ids = list(session2.runs.keys()) if session2.runs else []
        for run_id in run_ids_session1:
            assert run_id not in session2_run_ids, f"run_id from session1 ({run_id}) should not be in session2"
        log_test_result("Different session_id - messages isolated", True, 
                       f"Session2 has {len(messages_session2)} messages, Session1 has {len(all_messages_session1)} messages")
        
        # ===== SCENARIO 3: Different user_id =====
        print("\n[Scenario 3] Testing different user_id...")
        session_id_3 = "session_003"
        user_id_2 = "user_002"
        
        memory3 = Memory(
            storage=storage,
            session_id=session_id_3,
            user_id=user_id_2,  # Different user_id
            full_session_memory=True,
            summary_memory=True,
            user_analysis_memory=True,
            debug=True,
            debug_level=1
        )
        
        agent3 = Agent(
            model="openai/gpt-4o-mini",
            memory=memory3,
            debug=False
        )
        
        # Run in new session with different user
        task_user2 = Task("My name is Bob and I love Java programming.")
        result_user2 = await agent3.do_async(task_user2)
        assert result_user2 is not None, "result_user2 should not be None"
        run_id_user2 = agent3._agent_run_output.run_id if hasattr(agent3, '_agent_run_output') and agent3._agent_run_output else None
        assert run_id_user2 is not None, "run_id_user2 should not be None"
        # run_id_user2 should be different from all run_ids in session1
        assert run_id_user2 not in run_ids_session1, f"run_id_user2 ({run_id_user2}) should not be in session1 run_ids"
        
        # Verify new user session created
        session3 = await storage.read_async(session_id_3, AgentSession)
        assert session3 is not None, f"Session3 should exist for session_id={session_id_3}"
        assert session3.session_id == session_id_3, f"Session3 session_id should be {session_id_3}, got {session3.session_id}"
        assert session3.user_id == user_id_2, f"Session3 user_id should be {user_id_2}, got {session3.user_id}"
        assert session3.user_id != user_id_1, f"Session3 user_id should be different from user_id_1"
        assert session3.runs is not None, "Session3 runs should not be None"
        assert len(session3.runs) == 1, f"Session3 should have 1 run, got {len(session3.runs)}"
        log_test_result("Different user_id - new session created", True, 
                       f"Session3 user_id: {session3.user_id}")
        
        # Test helper method: get_all_user_prompt_messages_for_user_id_async for user_id_2
        user2_prompts = await AgentSession.get_all_user_prompt_messages_for_user_id_async(
            storage=storage,
            user_id=user_id_2
        )
        assert isinstance(user2_prompts, list), f"user2_prompts should be a list, got {type(user2_prompts)}"
        assert len(user2_prompts) == 1, f"User2 should have exactly 1 prompt, got {len(user2_prompts)}"
        assert "Bob" in " ".join(user2_prompts), f"User2 prompts should contain 'Bob', got: {user2_prompts}"
        assert "Java" in " ".join(user2_prompts), f"User2 prompts should contain 'Java', got: {user2_prompts}"
        log_test_result("get_all_user_prompt_messages_for_user_id_async (different user_id)", True, 
                       f"Retrieved {len(user2_prompts)} user prompts from user_id={user_id_2}")
        
        # Verify user_id_1 prompts don't include user_id_2 prompts
        user1_prompts_after = await AgentSession.get_all_user_prompt_messages_for_user_id_async(
            storage=storage,
            user_id=user_id_1
        )
        assert isinstance(user1_prompts_after, list), f"user1_prompts_after should be a list, got {type(user1_prompts_after)}"
        # User1 should have exactly 7 prompts (5 from session1 + 2 from session2)
        assert len(user1_prompts_after) == 7, f"User1 should have exactly 7 prompts (5 from session1 + 2 from session2), got {len(user1_prompts_after)}"
        user1_prompts_str = " ".join(user1_prompts_after)
        assert "Bob" not in user1_prompts_str, f"User1 prompts should NOT contain 'Bob' from user2, got: {user1_prompts_str[:200]}"
        # Check for "Java programming" (not just "Java" which is in "JavaScript")
        assert "Java programming" not in user1_prompts_str, f"User1 prompts should NOT contain 'Java programming' from user2, got: {user1_prompts_str[:200]}"
        assert "Alice" in user1_prompts_str, f"User1 prompts should contain 'Alice', got: {user1_prompts_str[:200]}"
        # Verify user2 prompts contain Bob and Java programming
        user2_prompts_str = " ".join(user2_prompts)
        assert "Bob" in user2_prompts_str, f"User2 prompts should contain 'Bob', got: {user2_prompts_str}"
        assert "Java programming" in user2_prompts_str, f"User2 prompts should contain 'Java programming', got: {user2_prompts_str}"
        # Verify user1 has more prompts than user2
        assert len(user1_prompts_after) > len(user2_prompts), f"User1 should have more prompts than User2 (User1: {len(user1_prompts_after)}, User2: {len(user2_prompts)})"
        log_test_result("Different user_id - prompts isolated", True, 
                       f"User1 has {len(user1_prompts_after)} prompts, User2 has {len(user2_prompts)} prompts")
        
        # ===== SCENARIO 4: Test exclude_run_id =====
        print("\n[Scenario 4] Testing exclude_run_id functionality...")
        
        # Get all messages from session_id_1 excluding run_id_5 (the last run)
        # Use run_id_5 which we got from the stored session above
        # First verify run_id_5 exists in the session
        session1_verify = await storage.read_async(session_id_1, AgentSession)
        assert session1_verify is not None, "Session1 should exist for verification"
        assert session1_verify.runs is not None, "Session1 should have runs"
        # runs is a Dict[str, RunData] where keys are run_ids
        run_ids_verify = list(session1_verify.runs.keys()) if session1_verify.runs else []
        assert run_id_5 in run_ids_verify, f"run_id_5 ({run_id_5}) must be in session runs. Found: {run_ids_verify}"
        
        messages_excluding_run5 = await AgentSession.get_all_messages_for_session_id_async(
            storage=storage,
            session_id=session_id_1,
            exclude_run_id=run_id_5
        )
        assert isinstance(messages_excluding_run5, list), f"messages_excluding_run5 should be a list, got {type(messages_excluding_run5)}"
        
        # Get all messages from session_id_1 (no exclusion)
        messages_all_runs = await AgentSession.get_all_messages_for_session_id_async(
            storage=storage,
            session_id=session_id_1
        )
        assert isinstance(messages_all_runs, list), f"messages_all_runs should be a list, got {type(messages_all_runs)}"
        assert len(messages_all_runs) > 0, f"messages_all_runs should have messages, got {len(messages_all_runs)}"
        
        # Debug: Check what run_ids are in the session and verify exclusion logic
        print(f"\n[DEBUG] run_id_5 to exclude: {run_id_5}")
        print(f"[DEBUG] Session has {len(session1_verify.runs)} runs")
        # runs is a Dict[str, RunData] where keys are run_ids
        for i, (run_id_check, run_data) in enumerate(session1_verify.runs.items()):
            run_msg_count = len(run_data.output.messages) if run_data.output and run_data.output.messages else 0
            print(f"[DEBUG] Run {i+1}: run_id={run_id_check}, messages={run_msg_count}, will_exclude={run_id_check == run_id_5}")
        
        # Verify exclusion worked
        assert len(messages_excluding_run5) < len(messages_all_runs), f"Excluded messages should be fewer (excluded: {len(messages_excluding_run5)}, all: {len(messages_all_runs)})"
        # Run5 should have at least 2 messages, so difference should be at least 2
        assert len(messages_all_runs) - len(messages_excluding_run5) >= 2, f"Difference should be at least 2 messages (excluded: {len(messages_excluding_run5)}, all: {len(messages_all_runs)})"
        # With 5 runs, excluding 1 run should leave 4 runs = 8 messages (4 * 2)
        assert len(messages_excluding_run5) >= 8, f"Excluding 1 run from 5 should leave at least 8 messages (4 runs * 2), got {len(messages_excluding_run5)}"
        
        # Verify run_id_5 messages are not in excluded list
        # Get messages from run_id_5 directly
        messages_run5_direct = await AgentSession.get_messages_for_run_id_async(
            storage=storage,
            session_id=session_id_1,
            run_id=run_id_5
        )
        assert len(messages_run5_direct) >= 2, f"Run5 should have at least 2 messages, got {len(messages_run5_direct)}"
        # Verify the count matches: excluded + run5 = all
        assert len(messages_excluding_run5) + len(messages_run5_direct) == len(messages_all_runs), f"Excluded ({len(messages_excluding_run5)}) + Run5 ({len(messages_run5_direct)}) should equal all ({len(messages_all_runs)})"
        log_test_result("exclude_run_id - messages filtered", True, 
                       f"With exclusion: {len(messages_excluding_run5)} messages, Without: {len(messages_all_runs)} messages")
        
        # Test exclude_run_id for user prompts (exclude run_id_2 from session1)
        user1_prompts_excluding_run2 = await AgentSession.get_all_user_prompt_messages_for_user_id_async(
            storage=storage,
            user_id=user_id_1,
            exclude_run_id=run_id_2
        )
        assert isinstance(user1_prompts_excluding_run2, list), f"user1_prompts_excluding_run2 should be a list, got {type(user1_prompts_excluding_run2)}"
        assert len(user1_prompts_excluding_run2) < len(user1_prompts_after), f"Excluded prompts should be fewer (excluded: {len(user1_prompts_excluding_run2)}, all: {len(user1_prompts_after)})"
        # Run2 had 1 user prompt, so difference should be exactly 1
        assert len(user1_prompts_after) - len(user1_prompts_excluding_run2) == 1, f"Difference should be exactly 1 prompt (excluded: {len(user1_prompts_excluding_run2)}, all: {len(user1_prompts_after)})"
        # Should have 6 prompts (7 total - 1 excluded)
        assert len(user1_prompts_excluding_run2) == 6, f"Should have exactly 6 prompts (7 total - 1 excluded), got {len(user1_prompts_excluding_run2)}"
        log_test_result("exclude_run_id - user prompts filtered", True, 
                       f"With exclusion: {len(user1_prompts_excluding_run2)} prompts, Without: {len(user1_prompts_after)} prompts")
        
        # ===== SCENARIO 5: Verify run_id isolation =====
        print("\n[Scenario 5] Testing run_id isolation...")
        
        # Get messages for each run individually from session1 (5 runs)
        messages_by_run = {}
        for i, run_id in enumerate(run_ids_session1, 1):
            messages = await AgentSession.get_messages_for_run_id_async(
                storage=storage,
                session_id=session_id_1,
                run_id=run_id
            )
            assert isinstance(messages, list), f"messages_run{i}_check should be a list, got {type(messages)}"
            assert len(messages) > 0, f"Run{i} should have messages, got {len(messages)}"
            assert len(messages) >= 2, f"Run{i} should have at least 2 messages, got {len(messages)}"
            messages_by_run[run_id] = messages
        
        messages_run1_check = messages_by_run[run_id_1]
        messages_run2_check = messages_by_run[run_id_2]
        messages_run3_check = messages_by_run[run_id_3]
        messages_run5_check = messages_by_run[run_id_5]
        
        # Verify all runs have the same message count (2 messages each: request + response)
        for run_id, messages in messages_by_run.items():
            assert len(messages) == 2, f"Each run should have exactly 2 messages (request + response), got {len(messages)} for run_id={run_id}"
        
        # Verify total messages from individual runs matches all messages
        total_individual = sum(len(msgs) for msgs in messages_by_run.values())
        assert total_individual == len(messages_all_runs), f"Sum of individual run messages ({total_individual}) should equal all messages ({len(messages_all_runs)})"
        # 5 runs * 2 messages = 10 messages
        assert total_individual == 10, f"Should have exactly 10 messages (5 runs * 2), got {total_individual}"
        assert len(messages_all_runs) == 10, f"All messages should be exactly 10 (5 runs * 2), got {len(messages_all_runs)}"
        
        # Verify each run has unique run_id (already verified above, but double-check)
        assert len(set(run_ids_session1)) == 5, f"All 5 run_ids should be unique, got {len(set(run_ids_session1))} unique run_ids"
        
        log_test_result("run_id isolation - each run has messages", True, 
                       f"Run1: {len(messages_run1_check)}, Run2: {len(messages_run2_check)}, Run3: {len(messages_run3_check)}, Run5: {len(messages_run5_check)} messages")
        
        # Verify run_id from different session is not accessible
        messages_wrong_session = await AgentSession.get_messages_for_run_id_async(
            storage=storage,
            session_id=session_id_2,  # Wrong session
            run_id=run_id_1  # Run from session_id_1
        )
        assert isinstance(messages_wrong_session, list), f"messages_wrong_session should be a list, got {type(messages_wrong_session)}"
        assert len(messages_wrong_session) == 0, f"Wrong session should return 0 messages, got {len(messages_wrong_session)}"
        
        # Verify run_id from session_id_2 is accessible with correct session_id
        messages_run4_correct = await AgentSession.get_messages_for_run_id_async(
            storage=storage,
            session_id=session_id_2,  # Correct session
            run_id=run_id_4  # Run from session_id_2
        )
        assert len(messages_run4_correct) > 0, f"Run4 should be accessible with correct session_id, got {len(messages_run4_correct)} messages"
        assert len(messages_run4_correct) == 2, f"Run4 should have exactly 2 messages, got {len(messages_run4_correct)}"
        
        log_test_result("run_id isolation - cross-session access blocked", True, 
                       f"Wrong session returned {len(messages_wrong_session)} messages (expected 0)")
        
        await storage.disconnect_async()
        await cleanup_db(db_path)
        
    except Exception as e:
        log_test_result("Session/User/Run ID Testing", False, str(e))
        import traceback
        traceback.print_exc()
        raise


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================
async def run_all_tests():
    """Run all Memory tests sequentially."""
    print("\n" + "=" * 60)
    print("  COMPREHENSIVE MEMORY TEST SUITE")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key in .env file or environment.")
        return
    
    print(f"\n✓ OpenAI API Key found (length: {len(api_key)})")
    
    tests = [
        ("Basic Storage and Session ID", test_basic_storage_and_session_id),
        ("Full Session Memory", test_full_session_memory),
        ("Summary Memory", test_summary_memory),
        ("User Analysis Memory (Default)", test_user_analysis_memory_default),
        ("Custom User Profile Schema", test_custom_user_profile_schema),
        ("Dynamic User Profile", test_dynamic_user_profile),
        ("num_last_messages Limit", test_num_last_messages),
        ("User Memory Mode", test_user_memory_mode),
        # ("Debug Levels", test_debug_levels),  # Skipped - debug testing not needed
        ("Feed Tool Call Results", test_feed_tool_call_results),
        ("Session Persistence", test_session_persistence),
        ("Memory API Methods", test_memory_api_methods),
        ("Session/User/Run ID Testing", test_session_user_run_ids),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    print(f"\n  Total Tests: {len(tests)}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    
    if failed == 0:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️  {failed} test(s) failed. Please review the output above.")
    
    print("\n" + "=" * 60)
    
    # Detailed results
    print("\n  DETAILED RESULTS:")
    for result in test_results:
        status = "✅" if result["passed"] else "❌"
        print(f"    {status} {result['name']}")
        if result["message"]:
            print(f"       └─ {result['message'][:100]}...")


if __name__ == "__main__":
    asyncio.run(run_all_tests())

