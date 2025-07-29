import os
import uuid
import time
import shutil
import asyncio
import traceback
from pathlib import Path


from upsonic.storage.factory import StorageFactory

from upsonic.storage.sessions import AgentSession, BaseSession

from pydantic_ai.messages import ModelMessagesTypeAdapter

from upsonic.agent.agent import Direct
from upsonic.tasks.tasks import Task
from upsonic.tools.tool import tool, ToolHooks



test_count = 0
failures = 0

def check(condition, success_message, failure_message):
    """A simple assertion helper that prints results."""
    global test_count, failures
    test_count += 1
    if condition:
        print(f"  [ OK ] {success_message}")
    else:
        print(f"  [FAIL] {failure_message}")
        failures += 1

def header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def subheader(title):
    print(f"\n--- {title} ---")

def deserialize_history(memory_list: list):
    if not memory_list:
        return []
    try:
        return ModelMessagesTypeAdapter.validate_python(memory_list)
    except Exception:
        return []


hook_events = []
cache_test_tool_execution_count = 0
def before_hook_func(*args, **kwargs): hook_events.append("before_hook_called")
def after_hook_func(result: any): hook_events.append("after_hook_called")
@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Gets the current system time as a string."""
    return f"The current time in {timezone} is {time.strftime('%H:%M:%S')}."
@tool(cache_results=True)
def cached_tool_with_side_effect(input: str) -> str:
    """A tool that simulates a side effect and caching behavior."""
    global cache_test_tool_execution_count
    cache_test_tool_execution_count += 1
    return f"Processed '{input}' for the {cache_test_tool_execution_count} time."
@tool(requires_confirmation=True)
def sensitive_operation(action: str) -> str:
    """A sensitive operation that requires user confirmation before execution."""
    return f"Sensitive action '{action}' was approved and executed."
@tool(requires_user_input=True, user_input_fields=['destination'])
def book_flight(destination: str, passengers: int = 1) -> str:
    """Books a flight to the specified destination with the given number of passengers."""
    return f"Booked a flight for {passengers} to '{destination}'."
@tool(tool_hooks=ToolHooks(before=before_hook_func, after=after_hook_func))
def tool_with_hooks() -> str:
    """A tool that demonstrates the use of hooks before and after execution."""
    hook_events.append("tool_executed")
    return "Tool has finished execution."


def test_storage_lifecycle(storage):
    """Tests the basic lifecycle operations of the storage provider."""
    subheader("Testing Storage Lifecycle (connect, disconnect, create, drop)")
    storage.connect()
    check(storage.is_connected(), "Provider connects successfully.", "Provider failed to connect.")
    storage.disconnect()
    check(not storage.is_connected(), "Provider disconnects successfully.", "Provider failed to disconnect.")
    storage.connect()
    try:
        storage.drop()
        storage.create()
        check(True, "Provider can drop and create its schema/structure.", "Provider failed on drop() or create().")
    except Exception as e:
        check(False, "", f"Provider failed on drop() or create() with error: {e}")

def test_crud_operations(storage):
    """Tests the core CRUD operations of the storage provider."""
    subheader("Testing Core CRUD Operations (upsert, read, delete)")
    session_id = str(uuid.uuid4())
    agent_id, user_id = "crud-agent", "crud-user"

    session1 = AgentSession(session_id=session_id, agent_id=agent_id, user_id=user_id, memory=[{"msg": "initial"}])
    storage.upsert(session1)
    retrieved1 = storage.read(session_id)
    check(retrieved1 is not None and retrieved1.memory == [{"msg": "initial"}], "upsert (create) and read() work as expected.", "Failed to create or read a session.")

    session2 = AgentSession(session_id=session_id, agent_id="crud-agent-updated", user_id=user_id)
    storage.upsert(session2)
    retrieved2 = storage.read(session_id)
    check(retrieved2 is not None and retrieved2.agent_id == "crud-agent-updated", "upsert (update) operation is working.", "Failed to update a session.")

    storage.delete_session(session_id)
    retrieved_deleted = storage.read(session_id)
    check(retrieved_deleted is None, "delete_session() successfully removed the session.", "delete_session() failed.")

def test_listing_and_filtering(storage):
    subheader("Testing Listing and Filtering (get_all, get_recent)")
    storage.drop()
    storage.create()

    base_timestamp = time.time()

    s1 = AgentSession(session_id=str(uuid.uuid4()), user_id="user1", agent_id="agentA", updated_at=base_timestamp + 0.001)
    storage.upsert(s1)
    
    s2 = AgentSession(session_id=str(uuid.uuid4()), user_id="user1", agent_id="agentB", updated_at=base_timestamp + 0.002)
    storage.upsert(s2)
    
    s3 = AgentSession(session_id=str(uuid.uuid4()), user_id="user2", agent_id="agentA", updated_at=base_timestamp + 0.003)
    storage.upsert(s3)
    
    s4 = AgentSession(session_id=str(uuid.uuid4()), user_id="user1", agent_id="agentC", updated_at=base_timestamp + 0.004)
    storage.upsert(s4)

    check(len(storage.get_all_sessions()) == 4, "get_all_sessions() returns the correct total count.", "get_all_sessions() count is wrong.")
    check(len(storage.get_all_sessions(user_id="user1")) == 3, "Filtering by user_id is working.", "user_id filter failed.")
    check(len(storage.get_all_sessions(entity_id="agentA")) == 2, "Filtering by entity_id is working.", "entity_id filter failed.")
    
    recent_sessions_user1 = storage.get_recent_sessions(user_id="user1", limit=2)
    check(len(recent_sessions_user1) == 2, "get_recent_sessions() returned the correct number of limited results.", "get_recent_sessions() limit failed.")
    is_ordered = recent_sessions_user1[0].session_id == s4.session_id and recent_sessions_user1[1].session_id == s2.session_id
    check(is_ordered, "get_recent_sessions() returned sessions in the correct order.", "get_recent_sessions() ordering failed.")

async def test_agent_and_tool_features(storage):
    subheader("INTEGRATION TEST: Agent, Tools & Persistence")
    session_id = str(uuid.uuid4())

    print("\n  --- 1. Simple Tool Use & Initial Save ---")
    agent1 = Direct(storage=storage, session_id=session_id, model="openai/gpt-4o")
    task1 = Task(description="What time is it in PST?", tools=[get_current_time])
    response1 = await agent1.do_async(task1)
    check(response1 is not None and "The current time" in response1, "Agent correctly used a simple tool.", "Agent failed to use a simple tool.")
    saved_session1 = storage.read(session_id)
    check(saved_session1 is not None, "Session was successfully saved after the first run.", "Session was NOT saved after the first run.")

    print("\n  --- 2. Persistence - Loading and Continuing ---")
    agent2 = Direct(storage=storage, session_id=session_id, model="openai/gpt-4o")
    task2 = Task(description="Thank you.")
    await agent2.do_async(task2)
    saved_session2 = storage.read(session_id)
    history = deserialize_history(saved_session2.memory)
    check(len(history) > 2, f"Agent loaded and appended to history (history length: {len(history)}).", "Agent failed to load/append history.")

    print("\n  --- 3. Caching Behavior ---")
    global cache_test_tool_execution_count
    cache_test_tool_execution_count = 0
    agent3 = Direct(storage=storage, session_id=str(uuid.uuid4()), model="openai/gpt-4o")
    cache_task = Task(description="Execute the cached_tool_with_side_effect tool using the input 'cached-test'. Do not ask for permission, just run the tool.", tools=[cached_tool_with_side_effect])
    await agent3.do_async(cache_task)
    check(cache_test_tool_execution_count == 1, "Cached tool executed for the first time.", "Cache tool count is wrong on first call.")
    await agent3.do_async(cache_task)
    check(cache_test_tool_execution_count == 1, "Cached tool was NOT re-executed (CACHE HIT).", "CACHE FAILED: Tool was re-executed.")

    print("\n  --- 4. Interactive Tools ---")
    agent4 = Direct(storage=storage, session_id=str(uuid.uuid4()), model="openai/gpt-4o")
    confirm_task_yes = Task(description="You must run the sensitive_operation tool with the action 'delete_database'.", tools=[sensitive_operation])
    print("\n  >>> INTERACTION REQUIRED: Type 'y' and press Enter to approve the sensitive operation. <<<")
    confirm_response_yes = await agent4.do_async(confirm_task_yes)
    check(confirm_response_yes is not None and "approved and executed" in confirm_response_yes, "Tool with requires_confirmation=True works with user approval.", "Confirmation (approved) failed.")

    confirm_task_no = Task("Delete the database again.", tools=[sensitive_operation])
    print("\n  >>> INTERACTION REQUIRED: Type 'n' and press Enter to deny the sensitive operation. <<<")
    confirm_response_no = await agent4.do_async(confirm_task_no)
    check(confirm_response_no is not None and "cancelled by the user" in confirm_response_no, "Tool with requires_confirmation=True works with user denial.", "Confirmation (denied) failed.")

    input_task = Task(description="You must run the book_flight tool. You will need to ask me for the destination.", tools=[book_flight])
    print("\n  >>> INTERACTION REQUIRED: When prompted for 'destination', type 'Tokyo' and press Enter. <<<")
    input_response = await agent4.do_async(input_task)
    check(input_response is not None and "'Tokyo'" in input_response, "Tool with requires_user_input works.", "User input tool failed.")

    print("\n  --- 5. Tool Hooks ---")
    global hook_events
    hook_events.clear()
    agent5 = Direct(storage=storage, session_id=str(uuid.uuid4()), model="openai/gpt-4o")
    hooks_task = Task("Run the tool with hooks.", tools=[tool_with_hooks])
    await agent5.do_async(hooks_task)
    expected_order = ["before_hook_called", "tool_executed", "after_hook_called"]
    check(hook_events == expected_order, f"Tool hooks executed in the correct order {hook_events}.", f"Tool hooks executed in wrong order: {hook_events}")



async def run_tests_for_provider(provider_type):
    header(f"STARTING TESTS FOR PROVIDER: {provider_type.upper()}")
    
    storage = None
    try:
        os.environ['STORAGE_TYPE'] = provider_type
        if provider_type == "sqlite":
            os.environ['SQLITE_DB_PATH'] = "test_storage.db"
            os.environ['SQLITE_TABLE_NAME'] = f"test_{str(uuid.uuid4()).replace('-', '')}"
        elif provider_type == "json":
            os.environ['JSON_DIRECTORY_PATH'] = "test_json_storage/"
        elif provider_type == "redis":
            os.environ['REDIS_PREFIX'] = f"test_{str(uuid.uuid4()).replace('-', '')}"
        elif provider_type == "postgres":
            os.environ['POSTGRES_TABLE_NAME'] = f"test_{str(uuid.uuid4()).replace('-', '')}"
            os.environ['POSTGRES_SCHEMA'] = "test_schema"
            
        storage = StorageFactory.get_storage()
        

        def run_with_connection(test_func):
            storage.connect()
            try:
                test_func(storage)
            finally:
                storage.disconnect()
        
        async def arun_with_connection(test_func):
            storage.connect()
            try:
                await test_func(storage)
            finally:
                storage.disconnect()

        run_with_connection(test_storage_lifecycle)
        run_with_connection(test_crud_operations)
        run_with_connection(test_listing_and_filtering)
        await arun_with_connection(test_agent_and_tool_features)
        
    except ConnectionError as e:
        print("\n" + "*"*80)
        print(f"  [ SKIP ] Could not connect to {provider_type.upper()}. Skipping tests for this provider.")
        print(f"  Reason: {e}")
        print("*"*80)
        
    except Exception as e:
        global failures
        failures += 1
        print("\n" + "#"*80)
        print(f"  [ FATAL ERROR ] A critical error occurred during tests for {provider_type.upper()}:")
        print(f"  {e}")
        traceback.print_exc()
        print("#"*80)
    finally:
        print(f"\n--- Tearing down for {provider_type.upper()} ---")
        if storage:
            try:
                storage.connect()
                storage.drop()
                storage.disconnect()
            except Exception:
                pass
        if provider_type == "sqlite" and os.path.exists("test_storage.db"):
            os.remove("test_storage.db")
        if provider_type == "json" and os.path.exists("test_json_storage"):
            shutil.rmtree("test_json_storage")
        
        header(f"COMPLETED TESTS FOR PROVIDER: {provider_type.upper()}")

async def main():
    global test_count, failures
    start_time = time.time()
    
    providers_to_test = [
        "in_memory",
        "sqlite",
        "json",
        "redis",
        "postgres",
    ]
    
    for provider in providers_to_test:
        await run_tests_for_provider(provider)
        
    end_time = time.time()
    header("TESTING SUMMARY")
    print(f"  Total checks run: {test_count}")
    print(f"  Total failures:   {failures}")
    print(f"  Total duration:   {end_time - start_time:.2f} seconds")
    if failures > 0:
        print("\n  [ RESULT: SOME TESTS FAILED ]")
    else:
        print("\n  [ RESULT: ALL TESTS PASSED SUCCESSFULLY ]")
    print("="*80)

if __name__ == "__main__":
    cache_dir = Path.home() / '.upsonic' / 'cache'
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        
    asyncio.run(main())