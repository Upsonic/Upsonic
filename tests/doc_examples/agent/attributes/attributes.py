from upsonic import Agent, Task
from upsonic.storage.providers.sqlite import SqliteStorage
from upsonic.storage import Memory

# Create storage and memory
storage = SqliteStorage(
    db_file="agent_memory.db",
    agent_sessions_table_name="sessions"
)

memory = Memory(
    storage=storage,
    session_id="session_001",
    user_id="user_001",
    full_session_memory=True,
    summary_memory=True,
    model="openai/gpt-4o-mini"
)

# Create agent with configuration
agent = Agent(
    model="openai/gpt-4o",
    name="Assistant",
    memory=memory,
    debug=True,
    role="AI Assistant",
    goal="Help users with their questions",
    show_tool_calls=True,
    tool_call_limit=10
)

# Execute a task
task = Task("Hello! What can you help me with?")
result = agent.do(task)
print(result)