from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.sqlite import SqliteStorage

# Create storage
storage = SqliteStorage(db_file="agent_memory.db", agent_sessions_table_name="sessions")

# Create memory with configuration
memory = Memory(
    storage=storage,
    session_id="session_001",
    user_id="user_123",
    full_session_memory=True,
    summary_memory=True,
    user_analysis_memory=True,
    num_last_messages=20,
    model="openai/gpt-4o-mini",
    feed_tool_call_results=False,
    debug=True
)

# Create agent with memory
agent = Agent("openai/gpt-4o", memory=memory)

# Use agent - memory is automatic
task = Task("Hello! I'm interested in learning Python")
result = agent.do(task)
print(result)