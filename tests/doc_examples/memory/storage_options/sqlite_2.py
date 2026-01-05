from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.sqlite import SqliteStorage

# In-memory SQLite (temporary) - omit db_file for in-memory
storage = SqliteStorage(
    agent_sessions_table_name="sessions"
)
memory = Memory(
    storage=storage,
    session_id="session_001",
    full_session_memory=True
)

agent = Agent("openai/gpt-4o", memory=memory)

task1 = Task("Hello, I'm learning about databases")
result1 = agent.do(task1)

task2 = Task("Which one are we using right now?")
result2 = agent.do(task2)
print(result2)