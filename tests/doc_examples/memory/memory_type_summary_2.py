from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.sqlite import SqliteStorage

# Combined with conversation memory
storage = SqliteStorage(
    db_file="agent_memory.db",
    agent_sessions_table_name="sessions"
)
memory = Memory(
    storage=storage,
    session_id="session_002",
    full_session_memory=True,
    summary_memory=True,
    model="openai/gpt-4o-mini"
)

agent = Agent("openai/gpt-4o", memory=memory)

task1 = Task("Let's talk about database optimization")
result1 = agent.do(task1)

task2 = Task("Continue with more advanced techniques")
result2 = agent.do(task2)
print(result2)