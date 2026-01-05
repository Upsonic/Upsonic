from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.postgres import PostgresStorage

# Create PostgreSQL storage and memory
storage = PostgresStorage(
    db_url="postgresql://user:pass@localhost:5432/dbname",
    schema="public",
    agent_sessions_table_name="agent_sessions"
)
memory = Memory(
    storage=storage,
    session_id="session_001",
    user_id="user_123",
    full_session_memory=True,
    summary_memory=True,
    user_analysis_memory=True,
    model="openai/gpt-4o"
)

# Create agent with memory
agent = Agent("openai/gpt-4o", memory=memory)

# First conversation
task1 = Task("My name is Alice and I love Python")
result1 = agent.do(task1)

# Second conversation - agent remembers
task2 = Task("What's my name and favorite language?")
result2 = agent.do(task2)
print(result2)  # Output: Your name is Alice and you love Python