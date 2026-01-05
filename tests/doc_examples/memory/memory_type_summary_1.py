from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.sqlite import SqliteStorage

# Create storage and memory with summary memory
storage = SqliteStorage(
    db_file="memory.db",
    agent_sessions_table_name="sessions"
)
memory = Memory(
    storage=storage,
    session_id="session_002",
    summary_memory=True,
    model="openai/gpt-4o-mini"  # Model for generating summaries
)

# Create agent with memory
agent = Agent("openai/gpt-4o", memory=memory)

# First conversation
task1 = Task("I'm learning about web development with React and Node.js")
result1 = agent.do(task1)

# Agent maintains evolving summary
task2 = Task("What have we been discussing?")
result2 = agent.do(task2)
print(result2)