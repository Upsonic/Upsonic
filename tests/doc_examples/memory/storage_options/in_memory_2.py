from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.in_memory import InMemoryStorage

# With LRU cache limits
storage = InMemoryStorage(
    max_sessions=100
)
memory = Memory(
    storage=storage,
    session_id="session_001",
    user_id="user_123",
    full_session_memory=True,
    user_analysis_memory=True,
    model="openai/gpt-4o"
)

agent = Agent("openai/gpt-4o", memory=memory)

task1 = Task("Tell me about machine learning")
result1 = agent.do(task1)

task2 = Task("What is the main concept we just discussed?")
result2 = agent.do(task2)
print(result2)