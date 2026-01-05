from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.in_memory import InMemoryStorage

# With message limiting
storage = InMemoryStorage()
memory = Memory(
    storage=storage,
    session_id="session_001",
    full_session_memory=True,
    num_last_messages=10  # Keep last 10 conversation turns
)

agent = Agent("openai/gpt-4o", memory=memory)

task1 = Task("Let's discuss machine learning")
result1 = agent.do(task1)

task2 = Task("What were we talking about?")
result2 = agent.do(task2)
print(result2)