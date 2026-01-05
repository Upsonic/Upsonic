from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.in_memory import InMemoryStorage

# Create storage and memory with conversation memory
storage = InMemoryStorage()
memory = Memory(
    storage=storage,
    session_id="session_001",
    full_session_memory=True
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