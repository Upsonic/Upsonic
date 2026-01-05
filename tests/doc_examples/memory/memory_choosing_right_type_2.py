from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.in_memory import InMemoryStorage

# All three memory types combined
storage = InMemoryStorage()
memory = Memory(
    storage=storage,
    session_id="session_001",
    user_id="user_001",
    full_session_memory=True,
    summary_memory=True,
    user_analysis_memory=True,
    model="openai/gpt-4o"
)

# Create agent with comprehensive memory
agent = Agent("openai/gpt-4o", memory=memory)

# First conversation - agent stores everything
task1 = Task("Hi! I'm Bob, a backend developer who loves microservices")
result1 = agent.do(task1)

# Agent uses all memory types for rich context
task2 = Task("What can you tell me about my background?")
result2 = agent.do(task2)
print(result2)