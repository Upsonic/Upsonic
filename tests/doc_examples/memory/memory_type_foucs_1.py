from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.in_memory import InMemoryStorage

# Create storage and memory with user analysis
storage = InMemoryStorage()
memory = Memory(
    storage=storage,
    session_id="session_003",
    user_id="user_001",
    user_analysis_memory=True,
    model="openai/gpt-4o"
)

# Create agent with memory
agent = Agent("openai/gpt-4o", memory=memory)

# First conversation - agent learns about user
task1 = Task("Hi! I'm a data scientist with 5 years of experience in ML")
result1 = agent.do(task1)

# Agent remembers user profile
task2 = Task("What do you know about me?")
result2 = agent.do(task2)
print(result2)  # Output: You're a data scientist with 5 years of ML experience