from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.json import JSONStorage

# Create JSON storage and memory
storage = JSONStorage(directory_path="./memory_data")
memory = Memory(
    storage=storage,
    session_id="session_001",
    full_session_memory=True
)

# Create agent with memory
agent = Agent("openai/gpt-4o", memory=memory)

# First conversation
task1 = Task("My name is Bob and I'm a software engineer")
result1 = agent.do(task1)

# Second conversation - agent remembers
task2 = Task("What's my profession?")
result2 = agent.do(task2)
print(result2)  # Output: You're a software engineer