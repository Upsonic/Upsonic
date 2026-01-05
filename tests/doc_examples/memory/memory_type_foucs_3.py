from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.in_memory import InMemoryStorage

# Dynamic user learning (agent creates custom fields)
storage = InMemoryStorage()
memory = Memory(
    storage=storage,
    session_id="session_003",
    user_id="user_001",
    user_analysis_memory=True,
    dynamic_user_profile=True,  # Agent creates custom profile fields
    model="openai/gpt-4o"
)

agent = Agent("openai/gpt-4o", memory=memory)

task1 = Task("I prefer concise answers and love working with Python")
result1 = agent.do(task1)

# Agent adapts to learned preferences
task2 = Task("How should you communicate with me?")
result2 = agent.do(task2)
print(result2)