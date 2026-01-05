from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.json import JSONStorage

# With custom formatting
storage = JSONStorage(
    directory_path="./memory_data",
    pretty_print=True
)
memory = Memory(
    storage=storage,
    session_id="session_001",
    user_id="user_123",
    full_session_memory=True,
    summary_memory=True,
    model="openai/gpt-4o"
)

agent = Agent("openai/gpt-4o", memory=memory)

task1 = Task("Help me understand Python decorators")
result1 = agent.do(task1)

task2 = Task("Can you give me a simple example of one?")
result2 = agent.do(task2)
print(result2)