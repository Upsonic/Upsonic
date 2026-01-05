from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.mem0 import Mem0Storage

# Mem0 Open Source
storage = Mem0Storage(
    local_config={
        "vector_store": {
            "provider": "chroma",
            "config": {
                "path": "./chroma_db"
            }
        }
    }
)
memory = Memory(
    storage=storage,
    session_id="session_001",
    full_session_memory=True
)

agent = Agent("openai/gpt-4o", memory=memory)

task1 = Task("Help me understand vector databases")
result1 = agent.do(task1)

task2 = Task("How do they differ from traditional SQL?")
result2 = agent.do(task2)
print(result2)