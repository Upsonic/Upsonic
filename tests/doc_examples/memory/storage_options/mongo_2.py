from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.mongo import MongoStorage

# MongoDB with authentication
storage = MongoStorage(
    db_url="mongodb://localhost:27017",
    database_name="production_memory",
    sessions_collection_name="sessions"
)
memory = Memory(
    storage=storage,
    session_id="session_001",
    full_session_memory=True
)

agent = Agent("openai/gpt-4o", memory=memory)

task1 = Task("Help me design a document schema")
result1 = agent.do(task1)

task2 = Task("How should I handle relationships in it?")
result2 = agent.do(task2)
print(result2)