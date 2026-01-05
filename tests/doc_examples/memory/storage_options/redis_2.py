from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.redis import RedisStorage

# Redis with custom configuration
storage = RedisStorage(
    host="redis.example.com",
    port=6379,
    db=1,
    ssl=True,
    prefix="prod:memory",
    expire=3600
)
memory = Memory(
    storage=storage,
    session_id="session_001",
    full_session_memory=True
)

agent = Agent("openai/gpt-4o", memory=memory)

task1 = Task("Help me with caching strategies")
result1 = agent.do(task1)

task2 = Task("Why is Redis good for this?")
result2 = agent.do(task2)
print(result2)