import asyncio
import asyncpg
from upsonic import Agent, Task
from upsonic.storage import Memory
from upsonic.storage.providers.postgres import PostgresStorage

async def setup_and_run():
    pool = await asyncpg.create_pool("postgresql://user:pass@localhost:5432/dbname")
    storage = PostgresStorage(pool=pool)
    memory = Memory(
        storage=storage,
        session_id="session_001",
        full_session_memory=True
    )
    
    agent = Agent("openai/gpt-4o", memory=memory)
    
    task1 = Task("Explain PostgreSQL indexing")
    result1 = await agent.do_async(task1)
    
    task2 = Task("What are the benefits of using it?")
    result2 = await agent.do_async(task2)
    return result2

if __name__ == "__main__":
    result = asyncio.run(setup_and_run())
    print(result)