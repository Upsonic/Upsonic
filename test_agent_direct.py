"""Test Agent.stream_async directly"""
import asyncio
import time
from upsonic import Agent, Task

async def main():
    agent = Agent("openai/gpt-4o", debug=False)
    task = Task(description="Write me a 500-word detailed story about a space explorer")

    print("\n=== DIRECT AGENT STREAMING TEST ===", flush=True)
    start_time = time.time()
    first_chunk_time = None
    chunk_count = 0

    stream_result = await agent.stream_async(task)

    async with stream_result:
        async for chunk in stream_result.stream_output():
            if first_chunk_time is None:
                first_chunk_time = time.time()
                print(f"\n[First chunk after {first_chunk_time - start_time:.2f}s]", flush=True)

            chunk_count += 1
            current_time = time.time()

            # Print every 100th chunk with timing
            if chunk_count % 100 == 0:
                print(f"[{current_time - start_time:.3f}s - Chunk {chunk_count}]", flush=True)

    end_time = time.time()
    print(f"\n[Total: {end_time - start_time:.2f}s, {chunk_count} chunks]")

if __name__ == "__main__":
    asyncio.run(main())
