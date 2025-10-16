"""Direct OpenAI streaming test to verify streaming works"""
import asyncio
import time
import os
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("\n=== DIRECT OPENAI STREAMING TEST ===", flush=True)
    start_time = time.time()
    first_chunk_time = None
    chunk_count = 0

    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Write me a 500-word detailed story about a space explorer"}
        ],
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_chunk_time is None:
                first_chunk_time = time.time()
                print(f"\n[First chunk after {first_chunk_time - start_time:.2f}s]", flush=True)

            chunk_count += 1
            current_time = time.time()

            # Print every 50th chunk with timing
            if chunk_count % 50 == 0:
                print(f"[{current_time - start_time:.3f}s - Chunk {chunk_count}] ", end='', flush=True)
                print(chunk.choices[0].delta.content, flush=True)

    end_time = time.time()
    print(f"\n[Total: {end_time - start_time:.2f}s, {chunk_count} chunks]")

if __name__ == "__main__":
    asyncio.run(main())
