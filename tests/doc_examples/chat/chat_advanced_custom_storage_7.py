import asyncio
from upsonic import Agent, Chat
from upsonic.storage.providers import Mem0Storage


async def main():
    # Mem0 Platform storage
    """
    storage = Mem0Storage(
        api_key="your-api-key",
        org_id="your-org-id",
        project_id="your-project-id",
        namespace="upsonic"
    )
    """

    # Or Mem0 Open Source (path must NOT be /tmp/chroma due to mem0 validation bug)
    storage = Mem0Storage(
        local_config={
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "mem0",
                    "path": "/tmp/upsonic_chroma_db"
                }
            }
        },
        namespace="upsonic"
    )

    agent = Agent("openai/gpt-4o")

    chat = Chat(
        session_id="session1",
        user_id="user1",
        agent=agent,
        storage=storage
    )

    # Invoke chat
    response = await chat.invoke("Explain AI")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())