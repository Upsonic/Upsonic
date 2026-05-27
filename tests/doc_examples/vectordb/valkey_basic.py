"""
Basic usage of ValkeyProvider for vector storage and search.

Requirements:
- Valkey 9.1+ with valkey-search module loaded
- pip install "upsonic[valkey]"
"""

import asyncio
from upsonic.vectordb import ValkeyProvider, ValkeyConfig
from upsonic.vectordb.config import (
    ConnectionConfig,
    Mode,
    DistanceMetric,
    HNSWIndexConfig,
)


async def main():
    # Configure the provider
    config = ValkeyConfig(
        vector_size=384,  # Match your embedding model's dimension
        collection_name="my_documents",
        key_prefix="docs:",
        connection=ConnectionConfig(
            mode=Mode.LOCAL,
            host="localhost",
            port=6379,
        ),
        distance_metric=DistanceMetric.COSINE,
        index=HNSWIndexConfig(m=16, ef_construction=200),
    )

    # Create provider and connect
    provider = ValkeyProvider(config)
    await provider.aconnect()

    # Create the search index
    await provider.acreate_collection()

    # Upsert documents (vectors would come from your embedding provider)
    await provider.aupsert(
        vectors=[
            [0.1] * 384,  # Replace with real embeddings
            [0.2] * 384,
            [0.3] * 384,
        ],
        ids=["chunk_1", "chunk_2", "chunk_3"],
        chunks=[
            "Valkey is a high-performance in-memory data store",
            "Vector search enables semantic similarity matching",
            "HNSW provides fast approximate nearest neighbor search",
        ],
        document_ids=["doc_1", "doc_1", "doc_2"],
        document_names=["valkey_intro.md", "valkey_intro.md", "vector_search.md"],
    )

    # Brief pause to allow the index to process ingested data
    await asyncio.sleep(0.5)

    # Dense (KNN) vector search
    query_vector = [0.15] * 384  # Replace with embedded query
    results = await provider.adense_search(
        query_vector=query_vector,
        top_k=2,
    )
    print("Dense search results:")
    for r in results:
        print(f"  [{r.score:.3f}] {r.text[:60]}...")

    # Full-text search
    results = await provider.afull_text_search(
        query_text="vector similarity",
        top_k=2,
    )
    print("\nFull-text search results:")
    for r in results:
        print(f"  [{r.score:.3f}] {r.text[:60]}...")

    # Hybrid search (combines vector + text via RRF)
    results = await provider.ahybrid_search(
        query_vector=query_vector,
        query_text="nearest neighbor",
        top_k=2,
    )
    print("\nHybrid search results:")
    for r in results:
        print(f"  [{r.score:.3f}] {r.text[:60]}...")

    # Search with filter
    results = await provider.adense_search(
        query_vector=query_vector,
        top_k=5,
        filter={"document_name": "valkey_intro.md"},
    )
    print("\nFiltered search (valkey_intro.md only):")
    for r in results:
        print(f"  [{r.score:.3f}] {r.text[:60]}...")

    # Check existence
    exists = await provider.achunk_content_hash_exists("some_hash")
    print(f"\nChunk exists: {exists}")

    # Cleanup
    await provider.adelete_collection()
    await provider.adisconnect()


if __name__ == "__main__":
    asyncio.run(main())
