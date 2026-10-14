"""
Comprehensive smoke tests for Valkey Search vector database provider.

Tests all methods, attributes, and search modes against a real Valkey 9.1+
instance with the valkey-search module loaded.

Requirements:
- Valkey 9.1+ running on localhost:6379
- valkey-search module loaded (v1.2+)
- valkey-glide>=2.3.1 installed
"""

import os
import pytest
import asyncio
from typing import List, Dict, Any

from upsonic.vectordb.providers.valkey import ValkeyProvider
from upsonic.vectordb.config import (
    ValkeyConfig,
    ConnectionConfig,
    Mode,
    DistanceMetric,
    HNSWIndexConfig,
    FlatIndexConfig,
)
from upsonic.utils.package.exception import (
    VectorDBConnectionError,
    ConfigurationError,
    CollectionDoesNotExistError,
    VectorDBError,
    SearchError,
    UpsertError,
)
from upsonic.schemas.vector_schemas import VectorSearchResult


# Test data
SAMPLE_VECTORS: List[List[float]] = [
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.6, 0.7, 0.8, 0.9, 1.0],
    [1.1, 1.2, 1.3, 1.4, 1.5],
    [1.6, 1.7, 1.8, 1.9, 2.0],
    [2.1, 2.2, 2.3, 2.4, 2.5],
]

SAMPLE_PAYLOADS: List[Dict[str, Any]] = [
    {"category": "science", "author": "Einstein", "year": 1905},
    {"category": "science", "author": "Newton", "year": 1687},
    {"category": "literature", "author": "Shakespeare", "year": 1600},
    {"category": "literature", "author": "Dickens", "year": 1850},
    {"category": "philosophy", "author": "Plato", "year": -400},
]

SAMPLE_CHUNKS: List[str] = [
    "The theory of relativity revolutionized physics",
    "Laws of motion and universal gravitation",
    "To be or not to be that is the question",
    "It was the best of times it was the worst of times",
    "The unexamined life is not worth living",
]

SAMPLE_IDS: List[str] = [
    "chunk_aaa111",
    "chunk_bbb222",
    "chunk_ccc333",
    "chunk_ddd444",
    "chunk_eee555",
]

QUERY_VECTOR: List[float] = [0.15, 0.25, 0.35, 0.45, 0.55]
QUERY_TEXT: str = "physics theory"

VALKEY_HOST = os.getenv("VALKEY_HOST", "localhost")
VALKEY_PORT = int(os.getenv("VALKEY_PORT", "6379"))


def _can_connect_to_valkey() -> bool:
    """Check if Valkey is reachable."""
    import socket
    try:
        s = socket.create_connection((VALKEY_HOST, VALKEY_PORT), timeout=2)
        s.close()
        return True
    except (socket.error, OSError):
        return False


pytestmark = [
    pytest.mark.skipif(
        not _can_connect_to_valkey(),
        reason=f"Valkey not available at {VALKEY_HOST}:{VALKEY_PORT}",
    ),
]


class TestValkeyProvider:
    """Test ValkeyProvider against a real Valkey instance."""

    @pytest.fixture
    def config(self) -> ValkeyConfig:
        """Create ValkeyConfig with unique collection name."""
        import uuid
        unique_name = f"test_valkey_{uuid.uuid4().hex[:8]}"
        return ValkeyConfig(
            vector_size=5,
            collection_name=unique_name,
            key_prefix=f"{unique_name}:",
            connection=ConnectionConfig(
                mode=Mode.LOCAL,
                host=VALKEY_HOST,
                port=VALKEY_PORT,
            ),
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig(m=16, ef_construction=200),
        )

    @pytest.fixture
    def provider(self, config: ValkeyConfig) -> ValkeyProvider:
        """Create ValkeyProvider instance."""
        return ValkeyProvider(config)

    @pytest.mark.asyncio
    async def test_initialization(self, provider: ValkeyProvider, config: ValkeyConfig):
        """Test provider initialization and attributes."""
        assert provider._config == config
        assert provider._config.vector_size == 5
        assert provider._config.distance_metric == DistanceMetric.COSINE
        assert not provider._is_connected
        assert provider.client is None
        assert provider.name is not None
        assert isinstance(provider.id, str)
        assert len(provider.id) > 0

    @pytest.mark.asyncio
    async def test_connect(self, provider: ValkeyProvider):
        """Test connection to Valkey."""
        await provider.aconnect()
        assert provider._is_connected
        assert provider.client is not None
        assert await provider.ais_ready()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_connect_sync(self, provider: ValkeyProvider):
        """Test synchronous connection."""
        provider.connect()
        assert provider._is_connected
        assert provider.client is not None
        assert provider.is_ready()
        provider.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect(self, provider: ValkeyProvider):
        """Test disconnection."""
        await provider.aconnect()
        assert provider._is_connected
        await provider.adisconnect()
        assert not provider._is_connected

    @pytest.mark.asyncio
    async def test_is_ready(self, provider: ValkeyProvider):
        """Test is_ready check."""
        assert not await provider.ais_ready()
        await provider.aconnect()
        assert await provider.ais_ready()
        await provider.adisconnect()
        assert not await provider.ais_ready()

    @pytest.mark.asyncio
    async def test_create_collection(self, provider: ValkeyProvider):
        """Test index creation."""
        await provider.aconnect()
        assert not await provider.acollection_exists()
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        # Cleanup
        await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_create_collection_sync(self, provider: ValkeyProvider):
        """Test synchronous index creation."""
        provider.connect()
        assert not provider.collection_exists()
        provider.create_collection()
        assert provider.collection_exists()
        provider.delete_collection()
        provider.disconnect()

    @pytest.mark.asyncio
    async def test_delete_collection(self, provider: ValkeyProvider):
        """Test index deletion."""
        await provider.aconnect()
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adelete_collection()
        assert not await provider.acollection_exists()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection(self, provider: ValkeyProvider):
        """Test deleting non-existent index raises error."""
        await provider.aconnect()
        with pytest.raises((CollectionDoesNotExistError, VectorDBError)):
            await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_upsert_and_fetch(self, provider: ValkeyProvider):
        """Test upsert and fetch operations."""
        await provider.aconnect()
        await provider.acreate_collection()

        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS,
        )

        # Fetch by IDs
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        assert all(isinstance(r, VectorSearchResult) for r in results)

        # Verify content
        for result in results:
            assert result.id in SAMPLE_IDS[:2]
            assert result.text in SAMPLE_CHUNKS[:2]
            assert result.vector is not None
            assert len(result.vector) == 5
            assert result.payload is not None

        # Cleanup
        await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_upsert_sync(self, provider: ValkeyProvider):
        """Test synchronous upsert."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS,
        )
        results = provider.fetch(ids=SAMPLE_IDS[:1])
        assert len(results) == 1
        provider.delete_collection()
        provider.disconnect()

    @pytest.mark.asyncio
    async def test_upsert_with_document_tracking(self, provider: ValkeyProvider):
        """Test upsert with document_name, document_id, and hashes."""
        await provider.aconnect()
        await provider.acreate_collection()

        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_ids=["doc_id_1", "doc_id_2"],
            document_names=["doc_alpha", "doc_beta"],
        )

        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        for result in results:
            assert result.payload["document_id"] in ["doc_id_1", "doc_id_2"]
            assert result.payload["document_name"] in ["doc_alpha", "doc_beta"]

        await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_upsert_validation_error(self, provider: ValkeyProvider):
        """Test upsert with mismatched lengths raises ValueError."""
        await provider.aconnect()
        await provider.acreate_collection()
        with pytest.raises(ValueError):
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:3],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2],
            )
        await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_delete(self, provider: ValkeyProvider):
        """Test delete operation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS,
        )
        await provider.adelete(ids=SAMPLE_IDS[:2])
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        # Remaining records still exist
        results = await provider.afetch(ids=SAMPLE_IDS[2:])
        assert len(results) == 3
        await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_dense_search(self, provider: ValkeyProvider):
        """Test KNN vector similarity search."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS,
        )

        # Allow index to update
        await asyncio.sleep(0.5)

        results = await provider.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=3,
        )
        assert len(results) > 0
        assert len(results) <= 3
        assert all(isinstance(r, VectorSearchResult) for r in results)
        # Results should be sorted by score descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        # First result should be closest to query vector (which is near SAMPLE_VECTORS[0])
        assert results[0].id == SAMPLE_IDS[0]

        await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_dense_search_with_filter(self, provider: ValkeyProvider):
        """Test dense search with TAG filter."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS,
            document_names=["science_doc"] * 2 + ["literature_doc"] * 2 + ["philosophy_doc"],
        )

        await asyncio.sleep(0.5)

        results = await provider.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=5,
            filter={"document_name": "science_doc"},
        )
        # Should only return science docs
        assert len(results) <= 2
        for r in results:
            assert r.payload["document_name"] == "science_doc"

        await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_full_text_search(self, provider: ValkeyProvider):
        """Test full-text search."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS,
        )

        await asyncio.sleep(0.5)

        results = await provider.afull_text_search(
            query_text="relativity physics",
            top_k=3,
        )
        assert len(results) > 0
        # The first chunk mentions "relativity" and "physics"
        found_texts = [r.text for r in results]
        assert any("relativity" in t for t in found_texts)

        await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_hybrid_search(self, provider: ValkeyProvider):
        """Test hybrid search combining vector + text."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS,
        )

        await asyncio.sleep(0.5)

        results = await provider.ahybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics theory",
            top_k=3,
        )
        assert len(results) > 0
        assert len(results) <= 3
        # Scores should be normalized [0, 1]
        for r in results:
            assert 0.0 <= r.score <= 1.0

        await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_existence_checks(self, provider: ValkeyProvider):
        """Test existence check methods."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_ids=["doc_id_1"],
            document_names=["test_doc"],
            chunk_content_hashes=["hash_abc123"],
        )

        await asyncio.sleep(0.5)

        assert await provider.achunk_id_exists(SAMPLE_IDS[0])
        assert not await provider.achunk_id_exists("nonexistent")

        assert await provider.adocument_name_exists("test_doc")
        assert not await provider.adocument_name_exists("nonexistent")

        assert await provider.adocument_id_exists("doc_id_1")
        assert not await provider.adocument_id_exists("nonexistent")

        assert await provider.achunk_content_hash_exists("hash_abc123")
        assert not await provider.achunk_content_hash_exists("nonexistent")

        await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_delete_by_document_name(self, provider: ValkeyProvider):
        """Test delete_by_document_name."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_names=["target_doc", "target_doc"],
        )

        await asyncio.sleep(0.5)

        deleted = await provider.adelete_by_document_name("target_doc")
        assert deleted is True

        # Verify they're gone
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0

        await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, provider: ValkeyProvider):
        """Test delete_by_document_id."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_ids=["doc_to_delete", "doc_to_delete"],
        )

        await asyncio.sleep(0.5)

        deleted = await provider.adelete_by_document_id("doc_to_delete")
        assert deleted is True

        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0

        await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_update_metadata(self, provider: ValkeyProvider):
        """Test metadata update."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=[{"initial_key": "initial_value"}],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )

        success = await provider.aupdate_metadata(
            SAMPLE_IDS[0], {"new_key": "new_value"}
        )
        assert success is True

        # Verify
        results = await provider.afetch(ids=SAMPLE_IDS[:1])
        assert len(results) == 1
        meta = results[0].payload.get("metadata", {})
        assert meta.get("new_key") == "new_value"
        assert meta.get("initial_key") == "initial_value"

        await provider.adelete_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_supported_search_types(self, provider: ValkeyProvider):
        """Test get_supported_search_types."""
        await provider.aconnect()
        types = await provider.aget_supported_search_types()
        assert "dense" in types
        assert "full_text" in types
        assert "hybrid" in types
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_optimize(self, provider: ValkeyProvider):
        """Test optimize (no-op for Valkey)."""
        await provider.aconnect()
        result = await provider.aoptimize()
        assert result is True
        await provider.adisconnect()
