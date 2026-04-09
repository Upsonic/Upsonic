"""
Comprehensive smoke tests for ChromaDB vector database provider.

Tests all methods, attributes, and connection modes (IN_MEMORY, EMBEDDED, LOCAL, CLOUD).
"""

import os
import pytest
import tempfile
import asyncio
from typing import List, Dict, Any, Optional
from hashlib import md5

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system env vars

from upsonic.vectordb.providers.chroma import ChromaProvider
from upsonic.vectordb.config import (
    ChromaConfig,
    ConnectionConfig,
    Mode,
    DistanceMetric,
    HNSWIndexConfig,
    FlatIndexConfig
)
from upsonic.utils.package.exception import (
    VectorDBConnectionError,
    ConfigurationError,
    CollectionDoesNotExistError,
    VectorDBError,
    UpsertError
)
from upsonic.schemas.vector_schemas import VectorSearchResult


# Test data
SAMPLE_VECTORS: List[List[float]] = [
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.6, 0.7, 0.8, 0.9, 1.0],
    [1.1, 1.2, 1.3, 1.4, 1.5],
    [1.6, 1.7, 1.8, 1.9, 2.0],
    [2.1, 2.2, 2.3, 2.4, 2.5]
]

SAMPLE_PAYLOADS: List[Dict[str, Any]] = [
    {"category": "science", "author": "Einstein", "year": 1905},
    {"category": "science", "author": "Newton", "year": 1687},
    {"category": "literature", "author": "Shakespeare", "year": 1600},
    {"category": "literature", "author": "Dickens", "year": 1850},
    {"category": "philosophy", "author": "Plato", "year": -400}
]

SAMPLE_CHUNKS: List[str] = [
    "The theory of relativity revolutionized physics",
    "Laws of motion and universal gravitation",
    "To be or not to be, that is the question",
    "It was the best of times, it was the worst of times",
    "The unexamined life is not worth living"
]

SAMPLE_IDS: List[str] = ["doc1", "doc2", "doc3", "doc4", "doc5"]

QUERY_VECTOR: List[float] = [0.15, 0.25, 0.35, 0.45, 0.55]
QUERY_TEXT: str = "physics theory"


def assert_vector_matches(actual_vector: Any, expected_vector: List[float], vector_id: str = "", tolerance: float = 1e-6) -> None:
    """
    Assert that a retrieved vector matches the expected vector.
    
    Args:
        actual_vector: The vector retrieved from the database (can be list, numpy array, etc.)
        expected_vector: The original vector that was inserted
        vector_id: Optional ID for better error messages
        tolerance: Floating point comparison tolerance
    """
    assert actual_vector is not None, f"Vector is None for {vector_id}"
    assert hasattr(actual_vector, '__len__'), f"Vector has no length for {vector_id}"
    assert len(actual_vector) == len(expected_vector), \
        f"Vector length mismatch for {vector_id}: {len(actual_vector)} != {len(expected_vector)}"
    
    # Convert to list of floats (handles numpy arrays and other types)
    vector_list = [float(x) for x in actual_vector]
    assert len(vector_list) == len(expected_vector), \
        f"Converted vector length mismatch for {vector_id}: {len(vector_list)} != {len(expected_vector)}"
    
    # Compare element by element
    for j, (actual, expected) in enumerate(zip(vector_list, expected_vector)):
        assert abs(actual - expected) < tolerance, \
            f"Vector element {j} mismatch for {vector_id}: {actual} != {expected} (diff: {abs(actual - expected)})"


def assert_result_vector_matches(result: VectorSearchResult, expected_vector: List[float], result_index: int = 0) -> None:
    """
    Assert that a search result's vector matches the expected vector.
    
    Args:
        result: The VectorSearchResult from search/fetch operations
        expected_vector: The original vector that was inserted
        result_index: Index for better error messages
    """
    assert_vector_matches(result.vector, expected_vector, vector_id=f"result[{result_index}] (id={result.id})")


def get_expected_vector_by_id(record_id: str) -> List[float]:
    """
    Get the expected vector for a given record ID.
    
    Args:
        record_id: The ID of the record (e.g., "doc1", "doc2", etc.)
    
    Returns:
        The original vector that was inserted for this ID
    """
    if record_id in SAMPLE_IDS:
        idx = SAMPLE_IDS.index(record_id)
        return SAMPLE_VECTORS[idx]
    raise ValueError(f"Unknown record ID: {record_id}")


class TestChromaProviderIN_MEMORY:
    """Test ChromaProvider in IN_MEMORY mode."""
    
    @pytest.fixture
    def config(self, request) -> ChromaConfig:
        """Create IN_MEMORY ChromaConfig with unique collection name."""
        import uuid
        unique_name = f"test_memory_{uuid.uuid4().hex[:8]}"
        return ChromaConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(mode=Mode.IN_MEMORY),
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig(m=16, ef_construction=200)
        )
    
    @pytest.fixture
    def provider(self, config: ChromaConfig) -> ChromaProvider:
        """Create ChromaProvider instance."""
        return ChromaProvider(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, provider: ChromaProvider, config: ChromaConfig):
        """Test provider initialization and attributes."""
        assert provider._config == config
        assert provider._config.collection_name.startswith("test_memory_")
        assert provider._config.vector_size == 5
        assert provider._config.distance_metric == DistanceMetric.COSINE
        assert not provider._is_connected
        assert provider.client is None
        assert provider._collection_instance is None
        
        # Test provider metadata attributes
        assert provider.name is not None
        assert isinstance(provider.id, str)
        assert len(provider.id) > 0
        assert provider.reranker is None
    
    @pytest.mark.asyncio
    async def test_connect(self, provider: ChromaProvider):
        """Test connection to ChromaDB."""
        await provider.aconnect()
        assert provider._is_connected
        assert provider.client is not None
        assert await provider.ais_ready()
    
    @pytest.mark.asyncio
    async def test_connect_sync(self, provider: ChromaProvider):
        """Test synchronous connection."""
        provider.connect()
        assert provider._is_connected
        assert provider.client is not None
        assert provider.is_ready()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, provider: ChromaProvider):
        """Test disconnection."""
        await provider.aconnect()
        assert provider._is_connected
        await provider.adisconnect()
        assert not provider._is_connected
        assert provider.client is None
        assert provider._collection_instance is None
    
    @pytest.mark.asyncio
    async def test_disconnect_sync(self, provider: ChromaProvider):
        """Test synchronous disconnection."""
        provider.connect()
        assert provider._is_connected
        provider.disconnect()
        assert not provider._is_connected
    
    @pytest.mark.asyncio
    async def test_is_ready(self, provider: ChromaProvider):
        """Test is_ready check."""
        assert not await provider.ais_ready()
        await provider.aconnect()
        assert await provider.ais_ready()
        await provider.adisconnect()
        assert not await provider.ais_ready()
    
    @pytest.mark.asyncio
    async def test_is_ready_sync(self, provider: ChromaProvider):
        """Test synchronous is_ready check."""
        assert not provider.is_ready()
        provider.connect()
        assert provider.is_ready()
        provider.disconnect()
        assert not provider.is_ready()
    
    @pytest.mark.asyncio
    async def test_create_collection(self, provider: ChromaProvider):
        """Test collection creation."""
        await provider.aconnect()
        assert not await provider.acollection_exists()
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_create_collection_sync(self, provider: ChromaProvider):
        """Test synchronous collection creation."""
        provider.connect()
        # Delete collection if it exists first
        try:
            if provider.collection_exists():
                provider.delete_collection()
        except Exception:
            pass
        assert not provider.collection_exists()
        provider.create_collection()
        assert provider.collection_exists()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_collection_exists(self, provider: ChromaProvider):
        """Test collection existence check."""
        await provider.aconnect()
        # Delete collection if it exists first
        try:
            if await provider.acollection_exists():
                await provider.adelete_collection()
        except Exception:
            pass
        assert not await provider.acollection_exists()
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_collection_exists_sync(self, provider: ChromaProvider):
        """Test synchronous collection existence check."""
        provider.connect()
        # Delete collection if it exists first
        try:
            if provider.collection_exists():
                provider.delete_collection()
        except Exception:
            pass
        assert not provider.collection_exists()
        provider.create_collection()
        assert provider.collection_exists()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_collection(self, provider: ChromaProvider):
        """Test collection deletion."""
        await provider.aconnect()
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adelete_collection()
        assert not await provider.acollection_exists()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_collection_sync(self, provider: ChromaProvider):
        """Test synchronous collection deletion."""
        provider.connect()
        provider.create_collection()
        assert provider.collection_exists()
        provider.delete_collection()
        assert not provider.collection_exists()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection(self, provider: ChromaProvider):
        """Test deleting non-existent collection raises error."""
        await provider.aconnect()
        with pytest.raises(CollectionDoesNotExistError):
            await provider.adelete_collection()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert(self, provider: ChromaProvider):
        """Test upsert operation with content validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # Verify data was actually stored with correct content
        results = await provider.afetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.id == SAMPLE_IDS[i]
            assert result.payload is not None
            assert result.payload["metadata"].get("category") == SAMPLE_PAYLOADS[i]["category"]
            assert result.payload["metadata"].get("author") == SAMPLE_PAYLOADS[i]["author"]
            assert result.payload["metadata"].get("year") == SAMPLE_PAYLOADS[i]["year"]
            assert result.text == SAMPLE_CHUNKS[i]
            # Validate vector matches exactly
            assert_result_vector_matches(result, SAMPLE_VECTORS[i], result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_sync(self, provider: ChromaProvider):
        """Test synchronous upsert."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_with_document_tracking(self, provider: ChromaProvider):
        """Test upsert with document_name, document_id, content_id and validate metadata."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_names=["doc1", "doc2"],
            document_ids=["doc_id_1", "doc_id_2"],
        )
        # Verify tracking metadata was stored correctly
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        for i, result in enumerate(results):
            assert result.payload.get("document_name") == f"doc{i+1}"
            assert result.payload.get("document_id") == f"doc_id_{i+1}"
            assert result.payload.get("chunk_id") == SAMPLE_IDS[i]
            assert result.payload["metadata"].get("category") == SAMPLE_PAYLOADS[i]["category"]
            assert result.text == SAMPLE_CHUNKS[i]
            # Validate vector matches exactly
            assert_result_vector_matches(result, SAMPLE_VECTORS[i], result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_with_sparse_vectors_ignored(self, provider: ChromaProvider):
        """Test that sparse vectors are ignored (ChromaDB doesn't support them)."""
        await provider.aconnect()
        await provider.acreate_collection()
        sparse_vectors = [
            {"indices": [0, 2, 4], "values": [0.5, 0.3, 0.2]},
            {"indices": [1, 3], "values": [0.4, 0.6]}
        ]
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            sparse_vectors=sparse_vectors
        )
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_validation_error(self, provider: ChromaProvider):
        """Test upsert with mismatched lengths raises error."""
        await provider.aconnect()
        await provider.acreate_collection()
        with pytest.raises(UpsertError):
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:3],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2]
            )
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_fetch(self, provider: ChromaProvider):
        """Test fetch operation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert results[0].id == "doc1"
        assert results[0].payload is not None
        assert results[0].text is not None
        # Validate vectors match exactly
        for i, result in enumerate(results):
            assert_result_vector_matches(result, SAMPLE_VECTORS[i], result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_fetch_sync(self, provider: ChromaProvider):
        """Test synchronous fetch."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        assert all(isinstance(r, VectorSearchResult) for r in results)
        # Validate vectors match exactly
        for i, result in enumerate(results):
            assert_result_vector_matches(result, SAMPLE_VECTORS[i], result_index=i)
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_fetch_by_id_sync(self, provider: ChromaProvider):
        """Test fetch_by_id_sync alias."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.fetch_by_id(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        # Validate vectors match exactly
        for i, result in enumerate(results):
            assert_result_vector_matches(result, SAMPLE_VECTORS[i], result_index=i)
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete(self, provider: ChromaProvider):
        """Test delete operation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        await provider.adelete(ids=SAMPLE_IDS[:2])
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_sync(self, provider: ChromaProvider):
        """Test synchronous delete."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        provider.delete(ids=SAMPLE_IDS[:2])
        results = provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_id_sync(self, provider: ChromaProvider):
        """Test delete_by_id_sync alias."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        provider.delete_by_id(ids=SAMPLE_IDS[:2])
        results = provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_filter(self, provider: ChromaProvider):
        """Test delete_by_filter."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        deleted = await provider.adelete_by_metadata({"category": "science"})
        assert deleted is True
        count = await provider.aget_count()
        assert count == 3
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_document_name(self, provider: ChromaProvider):
        """Test delete_by_document_name."""
        await provider.aconnect()
        await provider.acreate_collection()
        # First ensure collection is empty
        initial_count = await provider.aget_count()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_names=["test_doc", "test_doc"]
        )
        count_before = await provider.aget_count()
        assert count_before == initial_count + 2
        deleted = provider.delete_by_document_name("test_doc")
        assert deleted is True
        count_after = await provider.aget_count()
        assert count_after == initial_count
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_document_name(self, provider: ChromaProvider):
        """Test async_delete_by_document_name."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_names=["test_doc", "test_doc"]
        )
        deleted = await provider.adelete_by_document_name("test_doc")
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, provider: ChromaProvider):
        """Test delete_by_document_id."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_ids=["doc_id_1", "doc_id_1"]
        )
        deleted = provider.delete_by_document_id("doc_id_1")
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_document_id(self, provider: ChromaProvider):
        """Test async_delete_by_document_id."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_ids=["doc_id_1", "doc_id_1"]
        )
        deleted = await provider.adelete_by_document_id("doc_id_1")
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_content_id(self, provider: ChromaProvider):
        """Test delete_by_chunk_id."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
        )
        deleted = provider.delete_by_chunk_id(SAMPLE_IDS[0])
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_content_id(self, provider: ChromaProvider):
        """Test async_delete_by_content_id."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
        )
        deleted = await provider.adelete_by_chunk_id(SAMPLE_IDS[0])
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_content_hash(self, provider: ChromaProvider):
        """Test delete_by_chunk_content_hash."""
        await provider.aconnect()
        await provider.acreate_collection()
        content_hash = md5(SAMPLE_CHUNKS[0].encode()).hexdigest()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        deleted = await provider.adelete_by_chunk_content_hash(content_hash)
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_metadata(self, provider: ChromaProvider):
        """Test delete_by_metadata."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        deleted = provider.delete_by_metadata({"category": "science"})
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_metadata(self, provider: ChromaProvider):
        """Test async_delete_by_metadata."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        deleted = await provider.adelete_by_metadata({"category": "science"})
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_id_exists(self, provider: ChromaProvider):
        """Test id_exists check."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.aid_exists("doc1")
        assert not await provider.aid_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_document_name_exists(self, provider: ChromaProvider):
        """Test document_name_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_names=["test_doc"]
        )
        assert provider.document_name_exists("test_doc")
        assert not provider.document_name_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_document_name_exists(self, provider: ChromaProvider):
        """Test async_document_name_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_names=["test_doc"]
        )
        assert await provider.adocument_name_exists("test_doc")
        assert not await provider.adocument_name_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_document_id_exists(self, provider: ChromaProvider):
        """Test document_id_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_ids=["doc_id_1"]
        )
        assert provider.document_id_exists("doc_id_1")
        assert not provider.document_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_document_id_exists(self, provider: ChromaProvider):
        """Test async_document_id_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_ids=["doc_id_1"]
        )
        assert await provider.adocument_id_exists("doc_id_1")
        assert not await provider.adocument_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_content_id_exists(self, provider: ChromaProvider):
        """Test chunk_id_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        assert provider.chunk_id_exists(SAMPLE_IDS[0])
        assert not provider.chunk_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_content_id_exists(self, provider: ChromaProvider):
        """Test async_content_id_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        assert await provider.achunk_id_exists(SAMPLE_IDS[0])
        assert not await provider.achunk_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_content_hash_exists(self, provider: ChromaProvider):
        """Test chunk_content_hash_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        content_hash = md5(SAMPLE_CHUNKS[0].encode()).hexdigest()
        assert await provider.achunk_content_hash_exists(content_hash)
        assert not await provider.achunk_content_hash_exists("nonexistent_hash")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_update_metadata(self, provider: ChromaProvider):
        """Test update_metadata."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        updated = provider.update_metadata(SAMPLE_IDS[0], {"new_field": "new_value"})
        assert updated is True
        results = await provider.afetch(ids=SAMPLE_IDS[:1])
        assert results[0].payload["metadata"].get("new_field") == "new_value"
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_update_metadata(self, provider: ChromaProvider):
        """Test async_update_metadata."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        updated = await provider.aupdate_metadata(SAMPLE_IDS[0], {"new_field": "new_value"})
        assert updated is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_get_count(self, provider: ChromaProvider):
        """Test get_count."""
        await provider.aconnect()
        await provider.acreate_collection()
        initial_count = await provider.aget_count()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        assert await provider.aget_count() == initial_count + 5
        await provider.adelete(ids=SAMPLE_IDS[:2])
        assert await provider.aget_count() == initial_count + 3
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_get_count_with_filter(self, provider: ChromaProvider):
        """Test get_count with filter."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        count = await provider.aget_count(filter={"category": "science"})
        assert count == 2
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_get_count_sync(self, provider: ChromaProvider):
        """Test synchronous get_count."""
        provider.connect()
        provider.create_collection()
        initial_count = provider.get_count()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        assert provider.get_count() == initial_count + 5
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_optimize(self, provider: ChromaProvider):
        """Test optimize operation."""
        result = provider.optimize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_async_optimize(self, provider: ChromaProvider):
        """Test async optimize."""
        result = await provider.aoptimize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_supported_search_types(self, provider: ChromaProvider):
        """Test get_supported_search_types."""
        supported = provider.get_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
        assert "full_text" in supported
        assert "hybrid" in supported
    
    @pytest.mark.asyncio
    async def test_async_get_supported_search_types(self, provider: ChromaProvider):
        """Test async_get_supported_search_types."""
        supported = await provider.aget_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
    
    @pytest.mark.asyncio
    async def test_dense_search(self, provider: ChromaProvider):
        """Test dense search."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert all(r.score >= 0.0 for r in results)
        # Validate vectors match exactly
        for i, result in enumerate(results):
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_dense_search_sync(self, provider: ChromaProvider):
        """Test synchronous dense search."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.dense_search(
            query_vector=QUERY_VECTOR,
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        # Validate vectors match exactly
        for i, result in enumerate(results):
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_full_text_search(self, provider: ChromaProvider):
        """Test full-text search."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.afull_text_search(
            query_text="physics",
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        # Validate vectors match exactly
        for i, result in enumerate(results):
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_full_text_search_sync(self, provider: ChromaProvider):
        """Test synchronous full-text search."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.full_text_search(
            query_text="physics",
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        # Validate vectors match exactly
        for i, result in enumerate(results):
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, provider: ChromaProvider):
        """Test hybrid search."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.ahybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            alpha=0.5,
            fusion_method="weighted",
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        # Validate vectors match exactly
        for i, result in enumerate(results):
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_rrf(self, provider: ChromaProvider):
        """Test hybrid search with RRF fusion."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.ahybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            fusion_method="rrf",
            similarity_threshold=0.0
        )
        assert len(results) > 0
        # Validate vectors match exactly
        for i, result in enumerate(results):
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_sync(self, provider: ChromaProvider):
        """Test synchronous hybrid search."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.hybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            alpha=0.5,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        # Validate vectors match exactly
        for i, result in enumerate(results):
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_search_master_method(self, provider: ChromaProvider):
        """Test master search method."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # Dense search
        results = await provider.asearch(
            query_vector=QUERY_VECTOR,
            top_k=3
        )
        assert len(results) > 0
        # Validate vectors match exactly
        for i, result in enumerate(results):
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        # Full-text search
        results = await provider.asearch(
            query_text="physics",
            top_k=3
        )
        assert len(results) > 0
        # Validate vectors match exactly
        for i, result in enumerate(results):
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        # Hybrid search
        results = await provider.asearch(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3
        )
        assert len(results) > 0
        # Validate vectors match exactly
        for i, result in enumerate(results):
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_search_sync(self, provider: ChromaProvider):
        """Test synchronous master search."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.search(
            query_vector=QUERY_VECTOR,
            top_k=3
        )
        assert len(results) > 0
        # Validate vectors match exactly
        for i, result in enumerate(results):
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_search_with_filter(self, provider: ChromaProvider):
        """Test search with metadata filter."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=5,
            filter={"category": "science"}
        )
        assert len(results) > 0
        assert all(r.payload["metadata"].get("category") == "science" for r in results)
        # Validate vectors match exactly
        for i, result in enumerate(results):
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_recreate_if_exists(self, provider: ChromaProvider):
        """Test recreate_if_exists configuration."""
        config = ChromaConfig(
            vector_size=5,
            collection_name="test_recreate",
            connection=ConnectionConfig(mode=Mode.IN_MEMORY),
            recreate_if_exists=True
        )
        provider2 = ChromaProvider(config)
        await provider2.aconnect()
        await provider2.acreate_collection()
        await provider2.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        # Create again with recreate_if_exists=True
        await provider2.acreate_collection()
        count = await provider2.aget_count()
        assert count == 0
        await provider2.adisconnect()
    
    @pytest.mark.asyncio
    async def test_flat_index_config(self, provider: ChromaProvider):
        """Test FlatIndexConfig."""
        config = ChromaConfig(
            vector_size=5,
            collection_name="test_flat",
            connection=ConnectionConfig(mode=Mode.IN_MEMORY),
            index=FlatIndexConfig()
        )
        provider2 = ChromaProvider(config)
        await provider2.aconnect()
        await provider2.acreate_collection()
        await provider2.adisconnect()
    
    @pytest.mark.asyncio
    async def test_distance_metrics(self, provider: ChromaProvider):
        """Test different distance metrics."""
        import uuid
        for metric in [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT]:
            unique_name = f"test_{metric.value}_{uuid.uuid4().hex[:8]}"
            config = ChromaConfig(
                vector_size=5,
                collection_name=unique_name,
                connection=ConnectionConfig(mode=Mode.IN_MEMORY),
                distance_metric=metric
            )
            provider2 = ChromaProvider(config)
            await provider2.aconnect()
            await provider2.acreate_collection()
            await provider2.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2]
            )
            results = await provider2.adense_search(
                query_vector=QUERY_VECTOR,
                top_k=2,
                similarity_threshold=0.0
            )
            assert len(results) > 0
            await provider2.adisconnect()


class TestChromaProviderEMBEDDED:
    """Comprehensive tests for ChromaProvider in EMBEDDED mode."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for embedded database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def config(self, temp_dir: str, request) -> ChromaConfig:
        """Create EMBEDDED ChromaConfig with unique collection name."""
        import uuid
        unique_name = f"test_embedded_{uuid.uuid4().hex[:8]}"
        db_path = os.path.join(temp_dir, "chroma_db")
        return ChromaConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=db_path),
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig(m=16, ef_construction=200)
        )
    
    @pytest.fixture
    def provider(self, config: ChromaConfig) -> ChromaProvider:
        """Create ChromaProvider instance."""
        return ChromaProvider(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, provider: ChromaProvider, config: ChromaConfig):
        """Test provider initialization and attributes with content validation."""
        assert provider._config == config
        assert provider._config.collection_name.startswith("test_embedded_")
        assert provider._config.vector_size == 5
        assert provider._config.distance_metric == DistanceMetric.COSINE
        assert not provider._is_connected
        assert provider.client is None
        assert provider._collection_instance is None
        assert provider.name is not None
        assert isinstance(provider.id, str)
        assert len(provider.id) > 0
        assert provider.reranker is None
    
    @pytest.mark.asyncio
    async def test_connect(self, provider: ChromaProvider):
        """Test connection to ChromaDB with validation."""
        await provider.aconnect()
        assert provider._is_connected is True
        assert provider.client is not None
        assert await provider.ais_ready() is True
    
    @pytest.mark.asyncio
    async def test_connect_sync(self, provider: ChromaProvider):
        """Test synchronous connection."""
        provider.connect()
        assert provider._is_connected is True
        assert provider.client is not None
        assert provider.is_ready() is True
    
    @pytest.mark.asyncio
    async def test_disconnect(self, provider: ChromaProvider):
        """Test disconnection."""
        await provider.aconnect()
        assert provider._is_connected is True
        await provider.adisconnect()
        assert provider._is_connected is False
        assert provider.client is None
        assert provider._collection_instance is None
    
    @pytest.mark.asyncio
    async def test_is_ready(self, provider: ChromaProvider):
        """Test is_ready check."""
        assert await provider.ais_ready() is False
        await provider.aconnect()
        assert await provider.ais_ready() is True
        await provider.adisconnect()
        assert await provider.ais_ready() is False
    
    @pytest.mark.asyncio
    async def test_create_collection(self, provider: ChromaProvider):
        """Test collection creation."""
        await provider.aconnect()
        try:
            if await provider.acollection_exists():
                await provider.adelete_collection()
        except Exception:
            pass
        assert not await provider.acollection_exists()
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_collection_exists(self, provider: ChromaProvider):
        """Test collection existence check."""
        await provider.aconnect()
        try:
            if await provider.acollection_exists():
                await provider.adelete_collection()
        except Exception:
            pass
        assert not await provider.acollection_exists()
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_collection(self, provider: ChromaProvider):
        """Test collection deletion."""
        await provider.aconnect()
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adelete_collection()
        assert not await provider.acollection_exists()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert(self, provider: ChromaProvider):
        """Test upsert operation with content validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # Verify data was actually stored
        results = await provider.afetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.id == SAMPLE_IDS[i]
            assert result.payload is not None
            assert result.payload["metadata"].get("category") == SAMPLE_PAYLOADS[i]["category"]
            assert result.payload["metadata"].get("author") == SAMPLE_PAYLOADS[i]["author"]
            assert result.payload["metadata"].get("year") == SAMPLE_PAYLOADS[i]["year"]
            assert result.text == SAMPLE_CHUNKS[i]
            assert result.vector is not None
            assert len(result.vector) == 5
            # Test that vectors match (with tolerance for floating-point precision in persistent storage)
            assert_result_vector_matches(result, SAMPLE_VECTORS[i], result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_with_document_tracking(self, provider: ChromaProvider):
        """Test upsert with document tracking and validate metadata."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_names=["doc1", "doc2"],
            document_ids=["doc_id_1", "doc_id_2"],
        )
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        for i, result in enumerate(results):
            assert result.payload.get("document_name") == f"doc{i+1}"
            assert result.payload.get("document_id") == f"doc_id_{i+1}"
            assert result.payload.get("chunk_id") == SAMPLE_IDS[i]
            # Validate vector matches exactly
            assert_result_vector_matches(result, SAMPLE_VECTORS[i], result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_fetch(self, provider: ChromaProvider):
        """Test fetch operation with detailed content validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.afetch(ids=SAMPLE_IDS[:3])
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, VectorSearchResult)
            assert result.id == SAMPLE_IDS[i]
            assert result.score == 1.0
            assert result.payload is not None
            assert isinstance(result.payload, dict)
            assert result.payload["metadata"]["category"] == SAMPLE_PAYLOADS[i]["category"]
            assert result.payload["metadata"]["author"] == SAMPLE_PAYLOADS[i]["author"]
            assert result.payload["metadata"]["year"] == SAMPLE_PAYLOADS[i]["year"]
            assert result.text == SAMPLE_CHUNKS[i]
            # Validate vector matches exactly
            assert_result_vector_matches(result, SAMPLE_VECTORS[i], result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete(self, provider: ChromaProvider):
        """Test delete operation with validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        await provider.adelete(ids=SAMPLE_IDS[:2])
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        # Verify remaining records still exist
        results = await provider.afetch(ids=SAMPLE_IDS[2:])
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.id == SAMPLE_IDS[i+2]
            assert result.payload["metadata"]["category"] == SAMPLE_PAYLOADS[i+2]["category"]
            # Validate vector matches exactly
            assert_result_vector_matches(result, SAMPLE_VECTORS[i+2], result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_filter(self, provider: ChromaProvider):
        """Test delete_by_filter with content validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        deleted = await provider.adelete_by_metadata({"category": "science"})
        assert deleted is True
        # Verify only science documents were deleted
        results = await provider.afetch(ids=SAMPLE_IDS)
        assert len(results) == 3
        for result in results:
            assert result.payload["metadata"].get("category") != "science"
            # Validate vector matches exactly
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=0)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_dense_search(self, provider: ChromaProvider):
        """Test dense search with detailed result validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for i, result in enumerate(results):
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert isinstance(result.score, float)
            assert 0.0 <= result.score <= 1.0
            assert result.payload is not None
            assert "category" in result.payload["metadata"]
            assert "author" in result.payload["metadata"]
            assert result.text is not None
            assert result.text in SAMPLE_CHUNKS
            # Validate vector matches exactly
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        # Verify results are sorted by score (descending)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_full_text_search(self, provider: ChromaProvider):
        """Test full-text search with content validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.afull_text_search(
            query_text="physics",
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        for i, result in enumerate(results):
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert isinstance(result.score, float)
            assert 0.0 <= result.score <= 1.0
            assert result.payload is not None
            assert result.text is not None
            assert "physics" in result.text.lower() or "theory" in result.text.lower()
            # Validate vector matches exactly
            expected_vector = get_expected_vector_by_id(result.id)
            assert_result_vector_matches(result, expected_vector, result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, provider: ChromaProvider):
        """Test hybrid search with detailed validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.ahybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            alpha=0.5,
            fusion_method="weighted",
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert isinstance(result.score, float)
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
            assert result.vector is not None
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_search_with_filter(self, provider: ChromaProvider):
        """Test search with metadata filter and validate filtered results."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=5,
            filter={"category": "science"}
        )
        assert len(results) > 0
        for result in results:
            assert result.payload["metadata"].get("category") == "science"
            assert result.payload["metadata"].get("author") in ["Einstein", "Newton"]
            assert result.id in ["doc1", "doc2"]
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_get_count(self, provider: ChromaProvider):
        """Test get_count with validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        initial_count = await provider.aget_count()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        assert await provider.aget_count() == initial_count + 5
        await provider.adelete(ids=SAMPLE_IDS[:2])
        assert await provider.aget_count() == initial_count + 3
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_update_metadata(self, provider: ChromaProvider):
        """Test update_metadata with content validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        updated = provider.update_metadata(SAMPLE_IDS[0], {"new_field": "new_value", "updated": True})
        assert updated is True
        results = await provider.afetch(ids=SAMPLE_IDS[:1])
        assert len(results) == 1
        assert results[0].payload["metadata"].get("new_field") == "new_value"
        assert results[0].payload["metadata"].get("updated") is True
        assert results[0].payload["metadata"].get("category") == SAMPLE_PAYLOADS[0]["category"]
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_embedded_persistence(self, provider: ChromaProvider, temp_dir: str):
        """Test that embedded mode persists data with full validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # Verify data before disconnect
        results = await provider.afetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        await provider.adisconnect()
        
        # Reconnect and verify data persists with exact content
        await provider.aconnect()
        assert await provider.acollection_exists()
        count = await provider.aget_count()
        assert count == 5
        results = await provider.afetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.id == SAMPLE_IDS[i]
            assert result.payload["metadata"]["category"] == SAMPLE_PAYLOADS[i]["category"]
            assert result.payload["metadata"]["author"] == SAMPLE_PAYLOADS[i]["author"]
            assert result.text == SAMPLE_CHUNKS[i]
            # Test that vectors match (with tolerance for floating-point precision in persistent storage)
            assert_result_vector_matches(result, SAMPLE_VECTORS[i], result_index=i)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_disconnect_sync(self, provider: ChromaProvider):
        """Test synchronous disconnection."""
        provider.connect()
        assert provider._is_connected is True
        provider.disconnect()
        assert provider._is_connected is False
    
    @pytest.mark.asyncio
    async def test_is_ready_sync(self, provider: ChromaProvider):
        """Test synchronous is_ready check."""
        assert provider.is_ready() is False
        provider.connect()
        assert provider.is_ready() is True
        provider.disconnect()
        assert provider.is_ready() is False
    
    @pytest.mark.asyncio
    async def test_create_collection_sync(self, provider: ChromaProvider):
        """Test synchronous collection creation."""
        provider.connect()
        try:
            if provider.collection_exists():
                provider.delete_collection()
        except Exception:
            pass
        assert not provider.collection_exists()
        provider.create_collection()
        assert provider.collection_exists()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_collection_exists_sync(self, provider: ChromaProvider):
        """Test synchronous collection existence check."""
        provider.connect()
        try:
            if provider.collection_exists():
                provider.delete_collection()
        except Exception:
            pass
        assert not provider.collection_exists()
        provider.create_collection()
        assert provider.collection_exists()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_collection_sync(self, provider: ChromaProvider):
        """Test synchronous collection deletion."""
        provider.connect()
        provider.create_collection()
        assert provider.collection_exists()
        provider.delete_collection()
        assert not provider.collection_exists()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection(self, provider: ChromaProvider):
        """Test deleting non-existent collection raises error."""
        await provider.aconnect()
        with pytest.raises(CollectionDoesNotExistError):
            await provider.adelete_collection()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_sync(self, provider: ChromaProvider):
        """Test synchronous upsert with content validation."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # Verify data was actually stored
        results = provider.fetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.id == SAMPLE_IDS[i]
            assert result.payload["metadata"].get("category") == SAMPLE_PAYLOADS[i]["category"]
            assert result.payload["metadata"].get("author") == SAMPLE_PAYLOADS[i]["author"]
            assert result.text == SAMPLE_CHUNKS[i]
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_with_sparse_vectors_ignored(self, provider: ChromaProvider):
        """Test that sparse vectors are ignored (ChromaDB doesn't support them)."""
        await provider.aconnect()
        await provider.acreate_collection()
        sparse_vectors = [
            {"indices": [0, 2, 4], "values": [0.5, 0.3, 0.2]},
            {"indices": [1, 3], "values": [0.4, 0.6]}
        ]
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            sparse_vectors=sparse_vectors
        )
        # Verify data was stored correctly despite sparse vectors
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_validation_error(self, provider: ChromaProvider):
        """Test upsert with mismatched lengths raises error."""
        await provider.aconnect()
        await provider.acreate_collection()
        with pytest.raises(UpsertError):
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:3],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2]
            )
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_fetch_sync(self, provider: ChromaProvider):
        """Test synchronous fetch with content validation."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert results[0].id == "doc1"
        assert results[0].payload is not None
        assert results[0].payload["metadata"].get("category") == SAMPLE_PAYLOADS[0]["category"]
        assert results[0].text == SAMPLE_CHUNKS[0]
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_fetch_by_id_sync(self, provider: ChromaProvider):
        """Test fetch_by_id_sync alias with content validation."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.fetch_by_id(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        for i, result in enumerate(results):
            assert result.id == SAMPLE_IDS[i]
            assert result.payload["metadata"].get("category") == SAMPLE_PAYLOADS[i]["category"]
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_sync(self, provider: ChromaProvider):
        """Test synchronous delete with validation."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        provider.delete(ids=SAMPLE_IDS[:2])
        results = provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        # Verify remaining records
        results = provider.fetch(ids=SAMPLE_IDS[2:])
        assert len(results) == 3
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_id_sync(self, provider: ChromaProvider):
        """Test delete_by_id_sync alias."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        provider.delete_by_id(ids=SAMPLE_IDS[:2])
        results = provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_document_name(self, provider: ChromaProvider):
        """Test delete_by_document_name with validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        initial_count = await provider.aget_count()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_names=["test_doc", "test_doc"]
        )
        count_before = await provider.aget_count()
        assert count_before == initial_count + 2
        deleted = provider.delete_by_document_name("test_doc")
        assert deleted is True
        count_after = await provider.aget_count()
        assert count_after == initial_count
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_document_name(self, provider: ChromaProvider):
        """Test async_delete_by_document_name."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_names=["test_doc", "test_doc"]
        )
        deleted = await provider.adelete_by_document_name("test_doc")
        assert deleted is True
        count = await provider.aget_count()
        assert count == 0
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, provider: ChromaProvider):
        """Test delete_by_document_id with validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_ids=["doc_id_1", "doc_id_1"]
        )
        deleted = provider.delete_by_document_id("doc_id_1")
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_document_id(self, provider: ChromaProvider):
        """Test async_delete_by_document_id."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_ids=["doc_id_1", "doc_id_1"]
        )
        deleted = await provider.adelete_by_document_id("doc_id_1")
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_content_id(self, provider: ChromaProvider):
        """Test delete_by_chunk_id with validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
        )
        deleted = provider.delete_by_chunk_id(SAMPLE_IDS[0])
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 1
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_async_delete_by_content_id(self, provider: ChromaProvider):
        """Test async_delete_by_content_id."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
        )
        deleted = await provider.adelete_by_chunk_id(SAMPLE_IDS[0])
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_content_hash(self, provider: ChromaProvider):
        """Test delete_by_chunk_content_hash with validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        content_hash = md5(SAMPLE_CHUNKS[0].encode()).hexdigest()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        deleted = await provider.adelete_by_chunk_content_hash(content_hash)
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS[:1])
        assert len(results) == 0
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_metadata(self, provider: ChromaProvider):
        """Test delete_by_metadata with validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        deleted = provider.delete_by_metadata({"category": "science"})
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS)
        assert len(results) == 3
        for result in results:
            assert result.payload["metadata"].get("category") != "science"
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_metadata(self, provider: ChromaProvider):
        """Test async_delete_by_metadata."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        deleted = await provider.adelete_by_metadata({"category": "science"})
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_id_exists(self, provider: ChromaProvider):
        """Test id_exists check."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.aid_exists("doc1")
        assert not await provider.aid_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_document_name_exists(self, provider: ChromaProvider):
        """Test document_name_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_names=["test_doc"]
        )
        assert provider.document_name_exists("test_doc")
        assert not provider.document_name_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_document_name_exists(self, provider: ChromaProvider):
        """Test async_document_name_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_names=["test_doc"]
        )
        assert await provider.adocument_name_exists("test_doc")
        assert not await provider.adocument_name_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_document_id_exists(self, provider: ChromaProvider):
        """Test document_id_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_ids=["doc_id_1"]
        )
        assert provider.document_id_exists("doc_id_1")
        assert not provider.document_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_document_id_exists(self, provider: ChromaProvider):
        """Test async_document_id_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_ids=["doc_id_1"]
        )
        assert await provider.adocument_id_exists("doc_id_1")
        assert not await provider.adocument_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_content_id_exists(self, provider: ChromaProvider):
        """Test chunk_id_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        assert provider.chunk_id_exists(SAMPLE_IDS[0])
        assert not provider.chunk_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_content_id_exists(self, provider: ChromaProvider):
        """Test async_content_id_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        assert await provider.achunk_id_exists(SAMPLE_IDS[0])
        assert not await provider.achunk_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_content_hash_exists(self, provider: ChromaProvider):
        """Test chunk_content_hash_exists."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        content_hash = md5(SAMPLE_CHUNKS[0].encode()).hexdigest()
        assert await provider.achunk_content_hash_exists(content_hash)
        assert not await provider.achunk_content_hash_exists("nonexistent_hash")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_update_metadata(self, provider: ChromaProvider):
        """Test async_update_metadata with validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        updated = await provider.aupdate_metadata(SAMPLE_IDS[0], {"new_field": "new_value", "updated": True})
        assert updated is True
        results = await provider.afetch(ids=SAMPLE_IDS[:1])
        assert results[0].payload["metadata"].get("new_field") == "new_value"
        assert results[0].payload["metadata"].get("updated") is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_get_count_with_filter(self, provider: ChromaProvider):
        """Test get_count with filter."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        count = await provider.aget_count(filter={"category": "science"})
        assert count == 2
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_get_count_sync(self, provider: ChromaProvider):
        """Test synchronous get_count."""
        provider.connect()
        provider.create_collection()
        initial_count = provider.get_count()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        assert provider.get_count() == initial_count + 5
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_optimize(self, provider: ChromaProvider):
        """Test optimize operation."""
        result = provider.optimize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_async_optimize(self, provider: ChromaProvider):
        """Test async optimize."""
        result = await provider.aoptimize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_supported_search_types(self, provider: ChromaProvider):
        """Test get_supported_search_types."""
        supported = provider.get_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
        assert "full_text" in supported
        assert "hybrid" in supported
    
    @pytest.mark.asyncio
    async def test_async_get_supported_search_types(self, provider: ChromaProvider):
        """Test async_get_supported_search_types."""
        supported = await provider.aget_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
        assert "full_text" in supported
        assert "hybrid" in supported
    
    @pytest.mark.asyncio
    async def test_dense_search_sync(self, provider: ChromaProvider):
        """Test synchronous dense search with content validation."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.dense_search(
            query_vector=QUERY_VECTOR,
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert isinstance(result.score, float)
            assert 0.0 <= result.score <= 1.0
            assert result.payload is not None
            assert result.text is not None
            assert result.vector is not None
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_full_text_search_sync(self, provider: ChromaProvider):
        """Test synchronous full-text search with content validation."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.full_text_search(
            query_text="physics",
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert isinstance(result.score, float)
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
            assert "physics" in result.text.lower() or "theory" in result.text.lower()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_rrf(self, provider: ChromaProvider):
        """Test hybrid search with RRF fusion."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.ahybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            fusion_method="rrf",
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_sync(self, provider: ChromaProvider):
        """Test synchronous hybrid search with validation."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.hybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            alpha=0.5,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert result.score >= 0.0
            assert result.payload is not None
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_search_master_method(self, provider: ChromaProvider):
        """Test master search method with content validation."""
        await provider.aconnect()
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # Dense search
        results = await provider.asearch(
            query_vector=QUERY_VECTOR,
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert all(r.id in SAMPLE_IDS for r in results)
        # Full-text search
        results = await provider.asearch(
            query_text="physics",
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        # Hybrid search
        results = await provider.asearch(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_search_sync(self, provider: ChromaProvider):
        """Test synchronous master search with validation."""
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.search(
            query_vector=QUERY_VECTOR,
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert all(r.payload is not None for r in results)
        assert all(r.text is not None for r in results)
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_recreate_if_exists(self, provider: ChromaProvider, temp_dir: str):
        """Test recreate_if_exists configuration."""
        import uuid
        unique_name = f"test_recreate_{uuid.uuid4().hex[:8]}"
        db_path = os.path.join(temp_dir, "chroma_db_recreate")
        config = ChromaConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=db_path),
            recreate_if_exists=True
        )
        provider2 = ChromaProvider(config)
        await provider2.aconnect()
        await provider2.acreate_collection()
        await provider2.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        # Create again with recreate_if_exists=True
        await provider2.acreate_collection()
        count = await provider2.aget_count()
        assert count == 0
        await provider2.adisconnect()
    
    @pytest.mark.asyncio
    async def test_flat_index_config(self, provider: ChromaProvider, temp_dir: str):
        """Test FlatIndexConfig."""
        import uuid
        unique_name = f"test_flat_{uuid.uuid4().hex[:8]}"
        db_path = os.path.join(temp_dir, "chroma_db_flat")
        config = ChromaConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=db_path),
            index=FlatIndexConfig()
        )
        provider2 = ChromaProvider(config)
        await provider2.aconnect()
        await provider2.acreate_collection()
        await provider2.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        results = await provider2.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=2,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        await provider2.adisconnect()
    
    @pytest.mark.asyncio
    async def test_distance_metrics(self, provider: ChromaProvider, temp_dir: str):
        """Test different distance metrics."""
        import uuid
        for metric in [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT]:
            unique_name = f"test_{metric.value}_{uuid.uuid4().hex[:8]}"
            db_path = os.path.join(temp_dir, f"chroma_db_{metric.value}")
            config = ChromaConfig(
                vector_size=5,
                collection_name=unique_name,
                connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=db_path),
                distance_metric=metric
            )
            provider2 = ChromaProvider(config)
            await provider2.aconnect()
            await provider2.acreate_collection()
            await provider2.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2]
            )
            results = await provider2.adense_search(
                query_vector=QUERY_VECTOR,
                top_k=2,
                similarity_threshold=0.0
            )
            assert len(results) > 0
            assert all(isinstance(r, VectorSearchResult) for r in results)
            assert all(r.score >= 0.0 for r in results)
            await provider2.adisconnect()


class TestChromaProviderLOCAL:
    """Comprehensive tests for ChromaProvider in LOCAL mode (requires running ChromaDB server)."""
    
    @pytest.fixture
    def config(self, request) -> Optional[ChromaConfig]:
        """Create LOCAL ChromaConfig if server available."""
        import uuid
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        unique_name = f"test_local_{uuid.uuid4().hex[:8]}"
        return ChromaConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(mode=Mode.LOCAL, host=host, port=port),
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig(m=16, ef_construction=200)
        )
    
    @pytest.fixture
    def provider(self, config: Optional[ChromaConfig]) -> Optional[ChromaProvider]:
        """Create ChromaProvider instance."""
        if config is None:
            return None
        return ChromaProvider(config)
    
    def _skip_if_unavailable(self, provider: Optional[ChromaProvider]):
        """Helper to skip tests if provider is not available."""
        if provider is None:
            pytest.skip("ChromaDB server not available")
    
    async def _ensure_connected(self, provider: ChromaProvider):
        """Helper to ensure connection, skip if unavailable."""
        try:
            await provider.aconnect()
            return True
        except VectorDBConnectionError:
            pytest.skip("ChromaDB server not accessible")
    
    def _ensure_connected_sync(self, provider: ChromaProvider):
        """Helper to ensure sync connection, skip if unavailable."""
        try:
            provider.connect()
            return True
        except VectorDBConnectionError:
            pytest.skip("ChromaDB server not accessible")
    
    @pytest.mark.asyncio
    async def test_initialization(self, provider: Optional[ChromaProvider], config: Optional[ChromaConfig]):
        """Test provider initialization and attributes."""
        self._skip_if_unavailable(provider)
        assert provider._config == config
        assert provider._config.collection_name.startswith("test_local_")
        assert provider._config.vector_size == 5
        assert provider._config.distance_metric == DistanceMetric.COSINE
        assert not provider._is_connected
        assert provider.client is None
    
    @pytest.mark.asyncio
    async def test_connect(self, provider: Optional[ChromaProvider]):
        """Test connection to ChromaDB."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        assert provider._is_connected is True
        assert provider.client is not None
        assert await provider.ais_ready() is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_create_collection(self, provider: Optional[ChromaProvider]):
        """Test collection creation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        try:
            if await provider.acollection_exists():
                await provider.adelete_collection()
        except Exception:
            pass
        assert not await provider.acollection_exists()
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert(self, provider: Optional[ChromaProvider]):
        """Test upsert operation with content validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.afetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.id == SAMPLE_IDS[i]
            assert result.payload["metadata"]["category"] == SAMPLE_PAYLOADS[i]["category"]
            assert result.payload["metadata"]["author"] == SAMPLE_PAYLOADS[i]["author"]
            assert result.text == SAMPLE_CHUNKS[i]
            # Test that vectors are exactly the same
            vector_list = [float(x) for x in result.vector]
            assert vector_list == SAMPLE_VECTORS[i], f"Vector mismatch for ID {result.id}: {vector_list} != {SAMPLE_VECTORS[i]}"
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_fetch(self, provider: Optional[ChromaProvider]):
        """Test fetch operation with detailed validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.afetch(ids=SAMPLE_IDS[:3])
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, VectorSearchResult)
            assert result.id == SAMPLE_IDS[i]
            assert result.score == 1.0
            assert result.payload is not None
            assert result.payload["metadata"]["category"] == SAMPLE_PAYLOADS[i]["category"]
            assert result.text == SAMPLE_CHUNKS[i]
            # Test that vectors are exactly the same
            vector_list = [float(x) for x in result.vector]
            assert vector_list == SAMPLE_VECTORS[i], f"Vector mismatch for ID {result.id}: {vector_list} != {SAMPLE_VECTORS[i]}"
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete(self, provider: Optional[ChromaProvider]):
        """Test delete operation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        await provider.adelete(ids=SAMPLE_IDS[:2])
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        results = await provider.afetch(ids=SAMPLE_IDS[2:])
        assert len(results) == 3
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_dense_search(self, provider: Optional[ChromaProvider]):
        """Test dense search with detailed result validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert isinstance(result.score, float)
            assert 0.0 <= result.score <= 1.0
            assert result.payload is not None
            assert result.text is not None
            assert result.vector is not None
            assert len(result.vector) == 5
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_full_text_search(self, provider: Optional[ChromaProvider]):
        """Test full-text search with content validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.afull_text_search(
            query_text="physics",
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
            assert "physics" in result.text.lower() or "theory" in result.text.lower()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, provider: Optional[ChromaProvider]):
        """Test hybrid search with detailed validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.ahybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            alpha=0.5,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
            assert result.vector is not None
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_search_with_filter(self, provider: Optional[ChromaProvider]):
        """Test search with metadata filter."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=5,
            filter={"category": "science"}
        )
        assert len(results) > 0
        for result in results:
            assert result.payload["metadata"].get("category") == "science"
            assert result.id in ["doc1", "doc2"]
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_get_count(self, provider: Optional[ChromaProvider]):
        """Test get_count."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        initial_count = await provider.aget_count()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        assert await provider.aget_count() == initial_count + 5
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_update_metadata(self, provider: Optional[ChromaProvider]):
        """Test update_metadata with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        updated = provider.update_metadata(SAMPLE_IDS[0], {"new_field": "new_value"})
        assert updated is True
        results = await provider.afetch(ids=SAMPLE_IDS[:1])
        assert results[0].payload["metadata"].get("new_field") == "new_value"
        assert results[0].payload["metadata"].get("category") == SAMPLE_PAYLOADS[0]["category"]
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_filter(self, provider: Optional[ChromaProvider]):
        """Test delete_by_filter."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        deleted = await provider.adelete_by_metadata({"category": "science"})
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS)
        for result in results:
            assert result.payload["metadata"].get("category") != "science"
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_connect_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous connection."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        assert provider._is_connected is True
        assert provider.client is not None
        assert provider.is_ready() is True
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, provider: Optional[ChromaProvider]):
        """Test disconnection."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        assert provider._is_connected is True
        await provider.adisconnect()
        assert provider._is_connected is False
        assert provider.client is None
        assert provider._collection_instance is None
    
    @pytest.mark.asyncio
    async def test_disconnect_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous disconnection."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        assert provider._is_connected is True
        provider.disconnect()
        assert provider._is_connected is False
    
    @pytest.mark.asyncio
    async def test_is_ready(self, provider: Optional[ChromaProvider]):
        """Test is_ready check."""
        self._skip_if_unavailable(provider)
        assert await provider.ais_ready() is False
        await self.a_ensure_connected(provider)
        assert await provider.ais_ready() is True
        await provider.adisconnect()
        assert await provider.ais_ready() is False
    
    @pytest.mark.asyncio
    async def test_is_ready_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous is_ready check."""
        self._skip_if_unavailable(provider)
        assert provider.is_ready() is False
        self._ensure_connected_sync(provider)
        assert provider.is_ready() is True
        provider.disconnect()
        assert provider.is_ready() is False
    
    @pytest.mark.asyncio
    async def test_create_collection_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous collection creation."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        try:
            if provider.collection_exists():
                provider.delete_collection()
        except Exception:
            pass
        assert not provider.collection_exists()
        provider.create_collection()
        assert provider.collection_exists()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_collection_exists(self, provider: Optional[ChromaProvider]):
        """Test collection existence check."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        try:
            if await provider.acollection_exists():
                await provider.adelete_collection()
        except Exception:
            pass
        assert not await provider.acollection_exists()
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_collection_exists_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous collection existence check."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        try:
            if provider.collection_exists():
                provider.delete_collection()
        except Exception:
            pass
        assert not provider.collection_exists()
        provider.create_collection()
        assert provider.collection_exists()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_collection(self, provider: Optional[ChromaProvider]):
        """Test collection deletion."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adelete_collection()
        assert not await provider.acollection_exists()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_collection_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous collection deletion."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        provider.create_collection()
        assert provider.collection_exists()
        provider.delete_collection()
        assert not provider.collection_exists()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection(self, provider: Optional[ChromaProvider]):
        """Test deleting non-existent collection raises error."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        with pytest.raises(CollectionDoesNotExistError):
            await provider.adelete_collection()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous upsert with content validation."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.fetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.id == SAMPLE_IDS[i]
            assert result.payload["metadata"].get("category") == SAMPLE_PAYLOADS[i]["category"]
            assert result.payload["metadata"].get("author") == SAMPLE_PAYLOADS[i]["author"]
            assert result.text == SAMPLE_CHUNKS[i]
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_with_document_tracking(self, provider: Optional[ChromaProvider]):
        """Test upsert with document tracking and validate metadata."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_names=["local_doc1", "local_doc2"],
            document_ids=["local_doc_id_1", "local_doc_id_2"],
        )
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        for i, result in enumerate(results):
            assert result.payload.get("document_name") == f"local_doc{i+1}"
            assert result.payload.get("document_id") == f"local_doc_id_{i+1}"
            assert result.payload.get("chunk_id") == f"local_content_{i+1}"
            assert result.payload["metadata"].get("category") == SAMPLE_PAYLOADS[i]["category"]
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_with_sparse_vectors_ignored(self, provider: Optional[ChromaProvider]):
        """Test that sparse vectors are ignored (ChromaDB doesn't support them)."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        sparse_vectors = [
            {"indices": [0, 2, 4], "values": [0.5, 0.3, 0.2]},
            {"indices": [1, 3], "values": [0.4, 0.6]}
        ]
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            sparse_vectors=sparse_vectors
        )
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_validation_error(self, provider: Optional[ChromaProvider]):
        """Test upsert with mismatched lengths raises error."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        with pytest.raises(UpsertError):
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:3],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2]
            )
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_fetch_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous fetch with content validation."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.fetch(ids=SAMPLE_IDS[:3])
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, VectorSearchResult)
            assert result.id == SAMPLE_IDS[i]
            assert result.score == 1.0
            assert result.payload is not None
            assert result.payload["metadata"]["category"] == SAMPLE_PAYLOADS[i]["category"]
            assert result.payload["metadata"]["author"] == SAMPLE_PAYLOADS[i]["author"]
            assert result.text == SAMPLE_CHUNKS[i]
            assert result.vector is not None
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_fetch_by_id_sync(self, provider: Optional[ChromaProvider]):
        """Test fetch_by_id_sync alias with content validation."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.fetch_by_id(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        for i, result in enumerate(results):
            assert result.id == SAMPLE_IDS[i]
            assert result.payload["metadata"].get("category") == SAMPLE_PAYLOADS[i]["category"]
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous delete with validation."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        provider.delete(ids=SAMPLE_IDS[:2])
        results = provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        results = provider.fetch(ids=SAMPLE_IDS[2:])
        assert len(results) == 3
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_id_sync(self, provider: Optional[ChromaProvider]):
        """Test delete_by_id_sync alias."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        provider.delete_by_id(ids=SAMPLE_IDS[:2])
        results = provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_document_name(self, provider: Optional[ChromaProvider]):
        """Test delete_by_document_name with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        initial_count = await provider.aget_count()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_names=["local_doc", "local_doc"]
        )
        deleted = provider.delete_by_document_name("local_doc")
        assert deleted is True
        count = await provider.aget_count()
        assert count == initial_count
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_document_name(self, provider: Optional[ChromaProvider]):
        """Test async_delete_by_document_name."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_names=["local_doc", "local_doc"]
        )
        deleted = await provider.adelete_by_document_name("local_doc")
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, provider: Optional[ChromaProvider]):
        """Test delete_by_document_id with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_ids=["local_doc_id_1", "local_doc_id_1"]
        )
        deleted = provider.delete_by_document_id("local_doc_id_1")
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_document_id(self, provider: Optional[ChromaProvider]):
        """Test async_delete_by_document_id."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_ids=["local_doc_id_1", "local_doc_id_1"]
        )
        deleted = await provider.adelete_by_document_id("local_doc_id_1")
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_content_id(self, provider: Optional[ChromaProvider]):
        """Test delete_by_chunk_id with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
        )
        deleted = provider.delete_by_chunk_id(SAMPLE_IDS[0])
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_content_id(self, provider: Optional[ChromaProvider]):
        """Test async_delete_by_content_id."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
        )
        deleted = await provider.adelete_by_chunk_id(SAMPLE_IDS[0])
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_content_hash(self, provider: Optional[ChromaProvider]):
        """Test delete_by_chunk_content_hash with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        content_hash = md5(SAMPLE_CHUNKS[0].encode()).hexdigest()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        deleted = await provider.adelete_by_chunk_content_hash(content_hash)
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS[:1])
        assert len(results) == 0
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_metadata(self, provider: Optional[ChromaProvider]):
        """Test delete_by_metadata with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        deleted = provider.delete_by_metadata({"category": "science"})
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS)
        assert len(results) == 3
        for result in results:
            assert result.payload["metadata"].get("category") != "science"
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_metadata(self, provider: Optional[ChromaProvider]):
        """Test async_delete_by_metadata."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        deleted = await provider.adelete_by_metadata({"category": "science"})
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_id_exists(self, provider: Optional[ChromaProvider]):
        """Test id_exists check."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.aid_exists("doc1")
        assert not await provider.aid_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_document_name_exists(self, provider: Optional[ChromaProvider]):
        """Test document_name_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_names=["local_doc"]
        )
        assert provider.document_name_exists("local_doc")
        assert not provider.document_name_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_document_name_exists(self, provider: Optional[ChromaProvider]):
        """Test async_document_name_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_names=["local_doc"]
        )
        assert await provider.adocument_name_exists("local_doc")
        assert not await provider.adocument_name_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_document_id_exists(self, provider: Optional[ChromaProvider]):
        """Test document_id_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_ids=["local_doc_id_1"]
        )
        assert provider.document_id_exists("local_doc_id_1")
        assert not provider.document_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_document_id_exists(self, provider: Optional[ChromaProvider]):
        """Test async_document_id_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_ids=["local_doc_id_1"]
        )
        assert await provider.adocument_id_exists("local_doc_id_1")
        assert not await provider.adocument_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_content_id_exists(self, provider: Optional[ChromaProvider]):
        """Test chunk_id_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        assert provider.chunk_id_exists(SAMPLE_IDS[0])
        assert not provider.chunk_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_content_id_exists(self, provider: Optional[ChromaProvider]):
        """Test async_content_id_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        assert await provider.achunk_id_exists(SAMPLE_IDS[0])
        assert not await provider.achunk_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_content_hash_exists(self, provider: Optional[ChromaProvider]):
        """Test chunk_content_hash_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        content_hash = md5(SAMPLE_CHUNKS[0].encode()).hexdigest()
        assert await provider.achunk_content_hash_exists(content_hash)
        assert not await provider.achunk_content_hash_exists("nonexistent_hash")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_update_metadata(self, provider: Optional[ChromaProvider]):
        """Test async_update_metadata with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        updated = await provider.aupdate_metadata(SAMPLE_IDS[0], {"new_field": "new_value", "updated": True})
        assert updated is True
        results = await provider.afetch(ids=SAMPLE_IDS[:1])
        assert results[0].payload["metadata"].get("new_field") == "new_value"
        assert results[0].payload["metadata"].get("updated") is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_get_count_with_filter(self, provider: Optional[ChromaProvider]):
        """Test get_count with filter."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        count = await provider.aget_count(filter={"category": "science"})
        assert count == 2
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_get_count_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous get_count."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        provider.create_collection()
        initial_count = provider.get_count()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        assert provider.get_count() == initial_count + 5
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_optimize(self, provider: Optional[ChromaProvider]):
        """Test optimize operation."""
        self._skip_if_unavailable(provider)
        result = provider.optimize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_async_optimize(self, provider: Optional[ChromaProvider]):
        """Test async optimize."""
        self._skip_if_unavailable(provider)
        result = await provider.aoptimize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_supported_search_types(self, provider: Optional[ChromaProvider]):
        """Test get_supported_search_types."""
        self._skip_if_unavailable(provider)
        supported = provider.get_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
        assert "full_text" in supported
        assert "hybrid" in supported
    
    @pytest.mark.asyncio
    async def test_async_get_supported_search_types(self, provider: Optional[ChromaProvider]):
        """Test async_get_supported_search_types."""
        self._skip_if_unavailable(provider)
        supported = await provider.aget_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
        assert "full_text" in supported
        assert "hybrid" in supported
    
    @pytest.mark.asyncio
    async def test_dense_search_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous dense search with content validation."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.dense_search(
            query_vector=QUERY_VECTOR,
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert isinstance(result.score, float)
            assert 0.0 <= result.score <= 1.0
            assert result.payload is not None
            assert result.text is not None
            assert result.vector is not None
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_full_text_search_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous full-text search with content validation."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.full_text_search(
            query_text="physics",
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert isinstance(result.score, float)
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
            assert "physics" in result.text.lower() or "theory" in result.text.lower()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_rrf(self, provider: Optional[ChromaProvider]):
        """Test hybrid search with RRF fusion."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.ahybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            fusion_method="rrf",
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous hybrid search with validation."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.hybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            alpha=0.5,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert result.score >= 0.0
            assert result.payload is not None
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_search_master_method(self, provider: Optional[ChromaProvider]):
        """Test master search method with content validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # Dense search
        results = await provider.asearch(
            query_vector=QUERY_VECTOR,
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert all(r.id in SAMPLE_IDS for r in results)
        assert all(r.payload is not None for r in results)
        # Full-text search
        results = await provider.asearch(
            query_text="physics",
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        # Hybrid search
        results = await provider.asearch(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_search_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous master search with validation."""
        self._skip_if_unavailable(provider)
        self._ensure_connected_sync(provider)
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.search(
            query_vector=QUERY_VECTOR,
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert all(r.payload is not None for r in results)
        assert all(r.text is not None for r in results)
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_recreate_if_exists(self, provider: Optional[ChromaProvider]):
        """Test recreate_if_exists configuration."""
        self._skip_if_unavailable(provider)
        import uuid
        unique_name = f"test_recreate_{uuid.uuid4().hex[:8]}"
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        config = ChromaConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(mode=Mode.LOCAL, host=host, port=port),
            recreate_if_exists=True
        )
        provider2 = ChromaProvider(config)
        await self.a_ensure_connected(provider2)
        await provider2.acreate_collection()
        await provider2.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        await provider2.acreate_collection()
        count = await provider2.aget_count()
        assert count == 0
        await provider2.adisconnect()
    
    @pytest.mark.asyncio
    async def test_flat_index_config(self, provider: Optional[ChromaProvider]):
        """Test FlatIndexConfig."""
        self._skip_if_unavailable(provider)
        import uuid
        unique_name = f"test_flat_{uuid.uuid4().hex[:8]}"
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        config = ChromaConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(mode=Mode.LOCAL, host=host, port=port),
            index=FlatIndexConfig()
        )
        provider2 = ChromaProvider(config)
        await self.a_ensure_connected(provider2)
        await provider2.acreate_collection()
        await provider2.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        results = await provider2.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=2,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        await provider2.adisconnect()
    
    @pytest.mark.asyncio
    async def test_distance_metrics(self, provider: Optional[ChromaProvider]):
        """Test different distance metrics."""
        self._skip_if_unavailable(provider)
        import uuid
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        for metric in [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT]:
            unique_name = f"test_{metric.value}_{uuid.uuid4().hex[:8]}"
            config = ChromaConfig(
                vector_size=5,
                collection_name=unique_name,
                connection=ConnectionConfig(mode=Mode.LOCAL, host=host, port=port),
                distance_metric=metric
            )
            provider2 = ChromaProvider(config)
            await self.a_ensure_connected(provider2)
            await provider2.acreate_collection()
            await provider2.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2]
            )
            results = await provider2.adense_search(
                query_vector=QUERY_VECTOR,
                top_k=2,
                similarity_threshold=0.0
            )
            assert len(results) > 0
            assert all(isinstance(r, VectorSearchResult) for r in results)
            assert all(r.score >= 0.0 for r in results)
            await provider2.adisconnect()


class TestChromaProviderCLOUD:
    """Comprehensive tests for ChromaProvider in CLOUD mode (requires API key)."""
    
    @pytest.fixture
    def config(self, request) -> Optional[ChromaConfig]:
        """Create CLOUD ChromaConfig if API key available."""
        import uuid
        api_key = os.getenv("CHROMA_CLOUD_API_KEY")
        tenant = os.getenv("CHROMA_CLOUD_TENANT")
        database = os.getenv("CHROMA_CLOUD_DATABASE")
        if not api_key:
            return None
        
        from pydantic import SecretStr
        unique_name = f"test_cloud_{uuid.uuid4().hex[:8]}"
        return ChromaConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(
                mode=Mode.CLOUD,
                api_key=SecretStr(api_key)
            ),
            tenant=tenant,
            database=database,
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig(m=16, ef_construction=200)
        )
    
    @pytest.fixture
    def provider(self, config: Optional[ChromaConfig]) -> Optional[ChromaProvider]:
        """Create ChromaProvider instance."""
        if config is None:
            return None
        return ChromaProvider(config)
    
    def _skip_if_unavailable(self, provider: Optional[ChromaProvider]):
        """Helper to skip tests if provider is not available."""
        if provider is None:
            pytest.skip("ChromaDB API key not available")
    
    async def _ensure_connected(self, provider: ChromaProvider):
        """Helper to ensure connection, skip if unavailable."""
        try:
            await provider.aconnect()
            return True
        except VectorDBConnectionError:
            pytest.skip("ChromaDB cloud connection failed")
    
    @pytest.mark.asyncio
    async def test_initialization(self, provider: Optional[ChromaProvider], config: Optional[ChromaConfig]):
        """Test provider initialization and attributes."""
        self._skip_if_unavailable(provider)
        assert provider._config == config
        assert provider._config.collection_name.startswith("test_cloud_")
        assert provider._config.vector_size == 5
        assert provider._config.distance_metric == DistanceMetric.COSINE
        assert not provider._is_connected
        assert provider.client is None
    
    @pytest.mark.asyncio
    async def test_connect(self, provider: Optional[ChromaProvider]):
        """Test connection to ChromaDB."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        assert provider._is_connected is True
        assert provider.client is not None
        assert await provider.ais_ready() is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_create_collection(self, provider: Optional[ChromaProvider]):
        """Test collection creation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        try:
            if await provider.acollection_exists():
                await provider.adelete_collection()
        except Exception:
            pass
        assert not await provider.acollection_exists()
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert(self, provider: Optional[ChromaProvider]):
        """Test upsert operation with content validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.afetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.id == SAMPLE_IDS[i]
            assert result.payload["metadata"]["category"] == SAMPLE_PAYLOADS[i]["category"]
            assert result.payload["metadata"]["author"] == SAMPLE_PAYLOADS[i]["author"]
            assert result.payload["metadata"]["year"] == SAMPLE_PAYLOADS[i]["year"]
            assert result.text == SAMPLE_CHUNKS[i]
            # Test that vectors are exactly the same
            vector_list = [float(x) for x in result.vector]
            assert vector_list == SAMPLE_VECTORS[i], f"Vector mismatch for ID {result.id}: {vector_list} != {SAMPLE_VECTORS[i]}"
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_fetch(self, provider: Optional[ChromaProvider]):
        """Test fetch operation with detailed validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.afetch(ids=SAMPLE_IDS[:3])
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, VectorSearchResult)
            assert result.id == SAMPLE_IDS[i]
            assert result.score == 1.0
            assert result.payload is not None
            assert isinstance(result.payload, dict)
            assert result.payload["metadata"]["category"] == SAMPLE_PAYLOADS[i]["category"]
            assert result.payload["metadata"]["author"] == SAMPLE_PAYLOADS[i]["author"]
            assert result.text == SAMPLE_CHUNKS[i]
            # Test that vectors are exactly the same
            vector_list = [float(x) for x in result.vector]
            assert vector_list == SAMPLE_VECTORS[i], f"Vector mismatch for ID {result.id}: {vector_list} != {SAMPLE_VECTORS[i]}"
            assert len(result.vector) == 5
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete(self, provider: Optional[ChromaProvider]):
        """Test delete operation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        await provider.adelete(ids=SAMPLE_IDS[:2])
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        results = await provider.afetch(ids=SAMPLE_IDS[2:])
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.id == SAMPLE_IDS[i+2]
            assert result.payload["metadata"]["category"] == SAMPLE_PAYLOADS[i+2]["category"]
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_dense_search(self, provider: Optional[ChromaProvider]):
        """Test dense search with detailed result validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert isinstance(result.score, float)
            assert 0.0 <= result.score <= 1.0
            assert result.payload is not None
            assert "category" in result.payload["metadata"]
            assert "author" in result.payload["metadata"]
            assert result.text is not None
            assert result.text in SAMPLE_CHUNKS
            assert result.vector is not None
            assert len(result.vector) == 5
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_full_text_search(self, provider: Optional[ChromaProvider]):
        """Test full-text search with content validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.afull_text_search(
            query_text="physics",
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert isinstance(result.score, float)
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
            assert "physics" in result.text.lower() or "theory" in result.text.lower()
            assert result.vector is not None
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, provider: Optional[ChromaProvider]):
        """Test hybrid search with detailed validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.ahybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            alpha=0.5,
            fusion_method="weighted",
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert isinstance(result.score, float)
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
            assert result.vector is not None
            assert len(result.vector) == 5
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_search_with_filter(self, provider: Optional[ChromaProvider]):
        """Test search with metadata filter."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=5,
            filter={"category": "science"}
        )
        assert len(results) > 0
        for result in results:
            assert result.payload["metadata"].get("category") == "science"
            assert result.payload["metadata"].get("author") in ["Einstein", "Newton"]
            assert result.id in ["doc1", "doc2"]
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_get_count(self, provider: Optional[ChromaProvider]):
        """Test get_count."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        initial_count = await provider.aget_count()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        assert await provider.aget_count() == initial_count + 5
        await provider.adelete(ids=SAMPLE_IDS[:2])
        assert await provider.aget_count() == initial_count + 3
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_update_metadata(self, provider: Optional[ChromaProvider]):
        """Test update_metadata with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        updated = provider.update_metadata(SAMPLE_IDS[0], {"new_field": "new_value", "updated": True})
        assert updated is True
        results = await provider.afetch(ids=SAMPLE_IDS[:1])
        assert len(results) == 1
        assert results[0].payload["metadata"].get("new_field") == "new_value"
        assert results[0].payload["metadata"].get("updated") is True
        assert results[0].payload["metadata"].get("category") == SAMPLE_PAYLOADS[0]["category"]
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_filter(self, provider: Optional[ChromaProvider]):
        """Test delete_by_filter."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        deleted = await provider.adelete_by_metadata({"category": "science"})
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS)
        assert len(results) == 3
        for result in results:
            assert result.payload["metadata"].get("category") != "science"
            assert result.id in ["doc3", "doc4", "doc5"]
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_with_document_tracking(self, provider: Optional[ChromaProvider]):
        """Test upsert with document tracking and validate metadata."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_names=["cloud_doc1", "cloud_doc2"],
            document_ids=["cloud_doc_id_1", "cloud_doc_id_2"],
        )
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        for i, result in enumerate(results):
            assert result.payload.get("document_name") == f"cloud_doc{i+1}"
            assert result.payload.get("document_id") == f"cloud_doc_id_{i+1}"
            assert result.payload.get("chunk_id") == f"cloud_content_{i+1}"
            assert result.payload["metadata"].get("category") == SAMPLE_PAYLOADS[i]["category"]
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_connect_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous connection."""
        self._skip_if_unavailable(provider)
        provider.connect()
        assert provider._is_connected is True
        assert provider.client is not None
        assert provider.is_ready() is True
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, provider: Optional[ChromaProvider]):
        """Test disconnection."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        assert provider._is_connected is True
        await provider.adisconnect()
        assert provider._is_connected is False
        assert provider.client is None
        assert provider._collection_instance is None
    
    @pytest.mark.asyncio
    async def test_disconnect_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous disconnection."""
        self._skip_if_unavailable(provider)
        provider.connect()
        assert provider._is_connected is True
        provider.disconnect()
        assert provider._is_connected is False
    
    @pytest.mark.asyncio
    async def test_is_ready(self, provider: Optional[ChromaProvider]):
        """Test is_ready check."""
        self._skip_if_unavailable(provider)
        assert await provider.ais_ready() is False
        await self.a_ensure_connected(provider)
        assert await provider.ais_ready() is True
        await provider.adisconnect()
        assert await provider.ais_ready() is False
    
    @pytest.mark.asyncio
    async def test_is_ready_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous is_ready check."""
        self._skip_if_unavailable(provider)
        assert provider.is_ready() is False
        provider.connect()
        assert provider.is_ready() is True
        provider.disconnect()
        assert provider.is_ready() is False
    
    @pytest.mark.asyncio
    async def test_create_collection_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous collection creation."""
        self._skip_if_unavailable(provider)
        provider.connect()
        try:
            if provider.collection_exists():
                provider.delete_collection()
        except Exception:
            pass
        assert not provider.collection_exists()
        provider.create_collection()
        assert provider.collection_exists()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_collection_exists(self, provider: Optional[ChromaProvider]):
        """Test collection existence check."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        try:
            if await provider.acollection_exists():
                await provider.adelete_collection()
        except Exception:
            pass
        assert not await provider.acollection_exists()
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_collection_exists_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous collection existence check."""
        self._skip_if_unavailable(provider)
        provider.connect()
        try:
            if provider.collection_exists():
                provider.delete_collection()
        except Exception:
            pass
        assert not provider.collection_exists()
        provider.create_collection()
        assert provider.collection_exists()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_collection(self, provider: Optional[ChromaProvider]):
        """Test collection deletion."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adelete_collection()
        assert not await provider.acollection_exists()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_collection_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous collection deletion."""
        self._skip_if_unavailable(provider)
        provider.connect()
        provider.create_collection()
        assert provider.collection_exists()
        provider.delete_collection()
        assert not provider.collection_exists()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection(self, provider: Optional[ChromaProvider]):
        """Test deleting non-existent collection raises error."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        with pytest.raises(CollectionDoesNotExistError):
            await provider.adelete_collection()
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous upsert with content validation."""
        self._skip_if_unavailable(provider)
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.fetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.id == SAMPLE_IDS[i]
            assert result.payload["metadata"].get("category") == SAMPLE_PAYLOADS[i]["category"]
            assert result.payload["metadata"].get("author") == SAMPLE_PAYLOADS[i]["author"]
            assert result.text == SAMPLE_CHUNKS[i]
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_with_sparse_vectors_ignored(self, provider: Optional[ChromaProvider]):
        """Test that sparse vectors are ignored (ChromaDB doesn't support them)."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        sparse_vectors = [
            {"indices": [0, 2, 4], "values": [0.5, 0.3, 0.2]},
            {"indices": [1, 3], "values": [0.4, 0.6]}
        ]
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            sparse_vectors=sparse_vectors
        )
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_validation_error(self, provider: Optional[ChromaProvider]):
        """Test upsert with mismatched lengths raises error."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        with pytest.raises(UpsertError):
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:3],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2]
            )
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_fetch_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous fetch with content validation."""
        self._skip_if_unavailable(provider)
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.fetch(ids=SAMPLE_IDS[:3])
        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, VectorSearchResult)
            assert result.id == SAMPLE_IDS[i]
            assert result.score == 1.0
            assert result.payload is not None
            assert result.payload["metadata"]["category"] == SAMPLE_PAYLOADS[i]["category"]
            assert result.payload["metadata"]["author"] == SAMPLE_PAYLOADS[i]["author"]
            assert result.text == SAMPLE_CHUNKS[i]
            assert result.vector is not None
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_fetch_by_id_sync(self, provider: Optional[ChromaProvider]):
        """Test fetch_by_id_sync alias with content validation."""
        self._skip_if_unavailable(provider)
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.fetch_by_id(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        for i, result in enumerate(results):
            assert result.id == SAMPLE_IDS[i]
            assert result.payload["metadata"].get("category") == SAMPLE_PAYLOADS[i]["category"]
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous delete with validation."""
        self._skip_if_unavailable(provider)
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        provider.delete(ids=SAMPLE_IDS[:2])
        results = provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        results = provider.fetch(ids=SAMPLE_IDS[2:])
        assert len(results) == 3
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_id_sync(self, provider: Optional[ChromaProvider]):
        """Test delete_by_id_sync alias."""
        self._skip_if_unavailable(provider)
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        provider.delete_by_id(ids=SAMPLE_IDS[:2])
        results = provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_document_name(self, provider: Optional[ChromaProvider]):
        """Test delete_by_document_name with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        initial_count = await provider.aget_count()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_names=["cloud_doc", "cloud_doc"]
        )
        deleted = provider.delete_by_document_name("cloud_doc")
        assert deleted is True
        count = await provider.aget_count()
        assert count == initial_count
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_document_name(self, provider: Optional[ChromaProvider]):
        """Test async_delete_by_document_name."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_names=["cloud_doc", "cloud_doc"]
        )
        deleted = await provider.adelete_by_document_name("cloud_doc")
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, provider: Optional[ChromaProvider]):
        """Test delete_by_document_id with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_ids=["cloud_doc_id_1", "cloud_doc_id_1"]
        )
        deleted = provider.delete_by_document_id("cloud_doc_id_1")
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_document_id(self, provider: Optional[ChromaProvider]):
        """Test async_delete_by_document_id."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_ids=["cloud_doc_id_1", "cloud_doc_id_1"]
        )
        deleted = await provider.adelete_by_document_id("cloud_doc_id_1")
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_content_id(self, provider: Optional[ChromaProvider]):
        """Test delete_by_chunk_id with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
        )
        deleted = provider.delete_by_chunk_id("cloud_content_1")
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_content_id(self, provider: Optional[ChromaProvider]):
        """Test async_delete_by_content_id."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
        )
        deleted = await provider.adelete_by_chunk_id("cloud_content_1")
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_content_hash(self, provider: Optional[ChromaProvider]):
        """Test delete_by_chunk_content_hash with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        content_hash = md5(SAMPLE_CHUNKS[0].encode()).hexdigest()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        deleted = await provider.adelete_by_chunk_content_hash(content_hash)
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS[:1])
        assert len(results) == 0
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_metadata(self, provider: Optional[ChromaProvider]):
        """Test delete_by_metadata with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        deleted = provider.delete_by_metadata({"category": "science"})
        assert deleted is True
        results = await provider.afetch(ids=SAMPLE_IDS)
        assert len(results) == 3
        for result in results:
            assert result.payload["metadata"].get("category") != "science"
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_metadata(self, provider: Optional[ChromaProvider]):
        """Test async_delete_by_metadata."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        deleted = await provider.adelete_by_metadata({"category": "science"})
        assert deleted is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_id_exists(self, provider: Optional[ChromaProvider]):
        """Test id_exists check."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.aid_exists("doc1")
        assert not await provider.aid_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_document_name_exists(self, provider: Optional[ChromaProvider]):
        """Test document_name_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_names=["cloud_doc"]
        )
        assert provider.document_name_exists("cloud_doc")
        assert not provider.document_name_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_document_name_exists(self, provider: Optional[ChromaProvider]):
        """Test async_document_name_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_names=["cloud_doc"]
        )
        assert await provider.adocument_name_exists("cloud_doc")
        assert not await provider.adocument_name_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_document_id_exists(self, provider: Optional[ChromaProvider]):
        """Test document_id_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_ids=["cloud_doc_id_1"]
        )
        assert provider.document_id_exists("cloud_doc_id_1")
        assert not provider.document_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_document_id_exists(self, provider: Optional[ChromaProvider]):
        """Test async_document_id_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_ids=["cloud_doc_id_1"]
        )
        assert await provider.adocument_id_exists("cloud_doc_id_1")
        assert not await provider.adocument_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_content_id_exists(self, provider: Optional[ChromaProvider]):
        """Test chunk_id_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        assert provider.chunk_id_exists("cloud_content_1")
        assert not provider.chunk_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_content_id_exists(self, provider: Optional[ChromaProvider]):
        """Test async_content_id_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        assert await provider.achunk_id_exists("cloud_content_1")
        assert not await provider.achunk_id_exists("nonexistent")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_content_hash_exists(self, provider: Optional[ChromaProvider]):
        """Test chunk_content_hash_exists."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        content_hash = md5(SAMPLE_CHUNKS[0].encode()).hexdigest()
        assert await provider.achunk_content_hash_exists(content_hash)
        assert not await provider.achunk_content_hash_exists("nonexistent_hash")
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_async_update_metadata(self, provider: Optional[ChromaProvider]):
        """Test async_update_metadata with validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )
        updated = await provider.aupdate_metadata("cloud_content_1", {"new_field": "new_value", "updated": True})
        assert updated is True
        results = await provider.afetch(ids=SAMPLE_IDS[:1])
        assert results[0].payload["metadata"].get("new_field") == "new_value"
        assert results[0].payload["metadata"].get("updated") is True
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_get_count_with_filter(self, provider: Optional[ChromaProvider]):
        """Test get_count with filter."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        count = await provider.aget_count(filter={"category": "science"})
        assert count == 2
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_get_count_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous get_count."""
        self._skip_if_unavailable(provider)
        provider.connect()
        provider.create_collection()
        initial_count = provider.get_count()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        assert provider.get_count() == initial_count + 5
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_optimize(self, provider: Optional[ChromaProvider]):
        """Test optimize operation."""
        self._skip_if_unavailable(provider)
        result = provider.optimize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_async_optimize(self, provider: Optional[ChromaProvider]):
        """Test async optimize."""
        self._skip_if_unavailable(provider)
        result = await provider.aoptimize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_supported_search_types(self, provider: Optional[ChromaProvider]):
        """Test get_supported_search_types."""
        self._skip_if_unavailable(provider)
        supported = provider.get_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
        assert "full_text" in supported
        assert "hybrid" in supported
    
    @pytest.mark.asyncio
    async def test_async_get_supported_search_types(self, provider: Optional[ChromaProvider]):
        """Test async_get_supported_search_types."""
        self._skip_if_unavailable(provider)
        supported = await provider.aget_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
        assert "full_text" in supported
        assert "hybrid" in supported
    
    @pytest.mark.asyncio
    async def test_dense_search_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous dense search with content validation."""
        self._skip_if_unavailable(provider)
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.dense_search(
            query_vector=QUERY_VECTOR,
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert isinstance(result.score, float)
            assert 0.0 <= result.score <= 1.0
            assert result.payload is not None
            assert result.text is not None
            assert result.vector is not None
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_full_text_search_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous full-text search with content validation."""
        self._skip_if_unavailable(provider)
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.full_text_search(
            query_text="physics",
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert isinstance(result.score, float)
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
            assert "physics" in result.text.lower() or "theory" in result.text.lower()
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_rrf(self, provider: Optional[ChromaProvider]):
        """Test hybrid search with RRF fusion."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.ahybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            fusion_method="rrf",
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous hybrid search with validation."""
        self._skip_if_unavailable(provider)
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.hybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            alpha=0.5,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id in SAMPLE_IDS
            assert result.score >= 0.0
            assert result.payload is not None
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_search_master_method(self, provider: Optional[ChromaProvider]):
        """Test master search method with content validation."""
        self._skip_if_unavailable(provider)
        await self.a_ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # Dense search
        results = await provider.asearch(
            query_vector=QUERY_VECTOR,
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert all(r.id in SAMPLE_IDS for r in results)
        assert all(r.payload is not None for r in results)
        # Full-text search
        results = await provider.asearch(
            query_text="physics",
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        # Hybrid search
        results = await provider.asearch(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_search_sync(self, provider: Optional[ChromaProvider]):
        """Test synchronous master search with validation."""
        self._skip_if_unavailable(provider)
        provider.connect()
        provider.create_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.search(
            query_vector=QUERY_VECTOR,
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert all(r.payload is not None for r in results)
        assert all(r.text is not None for r in results)
        provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_recreate_if_exists(self, provider: Optional[ChromaProvider]):
        """Test recreate_if_exists configuration."""
        self._skip_if_unavailable(provider)
        import uuid
        api_key = os.getenv("CHROMA_CLOUD_API_KEY")
        tenant = os.getenv("CHROMA_CLOUD_TENANT")
        database = os.getenv("CHROMA_CLOUD_DATABASE")
        if not api_key:
            pytest.skip("ChromaDB API key not available")
        from pydantic import SecretStr
        unique_name = f"test_recreate_{uuid.uuid4().hex[:8]}"
        config = ChromaConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(
                mode=Mode.CLOUD,
                api_key=SecretStr(api_key)
            ),
            tenant=tenant,
            database=database,
            recreate_if_exists=True
        )
        provider2 = ChromaProvider(config)
        await self.a_ensure_connected(provider2)
        await provider2.acreate_collection()
        await provider2.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        await provider2.acreate_collection()
        count = await provider2.aget_count()
        assert count == 0
        await provider2.adisconnect()
    
    @pytest.mark.asyncio
    async def test_flat_index_config(self, provider: Optional[ChromaProvider]):
        """Test FlatIndexConfig."""
        self._skip_if_unavailable(provider)
        import uuid
        api_key = os.getenv("CHROMA_CLOUD_API_KEY")
        tenant = os.getenv("CHROMA_CLOUD_TENANT")
        database = os.getenv("CHROMA_CLOUD_DATABASE")
        if not api_key:
            pytest.skip("ChromaDB API key not available")
        from pydantic import SecretStr
        unique_name = f"test_flat_{uuid.uuid4().hex[:8]}"
        config = ChromaConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(
                mode=Mode.CLOUD,
                api_key=SecretStr(api_key)
            ),
            tenant=tenant,
            database=database,
            index=FlatIndexConfig()
        )
        provider2 = ChromaProvider(config)
        await self.a_ensure_connected(provider2)
        await provider2.acreate_collection()
        await provider2.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        results = await provider2.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=2,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        await provider2.adisconnect()
    
    @pytest.mark.asyncio
    async def test_distance_metrics(self, provider: Optional[ChromaProvider]):
        """Test different distance metrics."""
        self._skip_if_unavailable(provider)
        import uuid
        api_key = os.getenv("CHROMA_CLOUD_API_KEY")
        tenant = os.getenv("CHROMA_CLOUD_TENANT")
        database = os.getenv("CHROMA_CLOUD_DATABASE")
        if not api_key:
            pytest.skip("ChromaDB API key not available")
        from pydantic import SecretStr
        for metric in [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT]:
            unique_name = f"test_{metric.value}_{uuid.uuid4().hex[:8]}"
            config = ChromaConfig(
                vector_size=5,
                collection_name=unique_name,
                connection=ConnectionConfig(
                    mode=Mode.CLOUD,
                    api_key=SecretStr(api_key)
                ),
                tenant=tenant,
                database=database,
                distance_metric=metric
            )
            provider2 = ChromaProvider(config)
            await self.a_ensure_connected(provider2)
            await provider2.acreate_collection()
            await provider2.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2]
            )
            results = await provider2.adense_search(
                query_vector=QUERY_VECTOR,
                top_k=2,
                similarity_threshold=0.0
            )
            assert len(results) > 0
            assert all(isinstance(r, VectorSearchResult) for r in results)
            assert all(r.score >= 0.0 for r in results)
            await provider2.adisconnect()


class TestChromaProviderConfigValidation:
    """Test ChromaProvider configuration validation."""
    
    @pytest.mark.asyncio
    async def test_invalid_ivf_index(self):
        """Test that IVF index raises error."""
        from upsonic.vectordb.config import IVFIndexConfig
        from pydantic import ValidationError
        with pytest.raises((ValueError, ValidationError)):
            try:
                config = ChromaConfig(
                    vector_size=5,
                    collection_name="test",
                    connection=ConnectionConfig(mode=Mode.IN_MEMORY),
                    index=IVFIndexConfig()
                )
            except ValidationError:
                # This is expected - Pydantic validates before the model_validator
                raise
            except ValueError as e:
                if "Chroma does not support IVF" in str(e):
                    raise
                raise
    
    @pytest.mark.asyncio
    async def test_embedded_missing_db_path(self):
        """Test that EMBEDDED mode requires db_path."""
        with pytest.raises(ValueError, match="db_path is required"):
            config = ChromaConfig(
                vector_size=5,
                collection_name="test",
                connection=ConnectionConfig(mode=Mode.EMBEDDED)
            )
    
    @pytest.mark.asyncio
    async def test_local_missing_host_port(self):
        """Test that LOCAL mode requires host and port."""
        with pytest.raises(ValueError, match="host and port are required"):
            config = ChromaConfig(
                vector_size=5,
                collection_name="test",
                connection=ConnectionConfig(mode=Mode.LOCAL)
            )
    
    @pytest.mark.asyncio
    async def test_cloud_missing_api_key(self):
        """Test that CLOUD mode requires api_key."""
        with pytest.raises(ValueError, match="api_key is required"):
            config = ChromaConfig(
                vector_size=5,
                collection_name="test",
                connection=ConnectionConfig(mode=Mode.CLOUD)
            )
    
    @pytest.mark.asyncio
    async def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "vector_size": 5,
            "collection_name": "test_dict",
            "connection": {
                "mode": "in_memory"
            }
        }
        config = ChromaConfig.from_dict(config_dict)
        assert config.vector_size == 5
        assert config.collection_name == "test_dict"
        assert config.connection.mode == Mode.IN_MEMORY


class TestChromaProviderErrorHandling:
    """Test ChromaProvider error handling."""
    
    @pytest.fixture
    def provider(self) -> ChromaProvider:
        """Create ChromaProvider instance."""
        config = ChromaConfig(
            vector_size=5,
            collection_name="test_error",
            connection=ConnectionConfig(mode=Mode.IN_MEMORY)
        )
        return ChromaProvider(config)
    
    @pytest.mark.asyncio
    async def test_operations_without_connection(self, provider: ChromaProvider):
        """Test operations fail without connection."""
        with pytest.raises(VectorDBConnectionError):
            await provider.acreate_collection()
        
        with pytest.raises(VectorDBConnectionError):
            await provider.acollection_exists()
    
    @pytest.mark.asyncio
    async def test_operations_without_collection(self, provider: ChromaProvider):
        """Test operations fail without collection."""
        await provider.aconnect()
        with pytest.raises(VectorDBError):
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS
            )
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_search_without_data(self, provider: ChromaProvider):
        """Test search returns empty results when no data."""
        await provider.aconnect()
        await provider.acreate_collection()
        results = await provider.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=5
        )
        assert len(results) == 0
        await provider.adisconnect()
    
    @pytest.mark.asyncio
    async def test_search_disabled_types(self, provider: ChromaProvider):
        """Test search fails when search type is disabled."""
        import uuid
        unique_name = f"test_disabled_{uuid.uuid4().hex[:8]}"
        config = ChromaConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(mode=Mode.IN_MEMORY),
            dense_search_enabled=False
        )
        provider2 = ChromaProvider(config)
        await provider2.aconnect()
        await provider2.acreate_collection()
        # The search method should check this, not dense_search directly
        with pytest.raises(ConfigurationError, match="Dense search is disabled"):
            await provider2.asearch(
                query_vector=QUERY_VECTOR,
                top_k=5
            )
        await provider2.adisconnect()
    
    @pytest.mark.asyncio
    async def test_search_missing_parameters(self, provider: ChromaProvider):
        """Test search fails without query_vector or query_text."""
        await provider.aconnect()
        await provider.acreate_collection()
        with pytest.raises(ConfigurationError, match="requires at least one"):
            await provider.asearch(top_k=5)
        await provider.adisconnect()
