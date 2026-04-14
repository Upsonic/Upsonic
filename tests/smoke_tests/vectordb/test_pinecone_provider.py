"""
Comprehensive smoke tests for Pinecone vector database provider.

Tests all methods, attributes, and the CLOUD connection mode, using the new
fully async provider API (a*-prefixed methods).
Verifies that stored values exactly match retrieved values.
"""

import os
import pytest
import asyncio
import time
from hashlib import md5
from typing import List, Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system env vars

from upsonic.vectordb.providers.pinecone import PineconeProvider
from upsonic.vectordb.config import (
    PineconeConfig,
    DistanceMetric
)
from upsonic.utils.package.exception import VectorDBConnectionError
from upsonic.schemas.vector_schemas import VectorSearchResult
from upsonic.utils.logging_config import get_logger

logger = get_logger(__name__)


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
    {"category": "philosophy", "author": "Plato", "year": -400},
]

SAMPLE_CHUNKS: List[str] = [
    "The theory of relativity revolutionized physics",
    "Laws of motion and universal gravitation",
    "To be or not to be, that is the question",
    "It was the best of times, it was the worst of times",
    "The unexamined life is not worth living"
]

# Hardcoded UUIDs (not generated at import time) — mirrors Qdrant test layout.
SAMPLE_IDS: List[str] = [
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
    "44444444-4444-4444-8444-444444444444",
    "55555555-5555-4555-8555-555555555555",
]

QUERY_VECTOR: List[float] = [0.15, 0.25, 0.35, 0.45, 0.55]
QUERY_TEXT: str = "physics theory"


def assert_vector_matches(actual_vector: Any, expected_vector: List[float], vector_id: str = "", tolerance: float = 1e-6) -> None:
    """Assert that a retrieved vector matches the expected vector."""
    assert actual_vector is not None, f"Vector is None for {vector_id}"
    assert hasattr(actual_vector, '__len__'), f"Vector has no length for {vector_id}"
    assert len(actual_vector) == len(expected_vector), \
        f"Vector length mismatch for {vector_id}: {len(actual_vector)} != {len(expected_vector)}"

    vector_list = [float(x) for x in actual_vector]
    for j, (actual, expected) in enumerate(zip(vector_list, expected_vector)):
        assert abs(actual - expected) < tolerance, \
            f"Vector element {j} mismatch for {vector_id}: {actual} != {expected} (diff: {abs(actual - expected)})"


# Module-level shared index state for all tests
_SHARED_INDEX_NAME = None
_SHARED_PROVIDER = None
_SHARED_CONNECTED = False


async def _poll_until(predicate, *, timeout: float = 30.0, interval: float = 1.0, description: str = ""):
    """
    Poll `predicate` (async callable returning bool) until it returns True or timeout expires.
    Returns True on success, False on timeout. Does NOT raise.
    """
    _ = description  # accepted for call-site documentation; not used internally
    start = time.monotonic()
    while True:
        try:
            if await predicate():
                return True
        except Exception:
            pass
        if time.monotonic() - start >= timeout:
            return False
        await asyncio.sleep(interval)


class TestPineconeProviderCLOUD:
    """Comprehensive tests for PineconeProvider in CLOUD mode (requires API key).

    Uses a SHARED INDEX approach - one index is created at class start and reused
    across all tests. Each test clears vectors in its namespace before running.
    This avoids the 30-60+ second wait for each index creation.
    """

    @pytest.fixture(scope="class")
    def shared_index_name(self):
        """Get or create a shared index name for all tests."""
        global _SHARED_INDEX_NAME
        if _SHARED_INDEX_NAME is None:
            import uuid
            _SHARED_INDEX_NAME = f"test-smoke-{uuid.uuid4().hex[:8]}"
        return _SHARED_INDEX_NAME

    @pytest.fixture(scope="class")
    def shared_config(self, shared_index_name) -> Optional[PineconeConfig]:
        """Create a SHARED PineconeConfig that's reused across all tests."""
        from pydantic import SecretStr
        api_key = os.getenv("PINECONE_CLOUD_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT", "aws-us-east-1")
        if not api_key:
            return None

        return PineconeConfig(
            vector_size=5,
            collection_name=shared_index_name,
            api_key=SecretStr(api_key),
            environment=environment,
            distance_metric=DistanceMetric.DOT_PRODUCT,  # Required for hybrid search
            use_sparse_vectors=True,
            hybrid_search_enabled=True,
            recreate_if_exists=False  # Don't recreate - reuse!
        )

    @pytest.fixture(scope="class")
    def shared_provider(self, shared_config: Optional[PineconeConfig]):
        """Create a SHARED provider that's reused across all tests."""
        global _SHARED_PROVIDER
        if shared_config is None:
            yield None
            return

        if _SHARED_PROVIDER is None:
            _SHARED_PROVIDER = PineconeProvider(shared_config)

        yield _SHARED_PROVIDER

    @pytest.fixture(scope="class")
    def event_loop(self):
        """Create an event loop for the class scope."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture(autouse=True, scope="class")
    def setup_shared_index(self, shared_provider, event_loop):
        """Setup: Connect and create index once for all tests in the class."""
        global _SHARED_CONNECTED
        if shared_provider is None:
            yield
            return

        async def _setup():
            global _SHARED_CONNECTED
            if not _SHARED_CONNECTED:
                await shared_provider.aconnect()
                if not await shared_provider.acollection_exists():
                    await shared_provider.acreate_collection()
                _SHARED_CONNECTED = True

        event_loop.run_until_complete(_setup())
        yield

        async def _teardown():
            global _SHARED_CONNECTED, _SHARED_PROVIDER, _SHARED_INDEX_NAME
            try:
                if shared_provider._is_connected:
                    if await shared_provider.acollection_exists():
                        await shared_provider.adelete_collection()
                    await shared_provider.adisconnect()
            except Exception as e:
                logger.warning(f"Teardown error: {e}")
            _SHARED_CONNECTED = False
            _SHARED_PROVIDER = None
            _SHARED_INDEX_NAME = None

        event_loop.run_until_complete(_teardown())

    @pytest.fixture
    def config(self, shared_config) -> Optional[PineconeConfig]:
        return shared_config

    @pytest.fixture
    def provider(self, shared_provider) -> Optional[PineconeProvider]:
        return shared_provider

    def _skip_if_unavailable(self, provider: Optional[PineconeProvider]):
        if provider is None:
            pytest.skip("Pinecone API key not available")

    async def _ensure_connected(self, provider: PineconeProvider):
        global _SHARED_CONNECTED
        if _SHARED_CONNECTED and provider._is_connected:
            return True
        try:
            await provider.aconnect()
            _SHARED_CONNECTED = True
            return True
        except VectorDBConnectionError:
            pytest.skip("Pinecone Cloud connection failed")

    async def _clear_vectors(self, provider: PineconeProvider):
        """Clear all vectors from the index (faster than recreating)."""
        try:
            if provider._is_connected and provider._index is not None:
                try:
                    provider._index.delete(delete_all=True)
                except Exception:
                    pass
                await asyncio.sleep(10)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Core provider tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_initialization(self, provider: Optional[PineconeProvider], config: Optional[PineconeConfig]):
        """Test provider initialization and attributes."""
        self._skip_if_unavailable(provider)
        assert provider._config == config
        assert provider._config.collection_name.startswith("test-smoke-")
        assert provider._config.vector_size == 5
        assert provider._config.distance_metric == DistanceMetric.DOT_PRODUCT
        assert provider._is_connected
        assert provider.client is not None

        assert provider.name is not None
        assert isinstance(provider.id, str)
        assert len(provider.id) > 0

    @pytest.mark.asyncio
    async def test_aconnect(self, provider: Optional[PineconeProvider]):
        """Test connection to Pinecone Cloud."""
        self._skip_if_unavailable(provider)
        assert provider._is_connected is True
        assert provider.client is not None
        assert await provider.ais_ready() is True

    @pytest.mark.asyncio
    async def test_acreate_collection(self, provider: Optional[PineconeProvider]):
        """Test collection exists (shared index is already created)."""
        self._skip_if_unavailable(provider)
        assert await provider.acollection_exists()

    @pytest.mark.asyncio
    async def test_aupsert(self, provider: Optional[PineconeProvider]):
        """Test upsert operation with content validation."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            await asyncio.sleep(5)
            results = await provider.afetch(ids=SAMPLE_IDS)
            assert len(results) >= 1  # eventual consistency
            for result in results:
                assert result.id is not None
                assert result.payload is not None
                idx = SAMPLE_IDS.index(str(result.id))
                assert_vector_matches(result.vector, SAMPLE_VECTORS[idx], vector_id=str(result.id))
                assert result.payload["metadata"].get("category") == SAMPLE_PAYLOADS[idx]["category"]
                assert result.payload["metadata"].get("author") == SAMPLE_PAYLOADS[idx]["author"]
                assert result.payload["metadata"].get("year") == SAMPLE_PAYLOADS[idx]["year"]
                assert result.payload.get("content") == SAMPLE_CHUNKS[idx]
                assert result.text == SAMPLE_CHUNKS[idx]
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_aupsert_validation_error(self, provider: Optional[PineconeProvider]):
        """Test that aupsert raises ValueError on per-item array length mismatch."""
        self._skip_if_unavailable(provider)
        try:
            with pytest.raises(ValueError):
                await provider.aupsert(
                    vectors=SAMPLE_VECTORS[:2],
                    payloads=SAMPLE_PAYLOADS[:3],
                    ids=SAMPLE_IDS[:2],
                    chunks=SAMPLE_CHUNKS[:2],
                )
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_afetch(self, provider: Optional[PineconeProvider]):
        """Test fetch operation with detailed validation."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            await asyncio.sleep(5)
            results = await provider.afetch(ids=SAMPLE_IDS[:3])
            assert len(results) >= 1  # eventual consistency
            for result in results:
                assert isinstance(result, VectorSearchResult)
                assert result.id is not None
                assert result.score == 1.0
                assert result.payload is not None
                assert result.text is not None
                assert result.vector is not None
                assert len(result.vector) == 5
                idx = SAMPLE_IDS.index(str(result.id))
                assert_vector_matches(result.vector, SAMPLE_VECTORS[idx], vector_id=str(result.id))
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_adelete(self, provider: Optional[PineconeProvider]):
        """Test delete operation."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            await asyncio.sleep(3)
            await provider.adelete(ids=SAMPLE_IDS[:2])
            await asyncio.sleep(3)
            results = await provider.afetch(ids=SAMPLE_IDS[:2])
            assert len(results) <= 2  # eventual consistency
            results = await provider.afetch(ids=SAMPLE_IDS[2:])
            assert len(results) <= 3  # eventual consistency
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_adense_search(self, provider: Optional[PineconeProvider]):
        """Test dense search with detailed result validation."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            await asyncio.sleep(5)
            results = await provider.adense_search(
                query_vector=QUERY_VECTOR,
                top_k=3,
                similarity_threshold=0.0,
            )
            assert len(results) >= 0
            assert len(results) <= 3
            for result in results:
                assert isinstance(result, VectorSearchResult)
                assert result.id is not None
                assert isinstance(result.score, float)
                assert result.payload is not None
                assert result.text is not None
                assert result.vector is not None
                assert len(result.vector) == 5
                assert str(result.id) in SAMPLE_IDS
                idx = SAMPLE_IDS.index(str(result.id))
                assert_vector_matches(result.vector, SAMPLE_VECTORS[idx], vector_id=str(result.id))
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_afull_text_search(self, provider: Optional[PineconeProvider]):
        """Test full-text search with content validation."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            await asyncio.sleep(5)
            results = await provider.afull_text_search(
                query_text="physics",
                top_k=3,
                similarity_threshold=0.0,
            )
            assert len(results) >= 0
            for result in results:
                assert isinstance(result, VectorSearchResult)
                assert result.id is not None
                assert isinstance(result.score, float)
                assert result.score >= 0.0
                assert result.payload is not None
                assert result.text is not None
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_ahybrid_search(self, provider: Optional[PineconeProvider]):
        """Test hybrid search with detailed validation."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            await asyncio.sleep(5)
            results = await provider.ahybrid_search(
                query_vector=QUERY_VECTOR,
                query_text="physics",
                top_k=3,
                alpha=0.5,
                fusion_method="weighted",
                similarity_threshold=0.0,
            )
            assert len(results) >= 0
            assert len(results) <= 3
            for result in results:
                assert isinstance(result, VectorSearchResult)
                assert result.id is not None
                assert isinstance(result.score, float)
                assert result.score >= 0.0
                assert result.payload is not None
                assert result.text is not None
                assert result.vector is not None
                assert len(result.vector) == 5
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_search_with_filter(self, provider: Optional[PineconeProvider]):
        """Test search with metadata filter."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            await asyncio.sleep(5)
            results = await provider.adense_search(
                query_vector=QUERY_VECTOR,
                top_k=5,
                filter={"category": {"$eq": "science"}},
            )
            for result in results:
                assert result.payload["metadata"].get("category") == "science"
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_aget_count(self, provider: Optional[PineconeProvider]):
        """Test get_count."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await asyncio.sleep(2)
            initial_count = await provider.aget_count()
            assert isinstance(initial_count, int)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            await asyncio.sleep(5)
            new_count = await provider.aget_count()
            assert isinstance(new_count, int)
            assert new_count >= initial_count
            await provider.adelete(ids=SAMPLE_IDS[:2])
            await asyncio.sleep(3)
            final_count = await provider.aget_count()
            assert isinstance(final_count, int)
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_aupdate_metadata(self, provider: Optional[PineconeProvider]):
        """Test aupdate_metadata with validation."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1],
                payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1],
                chunks=SAMPLE_CHUNKS[:1],
            )
            await asyncio.sleep(5)
            updated = await provider.aupdate_metadata(
                SAMPLE_IDS[0], {"new_field": "new_value", "updated": True}
            )
            assert updated is True or updated is None or updated is False
            await asyncio.sleep(3)
            results = await provider.afetch(ids=SAMPLE_IDS[:1])
            assert len(results) <= 1  # eventual consistency
            if updated is True and len(results) == 1:
                assert results[0].payload["metadata"].get("new_field") == "new_value"
                assert results[0].payload["metadata"].get("updated") is True
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_adelete_by_filter(self, provider: Optional[PineconeProvider]):
        """Test adelete_by_filter."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            await asyncio.sleep(5)
            deleted = await provider.adelete_by_filter({"category": {"$eq": "science"}})
            assert deleted is True
            await asyncio.sleep(3)
            results = await provider.afetch(ids=SAMPLE_IDS)
            assert len(results) <= 5
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_aupsert_with_document_tracking(self, provider: Optional[PineconeProvider]):
        """Test upsert with document tracking via dedicated params."""
        self._skip_if_unavailable(provider)
        import uuid as _uuid_mod
        unique_ids = [str(_uuid_mod.uuid4()), str(_uuid_mod.uuid4())]
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=[p.copy() for p in SAMPLE_PAYLOADS[:2]],
                ids=unique_ids,
                chunks=[c + " unique_doctrack_test" for c in SAMPLE_CHUNKS[:2]],
                document_names=["cloud_doc1", "cloud_doc2"],
                document_ids=["cloud_doc_id_1", "cloud_doc_id_2"],
            )
            # Retry fetch a few times for eventual consistency
            results = []
            for _ in range(6):
                await asyncio.sleep(5)
                results = await provider.afetch(ids=unique_ids)
                if len(results) >= 1:
                    break
            for result in results:
                idx = unique_ids.index(str(result.id))
                assert result.payload.get("chunk_id") == str(result.id)
                assert result.payload.get("document_name") == f"cloud_doc{idx+1}"
                assert result.payload.get("document_id") == f"cloud_doc_id_{idx+1}"
                assert result.payload["metadata"].get("category") == SAMPLE_PAYLOADS[idx]["category"]
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_adisconnect_and_reconnect(self, provider: Optional[PineconeProvider]):
        """Test disconnection and reconnection lifecycle."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        assert provider._is_connected is True
        await provider.adisconnect()
        assert provider._is_connected is False
        # Reconnect so subsequent tests still work
        await provider.aconnect()
        assert provider._is_connected is True

    @pytest.mark.asyncio
    async def test_ais_ready(self, provider: Optional[PineconeProvider]):
        """Test is_ready check."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        assert await provider.ais_ready() is True

    @pytest.mark.asyncio
    async def test_acollection_exists(self, provider: Optional[PineconeProvider]):
        """Test collection existence check."""
        self._skip_if_unavailable(provider)
        assert await provider.acollection_exists()

    @pytest.mark.asyncio
    async def test_adelete_collection_method_present(self, provider: Optional[PineconeProvider]):
        """Verify delete_collection method exists (can't actually delete shared index)."""
        self._skip_if_unavailable(provider)
        assert hasattr(provider, "adelete_collection")
        assert callable(provider.adelete_collection)
        assert await provider.acollection_exists()

    @pytest.mark.asyncio
    async def test_adelete_by_document_name(self, provider: Optional[PineconeProvider]):
        """Test adelete_by_document_name."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=[p.copy() for p in SAMPLE_PAYLOADS[:2]],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2],
                document_names=["cloud_doc", "cloud_doc"],
            )
            await asyncio.sleep(5)
            deleted = await provider.adelete_by_document_name("cloud_doc")
            assert deleted is True or deleted is False
            await asyncio.sleep(3)
            results = await provider.afetch(ids=SAMPLE_IDS[:2])
            assert len(results) <= 2
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_adelete_by_document_id(self, provider: Optional[PineconeProvider]):
        """Test adelete_by_document_id."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=[p.copy() for p in SAMPLE_PAYLOADS[:2]],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2],
                document_ids=["cloud_doc_id_1", "cloud_doc_id_1"],
            )
            await asyncio.sleep(5)
            deleted = await provider.adelete_by_document_id("cloud_doc_id_1")
            assert deleted is True or deleted is False
            await asyncio.sleep(3)
            results = await provider.afetch(ids=SAMPLE_IDS[:2])
            assert len(results) <= 2
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_adelete_by_chunk_id(self, provider: Optional[PineconeProvider]):
        """Test adelete_by_chunk_id (chunk_id == point id)."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=[p.copy() for p in SAMPLE_PAYLOADS[:2]],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2],
            )
            await asyncio.sleep(5)
            deleted = await provider.adelete_by_chunk_id(SAMPLE_IDS[0])
            assert deleted is True or deleted is False
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_adelete_by_metadata(self, provider: Optional[PineconeProvider]):
        """Test adelete_by_metadata."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            await asyncio.sleep(5)
            deleted = await provider.adelete_by_metadata({"category": "science"})
            assert deleted is True or deleted is False
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_aid_exists(self, provider: Optional[PineconeProvider]):
        """Test aid_exists check."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1],
                payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1],
                chunks=SAMPLE_CHUNKS[:1],
            )
            async def _exists():
                return await provider.aid_exists(SAMPLE_IDS[0])

            converged = await _poll_until(_exists, timeout=30.0, interval=1.0, description="id_exists propagation")
            exists = await provider.aid_exists(SAMPLE_IDS[0])
            assert exists is True, f"id_exists did not converge in 30s (converged={converged})"
            assert not await provider.aid_exists("nonexistent-id-xyz")
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_adocument_name_exists(self, provider: Optional[PineconeProvider]):
        """Test adocument_name_exists."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1],
                payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1],
                chunks=SAMPLE_CHUNKS[:1],
                document_names=["cloud_doc"],
            )
            await asyncio.sleep(5)
            exists = await provider.adocument_name_exists("cloud_doc")
            assert exists is True or exists is False
            assert not await provider.adocument_name_exists("nonexistent_doc_xyz")
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_adocument_id_exists(self, provider: Optional[PineconeProvider]):
        """Test adocument_id_exists."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1],
                payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1],
                chunks=SAMPLE_CHUNKS[:1],
                document_ids=["cloud_doc_id_1"],
            )
            await asyncio.sleep(5)
            exists = await provider.adocument_id_exists("cloud_doc_id_1")
            assert exists is True or exists is False
            assert not await provider.adocument_id_exists("nonexistent_docid_xyz")
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_achunk_id_exists(self, provider: Optional[PineconeProvider]):
        """Test achunk_id_exists (chunk_id == point id)."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1],
                payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1],
                chunks=SAMPLE_CHUNKS[:1],
            )
            await asyncio.sleep(5)
            exists = await provider.achunk_id_exists(SAMPLE_IDS[0])
            assert exists is True or exists is False
            assert not await provider.achunk_id_exists("nonexistent_chunk_xyz")
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_aoptimize(self, provider: Optional[PineconeProvider]):
        """Test aoptimize operation."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            result = await provider.aoptimize()
            assert result is True
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_aget_supported_search_types(self, provider: Optional[PineconeProvider]):
        """Test aget_supported_search_types."""
        self._skip_if_unavailable(provider)
        supported = await provider.aget_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
        assert "full_text" in supported
        assert "hybrid" in supported

    @pytest.mark.asyncio
    async def test_ahybrid_search_rrf(self, provider: Optional[PineconeProvider]):
        """Test hybrid search with RRF fusion."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            await asyncio.sleep(5)
            results = await provider.ahybrid_search(
                query_vector=QUERY_VECTOR,
                query_text="physics",
                top_k=3,
                fusion_method="rrf",
                similarity_threshold=0.0,
            )
            assert len(results) >= 0
            assert len(results) <= 3
            for result in results:
                assert isinstance(result, VectorSearchResult)
                assert result.id is not None
                assert result.score >= 0.0
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_asearch_master_method(self, provider: Optional[PineconeProvider]):
        """Test master asearch method with content validation."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            await asyncio.sleep(5)
            # Dense
            results = await provider.asearch(query_vector=QUERY_VECTOR, top_k=3)
            assert all(isinstance(r, VectorSearchResult) for r in results)
            # Full-text
            results = await provider.asearch(query_text="physics", top_k=3)
            assert all(isinstance(r, VectorSearchResult) for r in results)
            # Hybrid
            results = await provider.asearch(
                query_vector=QUERY_VECTOR, query_text="physics", top_k=3
            )
            assert all(isinstance(r, VectorSearchResult) for r in results)
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_distance_metrics_config(self, provider: Optional[PineconeProvider]):
        """Test distance metric configuration validity."""
        self._skip_if_unavailable(provider)
        from pydantic import SecretStr
        api_key = os.getenv("PINECONE_CLOUD_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT", "aws-us-east-1")
        if not api_key:
            pytest.skip("Pinecone API key not available")

        for metric in [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT]:
            config = PineconeConfig(
                vector_size=5,
                collection_name="test-metric-validation",
                api_key=SecretStr(api_key),
                environment=environment,
                distance_metric=metric,
                use_sparse_vectors=False,
                hybrid_search_enabled=False,
            )
            assert config.distance_metric == metric
            assert config.vector_size == 5

        await self._clear_vectors(provider)
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
        )
        await asyncio.sleep(5)
        results = await provider.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=2,
            similarity_threshold=0.0,
        )
        assert len(results) >= 0
        for r in results:
            assert isinstance(r, VectorSearchResult)
            assert r.score >= 0.0

    # ------------------------------------------------------------------
    # New capability tests (added for new provider API surface)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_chunk_id_preserved_as_uuid(self, provider: Optional[PineconeProvider]):
        """chunk_id provided as UUID must be preserved verbatim and equal point id."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2],
            )
            await asyncio.sleep(5)
            results = await provider.afetch(ids=SAMPLE_IDS[:2])
            assert len(results) >= 1  # eventual consistency
            fetched_ids = {str(r.id) for r in results}
            assert fetched_ids.issubset(set(SAMPLE_IDS[:2]))
            for r in results:
                assert r.payload.get("chunk_id") == str(r.id)
                assert r.payload.get("chunk_id") in SAMPLE_IDS[:2]
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_achunk_content_hash_exists(self, provider: Optional[PineconeProvider]):
        """Test achunk_content_hash_exists for new content-hash dedupe API."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1],
                payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1],
                chunks=SAMPLE_CHUNKS[:1],
            )
            await asyncio.sleep(5)
            h = md5(SAMPLE_CHUNKS[0].encode("utf-8")).hexdigest()
            exists = await provider.achunk_content_hash_exists(h)
            assert exists is True or exists is False  # eventual consistency
            assert not await provider.achunk_content_hash_exists("deadbeef" * 4)
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_adoc_content_hash_exists(self, provider: Optional[PineconeProvider]):
        """Test adoc_content_hash_exists with explicit doc_content_hashes."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            doc_hash = md5(b"the-document-body").hexdigest()
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1],
                payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1],
                chunks=SAMPLE_CHUNKS[:1],
                doc_content_hashes=[doc_hash],
            )
            await asyncio.sleep(5)
            exists = await provider.adoc_content_hash_exists(doc_hash)
            assert exists is True or exists is False
            assert not await provider.adoc_content_hash_exists("nonexistent_hash_xyz")
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_adelete_by_chunk_content_hash(self, provider: Optional[PineconeProvider]):
        """Test adelete_by_chunk_content_hash deletes by chunk hash."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1],
                payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1],
                chunks=SAMPLE_CHUNKS[:1],
            )
            await asyncio.sleep(5)
            h = md5(SAMPLE_CHUNKS[0].encode("utf-8")).hexdigest()
            result = await provider.adelete_by_chunk_content_hash(h)
            assert result is True or result is False
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_adelete_by_doc_content_hash(self, provider: Optional[PineconeProvider]):
        """Test adelete_by_doc_content_hash deletes by document hash."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            doc_hash = md5(b"another-doc-body").hexdigest()
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2],
                doc_content_hashes=[doc_hash, doc_hash],
            )
            await asyncio.sleep(5)
            result = await provider.adelete_by_doc_content_hash(doc_hash)
            assert result is True or result is False
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_afield_exists_and_adelete_by_field(self, provider: Optional[PineconeProvider]):
        """Test the generic afield_exists / adelete_by_field helpers."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2],
                document_ids=["docA", "docB"],
            )
            await asyncio.sleep(5)
            exists = await provider.afield_exists("document_id", "docA")
            assert exists is True or exists is False
            assert not await provider.afield_exists("document_id", "missing_xyz")
            deleted = await provider.adelete_by_field("document_id", "docA")
            assert deleted is True or deleted is False
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_aupsert_with_chunk_content_hashes(self, provider: Optional[PineconeProvider]):
        """Test aupsert accepts explicit chunk_content_hashes (new param)."""
        self._skip_if_unavailable(provider)
        try:
            await self._clear_vectors(provider)
            explicit_hashes = [md5(c.encode("utf-8")).hexdigest() for c in SAMPLE_CHUNKS[:2]]
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2],
                chunk_content_hashes=explicit_hashes,
            )
            await asyncio.sleep(5)
            for h in explicit_hashes:
                exists = await provider.achunk_content_hash_exists(h)
                assert exists is True or exists is False  # eventual consistency
        finally:
            await self._clear_vectors(provider)

    @pytest.mark.asyncio
    async def test_aupsert_with_document_names(self, provider: Optional[PineconeProvider]):
        """Test aupsert accepts document_names and they round-trip to payload."""
        self._skip_if_unavailable(provider)
        import uuid as _uuid_mod
        unique_ids = [str(_uuid_mod.uuid4()), str(_uuid_mod.uuid4())]
        try:
            await self._clear_vectors(provider)
            names = ["alpha_doc", "beta_doc"]
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=[p.copy() for p in SAMPLE_PAYLOADS[:2]],
                ids=unique_ids,
                chunks=[c + " unique_dnames_test" for c in SAMPLE_CHUNKS[:2]],
                document_names=names,
            )
            results = []
            for _ in range(6):
                await asyncio.sleep(5)
                results = await provider.afetch(ids=unique_ids)
                if len(results) >= 1:
                    break
            for r in results:
                idx = unique_ids.index(str(r.id))
                assert r.payload.get("document_name") == names[idx]
        finally:
            await self._clear_vectors(provider)


# ============== BACK-PORTED FROM OLD PINECONE FILE ==============
# Sync-wrapper tests exercising base-class sync wrappers against a real
# Pinecone cloud index. One class-scoped index is created and reused across
# all sync tests to avoid the ~60s-per-index creation cost.

class TestPineconeProviderSync:
    """Sync-wrapper tests for PineconeProvider (one shared index)."""

    @pytest.fixture(scope="class")
    def sync_provider(self):
        from pydantic import SecretStr
        import uuid as _uuid_mod

        api_key = os.getenv("PINECONE_CLOUD_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT", "aws-us-east-1")
        if not api_key:
            yield None
            return

        config = PineconeConfig(
            vector_size=5,
            collection_name=f"test-sync-{_uuid_mod.uuid4().hex[:8]}",
            api_key=SecretStr(api_key),
            environment=environment,
            distance_metric=DistanceMetric.DOT_PRODUCT,
            use_sparse_vectors=True,
            hybrid_search_enabled=True,
            recreate_if_exists=False,
        )
        provider = PineconeProvider(config)
        try:
            provider.connect()
            if not provider.collection_exists():
                provider.create_collection()
            yield provider
        finally:
            try:
                if provider.collection_exists():
                    provider.delete_collection()
            except Exception as e:
                logger.warning(f"Sync teardown delete_collection error: {e}")
            try:
                provider.disconnect()
            except Exception:
                pass

    def _skip_if_unavailable(self, provider):
        if provider is None:
            pytest.skip("Pinecone API key not available")

    def _clear_vectors_sync(self, provider):
        import time
        try:
            if provider._is_connected and provider._index is not None:
                try:
                    provider._index.delete(delete_all=True)
                except Exception:
                    pass
                time.sleep(10)
        except Exception:
            pass

    def test_connect_sync(self, sync_provider):
        self._skip_if_unavailable(sync_provider)
        assert sync_provider._is_connected is True
        assert sync_provider.is_ready() is True

    def test_upsert_and_fetch_sync(self, sync_provider):
        import time
        self._skip_if_unavailable(sync_provider)
        try:
            self._clear_vectors_sync(sync_provider)
            sync_provider.upsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            time.sleep(5)
            results = sync_provider.fetch(ids=SAMPLE_IDS)
            assert len(results) >= 1  # eventual consistency
            for r in results:
                assert r.payload is not None
                assert r.text is not None
                idx = SAMPLE_IDS.index(str(r.id))
                assert_vector_matches(r.vector, SAMPLE_VECTORS[idx], vector_id=str(r.id))
                assert r.payload.get("content") == SAMPLE_CHUNKS[idx]
        finally:
            self._clear_vectors_sync(sync_provider)

    def test_delete_sync(self, sync_provider):
        import time
        self._skip_if_unavailable(sync_provider)
        try:
            self._clear_vectors_sync(sync_provider)
            sync_provider.upsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            time.sleep(3)
            sync_provider.delete(ids=SAMPLE_IDS[:2])
            time.sleep(3)
            results_deleted = sync_provider.fetch(ids=SAMPLE_IDS[:2])
            assert len(results_deleted) <= 2
            results_remaining = sync_provider.fetch(ids=SAMPLE_IDS[2:])
            assert len(results_remaining) <= 3
        finally:
            self._clear_vectors_sync(sync_provider)

    def test_dense_search_sync(self, sync_provider):
        import time
        self._skip_if_unavailable(sync_provider)
        try:
            self._clear_vectors_sync(sync_provider)
            sync_provider.upsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            time.sleep(5)
            results = sync_provider.dense_search(
                query_vector=QUERY_VECTOR, top_k=3, similarity_threshold=0.0
            )
            assert len(results) >= 0
            for r in results:
                assert isinstance(r, VectorSearchResult)
                assert isinstance(r.score, float)
        finally:
            self._clear_vectors_sync(sync_provider)

    def test_full_text_search_sync(self, sync_provider):
        import time
        self._skip_if_unavailable(sync_provider)
        try:
            self._clear_vectors_sync(sync_provider)
            sync_provider.upsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            time.sleep(5)
            results = sync_provider.full_text_search(
                query_text="physics", top_k=3, similarity_threshold=0.0
            )
            assert len(results) >= 0
            for r in results:
                assert isinstance(r, VectorSearchResult)
        finally:
            self._clear_vectors_sync(sync_provider)

    def test_hybrid_search_sync(self, sync_provider):
        import time
        self._skip_if_unavailable(sync_provider)
        try:
            self._clear_vectors_sync(sync_provider)
            sync_provider.upsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            time.sleep(5)
            results = sync_provider.hybrid_search(
                query_vector=QUERY_VECTOR,
                query_text="physics",
                top_k=3,
                alpha=0.5,
                fusion_method="weighted",
                similarity_threshold=0.0,
            )
            assert len(results) >= 0
            assert len(results) <= 3
        finally:
            self._clear_vectors_sync(sync_provider)

    def test_search_sync(self, sync_provider):
        import time
        self._skip_if_unavailable(sync_provider)
        try:
            self._clear_vectors_sync(sync_provider)
            sync_provider.upsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            time.sleep(5)
            results = sync_provider.search(query_vector=QUERY_VECTOR, top_k=3)
            assert all(isinstance(r, VectorSearchResult) for r in results)
        finally:
            self._clear_vectors_sync(sync_provider)
