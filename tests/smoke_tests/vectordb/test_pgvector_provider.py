"""
Comprehensive smoke tests for PgVector vector database provider.

Mirrors the Qdrant smoke-test structure. Adapted to the PgVectorProvider
async-first API surface (a* methods, sync wrappers provided by base class).
"""

import os
import pytest
from hashlib import md5
from typing import List, Dict, Any, Optional
from pydantic import SecretStr

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from upsonic.vectordb.providers.pgvector import PgVectorProvider
from upsonic.vectordb.config import (
    PgVectorConfig,
    DistanceMetric,
    HNSWIndexConfig,
    IVFIndexConfig,
)
from upsonic.utils.package.exception import (
    VectorDBConnectionError,
    UpsertError,
)
from upsonic.schemas.vector_schemas import VectorSearchResult


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

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
    "To be or not to be, that is the question",
    "It was the best of times, it was the worst of times",
    "The unexamined life is not worth living",
]

# Real UUID strings, preserved verbatim.
SAMPLE_IDS: List[str] = [
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
    "44444444-4444-4444-8444-444444444444",
    "55555555-5555-4555-8555-555555555555",
]

QUERY_VECTOR: List[float] = [0.15, 0.25, 0.35, 0.45, 0.55]
QUERY_TEXT: str = "physics theory"


def assert_vector_matches(actual_vector: Any, expected_vector: List[float],
                          vector_id: str = "", tolerance: float = 1e-6) -> None:
    assert actual_vector is not None, f"Vector is None for {vector_id}"
    assert len(actual_vector) == len(expected_vector)
    for j, (a, e) in enumerate(zip([float(x) for x in actual_vector], expected_vector)):
        assert abs(a - e) < tolerance, \
            f"Vector element {j} mismatch for {vector_id}: {a} != {e}"


def get_connection_string() -> Optional[str]:
    conn_str = (os.getenv("POSTGRES_CONNECTION_STRING")
                or os.getenv("PGVECTOR_CONNECTION_STRING")
                or os.getenv("DATABASE_URL")
                or os.getenv("POSTGRES_URL"))
    if conn_str:
        if conn_str.startswith("postgresql://"):
            conn_str = conn_str.replace("postgresql://", "postgresql+psycopg://", 1)
        return conn_str

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5434")
    user = os.getenv("POSTGRES_USER", "upsonic_test")
    password = os.getenv("POSTGRES_PASSWORD", "test_password")
    dbname = os.getenv("POSTGRES_DB", "upsonic_test")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"


class TestPgVectorProvider:
    """Comprehensive tests for PgVectorProvider."""

    @pytest.fixture
    def config(self) -> Optional[PgVectorConfig]:
        import uuid
        conn_str = get_connection_string()
        if not conn_str:
            return None
        unique_name = f"test_pgvector_{uuid.uuid4().hex[:8]}"
        return PgVectorConfig(
            vector_size=5,
            collection_name=unique_name,
            connection_string=SecretStr(conn_str),
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig(m=16, ef_construction=200),
        )

    @pytest.fixture
    def provider(self, config: Optional[PgVectorConfig]) -> Optional[PgVectorProvider]:
        if config is None:
            return None
        return PgVectorProvider(config)

    def _skip_if_unavailable(self, provider: Optional[PgVectorProvider]):
        if provider is None:
            pytest.skip("PostgreSQL connection string not available.")

    async def _ensure_connected(self, provider: PgVectorProvider):
        try:
            await provider.aconnect()
        except VectorDBConnectionError:
            pytest.skip("PostgreSQL connection failed. Ensure pgvector is running.")

    # ----------------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_initialization(self, provider, config):
        self._skip_if_unavailable(provider)
        assert provider._config == config
        assert provider._config.collection_name.startswith("test_pgvector_")
        assert provider._config.vector_size == 5
        assert provider._config.distance_metric == DistanceMetric.COSINE
        assert not provider._is_connected
        assert provider._engine is None
        assert provider._session_factory is None
        assert provider.name is not None
        assert isinstance(provider.id, str)
        assert len(provider.id) > 0

    @pytest.mark.asyncio
    async def test_connect(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        assert provider._is_connected is True
        assert provider._engine is not None
        assert provider._session_factory is not None
        assert await provider.ais_ready() is True
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_connect_sync(self, provider):
        self._skip_if_unavailable(provider)
        try:
            provider.connect()
        except VectorDBConnectionError:
            pytest.skip("PostgreSQL connection failed.")
        assert provider._is_connected is True
        assert provider.is_ready() is True
        provider.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        assert provider._is_connected is True
        await provider.adisconnect()
        assert provider._is_connected is False

    @pytest.mark.asyncio
    async def test_is_ready(self, provider):
        self._skip_if_unavailable(provider)
        assert await provider.ais_ready() is False
        await self._ensure_connected(provider)
        assert await provider.ais_ready() is True
        await provider.adisconnect()

    # ----------------------------------------------------------------------
    # Collection
    # ----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_collection(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
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
    async def test_collection_exists(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        assert not await provider.acollection_exists()
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_delete_collection(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        assert await provider.acollection_exists()
        await provider.adelete_collection()
        assert not await provider.acollection_exists()
        await provider.adisconnect()

    # ----------------------------------------------------------------------
    # Upsert / fetch / delete
    # ----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_upsert(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS,
        )
        results = await provider.afetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        for result in results:
            assert result.id is not None
            assert result.payload is not None
            content = result.text
            assert content in SAMPLE_CHUNKS
            idx = SAMPLE_CHUNKS.index(content)
            assert result.payload["metadata"]["category"] == SAMPLE_PAYLOADS[idx]["category"]
            assert result.payload["metadata"]["author"] == SAMPLE_PAYLOADS[idx]["author"]
            assert result.payload["metadata"]["year"] == SAMPLE_PAYLOADS[idx]["year"]
            assert result.text == SAMPLE_CHUNKS[idx]
            assert result.vector is not None
            assert_vector_matches(result.vector, SAMPLE_VECTORS[idx], vector_id=str(result.id))
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_upsert_sync(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS,
        )
        results = await provider.afetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_upsert_with_document_tracking(self, provider):
        """document_name / document_id / chunk_id must come from dedicated params."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
            document_ids=["doc_id_1", "doc_id_2"],
            document_names=["doc1", "doc2"],
        )
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        for result in results:
            chunk_id = result.payload.get("chunk_id")
            assert chunk_id in SAMPLE_IDS[:2]
            idx = SAMPLE_IDS.index(chunk_id)
            assert result.payload.get("document_name") == f"doc{idx+1}"
            assert result.payload.get("document_id") == f"doc_id_{idx+1}"
            assert result.payload["metadata"]["category"] == SAMPLE_PAYLOADS[idx]["category"]
            assert result.text == SAMPLE_CHUNKS[idx]
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_upsert_validation_error(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        with pytest.raises((ValueError, UpsertError)):
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:3],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2],
            )
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_fetch(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
        )
        results = await provider.afetch(ids=SAMPLE_IDS[:3])
        assert len(results) == 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.score == 1.0
            assert result.payload is not None
            assert result.text is not None
            assert result.vector is not None
            assert len(result.vector) == 5
            content = result.text
            idx = SAMPLE_CHUNKS.index(content)
            assert_vector_matches(result.vector, SAMPLE_VECTORS[idx])
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_delete(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
        )
        await provider.adelete(ids=SAMPLE_IDS[:2])
        assert len(await provider.afetch(ids=SAMPLE_IDS[:2])) == 0
        assert len(await provider.afetch(ids=SAMPLE_IDS[2:])) == 3
        await provider.adisconnect()

    # ----------------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_dense_search(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
        )
        results = await provider.adense_search(
            query_vector=QUERY_VECTOR, top_k=3, similarity_threshold=0.0
        )
        assert 0 < len(results) <= 3
        for r in results:
            assert isinstance(r, VectorSearchResult)
            assert 0.0 <= r.score <= 1.0
            assert r.payload is not None
            assert r.text is not None
            assert r.vector is not None
            assert len(r.vector) == 5
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_full_text_search(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
        )
        results = await provider.afull_text_search(
            query_text="physics", top_k=3, similarity_threshold=0.0
        )
        assert len(results) > 0
        for r in results:
            assert r.text is not None
            assert "physics" in r.text.lower() or "theory" in r.text.lower()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_hybrid_search(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
        )
        results = await provider.ahybrid_search(
            query_vector=QUERY_VECTOR, query_text="physics",
            top_k=3, alpha=0.5, fusion_method="weighted",
            similarity_threshold=0.0,
        )
        assert 0 < len(results) <= 3
        for r in results:
            assert r.score >= 0.0
            assert r.payload is not None
            assert r.vector is not None
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_hybrid_search_rrf(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
        )
        results = await provider.ahybrid_search(
            query_vector=QUERY_VECTOR, query_text="physics",
            top_k=3, fusion_method="rrf", similarity_threshold=0.0,
        )
        assert len(results) > 0
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_search_master_method(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
        )
        results = await provider.asearch(query_vector=QUERY_VECTOR, top_k=3)
        assert len(results) > 0
        results = await provider.asearch(query_text="physics", top_k=3)
        assert len(results) > 0
        results = await provider.asearch(
            query_vector=QUERY_VECTOR, query_text="physics", top_k=3
        )
        assert len(results) > 0
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_search_with_filter(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        payloads = [{**p, "metadata": {"category": p["category"]}} for p in SAMPLE_PAYLOADS]
        await provider.aupsert(
            vectors=SAMPLE_VECTORS, payloads=payloads,
            ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
        )
        # Nested filter path
        results = await provider.adense_search(
            query_vector=QUERY_VECTOR, top_k=5,
            filter={"metadata.category": "science"},
        )
        assert len(results) > 0
        for r in results:
            # pgvector flattens JSONB metadata into top-level payload
            category = r.payload.get("category") or (r.payload.get("metadata") or {}).get("category")
            assert category == "science"
        await provider.adisconnect()

    # ----------------------------------------------------------------------
    # Count / existence / deletes by field
    # ----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_count(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        initial = await provider.aget_count()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
        )
        assert await provider.aget_count() == initial + 5
        await provider.adelete(ids=SAMPLE_IDS[:2])
        assert await provider.aget_count() == initial + 3
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_aid_exists(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
        )
        assert await provider.aid_exists(SAMPLE_IDS[0])
        assert not await provider.aid_exists("99999999-9999-4999-8999-999999999999")
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_chunk_id_preserved_as_uuid(self, provider):
        """chunk_ids provided as UUID strings must be preserved verbatim."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
        )
        results = await provider.afetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        fetched_ids = {str(r.id) for r in results}
        assert fetched_ids == set(SAMPLE_IDS[:2])
        for r in results:
            assert r.payload.get("chunk_id") in SAMPLE_IDS[:2]
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_achunk_id_exists(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
        )
        assert await provider.achunk_id_exists(SAMPLE_IDS[0])
        assert not await provider.achunk_id_exists("nonexistent")
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_adocument_name_exists(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
            document_names=["pg_doc"],
        )
        assert await provider.adocument_name_exists("pg_doc")
        assert not await provider.adocument_name_exists("nonexistent")
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_adocument_id_exists(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
            document_ids=["pg_doc_id_1"],
        )
        assert await provider.adocument_id_exists("pg_doc_id_1")
        assert not await provider.adocument_id_exists("nonexistent")
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_achunk_content_hash_exists(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
        )
        h = md5(SAMPLE_CHUNKS[0].encode("utf-8")).hexdigest()
        assert await provider.achunk_content_hash_exists(h)
        assert not await provider.achunk_content_hash_exists("deadbeef" * 4)
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_adoc_content_hash_exists(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        doc_hash = md5(b"the-document-body").hexdigest()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
            doc_content_hashes=[doc_hash],
        )
        assert await provider.adoc_content_hash_exists(doc_hash)
        assert not await provider.adoc_content_hash_exists("nonexistent_hash")
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_adelete_by_document_name(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        initial = await provider.aget_count()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
            document_names=["pg_doc"] * 2,
        )
        assert await provider.aget_count() == initial + 2
        assert await provider.adelete_by_document_name("pg_doc") is True
        assert await provider.aget_count() == initial
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_adelete_by_document_id(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
            document_ids=["pg_doc_id_1", "pg_doc_id_1"],
        )
        assert await provider.adelete_by_document_id("pg_doc_id_1") is True
        assert len(await provider.afetch(ids=SAMPLE_IDS[:2])) == 0
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_adelete_by_chunk_id(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
        )
        assert await provider.adelete_by_chunk_id(SAMPLE_IDS[0]) is True
        assert not await provider.achunk_id_exists(SAMPLE_IDS[0])
        assert await provider.achunk_id_exists(SAMPLE_IDS[1])
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_adelete_by_chunk_content_hash(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
        )
        h = md5(SAMPLE_CHUNKS[0].encode("utf-8")).hexdigest()
        assert await provider.achunk_content_hash_exists(h)
        assert await provider.adelete_by_chunk_content_hash(h) is True
        assert not await provider.achunk_content_hash_exists(h)
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_adelete_by_doc_content_hash(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        doc_hash = md5(b"another-doc-body").hexdigest()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
            doc_content_hashes=[doc_hash, doc_hash],
        )
        assert await provider.adoc_content_hash_exists(doc_hash)
        assert await provider.adelete_by_doc_content_hash(doc_hash) is True
        assert not await provider.adoc_content_hash_exists(doc_hash)
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_adelete_by_metadata(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
        )
        # Non-standard 'category' went into JSONB metadata column.
        assert await provider.adelete_by_metadata({"category": "science"}) is True
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_adelete_by_metadata_nested_filter(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
        )
        initial = await provider.aget_count()
        assert initial == 5

        assert await provider.adelete_by_metadata({"metadata.category": "science"}) is True

        remaining_count = await provider.aget_count()
        assert remaining_count == initial - 2

        remaining = await provider.afetch(ids=SAMPLE_IDS)
        for r in remaining:
            category = r.payload.get("category") or (r.payload.get("metadata") or {}).get("category")
            assert category != "science"
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_afield_exists_and_adelete_by_field(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
            document_ids=["docA", "docB"],
        )
        assert await provider.afield_exists("document_id", "docA")
        assert not await provider.afield_exists("document_id", "missing")
        assert await provider.adelete_by_field("document_id", "docA") is True
        assert not await provider.afield_exists("document_id", "docA")
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_aupsert_with_chunk_content_hashes(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        explicit = [md5(c.encode("utf-8")).hexdigest() for c in SAMPLE_CHUNKS[:2]]
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
            chunk_content_hashes=explicit,
        )
        for h in explicit:
            assert await provider.achunk_content_hash_exists(h)
        await provider.adisconnect()

    # ----------------------------------------------------------------------
    # Metadata update / optimize / search types
    # ----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_aupdate_metadata(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
        )
        updated = await provider.aupdate_metadata(
            SAMPLE_IDS[0], {"new_field": "new_value", "updated": True}
        )
        assert updated is True
        results = await provider.afetch(ids=SAMPLE_IDS[:1])
        assert results[0].payload["metadata"].get("new_field") == "new_value"
        assert results[0].payload["metadata"].get("updated") is True
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_aoptimize(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        assert await provider.aoptimize() is True
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_aget_supported_search_types(self, provider):
        self._skip_if_unavailable(provider)
        supported = await provider.aget_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
        assert "full_text" in supported
        assert "hybrid" in supported

    # ----------------------------------------------------------------------
    # Config variants
    # ----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_recreate_if_exists(self, provider):
        self._skip_if_unavailable(provider)
        import uuid
        conn_str = get_connection_string()
        if not conn_str:
            pytest.skip("PostgreSQL connection string not available")
        unique_name = f"test_recreate_{uuid.uuid4().hex[:8]}"
        config = PgVectorConfig(
            vector_size=5,
            collection_name=unique_name,
            connection_string=SecretStr(conn_str),
            recreate_if_exists=True,
        )
        provider2 = PgVectorProvider(config)
        await self._ensure_connected(provider2)
        await provider2.acreate_collection()
        await provider2.aupsert(
            vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
        )
        await provider2.acreate_collection()
        assert await provider2.aget_count() == 0
        await provider2.adisconnect()

    @pytest.mark.asyncio
    async def test_ivfflat_index_config(self, provider):
        self._skip_if_unavailable(provider)
        import uuid
        conn_str = get_connection_string()
        if not conn_str:
            pytest.skip("PostgreSQL connection string not available")
        unique_name = f"test_ivfflat_{uuid.uuid4().hex[:8]}"
        config = PgVectorConfig(
            vector_size=5,
            collection_name=unique_name,
            connection_string=SecretStr(conn_str),
            index=IVFIndexConfig(nlist=10),
        )
        provider2 = PgVectorProvider(config)
        await self._ensure_connected(provider2)
        await provider2.acreate_collection()
        await provider2.aupsert(
            vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
        )
        results = await provider2.adense_search(
            query_vector=QUERY_VECTOR, top_k=2, similarity_threshold=0.0
        )
        assert len(results) > 0
        await provider2.adisconnect()

    @pytest.mark.asyncio
    async def test_distance_metrics(self, provider):
        self._skip_if_unavailable(provider)
        import uuid
        conn_str = get_connection_string()
        if not conn_str:
            pytest.skip("PostgreSQL connection string not available")
        for metric in [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT]:
            unique_name = f"test_{metric.value.lower()}_{uuid.uuid4().hex[:8]}"
            config = PgVectorConfig(
                vector_size=5,
                collection_name=unique_name,
                connection_string=SecretStr(conn_str),
                distance_metric=metric,
            )
            provider2 = PgVectorProvider(config)
            await self._ensure_connected(provider2)
            await provider2.acreate_collection()
            await provider2.aupsert(
                vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
            )
            results = await provider2.adense_search(
                query_vector=QUERY_VECTOR, top_k=2, similarity_threshold=0.0
            )
            assert len(results) > 0
            for r in results:
                assert 0.0 <= r.score <= 1.0
            await provider2.adisconnect()

    @pytest.mark.asyncio
    async def test_clear(self, provider):
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.aupsert(
            vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
        )
        assert await provider.aget_count() == 5
        await provider.aclear()
        assert await provider.aget_count() == 0
        assert await provider.acollection_exists()
        await provider.adisconnect()


# ============== BACK-PORTED FROM OLD PGVECTOR FILE ==============
# Dedicated synchronous test class that exercises the base-class sync
# wrappers end-to-end against a real pgvector backend. These tests are
# plain (non-async) functions; _run_async_from_sync transparently dispatches
# to a worker thread if a running loop is detected.

class TestPgVectorProviderSync:
    """Sync-wrapper tests for PgVectorProvider."""

    def _make_provider(self) -> Optional[PgVectorProvider]:
        import uuid
        conn_str = get_connection_string()
        if not conn_str:
            return None
        unique_name = f"test_pgvector_sync_{uuid.uuid4().hex[:8]}"
        config = PgVectorConfig(
            vector_size=5,
            collection_name=unique_name,
            connection_string=SecretStr(conn_str),
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig(m=16, ef_construction=200),
        )
        return PgVectorProvider(config)

    def _connect_or_skip(self, provider: PgVectorProvider) -> None:
        try:
            provider.connect()
        except VectorDBConnectionError:
            pytest.skip("PostgreSQL connection failed. Ensure pgvector is running.")

    def test_connect_sync(self):
        provider = self._make_provider()
        if provider is None:
            pytest.skip("PostgreSQL connection string not available.")
        self._connect_or_skip(provider)
        try:
            assert provider._is_connected is True
            assert provider.is_ready() is True
        finally:
            provider.disconnect()
            assert provider._is_connected is False

    def test_upsert_and_fetch_sync(self):
        provider = self._make_provider()
        if provider is None:
            pytest.skip("PostgreSQL connection string not available.")
        self._connect_or_skip(provider)
        try:
            provider.create_collection()
            provider.upsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            results = provider.fetch(ids=SAMPLE_IDS)
            assert len(results) == 5
            for r in results:
                assert r.payload is not None
                assert r.text in SAMPLE_CHUNKS
                idx = SAMPLE_CHUNKS.index(r.text)
                assert_vector_matches(r.vector, SAMPLE_VECTORS[idx],
                                       vector_id=str(r.id))
        finally:
            try:
                provider.delete_collection()
            except Exception:
                pass
            provider.disconnect()

    def test_delete_sync(self):
        provider = self._make_provider()
        if provider is None:
            pytest.skip("PostgreSQL connection string not available.")
        self._connect_or_skip(provider)
        try:
            provider.create_collection()
            provider.upsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
            )
            provider.delete(ids=SAMPLE_IDS[:2])
            assert len(provider.fetch(ids=SAMPLE_IDS[:2])) == 0
            assert len(provider.fetch(ids=SAMPLE_IDS[2:])) == 3
        finally:
            try:
                provider.delete_collection()
            except Exception:
                pass
            provider.disconnect()

    def test_dense_search_sync(self):
        provider = self._make_provider()
        if provider is None:
            pytest.skip("PostgreSQL connection string not available.")
        self._connect_or_skip(provider)
        try:
            provider.create_collection()
            provider.upsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
            )
            results = provider.dense_search(
                query_vector=QUERY_VECTOR, top_k=3, similarity_threshold=0.0
            )
            assert len(results) > 0
            for r in results:
                assert isinstance(r, VectorSearchResult)
                assert r.vector is not None
                idx = SAMPLE_CHUNKS.index(r.text)
                assert_vector_matches(r.vector, SAMPLE_VECTORS[idx],
                                       vector_id=str(r.id))
        finally:
            try:
                provider.delete_collection()
            except Exception:
                pass
            provider.disconnect()

    def test_full_text_search_sync(self):
        provider = self._make_provider()
        if provider is None:
            pytest.skip("PostgreSQL connection string not available.")
        self._connect_or_skip(provider)
        try:
            provider.create_collection()
            provider.upsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
            )
            results = provider.full_text_search(
                query_text="physics", top_k=3, similarity_threshold=0.0
            )
            assert len(results) > 0
            for r in results:
                assert r.text is not None
        finally:
            try:
                provider.delete_collection()
            except Exception:
                pass
            provider.disconnect()

    def test_hybrid_search_sync(self):
        provider = self._make_provider()
        if provider is None:
            pytest.skip("PostgreSQL connection string not available.")
        self._connect_or_skip(provider)
        try:
            provider.create_collection()
            provider.upsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
            )
            results = provider.hybrid_search(
                query_vector=QUERY_VECTOR, query_text="physics",
                top_k=3, alpha=0.5, fusion_method="weighted",
                similarity_threshold=0.0,
            )
            assert len(results) > 0
        finally:
            try:
                provider.delete_collection()
            except Exception:
                pass
            provider.disconnect()

    def test_search_sync(self):
        provider = self._make_provider()
        if provider is None:
            pytest.skip("PostgreSQL connection string not available.")
        self._connect_or_skip(provider)
        try:
            provider.create_collection()
            provider.upsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
            )
            results = provider.search(query_vector=QUERY_VECTOR, top_k=3)
            assert len(results) > 0
        finally:
            try:
                provider.delete_collection()
            except Exception:
                pass
            provider.disconnect()
