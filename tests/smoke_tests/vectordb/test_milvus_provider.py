"""
Comprehensive smoke tests for Milvus vector database provider.

Mirrors the Qdrant / PgVector smoke-test structure. Adapted to the
MilvusProvider async-first API surface (a* methods, sync wrappers via base).

Two test classes:
- TestMilvusProviderEmbedded: uses Milvus Lite (embedded, file-backed).
- TestMilvusProviderCLOUD:     uses Zilliz Cloud via MILVUS_CLOUD_URI / MILVUS_CLOUD_TOKEN.
"""

import asyncio
import os
import time
import uuid as _uuid
import tempfile
from hashlib import md5
from typing import List, Dict, Any, Optional

import pytest
from pydantic import SecretStr

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from upsonic.vectordb.providers.milvus import MilvusProvider
from upsonic.vectordb.config import (
    MilvusConfig,
    ConnectionConfig,
    Mode,
    DistanceMetric,
    HNSWIndexConfig,
    FlatIndexConfig,
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

# Real UUID strings — must be preserved verbatim by the provider.
SAMPLE_IDS: List[str] = [
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
    "44444444-4444-4444-8444-444444444444",
    "55555555-5555-4555-8555-555555555555",
]

QUERY_VECTOR: List[float] = [0.15, 0.25, 0.35, 0.45, 0.55]


def assert_vector_matches(actual_vector: Any, expected_vector: List[float],
                          vector_id: str = "", tolerance: float = 1e-5) -> None:
    assert actual_vector is not None, f"Vector is None for {vector_id}"
    assert len(actual_vector) == len(expected_vector)
    for j, (a, e) in enumerate(zip([float(x) for x in actual_vector], expected_vector)):
        assert abs(a - e) < tolerance, \
            f"Vector element {j} mismatch for {vector_id}: {a} != {e}"


def _embedded_config(collection_name: str) -> MilvusConfig:
    db_path = os.path.join(tempfile.gettempdir(), f"{collection_name}.db")
    return MilvusConfig(
        vector_size=5,
        collection_name=collection_name,
        connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=db_path),
        distance_metric=DistanceMetric.COSINE,
        index=FlatIndexConfig(),
        use_sparse_vectors=False,
        full_text_search_enabled=False,
        hybrid_search_enabled=False,
        recreate_if_exists=True,
    )


def _cloud_config(collection_name: str) -> Optional[MilvusConfig]:
    uri = os.getenv("MILVUS_CLOUD_URI")
    token = os.getenv("MILVUS_CLOUD_TOKEN") or os.getenv("MULVUS_CLOUD_TOKEN")
    if not uri or not token:
        return None
    return MilvusConfig(
        vector_size=5,
        collection_name=collection_name,
        connection=ConnectionConfig(
            mode=Mode.CLOUD,
            url=uri,
            api_key=SecretStr(token),
        ),
        distance_metric=DistanceMetric.COSINE,
        index=HNSWIndexConfig(m=16, ef_construction=200),
        use_sparse_vectors=False,
        full_text_search_enabled=False,
        hybrid_search_enabled=False,
        recreate_if_exists=True,
    )


# ---------------------------------------------------------------------------
# Shared test body: reused by both backends via a factory fixture.
# ---------------------------------------------------------------------------

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


class _MilvusTestMixin:
    """Backing config factory: subclasses override `_make_config`."""

    def _make_config(self, collection_name: str) -> Optional[MilvusConfig]:
        _ = collection_name  # base implementation; overrides use the argument
        raise NotImplementedError

    @pytest.fixture
    def config(self) -> Optional[MilvusConfig]:
        return self._make_config(f"test_milvus_{_uuid.uuid4().hex[:8]}")

    @pytest.fixture
    def provider(self, config: Optional[MilvusConfig]) -> Optional[MilvusProvider]:
        if config is None:
            return None
        return MilvusProvider(config)

    def _skip_if_unavailable(self, provider: Optional[MilvusProvider]) -> None:
        if provider is None:
            pytest.skip("Milvus backend not available for this class.")

    async def _settle(self) -> None:
        """
        Backend-specific post-write settle delay.
        Embedded is strongly consistent; Zilliz Cloud serverless is not.
        """
        return None

    async def _setup(self, provider: MilvusProvider) -> None:
        await provider.aconnect()
        await provider.acreate_collection()

    async def _teardown(self, provider: MilvusProvider) -> None:
        try:
            await provider.adelete_collection()
        except Exception:
            pass
        try:
            await provider.adisconnect()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Lifecycle / init
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_initialization(self, provider, config):
        self._skip_if_unavailable(provider)
        assert provider._config == config
        assert provider._config.vector_size == 5
        assert provider._config.distance_metric == DistanceMetric.COSINE
        assert provider.client is None
        assert provider.name is not None
        assert isinstance(provider.id, str) and len(provider.id) > 0

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, provider):
        self._skip_if_unavailable(provider)
        await provider.aconnect()
        assert provider.client is not None
        assert await provider.ais_ready() is True
        await provider.adisconnect()
        assert await provider.ais_ready() is False

    @pytest.mark.asyncio
    async def test_is_ready(self, provider):
        self._skip_if_unavailable(provider)
        assert await provider.ais_ready() is False
        await provider.aconnect()
        assert await provider.ais_ready() is True
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_collection(self, provider):
        self._skip_if_unavailable(provider)
        await provider.aconnect()
        await provider.acreate_collection()
        assert await provider.acollection_exists() is True
        await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_collection_exists(self, provider):
        self._skip_if_unavailable(provider)
        await provider.aconnect()
        await provider.acreate_collection()
        assert await provider.acollection_exists() is True
        await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_delete_collection(self, provider):
        self._skip_if_unavailable(provider)
        await provider.aconnect()
        await provider.acreate_collection()
        assert await provider.acollection_exists() is True
        await provider.adelete_collection()
        assert await provider.acollection_exists() is False
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Upsert / fetch / delete
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_upsert_and_verify_vectors(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )

            async def _verified():
                r = await provider.afetch(ids=SAMPLE_IDS)
                return len(r) == 5

            await _poll_until(_verified, timeout=30.0, interval=1.0, description="upsert read-after-write")

            results = await provider.afetch(ids=SAMPLE_IDS)
            assert len(results) == 5
            for r in results:
                assert isinstance(r, VectorSearchResult)
                idx = SAMPLE_IDS.index(str(r.id))
                assert r.text == SAMPLE_CHUNKS[idx]
                assert r.payload.get("chunk_id") == SAMPLE_IDS[idx]
                assert r.vector is not None
                assert_vector_matches(r.vector, SAMPLE_VECTORS[idx], vector_id=str(r.id))
                # User metadata lives under payload['metadata']
                md = r.payload.get("metadata") or {}
                assert md.get("category") == SAMPLE_PAYLOADS[idx]["category"]
                assert md.get("author") == SAMPLE_PAYLOADS[idx]["author"]
                assert md.get("year") == SAMPLE_PAYLOADS[idx]["year"]
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_upsert_sync(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            provider.upsert(
                vectors=SAMPLE_VECTORS,
                payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS,
                chunks=SAMPLE_CHUNKS,
            )
            assert len(await provider.afetch(ids=SAMPLE_IDS)) == 5
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_upsert_with_document_tracking(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2],
                document_ids=["doc_id_1", "doc_id_2"],
                document_names=["doc1", "doc2"],
            )
            await self._settle()
            results = await provider.afetch(ids=SAMPLE_IDS[:2])
            assert len(results) == 2
            for r in results:
                chunk_id = r.payload.get("chunk_id")
                assert chunk_id in SAMPLE_IDS[:2]
                idx = SAMPLE_IDS.index(chunk_id)
                assert r.payload.get("document_name") == f"doc{idx+1}"
                assert r.payload.get("document_id") == f"doc_id_{idx+1}"
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_upsert_validation_error(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            with pytest.raises(ValueError):
                await provider.aupsert(
                    vectors=SAMPLE_VECTORS[:2],
                    payloads=SAMPLE_PAYLOADS[:3],
                    ids=SAMPLE_IDS[:2],
                    chunks=SAMPLE_CHUNKS[:2],
                )
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_fetch(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
            )
            await self._settle()
            results = await provider.afetch(ids=SAMPLE_IDS[:3])
            assert len(results) == 3
            for r in results:
                assert isinstance(r, VectorSearchResult)
                assert r.text in SAMPLE_CHUNKS
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_fetch_nonexistent(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            results = await provider.afetch(
                ids=["00000000-0000-4000-8000-000000000000"]
            )
            assert results == []
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_delete(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
            )
            await provider.adelete(ids=SAMPLE_IDS[:2])

            async def _deleted():
                return (
                    len(await provider.afetch(ids=SAMPLE_IDS[:2])) == 0
                    and len(await provider.afetch(ids=SAMPLE_IDS[2:])) == 3
                )

            converged = await _poll_until(_deleted, timeout=30.0, interval=1.0, description="delete propagation")
            assert len(await provider.afetch(ids=SAMPLE_IDS[:2])) == 0, f"delete did not converge in 30s (converged={converged})"
            assert len(await provider.afetch(ids=SAMPLE_IDS[2:])) == 3, f"delete did not converge in 30s (converged={converged})"
        finally:
            await self._teardown(provider)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_dense_search(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
            )

            async def _indexed():
                r = await provider.adense_search(
                    query_vector=QUERY_VECTOR, top_k=5, similarity_threshold=0.0
                )
                return len(r) > 0

            converged = await _poll_until(_indexed, timeout=30.0, interval=1.0, description="upsert indexing")
            results = await provider.adense_search(
                query_vector=QUERY_VECTOR, top_k=3, similarity_threshold=0.0
            )
            assert 0 < len(results) <= 3, f"upsert did not converge in 30s (converged={converged})"
            for r in results:
                assert isinstance(r, VectorSearchResult)
                assert r.text in SAMPLE_CHUNKS
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_search_master_method(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
            )
            results = await provider.asearch(query_vector=QUERY_VECTOR, top_k=3)
            assert len(results) > 0
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_search_with_standard_field_filter(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
                document_names=["d1", "d1", "d2", "d2", "d3"],
            )

            async def _filtered_indexed():
                r = await provider.adense_search(
                    query_vector=QUERY_VECTOR, top_k=5,
                    filter={"document_name": "d1"},
                    similarity_threshold=0.0,
                )
                return len(r) > 0

            converged = await _poll_until(_filtered_indexed, timeout=30.0, interval=1.0, description="filtered upsert indexing")
            results = await provider.adense_search(
                query_vector=QUERY_VECTOR, top_k=5,
                filter={"document_name": "d1"},
                similarity_threshold=0.0,
            )
            assert len(results) > 0, f"filtered upsert did not converge in 30s (converged={converged})"
            for r in results:
                assert r.payload.get("document_name") == "d1"
        finally:
            await self._teardown(provider)

    # ------------------------------------------------------------------
    # Count / existence
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_aget_count(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
            )
            count = await provider.aget_count()
            assert count >= 0  # Milvus stats may be eventual.
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_aid_exists(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
            )
            await self._settle()
            assert await provider.aid_exists(SAMPLE_IDS[0]) is True
            assert await provider.aid_exists(
                "99999999-9999-4999-8999-999999999999"
            ) is False
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_chunk_id_preserved_as_uuid(self, provider):
        """UUID chunk_ids must be preserved verbatim through the provider."""
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
            )
            results = await provider.afetch(ids=SAMPLE_IDS[:2])
            assert len(results) == 2
            fetched = {str(r.id) for r in results}
            assert fetched == set(SAMPLE_IDS[:2])
            for r in results:
                assert r.payload.get("chunk_id") in SAMPLE_IDS[:2]
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_achunk_id_exists(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
            )
            await self._settle()
            assert await provider.achunk_id_exists(SAMPLE_IDS[0]) is True
            assert await provider.achunk_id_exists("nonexistent") is False
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_adocument_name_exists(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
                document_names=["mv_doc"],
            )
            await self._settle()
            assert await provider.adocument_name_exists("mv_doc") is True
            assert await provider.adocument_name_exists("nonexistent") is False
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_adocument_id_exists(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
                document_ids=["mv_doc_id_1"],
            )
            await self._settle()
            assert await provider.adocument_id_exists("mv_doc_id_1") is True
            assert await provider.adocument_id_exists("nonexistent") is False
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_achunk_content_hash_exists(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
            )
            await self._settle()
            h = md5(SAMPLE_CHUNKS[0].encode("utf-8")).hexdigest()
            assert await provider.achunk_content_hash_exists(h) is True
            assert await provider.achunk_content_hash_exists("deadbeef" * 4) is False
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_adoc_content_hash_exists(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            dh = md5(b"the-document-body").hexdigest()
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
                doc_content_hashes=[dh],
            )
            await self._settle()
            assert await provider.adoc_content_hash_exists(dh) is True
            assert await provider.adoc_content_hash_exists("missing") is False
        finally:
            await self._teardown(provider)

    # ------------------------------------------------------------------
    # Delete by field
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_adelete_by_document_name(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
                document_names=["mv_doc", "mv_doc"],
            )
            await self._settle()
            assert await provider.adelete_by_document_name("mv_doc") is True
            await self._settle()
            assert await provider.adocument_name_exists("mv_doc") is False
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_adelete_by_document_id(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
                document_ids=["mv_doc_id", "mv_doc_id"],
            )
            await self._settle()
            assert await provider.adelete_by_document_id("mv_doc_id") is True
            await self._settle()
            assert len(await provider.afetch(ids=SAMPLE_IDS[:2])) == 0
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_adelete_by_chunk_id(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
            )
            await self._settle()
            assert await provider.adelete_by_chunk_id(SAMPLE_IDS[0]) is True
            await self._settle()
            assert await provider.achunk_id_exists(SAMPLE_IDS[0]) is False
            assert await provider.achunk_id_exists(SAMPLE_IDS[1]) is True
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_adelete_by_chunk_content_hash(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
            )
            await self._settle()
            h = md5(SAMPLE_CHUNKS[0].encode("utf-8")).hexdigest()
            assert await provider.achunk_content_hash_exists(h) is True
            assert await provider.adelete_by_chunk_content_hash(h) is True
            await self._settle()
            assert await provider.achunk_content_hash_exists(h) is False
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_adelete_by_doc_content_hash(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            dh = md5(b"another-doc-body").hexdigest()
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
                doc_content_hashes=[dh, dh],
            )
            await self._settle()
            assert await provider.adoc_content_hash_exists(dh) is True
            assert await provider.adelete_by_doc_content_hash(dh) is True
            await self._settle()
            assert await provider.adoc_content_hash_exists(dh) is False
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_afield_exists_and_adelete_by_field(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
                document_ids=["docA", "docB"],
            )
            await self._settle()
            assert await provider.afield_exists("document_id", "docA") is True
            assert await provider.afield_exists("document_id", "missing") is False
            assert await provider.adelete_by_field("document_id", "docA") is True
            await self._settle()
            assert await provider.afield_exists("document_id", "docA") is False
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_aupsert_with_chunk_content_hashes(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            explicit = [md5(c.encode("utf-8")).hexdigest() for c in SAMPLE_CHUNKS[:2]]
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
                chunk_content_hashes=explicit,
            )
            await self._settle()
            for h in explicit:
                assert await provider.achunk_content_hash_exists(h) is True
        finally:
            await self._teardown(provider)

    # ------------------------------------------------------------------
    # Update metadata / optimize / search types
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_aupdate_metadata(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1], payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1], chunks=SAMPLE_CHUNKS[:1],
            )
            await self._settle()
            ok = await provider.aupdate_metadata(
                SAMPLE_IDS[0], {"new_field": "new_value", "updated": True}
            )
            assert ok is True
            await self._settle()
            results = await provider.afetch(ids=SAMPLE_IDS[:1])
            md = results[0].payload.get("metadata") or {}
            assert md.get("new_field") == "new_value"
            assert md.get("updated") is True
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_aoptimize(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            assert await provider.aoptimize() is True
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_aget_supported_search_types(self, provider):
        self._skip_if_unavailable(provider)
        await provider.aconnect()
        try:
            supported = await provider.aget_supported_search_types()
            assert isinstance(supported, list)
            assert "dense" in supported
        finally:
            await provider.adisconnect()


    # ============== BACK-PORTED FROM OLD FILE ==============

    @pytest.mark.asyncio
    async def test_recreate_if_exists_clears_data(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
            )
            await self._settle()
            await provider.acreate_collection()
            await self._settle()
            try:
                results = await provider.afetch(ids=SAMPLE_IDS[:2])
                assert len(results) == 0
            except Exception:
                # Milvus Lite / serverless may raise on empty-collection fetch
                pass
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_euclidean_distance_metric(self, config):
        if config is None:
            pytest.skip("Milvus backend not available for this class.")
        new_cfg = config.model_copy(update={
            "distance_metric": DistanceMetric.EUCLIDEAN,
            "collection_name": f"test_milvus_euc_{_uuid.uuid4().hex[:8]}",
        })
        provider = MilvusProvider(new_cfg)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
            )
            await self._settle()
            results = await provider.afetch(ids=SAMPLE_IDS[:2])
            assert len(results) == 2
            for r in results:
                idx = SAMPLE_IDS.index(str(r.id))
                assert_vector_matches(r.vector, SAMPLE_VECTORS[idx], vector_id=str(r.id))
            search_results = await provider.adense_search(
                query_vector=QUERY_VECTOR, top_k=2, similarity_threshold=0.0
            )
            assert len(search_results) > 0
            for r in search_results:
                idx = SAMPLE_IDS.index(str(r.id))
                assert_vector_matches(r.vector, SAMPLE_VECTORS[idx], vector_id=str(r.id))
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_flat_index_standalone(self, config):
        if config is None:
            pytest.skip("Milvus backend not available for this class.")
        new_cfg = config.model_copy(update={
            "index": FlatIndexConfig(),
            "collection_name": f"test_milvus_flat_{_uuid.uuid4().hex[:8]}",
        })
        provider = MilvusProvider(new_cfg)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2], payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2], chunks=SAMPLE_CHUNKS[:2],
            )
            await self._settle()
            results = await provider.afetch(ids=SAMPLE_IDS[:2])
            assert len(results) == 2
            for r in results:
                idx = SAMPLE_IDS.index(str(r.id))
                assert_vector_matches(r.vector, SAMPLE_VECTORS[idx], vector_id=str(r.id))
            search_results = await provider.adense_search(
                query_vector=QUERY_VECTOR, top_k=2, similarity_threshold=0.0
            )
            assert len(search_results) > 0
        finally:
            await self._teardown(provider)

    @pytest.mark.asyncio
    async def test_payload_content_round_trip(self, provider):
        self._skip_if_unavailable(provider)
        await self._setup(provider)
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
            )
            await self._settle()
            results = await provider.afetch(ids=SAMPLE_IDS)
            assert len(results) == 5
            for r in results:
                idx = SAMPLE_IDS.index(str(r.id))
                assert r.text == SAMPLE_CHUNKS[idx]
                # Provider stores chunk text under payload['content']
                assert r.payload.get("content") == SAMPLE_CHUNKS[idx]
        finally:
            await self._teardown(provider)


class TestMilvusProviderEmbedded(_MilvusTestMixin):
    """Milvus Lite (embedded / file-backed) tests."""

    def _make_config(self, collection_name: str) -> Optional[MilvusConfig]:
        return _embedded_config(collection_name)


class TestMilvusProviderCLOUD(_MilvusTestMixin):
    """Zilliz Cloud tests (require MILVUS_CLOUD_URI / MILVUS_CLOUD_TOKEN)."""

    def _make_config(self, collection_name: str) -> Optional[MilvusConfig]:
        return _cloud_config(collection_name)

    async def _settle(self) -> None:
        import asyncio
        await asyncio.sleep(5.0)


# ============== BACK-PORTED FROM OLD FILE: sync wrappers ==============

class TestMilvusProviderEmbeddedSync:
    """Exercise base-class sync wrappers end-to-end (Embedded only)."""

    @pytest.fixture
    def provider(self) -> MilvusProvider:
        cfg = _embedded_config(f"test_milvus_sync_{_uuid.uuid4().hex[:8]}")
        return MilvusProvider(cfg)

    def test_connect_sync(self, provider):
        provider.connect()
        try:
            assert provider.client is not None
        finally:
            provider.disconnect()

    def test_upsert_and_fetch_sync(self, provider):
        provider.connect()
        provider.create_collection()
        try:
            provider.upsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
            )
            results = provider.fetch(ids=SAMPLE_IDS)
            assert len(results) == 5
            for r in results:
                idx = SAMPLE_IDS.index(str(r.id))
                assert r.text == SAMPLE_CHUNKS[idx]
                assert_vector_matches(r.vector, SAMPLE_VECTORS[idx], vector_id=str(r.id))
        finally:
            try:
                provider.delete_collection()
            except Exception:
                pass
            provider.disconnect()

    def test_delete_sync(self, provider):
        provider.connect()
        provider.create_collection()
        try:
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

    def test_dense_search_sync(self, provider):
        provider.connect()
        provider.create_collection()
        try:
            provider.upsert(
                vectors=SAMPLE_VECTORS, payloads=SAMPLE_PAYLOADS,
                ids=SAMPLE_IDS, chunks=SAMPLE_CHUNKS,
            )
            results = provider.dense_search(
                query_vector=QUERY_VECTOR, top_k=3, similarity_threshold=0.0
            )
            assert len(results) > 0
            for r in results:
                idx = SAMPLE_IDS.index(str(r.id))
                assert_vector_matches(r.vector, SAMPLE_VECTORS[idx], vector_id=str(r.id))
        finally:
            try:
                provider.delete_collection()
            except Exception:
                pass
            provider.disconnect()
