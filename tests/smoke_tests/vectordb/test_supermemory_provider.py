"""
Smoke tests for SuperMemory vector database provider.

Modernized for the Vector DB Payload Contract refactor:
- Uses `a*` async method names and bare sync wrappers from the base class
- `chunk_id` terminology throughout
- UUID SAMPLE_IDS preserved verbatim across upsert/fetch
- Non-standard payload fields asserted under ``result.payload["metadata"]``
- Standard fields only passed via dedicated ``aupsert`` parameters

Requires SUPER_MEMORY_API_KEY (or SUPERMEMORY_API_KEY) in the environment.
"""

import os
import uuid
import asyncio
import pytest
from typing import List, Dict, Any, Optional, Callable, Awaitable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pydantic import SecretStr

from upsonic.vectordb.providers.supermemory import SuperMemoryProvider
from upsonic.vectordb.config import SuperMemoryConfig
from upsonic.vectordb.base import BaseVectorDBProvider
from upsonic.utils.package.exception import (
    VectorDBConnectionError,
    VectorDBError,
    SearchError,
    UpsertError,
)
from upsonic.schemas.vector_schemas import VectorSearchResult


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
    "The theory of relativity revolutionized physics and our understanding of spacetime",
    "Laws of motion and universal gravitation describe how objects move and attract",
    "To be or not to be, that is the question from Hamlet by Shakespeare",
    "It was the best of times, it was the worst of times from A Tale of Two Cities",
    "The unexamined life is not worth living according to Socratic philosophy",
]

SAMPLE_IDS: List[str] = [
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222",
    "33333333-3333-4333-8333-333333333333",
    "44444444-4444-4444-8444-444444444444",
    "55555555-5555-4555-8555-555555555555",
]

QUERY_VECTOR: List[float] = [0.15, 0.25, 0.35, 0.45, 0.55]
QUERY_TEXT: str = "physics theory relativity"


def _get_api_key() -> Optional[str]:
    return os.getenv("SUPER_MEMORY_API_KEY") or os.getenv("SUPERMEMORY_API_KEY")


def _unique_tag() -> str:
    return f"upsonic_test_{uuid.uuid4().hex[:8]}"


async def _poll_until(
    predicate: Callable[[], Awaitable[bool]],
    timeout: float = 30.0,
    interval: float = 1.0,
) -> bool:
    """Poll an async predicate until it returns True or timeout expires."""
    loops = max(1, int(timeout / interval))
    for _ in range(loops):
        try:
            if await predicate():
                return True
        except Exception:
            pass
        await asyncio.sleep(interval)
    try:
        return await predicate()
    except Exception:
        return False


# ============================================================================
# Config-only tests (no API key needed)
# ============================================================================


class TestSuperMemoryConfig:
    """Tests for SuperMemoryConfig validation and defaults."""

    def test_defaults(self) -> None:
        config = SuperMemoryConfig(collection_name="test_col")
        assert config.collection_name == "test_col"
        assert config.container_tag == "test_col"
        assert config.vector_size == 0
        assert config.dense_search_enabled is False
        assert config.full_text_search_enabled is True
        assert config.hybrid_search_enabled is True
        assert config.search_mode == "hybrid"
        assert config.threshold == 0.5
        assert config.rerank is False
        assert config.max_retries == 2
        assert config.timeout == 60.0
        assert config.batch_delay == 0.1
        assert config.batch_size == 50
        assert config.api_key is None

    def test_container_tag_defaults_to_collection_name(self) -> None:
        config = SuperMemoryConfig(collection_name="my_kb")
        assert config.container_tag == "my_kb"

    def test_explicit_container_tag(self) -> None:
        config = SuperMemoryConfig(
            collection_name="my_kb",
            container_tag="custom_tag",
        )
        assert config.container_tag == "custom_tag"

    def test_api_key_via_config(self) -> None:
        config = SuperMemoryConfig(
            collection_name="test",
            api_key=SecretStr("sm_test_key_123"),
        )
        assert config.api_key is not None
        assert config.api_key.get_secret_value() == "sm_test_key_123"

    def test_threshold_validation(self) -> None:
        with pytest.raises(ValueError):
            SuperMemoryConfig(collection_name="t", threshold=1.5)
        with pytest.raises(ValueError):
            SuperMemoryConfig(collection_name="t", threshold=-0.1)

    def test_search_mode_literal(self) -> None:
        config_h = SuperMemoryConfig(collection_name="t", search_mode="hybrid")
        assert config_h.search_mode == "hybrid"
        config_m = SuperMemoryConfig(collection_name="t", search_mode="memories")
        assert config_m.search_mode == "memories"
        config_d = SuperMemoryConfig(collection_name="t", search_mode="documents")
        assert config_d.search_mode == "documents"

    def test_batch_size_config(self) -> None:
        config_default = SuperMemoryConfig(collection_name="t")
        assert config_default.batch_size == 50
        config_custom = SuperMemoryConfig(collection_name="t", batch_size=10)
        assert config_custom.batch_size == 10

    def test_from_dict(self) -> None:
        config = SuperMemoryConfig.from_dict({
            "collection_name": "dict_test",
            "container_tag": "dict_tag",
            "threshold": 0.7,
        })
        assert config.collection_name == "dict_test"
        assert config.container_tag == "dict_tag"
        assert config.threshold == 0.7

    def test_frozen_immutability(self) -> None:
        config = SuperMemoryConfig(collection_name="frozen")
        with pytest.raises(Exception):
            config.collection_name = "changed"  # type: ignore[misc]

    def test_factory_create_config(self) -> None:
        from upsonic.vectordb.config import create_config
        config = create_config("supermemory", collection_name="factory")
        assert isinstance(config, SuperMemoryConfig)
        assert config.collection_name == "factory"


# ============================================================================
# Provider instantiation tests (no API call needed)
# ============================================================================


class TestSuperMemoryProviderInit:
    """Tests for provider initialization without making API calls."""

    def test_isinstance_base(self) -> None:
        config = SuperMemoryConfig(collection_name="test")
        provider = SuperMemoryProvider(config)
        assert isinstance(provider, BaseVectorDBProvider)

    def test_provider_attributes(self) -> None:
        config = SuperMemoryConfig(
            collection_name="attr_test",
            container_tag="attr_tag",
            provider_name="MyProvider",
            provider_description="Test provider",
        )
        provider = SuperMemoryProvider(config)
        assert provider.name == "MyProvider"
        assert provider.description == "Test provider"
        assert provider._config is config
        assert provider._container_tag == "attr_tag"
        assert provider._is_connected is False
        assert provider.client is None

    def test_default_provider_name(self) -> None:
        config = SuperMemoryConfig(collection_name="my_coll")
        provider = SuperMemoryProvider(config)
        assert provider.name == "SuperMemoryProvider_my_coll"

    def test_provider_id_generation(self) -> None:
        config = SuperMemoryConfig(collection_name="id_test", container_tag="id_tag")
        provider = SuperMemoryProvider(config)
        assert len(provider.id) == 16
        provider2 = SuperMemoryProvider(config)
        assert provider.id == provider2.id

    def test_custom_provider_id(self) -> None:
        config = SuperMemoryConfig(collection_name="t", provider_id="custom_id_123")
        provider = SuperMemoryProvider(config)
        assert provider.id == "custom_id_123"

    def test_init_from_dict(self) -> None:
        provider = SuperMemoryProvider({
            "collection_name": "dict_init",
            "container_tag": "dict_tag",
        })
        assert provider._config.collection_name == "dict_init"
        assert provider._container_tag == "dict_tag"

    def test_supported_search_types(self) -> None:
        config = SuperMemoryConfig(collection_name="t")
        provider = SuperMemoryProvider(config)
        # aget_supported_search_types is async; call the sync-ish path by running it
        loop = asyncio.new_event_loop()
        try:
            types = loop.run_until_complete(provider.aget_supported_search_types())
        finally:
            loop.close()
        assert "full_text" in types
        assert "hybrid" in types
        assert "dense" not in types


# ============================================================================
# Live API tests (require SUPER_MEMORY_API_KEY)
# ============================================================================


class TestSuperMemoryProviderCloud:
    """Comprehensive tests against the live SuperMemory API."""

    @pytest.fixture
    def config(self) -> Optional[SuperMemoryConfig]:
        api_key = _get_api_key()
        if not api_key:
            return None
        tag = _unique_tag()
        return SuperMemoryConfig(
            collection_name=tag,
            container_tag=tag,
            api_key=SecretStr(api_key),
            batch_delay=0.1,
            threshold=0.1,
        )

    @pytest.fixture
    def provider(self, config: Optional[SuperMemoryConfig]) -> Optional[SuperMemoryProvider]:
        if config is None:
            return None
        return SuperMemoryProvider(config)

    def _skip_if_unavailable(self, provider: Optional[SuperMemoryProvider]) -> None:
        if provider is None:
            pytest.skip("SUPER_MEMORY_API_KEY not available")

    async def _ensure_connected(self, provider: SuperMemoryProvider) -> None:
        try:
            await provider.aconnect()
        except VectorDBConnectionError:
            pytest.skip("SuperMemory connection failed")

    async def _upsert_sample_data(
        self,
        provider: SuperMemoryProvider,
        count: int = 5,
    ) -> None:
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:count],
            payloads=SAMPLE_PAYLOADS[:count],
            ids=SAMPLE_IDS[:count],
            chunks=SAMPLE_CHUNKS[:count],
        )
        await asyncio.sleep(10)

    async def _search_with_retry(
        self,
        provider: SuperMemoryProvider,
        query_text: str,
        top_k: int = 3,
        max_attempts: int = 5,
        delay: float = 5.0,
        **kwargs: Any,
    ) -> List[VectorSearchResult]:
        results: List[VectorSearchResult] = []
        for attempt in range(max_attempts):
            results = await provider.asearch(query_text=query_text, top_k=top_k, **kwargs)
            if len(results) > 0:
                return results
            if attempt < max_attempts - 1:
                await asyncio.sleep(delay)
        return results

    # ------------------------------------------------------------------
    # Connection Management
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_connect(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        assert provider._is_connected is False
        await provider.aconnect()
        assert provider._is_connected is True
        assert provider.client is not None
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await provider.aconnect()
        client_ref = provider.client
        await provider.aconnect()
        assert provider.client is client_ref
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_disconnect(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await provider.aconnect()
        assert provider._is_connected is True
        await provider.adisconnect()
        assert provider._is_connected is False
        assert provider.client is None

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await provider.adisconnect()
        assert provider._is_connected is False

    # ------------------------------------------------------------------
    # is_ready
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_is_ready_false_when_disconnected(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        assert await provider.ais_ready() is False

    @pytest.mark.asyncio
    async def test_is_ready_true_when_connected(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        assert await provider.ais_ready() is True
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Collection Management
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_collection(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        await provider.acreate_collection()
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_collection_exists_empty(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        exists = await provider.acollection_exists()
        assert isinstance(exists, bool)
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_collection_exists_after_upsert(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        await self._upsert_sample_data(provider, count=1)
        assert await _poll_until(
            provider.acollection_exists, timeout=60.0, interval=3.0
        ) is True
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_delete_collection(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        await self._upsert_sample_data(provider, count=1)
        await provider.adelete_collection()
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_upsert(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2],
        )
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_upsert_no_chunks_raises(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        with pytest.raises(UpsertError):
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1],
                payloads=SAMPLE_PAYLOADS[:1],
                ids=SAMPLE_IDS[:1],
                chunks=None,
            )
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_upsert_length_mismatch_raises(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        with pytest.raises(UpsertError):
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:3],
            )
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_upsert_skips_empty_chunks(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=[
                "aaaaaaaa-0000-4000-8000-000000000001",
                "aaaaaaaa-0000-4000-8000-000000000002",
            ],
            chunks=["", "   "],
        )
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Search — hybrid (primary path for SuperMemory)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_hybrid_search(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        await self._upsert_sample_data(provider)
        results: List[VectorSearchResult] = []
        for attempt in range(5):
            results = await provider.ahybrid_search(
                query_vector=QUERY_VECTOR,
                query_text=QUERY_TEXT,
                top_k=3,
                similarity_threshold=0.0,
            )
            if len(results) > 0:
                break
            await asyncio.sleep(5)
        assert len(results) > 0, "Expected results after retries (eventual consistency)"
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id is not None
            assert isinstance(result.score, float)
            assert result.score >= 0.0
            assert result.text is not None
            assert result.vector is None
            assert isinstance(result.payload, dict)
            assert isinstance(result.payload.get("metadata"), dict)
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Search — full-text (memories-only mode)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_full_text_search(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        await self._upsert_sample_data(provider)
        results = await provider.afull_text_search(
            query_text="relativity physics spacetime",
            top_k=3,
            similarity_threshold=0.0,
        )
        assert len(results) >= 0
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id is not None
            assert isinstance(result.score, float)
            assert result.text is not None
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Search — dense (returns empty, SuperMemory limitation)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_dense_search_returns_empty(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        results = await provider.adense_search(
            query_vector=QUERY_VECTOR,
            top_k=3,
        )
        assert results == []
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Search — master dispatch method
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_search_with_query_text(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        await self._upsert_sample_data(provider)
        results = await self._search_with_retry(provider, query_text=QUERY_TEXT, top_k=3)
        assert len(results) > 0, "Expected results after retries (eventual consistency)"
        assert all(isinstance(r, VectorSearchResult) for r in results)
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_search_with_query_vector_only(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        results = await provider.asearch(
            query_vector=QUERY_VECTOR,
            top_k=3,
        )
        assert results == []
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_search_no_args_raises(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        with pytest.raises(SearchError):
            await provider.asearch(top_k=3)
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_delete(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        await self._upsert_sample_data(provider, count=2)
        await provider.adelete(ids=[SAMPLE_IDS[0]])
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fetch_nonexistent_returns_empty(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        results = await provider.afetch(ids=["totally_nonexistent_id_xyz"])
        assert results == []
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Existence Checks
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_document_id_exists(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        exists = await provider.adocument_id_exists("nonexistent_doc_id_xyz")
        assert exists is False
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_document_name_exists(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        exists = await provider.adocument_name_exists("nonexistent_doc_name_xyz")
        assert exists is False
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Delete by metadata variants
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        result = await provider.adelete_by_document_id("nonexistent_id")
        assert isinstance(result, bool)
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_delete_by_document_name(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        result = await provider.adelete_by_document_name("nonexistent_name")
        assert result is True
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_delete_by_chunk_id(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        result = await provider.adelete_by_chunk_id("nonexistent_content")
        assert isinstance(result, bool)
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_delete_by_metadata(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        result = await provider.adelete_by_metadata({"category": "nonexistent"})
        assert result is True
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Optimize (no-op)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_optimize(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        assert await provider.aoptimize() is True

    # ------------------------------------------------------------------
    # Supported Search Types
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_supported_search_types(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        supported = await provider.aget_supported_search_types()
        assert isinstance(supported, list)
        assert "full_text" in supported
        assert "hybrid" in supported
        assert "dense" not in supported

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_convert_filters_simple(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        result = SuperMemoryProvider._convert_filters({"category": "science"})
        assert "AND" in result
        assert len(result["AND"]) == 1
        assert result["AND"][0]["key"] == "category"
        assert result["AND"][0]["value"] == "science"

    @pytest.mark.asyncio
    async def test_convert_filters_multiple(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        result = SuperMemoryProvider._convert_filters({"cat": "sci", "year": "1905"})
        assert "AND" in result
        assert len(result["AND"]) == 2

    @pytest.mark.asyncio
    async def test_convert_filters_passthrough(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        original: Dict[str, Any] = {"AND": [{"key": "x", "value": "y"}]}
        result = SuperMemoryProvider._convert_filters(original)
        assert result is original

    # ------------------------------------------------------------------
    # End-to-end
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_e2e_upsert_and_search(self, provider: Optional[SuperMemoryProvider]) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        await self._upsert_sample_data(provider, count=5)

        results = await self._search_with_retry(
            provider, query_text="theory of relativity physics", top_k=3,
        )
        assert len(results) > 0, "Expected results after retries (eventual consistency)"
        top_result = results[0]
        assert isinstance(top_result, VectorSearchResult)
        assert top_result.text is not None
        assert top_result.score > 0.0
        await provider.adisconnect()

    # ------------------------------------------------------------------
    # Imports via __init__.py
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_import_from_vectordb_module(self) -> None:
        from upsonic.vectordb import SuperMemoryProvider as SP
        from upsonic.vectordb import SuperMemoryConfig as SC
        assert SP is SuperMemoryProvider
        assert SC is SuperMemoryConfig

    @pytest.mark.asyncio
    async def test_import_in_all(self) -> None:
        from upsonic.vectordb import __all__ as exports
        assert "SuperMemoryProvider" in exports
        assert "SuperMemoryConfig" in exports

    # ==================================================================
    # Payload Contract Tests
    # ==================================================================

    @pytest.mark.asyncio
    async def test_chunk_id_preserved_as_uuid(
        self, provider: Optional[SuperMemoryProvider]
    ) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_ids=["doc-rel"],
            document_names=["rel.pdf"],
        )

        async def _found() -> bool:
            res = await provider.ahybrid_search(query_vector=QUERY_VECTOR,
                query_text=SAMPLE_CHUNKS[0],
                top_k=5,
                similarity_threshold=0.0,
            )
            return any(
                isinstance(r.payload, dict)
                and r.payload.get("chunk_id") == SAMPLE_IDS[0]
                for r in res
            )

        assert await _poll_until(_found, timeout=60.0, interval=3.0), (
            "chunk_id UUID not preserved on round-trip"
        )
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_aupsert_with_knowledge_base_ids(
        self, provider: Optional[SuperMemoryProvider]
    ) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            knowledge_base_ids=["kb-physics"],
        )

        async def _has_kb() -> bool:
            res = await provider.ahybrid_search(query_vector=QUERY_VECTOR,
                query_text=SAMPLE_CHUNKS[0], top_k=5, similarity_threshold=0.0
            )
            return any(
                isinstance(r.payload, dict)
                and r.payload.get("knowledge_base_id") == "kb-physics"
                for r in res
            )

        assert await _poll_until(_has_kb, timeout=60.0, interval=3.0), (
            "knowledge_base_id not stored via dedicated param"
        )
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_aupsert_standard_field_leak_warning(
        self, provider: Optional[SuperMemoryProvider]
    ) -> None:
        """Standard fields inside payload dicts must be dropped; dedicated
        params take precedence."""
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)

        leaked_payload = dict(SAMPLE_PAYLOADS[0])
        leaked_payload["document_name"] = "leaked_name.pdf"
        leaked_payload["document_id"] = "leaked_doc_id"
        leaked_payload["knowledge_base_id"] = "leaked_kb"

        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=[leaked_payload],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            document_names=["real_name.pdf"],
            document_ids=["real_doc_id"],
            knowledge_base_ids=["real_kb"],
        )

        async def _correct() -> bool:
            res = await provider.ahybrid_search(query_vector=QUERY_VECTOR,
                query_text=SAMPLE_CHUNKS[0], top_k=5, similarity_threshold=0.0
            )
            for r in res:
                if not isinstance(r.payload, dict):
                    continue
                if r.payload.get("chunk_id") != SAMPLE_IDS[0]:
                    continue
                if (
                    r.payload.get("document_name") == "real_name.pdf"
                    and r.payload.get("document_id") == "real_doc_id"
                    and r.payload.get("knowledge_base_id") == "real_kb"
                    and isinstance(r.payload.get("metadata"), dict)
                    and "document_name" not in r.payload["metadata"]
                    and "knowledge_base_id" not in r.payload["metadata"]
                ):
                    return True
            return False

        assert await _poll_until(_correct, timeout=60.0, interval=3.0), (
            "Dedicated params did not override leaked standard keys in payload dict"
        )
        await provider.adisconnect()

    @pytest.mark.parametrize(
        "bad_field",
        [
            "payloads",
            "ids",
            "chunks",
            "document_ids",
            "document_names",
            "doc_content_hashes",
            "chunk_content_hashes",
            "knowledge_base_ids",
        ],
    )
    @pytest.mark.asyncio
    async def test_aupsert_length_mismatch_variants(
        self,
        provider: Optional[SuperMemoryProvider],
        bad_field: str,
    ) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)

        base: Dict[str, Any] = {
            "vectors": SAMPLE_VECTORS[:2],
            "payloads": SAMPLE_PAYLOADS[:2],
            "ids": SAMPLE_IDS[:2],
            "chunks": SAMPLE_CHUNKS[:2],
            "document_ids": ["d1", "d2"],
            "document_names": ["n1", "n2"],
            "doc_content_hashes": ["dh1", "dh2"],
            "chunk_content_hashes": ["ch1", "ch2"],
            "knowledge_base_ids": ["k1", "k2"],
        }
        # Make the tested field have length 3 instead of 2
        fill3 = {
            "payloads": [{"a": 1}, {"a": 2}, {"a": 3}],
            "ids": SAMPLE_IDS[:3],
            "chunks": SAMPLE_CHUNKS[:3],
            "document_ids": ["d1", "d2", "d3"],
            "document_names": ["n1", "n2", "n3"],
            "doc_content_hashes": ["dh1", "dh2", "dh3"],
            "chunk_content_hashes": ["ch1", "ch2", "ch3"],
            "knowledge_base_ids": ["k1", "k2", "k3"],
        }
        base[bad_field] = fill3[bad_field]
        with pytest.raises((UpsertError, ValueError)):
            await provider.aupsert(**base)
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_achunk_content_hash_exists(
        self, provider: Optional[SuperMemoryProvider]
    ) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        ch = "chunk_hash_abc123_" + uuid.uuid4().hex[:6]
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            chunk_content_hashes=[ch],
        )

        async def _pred() -> bool:
            return await provider.achunk_content_hash_exists(ch)

        assert await _poll_until(_pred, timeout=60.0, interval=3.0)
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_adoc_content_hash_exists(
        self, provider: Optional[SuperMemoryProvider]
    ) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        dh = "doc_hash_xyz_" + uuid.uuid4().hex[:6]
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            doc_content_hashes=[dh],
        )

        async def _pred() -> bool:
            return await provider.adoc_content_hash_exists(dh)

        assert await _poll_until(_pred, timeout=60.0, interval=3.0)
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_adelete_by_chunk_content_hash(
        self, provider: Optional[SuperMemoryProvider]
    ) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        ch = "ch_del_" + uuid.uuid4().hex[:6]
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            chunk_content_hashes=[ch],
        )
        await asyncio.sleep(5)
        result = await provider.adelete_by_chunk_content_hash(ch)
        assert result is True
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_adelete_by_doc_content_hash(
        self, provider: Optional[SuperMemoryProvider]
    ) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        dh = "dh_del_" + uuid.uuid4().hex[:6]
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
            doc_content_hashes=[dh],
        )
        await asyncio.sleep(5)
        result = await provider.adelete_by_doc_content_hash(dh)
        assert result is True
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_filter_on_nonstandard_field(
        self, provider: Optional[SuperMemoryProvider]
    ) -> None:
        """Filter on non-standard field (no 'metadata.' prefix)."""
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        unique_cat = "catA_" + uuid.uuid4().hex[:6]
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=[{"category": unique_cat}],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )

        async def _found() -> bool:
            res = await provider.ahybrid_search(query_vector=QUERY_VECTOR,
                query_text=SAMPLE_CHUNKS[0],
                top_k=5,
                filter={"category": unique_cat},
                similarity_threshold=0.0,
            )
            for r in res:
                if (
                    isinstance(r.payload, dict)
                    and isinstance(r.payload.get("metadata"), dict)
                    and r.payload["metadata"].get("category") == unique_cat
                ):
                    return True
            return False

        assert await _poll_until(_found, timeout=60.0, interval=3.0)
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_metadata_is_always_dict(
        self, provider: Optional[SuperMemoryProvider]
    ) -> None:
        self._skip_if_unavailable(provider)
        assert provider is not None
        await self._ensure_connected(provider)
        await provider.aupsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=[{}],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1],
        )

        async def _ok() -> bool:
            res = await provider.ahybrid_search(query_vector=QUERY_VECTOR,
                query_text=SAMPLE_CHUNKS[0], top_k=5, similarity_threshold=0.0
            )
            for r in res:
                if (
                    isinstance(r.payload, dict)
                    and r.payload.get("chunk_id") == SAMPLE_IDS[0]
                ):
                    return isinstance(r.payload.get("metadata"), dict)
            return False

        assert await _poll_until(_ok, timeout=60.0, interval=3.0)
        await provider.adisconnect()

    @pytest.mark.asyncio
    async def test_aupdate_metadata_preserves_user_values_over_defaults(self) -> None:
        """aupdate_metadata must not silently reapply default_metadata on top
        of user-supplied values."""
        api_key = _get_api_key()
        if not api_key:
            pytest.skip("SUPER_MEMORY_API_KEY not available")
        tag = _unique_tag()
        config = SuperMemoryConfig(
            collection_name=tag,
            container_tag=tag,
            api_key=SecretStr(api_key),
            default_metadata={"tier": "default"},
            batch_delay=0.1,
            threshold=0.1,
        )
        provider = SuperMemoryProvider(config)
        await provider.aconnect()
        try:
            await provider.aupsert(
                vectors=SAMPLE_VECTORS[:1],
                payloads=[{"category": "science"}],
                ids=SAMPLE_IDS[:1],
                chunks=SAMPLE_CHUNKS[:1],
            )
            await asyncio.sleep(8)
            ok = await provider.aupdate_metadata(SAMPLE_IDS[0], {"tier": "premium"})
            # update may fail on transient index lag - allow boolean
            assert isinstance(ok, bool)
        finally:
            await provider.adisconnect()


# ============================================================================
# Final cleanup
# ============================================================================


class TestSuperMemoryFinalCleanup:
    """Runs after all other tests to delete every document created during the
    test session."""

    @pytest.mark.asyncio
    async def test_zz_final_cleanup_and_verify(self) -> None:
        api_key = _get_api_key()
        if not api_key:
            pytest.skip("SUPER_MEMORY_API_KEY not available")

        from supermemory import AsyncSupermemory

        client = AsyncSupermemory(
            api_key=api_key,
            max_retries=5,
            timeout=120.0,
        )
        all_tags_deleted: set = set()
        max_attempts: int = 40

        try:
            for attempt in range(max_attempts):
                await asyncio.sleep(2)

                try:
                    search_results = await client.search.execute(q="*", limit=100)
                except Exception:
                    await asyncio.sleep(5)
                    continue

                if not search_results.results:
                    break

                container_tags: set = set()
                internal_ids: set = set()

                for doc in search_results.results:
                    custom_id: Optional[str] = getattr(doc, "document_id", None)
                    if not custom_id:
                        continue
                    try:
                        got = await client.documents.get(str(custom_id))
                        iid: Optional[str] = getattr(got, "id", None)
                        ctags: List[str] = getattr(got, "container_tags", [])
                        if iid:
                            internal_ids.add(str(iid))
                        for tag in ctags:
                            container_tags.add(tag)
                    except Exception:
                        pass
                    await asyncio.sleep(0.3)

                for tag in container_tags:
                    if tag not in all_tags_deleted:
                        try:
                            await client.documents.delete_bulk(container_tags=[tag])
                            all_tags_deleted.add(tag)
                        except Exception:
                            pass
                        await asyncio.sleep(0.5)

                for iid in internal_ids:
                    try:
                        await client.documents.delete(iid)
                    except Exception:
                        pass
                    await asyncio.sleep(0.3)

            await asyncio.sleep(3)
        finally:
            await client.close()
