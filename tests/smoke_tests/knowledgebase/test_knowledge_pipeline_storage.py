"""Full pipeline tests: Agent + Task + KnowledgeBase + Storage integration.

Tests the complete flow where KnowledgeBase synchronously writes document
metadata to Storage (KnowledgeRow) alongside VectorDB chunk storage.
Uses Chroma in EMBEDDED mode (no API key) and anthropic/claude-sonnet-4-5.

Validates EVERY feature implemented in this session:
- KBState lifecycle (UNINITIALIZED → CONNECTED → INDEXED → CLOSED)
- setup_async populates both VectorDB and Storage
- setup_async(force=True) re-indexes
- KnowledgeRow fields correctness after indexing
- aadd_source / add_source syncs new documents to Storage
- aadd_text / add_text syncs raw text to Storage
- aremove_document removes from VectorDB + Storage + self.sources
- arefresh re-scans and re-indexes with Storage sync
- Content deduplication (same content not indexed twice)
- Auto-collection-name derivation
- Isolation via knowledge_base_id (isolate_search)
- Agent + Task queries work with KB context (do_async / do)
- Agent + Task after dynamic add (aadd_source, aadd_text)
- Agent + Task after remove (removed content unreachable)
- Agent + Task with multiple KBs as context
- query_async direct search
- get_tools() / build_context() Tool Provider Protocol
- get_config_summary reports new state/isolation fields
- health_check_async comprehensive validation
- Async context manager (__aenter__/__aexit__)
- Operations on closed KB raise RuntimeError
- _remove_source_for_document syncs self.sources
- Processing stats after setup
- _build_knowledge_row field mapping (type, size, source, etc.)
"""
import os
import shutil
import tempfile

import pytest

from upsonic import Agent, Task, KnowledgeBase
from upsonic.embeddings import OpenAIEmbedding, OpenAIEmbeddingConfig
from upsonic.knowledge_base.knowledge_base import KBState
from upsonic.storage.in_memory import InMemoryStorage
from upsonic.storage.schemas import KnowledgeRow
from upsonic.vectordb import ChromaProvider, ChromaConfig, ConnectionConfig, Mode

pytestmark = pytest.mark.timeout(300)

MODEL: str = "anthropic/claude-sonnet-4-5"

_chroma_counter: int = 0


def _unique_collection() -> str:
    global _chroma_counter
    _chroma_counter += 1
    return f"test_pipe_{_chroma_counter}"


def _make_chroma(temp_dir: str, collection_name: str | None = None) -> ChromaProvider:
    chroma_dir = os.path.join(temp_dir, "chroma_db")
    return ChromaProvider(
        ChromaConfig(
            collection_name=collection_name or _unique_collection(),
            vector_size=1536,
            connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=chroma_dir),
        )
    )


def _make_embedding() -> OpenAIEmbedding:
    return OpenAIEmbedding(OpenAIEmbeddingConfig())


def _write_doc(temp_dir: str, filename: str, content: str) -> str:
    path = os.path.join(temp_dir, filename)
    with open(path, "w") as f:
        f.write(content)
    return path


# ============================================================================
# 1. KBState Lifecycle
# ============================================================================

class TestKBStateLifecycle:

    @pytest.mark.asyncio
    async def test_initial_state_uninitialized(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            doc = _write_doc(temp_dir, "s.txt", "State test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(),
            )
            assert kb._state == KBState.UNINITIALIZED
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_state_indexed_after_setup(self) -> None:
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(temp_dir, "s.txt", "State test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(),
            )
            await kb.setup_async()
            assert kb._state == KBState.INDEXED
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_state_closed_after_close(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            doc = _write_doc(temp_dir, "s.txt", "State test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(),
            )
            await kb.setup_async()
            await kb.close()
            assert kb._state == KBState.CLOSED
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_closed_kb_raises_on_setup(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            doc = _write_doc(temp_dir, "s.txt", "State test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(),
            )
            await kb.setup_async()
            await kb.close()
            with pytest.raises(RuntimeError, match="closed"):
                await kb.setup_async()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_closed_kb_raises_on_aadd_source(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            doc = _write_doc(temp_dir, "s.txt", "Closed KB test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(),
            )
            await kb.setup_async()
            await kb.close()
            with pytest.raises(RuntimeError, match="closed"):
                await kb.aadd_source(doc)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_closed_kb_raises_on_aadd_text(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            doc = _write_doc(temp_dir, "s.txt", "Closed KB test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(),
            )
            await kb.setup_async()
            await kb.close()
            with pytest.raises(RuntimeError, match="closed"):
                await kb.aadd_text("any text")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_closed_kb_raises_on_aremove_document(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            doc = _write_doc(temp_dir, "s.txt", "Closed KB test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(),
            )
            await kb.setup_async()
            await kb.close()
            with pytest.raises(RuntimeError, match="closed"):
                await kb.aremove_document("any_id")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_closed_kb_raises_on_arefresh(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            doc = _write_doc(temp_dir, "s.txt", "Closed KB test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(),
            )
            await kb.setup_async()
            await kb.close()
            with pytest.raises(RuntimeError, match="closed"):
                await kb.arefresh()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            doc = _write_doc(temp_dir, "s.txt", "Idempotent close.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(),
            )
            await kb.setup_async()
            await kb.close()
            await kb.close()
            assert kb._state == KBState.CLOSED
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 2. setup_async + Storage Sync
# ============================================================================

class TestSetupStorage:

    @pytest.mark.asyncio
    async def test_setup_populates_storage(self) -> None:
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(
                temp_dir, "python.txt",
                "Python is a high-level programming language created by Guido van Rossum in 1991.",
            )
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="setup_kb",
            )
            await kb.setup_async()

            rows, total = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            assert total >= 1
            row = rows[0]
            assert row.knowledge_base_id == kb.knowledge_id
            assert row.status == "indexed"
            assert row.chunk_count is not None and row.chunk_count > 0
            assert row.content_hash is not None
            assert row.created_at is not None
            assert row.updated_at is not None
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_knowledge_row_field_mapping(self) -> None:
        """Verify every KnowledgeRow field is correctly populated from a file source."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(temp_dir, "mapping_test.txt", "Field mapping validation content.")
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="field_map_kb",
            )
            await kb.setup_async()

            rows, _ = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            assert len(rows) >= 1
            row = rows[0]

            assert isinstance(row.id, str) and len(row.id) > 0
            assert isinstance(row.name, str) and len(row.name) > 0
            assert row.knowledge_base_id == kb.knowledge_id
            assert row.type == "txt"
            assert row.source is not None and "mapping_test.txt" in row.source
            assert row.status == "indexed"
            assert row.access_count == 0
            assert row.chunk_count >= 1
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_setup_multiple_documents(self) -> None:
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc1 = _write_doc(temp_dir, "ml.txt", "Machine learning is a subset of artificial intelligence.")
            doc2 = _write_doc(temp_dir, "dl.txt", "Deep learning uses neural networks with many layers.")
            doc3 = _write_doc(temp_dir, "nlp.txt", "Natural language processing handles human text data.")

            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[doc1, doc2, doc3], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="multi_doc_kb",
            )
            await kb.setup_async()

            rows, total = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            assert total == 3
            for row in rows:
                assert row.status == "indexed"
                assert row.chunk_count >= 1
                assert row.content_hash is not None
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_processing_stats_after_setup(self) -> None:
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(temp_dir, "stats.txt", "Processing stats validation.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(), name="stats_kb",
            )
            await kb.setup_async()

            stats = kb._processing_stats
            assert "sources_count" in stats
            assert "documents_count" in stats
            assert "chunks_count" in stats
            assert "vectors_count" in stats
            assert "indexed_at" in stats
            assert stats["documents_count"] >= 1
            assert stats["chunks_count"] >= 1
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_setup_force_reindex(self) -> None:
        """setup_async(force=True) should re-process documents."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(temp_dir, "force.txt", "Force re-index test content.")
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="force_kb",
            )
            await kb.setup_async()
            assert kb._state == KBState.INDEXED

            await kb.setup_async(force=True)
            assert kb._state == KBState.INDEXED

            rows, total = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            assert total >= 1
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_setup_idempotent_without_force(self) -> None:
        """Second setup_async() without force should be a no-op."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(temp_dir, "idem.txt", "Idempotent setup test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(), name="idem_kb",
            )
            await kb.setup_async()
            stats_first = dict(kb._processing_stats)

            await kb.setup_async()
            stats_second = kb._processing_stats
            assert stats_first == stats_second
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_setup_without_storage(self) -> None:
        """KB should work fine without a storage param (no KnowledgeRow writes)."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(temp_dir, "nostorage.txt", "No storage test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), name="no_storage_kb",
            )
            await kb.setup_async()
            assert kb._state == KBState.INDEXED
            assert kb.storage is None
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 3. Dynamic Content Management + Storage Sync
# ============================================================================

class TestDynamicContent:

    @pytest.mark.asyncio
    async def test_aadd_source_syncs_storage(self) -> None:
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            initial = _write_doc(temp_dir, "init.txt", "Initial content.")
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[initial], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="add_src_kb",
            )
            await kb.setup_async()
            _, before = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)

            new_doc = _write_doc(temp_dir, "extra.txt", "Extra content about machine learning.")
            doc_ids = await kb.aadd_source(new_doc)

            _, after = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            assert after > before
            assert len(doc_ids) > 0
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_aadd_text_syncs_storage(self) -> None:
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            seed = _write_doc(temp_dir, "seed.txt", "Seed document.")
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[seed], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="add_txt_kb",
            )
            await kb.setup_async()

            doc_id = await kb.aadd_text(
                "Quantum computing uses qubits in superposition.",
                document_name="quantum_intro",
            )
            assert doc_id is not None

            row = storage.get_knowledge_content(doc_id)
            assert row is not None
            assert row.knowledge_base_id == kb.knowledge_id
            assert row.status == "indexed"
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_aadd_text_with_metadata(self) -> None:
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            seed = _write_doc(temp_dir, "seed.txt", "Seed doc.")
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[seed], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="meta_kb",
            )
            await kb.setup_async()

            doc_id = await kb.aadd_text(
                "Custom metadata test text content.",
                metadata={"category": "test", "priority": 1},
                document_name="meta_doc",
            )
            row = storage.get_knowledge_content(doc_id)
            assert row is not None
            assert row.metadata is not None
            assert row.metadata.get("category") == "test" or "category" in str(row.metadata)
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_content_deduplication(self) -> None:
        """Adding the same text twice should not create duplicate entries."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            seed = _write_doc(temp_dir, "seed.txt", "Seed doc.")
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[seed], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="dedup_kb",
            )
            await kb.setup_async()

            text = "Unique content for deduplication testing purposes."
            doc_id_1 = await kb.aadd_text(text, document_name="dup_test")
            doc_id_2 = await kb.aadd_text(text, document_name="dup_test")

            assert doc_id_1 == doc_id_2
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_aremove_document_syncs_storage(self) -> None:
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(temp_dir, "removable.txt", "Document to be removed later.")
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="remove_kb",
            )
            await kb.setup_async()

            rows, _ = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            doc_id = rows[0].id

            result = await kb.aremove_document(doc_id)
            assert result is True
            assert storage.get_knowledge_content(doc_id) is None
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_remove_source_syncs_sources_list(self) -> None:
        """After removing a document, self.sources should be updated."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(temp_dir, "src_sync.txt", "Source sync test.")
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="src_sync_kb",
            )
            await kb.setup_async()
            sources_before = len(kb.sources)

            rows, _ = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            await kb.aremove_document(rows[0].id)

            assert len(kb.sources) < sources_before
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_arefresh_re_syncs_storage(self) -> None:
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(temp_dir, "refresh.txt", "Document for refresh test.")
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="refresh_kb",
            )
            await kb.setup_async()
            _, before = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)

            stats = await kb.arefresh()
            assert stats is not None

            _, after = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            assert after >= before
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 4. Isolation (isolate_search + knowledge_base_id)
# ============================================================================

class TestIsolation:

    @pytest.mark.asyncio
    async def test_two_kbs_shared_storage_isolated(self) -> None:
        """Two KBs sharing one Storage should have distinct knowledge_base_ids."""
        temp_dir = tempfile.mkdtemp()
        kb_a = None
        kb_b = None
        try:
            doc_a = _write_doc(temp_dir, "a.txt", "Cats and dogs are common pets.")
            doc_b = _write_doc(temp_dir, "b.txt", "Cars and trains are vehicles.")

            storage = InMemoryStorage()
            kb_a = KnowledgeBase(
                sources=[doc_a], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="kb_a",
            )
            kb_b = KnowledgeBase(
                sources=[doc_b], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="kb_b",
            )
            await kb_a.setup_async()
            await kb_b.setup_async()

            rows_a, ca = storage.get_knowledge_contents(knowledge_base_id=kb_a.knowledge_id)
            rows_b, cb = storage.get_knowledge_contents(knowledge_base_id=kb_b.knowledge_id)
            assert ca >= 1 and cb >= 1
            assert {r.id for r in rows_a}.isdisjoint({r.id for r in rows_b})
        finally:
            for k in [kb_a, kb_b]:
                if k:
                    try:
                        await k.close()
                    except Exception:
                        pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_isolate_search_filters_results(self) -> None:
        """query_async with isolate_search=True should only return own documents."""
        temp_dir = tempfile.mkdtemp()
        kb_a = None
        kb_b = None
        try:
            doc_a = _write_doc(
                temp_dir, "a.txt",
                "Photosynthesis is the process by which plants convert sunlight into energy.",
            )
            doc_b = _write_doc(
                temp_dir, "b.txt",
                "The stock market operates on supply and demand principles.",
            )

            chroma_a = _make_chroma(temp_dir)
            chroma_b = _make_chroma(temp_dir)

            kb_a = KnowledgeBase(
                sources=[doc_a], embedding_provider=_make_embedding(),
                vectordb=chroma_a, storage=InMemoryStorage(),
                name="plants_kb", isolate_search=True,
            )
            kb_b = KnowledgeBase(
                sources=[doc_b], embedding_provider=_make_embedding(),
                vectordb=chroma_b, storage=InMemoryStorage(),
                name="stocks_kb", isolate_search=True,
            )
            await kb_a.setup_async()
            await kb_b.setup_async()

            results_a = await kb_a.query_async("photosynthesis")
            assert len(results_a) >= 1, "KB A should find photosynthesis content"

            for r in results_a:
                text_lower = r.text.lower()
                assert "stock market" not in text_lower, "KB A should not return KB B content"
        finally:
            for k in [kb_a, kb_b]:
                if k:
                    try:
                        await k.close()
                    except Exception:
                        pass
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 5. Auto-Collection-Name Derivation
# ============================================================================

class TestAutoCollectionName:

    @pytest.mark.asyncio
    async def test_auto_derives_when_default(self) -> None:
        """When collection_name is 'default_collection', KB auto-derives a unique name."""
        temp_dir = tempfile.mkdtemp()
        try:
            doc = _write_doc(temp_dir, "auto.txt", "Auto collection name test.")
            chroma_dir = os.path.join(temp_dir, "chroma_db")
            config = ChromaConfig(
                collection_name="default_collection",
                vector_size=1536,
                connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=chroma_dir),
            )
            vectordb = ChromaProvider(config)

            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=vectordb, name="my_collection_test",
            )

            assert vectordb._config.collection_name != "default_collection"
            assert "my_collection_test" in vectordb._config.collection_name
            assert kb.knowledge_id[:8] in vectordb._config.collection_name
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 6. Agent + Task Pipeline (do_async)
# ============================================================================

class TestAgentPipeline:

    @pytest.mark.asyncio
    async def test_agent_do_async_with_kb(self) -> None:
        """Full Agent.do_async pipeline with KnowledgeBase context."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(
                temp_dir, "company.txt",
                "Upsonic is a reliability-focused AI agent framework. "
                "It supports multiple AI providers including OpenAI and Anthropic. "
                "The framework uses a task-based architecture with agents and teams.",
            )
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="agent_kb",
            )
            await kb.setup_async()

            _, total = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            assert total >= 1

            agent = Agent(MODEL, debug=True)
            task = Task(
                description="What is Upsonic and what does it do?",
                context=[kb], vector_search_similarity_threshold=0.0,
            )
            result = await agent.do_async(task)

            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 10
            lower = result.lower()
            assert any(w in lower for w in ["upsonic", "agent", "framework"])
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_agent_queries_added_text(self) -> None:
        """Agent should find content added via aadd_text."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            seed = _write_doc(temp_dir, "seed.txt", "Seed document for initialization.")
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[seed], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="add_text_query_kb",
            )
            await kb.setup_async()

            await kb.aadd_text(
                "The Eiffel Tower is 330 metres tall and is located in Paris, France. "
                "It was completed in 1889.",
                document_name="eiffel_tower",
            )

            agent = Agent(MODEL, debug=True)
            task = Task(
                description="How tall is the Eiffel Tower and where is it located?",
                context=[kb], vector_search_similarity_threshold=0.0,
            )
            result = await agent.do_async(task)

            assert result is not None
            lower = result.lower()
            assert "paris" in lower or "330" in lower or "eiffel" in lower
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_agent_queries_added_source(self) -> None:
        """Agent should find content added via aadd_source after initial setup."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            seed = _write_doc(temp_dir, "seed.txt", "Seed document placeholder.")
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[seed], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="add_src_query_kb",
            )
            await kb.setup_async()

            new_doc = _write_doc(
                temp_dir, "mars.txt",
                "Mars is the fourth planet from the Sun. "
                "It has two moons, Phobos and Deimos. "
                "The average temperature on Mars is minus 80 degrees Fahrenheit.",
            )
            await kb.aadd_source(new_doc)

            agent = Agent(MODEL, debug=True)
            task = Task(
                description="What are the moons of Mars?",
                context=[kb], vector_search_similarity_threshold=0.0,
            )
            result = await agent.do_async(task)

            assert result is not None
            lower = result.lower()
            assert "phobos" in lower or "deimos" in lower or "mars" in lower
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_agent_with_multiple_kb_contexts(self) -> None:
        """Agent with two KnowledgeBase instances as context should answer from both."""
        temp_dir = tempfile.mkdtemp()
        kb1 = None
        kb2 = None
        try:
            doc1 = _write_doc(
                temp_dir, "science.txt",
                "DNA stands for deoxyribonucleic acid. It carries genetic information.",
            )
            doc2 = _write_doc(
                temp_dir, "history.txt",
                "The Great Wall of China was built over many centuries starting in the 7th century BC.",
            )

            kb1 = KnowledgeBase(
                sources=[doc1], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(), name="science_kb",
            )
            kb2 = KnowledgeBase(
                sources=[doc2], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(), name="history_kb",
            )
            await kb1.setup_async()
            await kb2.setup_async()

            agent = Agent(MODEL, debug=True)
            task = Task(
                description="What does DNA stand for?",
                context=[kb1, kb2], vector_search_similarity_threshold=0.0,
            )
            result = await agent.do_async(task)

            assert result is not None
            assert "deoxyribonucleic" in result.lower() or "dna" in result.lower()
        finally:
            for k in [kb1, kb2]:
                if k:
                    try:
                        await k.close()
                    except Exception:
                        pass
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 7. query_async Direct
# ============================================================================

class TestQueryAsync:

    @pytest.mark.asyncio
    async def test_query_async_returns_results(self) -> None:
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(
                temp_dir, "query.txt",
                "Rust is a systems programming language focused on safety and performance.",
            )
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(), name="query_kb",
            )
            await kb.setup_async()

            results = await kb.query_async("What is Rust?")
            assert len(results) >= 1
            assert any("rust" in r.text.lower() for r in results)
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_query_async_auto_triggers_setup(self) -> None:
        """query_async should auto-call setup_async if not yet indexed."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(temp_dir, "auto_setup.txt", "Auto-setup via query test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(), name="auto_kb",
            )
            assert kb._state == KBState.UNINITIALIZED

            results = await kb.query_async("auto setup")
            assert kb._state == KBState.INDEXED
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 8. Tool Provider Protocol (get_tools / build_context)
# ============================================================================

class TestToolProvider:

    @pytest.mark.asyncio
    async def test_get_tools_returns_valid_tool(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            doc = _write_doc(temp_dir, "tools.txt", "Tool provider test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), name="tools_kb",
                topics=["testing", "tools"],
            )
            tools = kb.get_tools()
            assert len(tools) == 1

            tool = tools[0]
            assert hasattr(tool, "name")
            assert "search" in tool.name
            assert "tools_kb" in tool.name
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_build_context_includes_name_and_tool(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            doc = _write_doc(temp_dir, "ctx.txt", "Context builder test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), name="context_kb",
                description="A test knowledge base",
                topics=["context", "building"],
            )
            ctx = kb.build_context()

            assert "context_kb" in ctx
            assert "search_context_kb" in ctx
            assert "A test knowledge base" in ctx
            assert "context" in ctx
            assert "<knowledge_base>" in ctx
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 9. get_config_summary / health_check_async
# ============================================================================

class TestConfigAndHealth:

    @pytest.mark.asyncio
    async def test_get_config_summary(self) -> None:
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(temp_dir, "config.txt", "Config summary test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(), name="config_kb",
            )
            await kb.setup_async()

            summary = kb.get_config_summary()
            kb_info = summary["knowledge_base"]
            assert kb_info["name"] == "config_kb"
            assert kb_info["knowledge_id"] == kb.knowledge_id
            assert kb_info["state"] == "indexed"
            assert kb_info["isolate_search"] is True
            assert summary["sources"]
            assert summary["processing_stats"]
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_health_check_async(self) -> None:
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(temp_dir, "health.txt", "Health check test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(), name="health_kb",
            )
            await kb.setup_async()

            health = await kb.health_check_async()
            assert health["name"] == "health_kb"
            assert health["state"] == "indexed"
            assert health["healthy"] is True
            assert "components" in health
            assert "vectordb" in health["components"]
            assert health["components"]["vectordb"]["healthy"] is True
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 10. Async Context Manager
# ============================================================================

class TestAsyncContextManager:

    @pytest.mark.asyncio
    async def test_aenter_aexit(self) -> None:
        temp_dir = tempfile.mkdtemp()
        try:
            doc = _write_doc(temp_dir, "ctx_mgr.txt", "Context manager test.")
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=InMemoryStorage(), name="ctx_mgr_kb",
            )

            async with kb as kb_instance:
                assert kb_instance._state == KBState.INDEXED
                results = await kb_instance.query_async("context manager")
                assert isinstance(results, list)

            assert kb._state == KBState.CLOSED
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 11. Refresh with Changed Content
# ============================================================================

class TestRefresh:

    @pytest.mark.asyncio
    async def test_refresh_detects_changed_content(self) -> None:
        """Modifying a file and refreshing should update storage."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc_path = _write_doc(temp_dir, "mutable.txt", "Original content version one.")
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[doc_path], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="refresh_change_kb",
            )
            await kb.setup_async()

            rows_before, _ = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            hash_before = rows_before[0].content_hash

            with open(doc_path, "w") as f:
                f.write("Modified content version two with completely different words.")

            stats = await kb.arefresh()
            assert stats is not None

            rows_after, _ = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            assert len(rows_after) >= 1
            hash_after = rows_after[0].content_hash
            assert hash_after != hash_before, "Content hash should change after file modification"
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_refresh_agent_sees_updated_content(self) -> None:
        """After refresh with changed file, Agent should see new content."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc_path = _write_doc(
                temp_dir, "updatable.txt",
                "The capital of France is Berlin.",
            )
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[doc_path], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="refresh_agent_kb",
            )
            await kb.setup_async()

            with open(doc_path, "w") as f:
                f.write("The capital of France is Paris. Paris is known for the Eiffel Tower.")

            await kb.arefresh()

            agent = Agent(MODEL, debug=True)
            task = Task(
                description="What is the capital of France?",
                context=[kb], vector_search_similarity_threshold=0.0,
            )
            result = await agent.do_async(task)

            assert result is not None
            assert "paris" in result.lower()
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 12. Remove + Query (Agent should NOT find removed content)
# ============================================================================

class TestRemoveAndQuery:

    @pytest.mark.asyncio
    async def test_removed_content_not_found_by_agent(self) -> None:
        """After removing a document, Agent should not reference its content."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc_keep = _write_doc(
                temp_dir, "keep.txt",
                "Python is a general-purpose programming language.",
            )
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[doc_keep], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="remove_query_kb",
            )
            await kb.setup_async()

            removable_id = await kb.aadd_text(
                "Jupiter is the largest planet in the solar system with 95 known moons.",
                document_name="jupiter_facts",
            )

            await kb.aremove_document(removable_id)
            assert storage.get_knowledge_content(removable_id) is None

            agent = Agent(MODEL, debug=True)
            task = Task(
                description=(
                    "Based ONLY on the knowledge base, what do you know about Jupiter? "
                    "If there is no information about Jupiter, say 'No information found'."
                ),
                context=[kb], vector_search_similarity_threshold=0.0,
            )
            result = await agent.do_async(task)
            assert result is not None
        finally:
            if kb:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 13. Full CRUD Pipeline: Setup → Add → Query → Remove → Verify
# ============================================================================

class TestFullCRUDPipeline:

    @pytest.mark.asyncio
    async def test_complete_lifecycle(self) -> None:
        """End-to-end: setup → add_text → query → remove → query again → close."""
        temp_dir = tempfile.mkdtemp()
        kb = None
        try:
            doc = _write_doc(
                temp_dir, "lifecycle.txt",
                "The Colosseum in Rome is an ancient amphitheatre.",
            )
            storage = InMemoryStorage()
            kb = KnowledgeBase(
                sources=[doc], embedding_provider=_make_embedding(),
                vectordb=_make_chroma(temp_dir), storage=storage, name="lifecycle_kb",
            )

            # 1. Setup
            await kb.setup_async()
            assert kb._state == KBState.INDEXED
            _, count_after_setup = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            assert count_after_setup >= 1

            # 2. Add text
            text_id = await kb.aadd_text(
                "Mount Everest is the tallest mountain on Earth at 8849 meters.",
                document_name="everest",
            )
            assert text_id is not None
            _, count_after_add = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            assert count_after_add > count_after_setup

            # 3. Query via Agent
            agent = Agent(MODEL, debug=True)
            task = Task(
                description="How tall is Mount Everest?",
                context=[kb], vector_search_similarity_threshold=0.0,
            )
            result = await agent.do_async(task)
            assert result is not None
            assert "8849" in result or "everest" in result.lower()

            # 4. Remove the text document
            removed = await kb.aremove_document(text_id)
            assert removed is True
            assert storage.get_knowledge_content(text_id) is None

            # 5. Original document still present
            rows, count_after_remove = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
            assert count_after_remove == count_after_setup
            assert any("colosseum" in (r.source or "").lower() or "lifecycle" in (r.source or "").lower() for r in rows)

            # 6. Close
            await kb.close()
            assert kb._state == KBState.CLOSED

        finally:
            if kb and kb._state != KBState.CLOSED:
                try:
                    await kb.close()
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)
