"""Full pipeline: Agent + Task + KnowledgeBase + SqliteStorage.

Tests KB pipeline with a real persistent SQLite database to validate actual
DB writes. Uses Chroma EMBEDDED mode and anthropic/claude-sonnet-4-5.

Covers: setup, add_source, add_text, remove, refresh, force re-index,
multiple documents, content dedup, Agent queries (do_async), and field
mapping validation -- all persisted to SQLite.
"""
import os
import shutil
import tempfile

import pytest

from upsonic import Agent, Task, KnowledgeBase
from upsonic.embeddings import OpenAIEmbedding, OpenAIEmbeddingConfig
from upsonic.knowledge_base.knowledge_base import KBState
from upsonic.storage.sqlite import SqliteStorage
from upsonic.vectordb import ChromaProvider, ChromaConfig, ConnectionConfig, Mode

pytestmark = pytest.mark.timeout(300)

MODEL: str = "anthropic/claude-sonnet-4-5"
_sq_counter: int = 0


def _unique_name() -> str:
    global _sq_counter
    _sq_counter += 1
    return f"sq_pipe_{_sq_counter}"


def _make_chroma(temp_dir: str, name: str | None = None) -> ChromaProvider:
    chroma_dir = os.path.join(temp_dir, "chroma_db")
    return ChromaProvider(
        ChromaConfig(
            collection_name=name or _unique_name(),
            vector_size=1536,
            connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=chroma_dir),
        )
    )


def _make_sqlite(temp_dir: str) -> SqliteStorage:
    db_path = os.path.join(temp_dir, "knowledge_test.db")
    s = SqliteStorage(db_url=f"sqlite:///{db_path}")
    s._create_all_tables()
    return s


def _emb() -> OpenAIEmbedding:
    return OpenAIEmbedding(OpenAIEmbeddingConfig())


def _write_doc(temp_dir: str, filename: str, content: str) -> str:
    path = os.path.join(temp_dir, filename)
    with open(path, "w") as f:
        f.write(content)
    return path


@pytest.mark.asyncio
async def test_sqlite_full_crud_pipeline() -> None:
    """End-to-end: setup → query → add_text → query → remove → verify in SQLite."""
    temp_dir = tempfile.mkdtemp()
    kb = None
    try:
        doc = _write_doc(
            temp_dir, "facts.txt",
            "The Eiffel Tower is located in Paris, France. "
            "It was constructed from 1887 to 1889 and stands 330 metres tall.",
        )
        storage = _make_sqlite(temp_dir)
        kb = KnowledgeBase(
            sources=[doc], embedding_provider=_emb(),
            vectordb=_make_chroma(temp_dir), storage=storage, name="sq_crud_kb",
        )

        # 1. Setup
        await kb.setup_async()
        rows, total = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
        assert total >= 1
        initial_id = rows[0].id
        assert storage.get_knowledge_content(initial_id) is not None

        # 2. Query via Agent
        agent = Agent(MODEL, debug=True)
        task = Task(
            description="Where is the Eiffel Tower?",
            context=[kb], vector_search_similarity_threshold=0.0,
        )
        result = await agent.do_async(task)
        assert result is not None
        assert "paris" in result.lower() or "france" in result.lower()

        # 3. Add text
        text_id = await kb.aadd_text(
            "The Colosseum in Rome is an ancient amphitheatre.",
            document_name="colosseum",
        )
        assert storage.get_knowledge_content(text_id) is not None
        _, cnt_after_add = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
        assert cnt_after_add > total

        # 4. Remove text document
        assert await kb.aremove_document(text_id) is True
        assert storage.get_knowledge_content(text_id) is None
        assert storage.get_knowledge_content(initial_id) is not None

    finally:
        if kb:
            try:
                await kb.close()
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_sqlite_multiple_documents() -> None:
    temp_dir = tempfile.mkdtemp()
    kb = None
    try:
        d1 = _write_doc(temp_dir, "d1.txt", "Machine learning is a subset of AI.")
        d2 = _write_doc(temp_dir, "d2.txt", "Deep learning uses neural networks.")
        d3 = _write_doc(temp_dir, "d3.txt", "NLP handles human text data.")

        storage = _make_sqlite(temp_dir)
        kb = KnowledgeBase(
            sources=[d1, d2, d3], embedding_provider=_emb(),
            vectordb=_make_chroma(temp_dir), storage=storage, name="sq_multi_kb",
        )
        await kb.setup_async()

        rows, total = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
        assert total == 3
        for r in rows:
            assert r.status == "indexed"
            assert r.chunk_count >= 1
            assert r.content_hash is not None

    finally:
        if kb:
            try:
                await kb.close()
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_sqlite_field_mapping() -> None:
    """All KnowledgeRow fields should be correctly persisted to SQLite."""
    temp_dir = tempfile.mkdtemp()
    kb = None
    try:
        doc = _write_doc(temp_dir, "fields.txt", "Field validation content for SQLite.")
        storage = _make_sqlite(temp_dir)
        kb = KnowledgeBase(
            sources=[doc], embedding_provider=_emb(),
            vectordb=_make_chroma(temp_dir), storage=storage, name="sq_fields_kb",
        )
        await kb.setup_async()

        rows, _ = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
        row = rows[0]
        assert isinstance(row.id, str)
        assert isinstance(row.name, str)
        assert row.knowledge_base_id == kb.knowledge_id
        assert row.type == "txt"
        assert row.source is not None and "fields.txt" in row.source
        assert row.status == "indexed"
        assert row.access_count == 0
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
async def test_sqlite_add_source_syncs() -> None:
    temp_dir = tempfile.mkdtemp()
    kb = None
    try:
        seed = _write_doc(temp_dir, "seed.txt", "Seed document.")
        storage = _make_sqlite(temp_dir)
        kb = KnowledgeBase(
            sources=[seed], embedding_provider=_emb(),
            vectordb=_make_chroma(temp_dir), storage=storage, name="sq_addsrc_kb",
        )
        await kb.setup_async()
        _, before = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)

        extra = _write_doc(temp_dir, "extra.txt", "Mars has two moons: Phobos and Deimos.")
        ids = await kb.aadd_source(extra)
        assert len(ids) > 0

        _, after = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
        assert after > before

    finally:
        if kb:
            try:
                await kb.close()
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_sqlite_deduplication() -> None:
    temp_dir = tempfile.mkdtemp()
    kb = None
    try:
        seed = _write_doc(temp_dir, "seed.txt", "Seed.")
        storage = _make_sqlite(temp_dir)
        kb = KnowledgeBase(
            sources=[seed], embedding_provider=_emb(),
            vectordb=_make_chroma(temp_dir), storage=storage, name="sq_dedup_kb",
        )
        await kb.setup_async()

        txt = "Unique dedup text for SQLite test."
        id1 = await kb.aadd_text(txt, document_name="dup")
        id2 = await kb.aadd_text(txt, document_name="dup")
        assert id1 == id2

    finally:
        if kb:
            try:
                await kb.close()
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_sqlite_force_reindex() -> None:
    temp_dir = tempfile.mkdtemp()
    kb = None
    try:
        doc = _write_doc(temp_dir, "force.txt", "Force re-index via SQLite.")
        storage = _make_sqlite(temp_dir)
        kb = KnowledgeBase(
            sources=[doc], embedding_provider=_emb(),
            vectordb=_make_chroma(temp_dir), storage=storage, name="sq_force_kb",
        )
        await kb.setup_async()
        assert kb._state == KBState.INDEXED

        await kb.setup_async(force=True)
        assert kb._state == KBState.INDEXED

        _, total = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
        assert total >= 1

    finally:
        if kb:
            try:
                await kb.close()
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_sqlite_refresh_detects_change() -> None:
    temp_dir = tempfile.mkdtemp()
    kb = None
    try:
        doc_path = _write_doc(temp_dir, "mutable.txt", "Original content.")
        storage = _make_sqlite(temp_dir)
        kb = KnowledgeBase(
            sources=[doc_path], embedding_provider=_emb(),
            vectordb=_make_chroma(temp_dir), storage=storage, name="sq_refresh_kb",
        )
        await kb.setup_async()
        rows_before, _ = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
        hash_before = rows_before[0].content_hash

        with open(doc_path, "w") as f:
            f.write("Completely changed content for refresh detection.")

        await kb.arefresh()
        rows_after, _ = storage.get_knowledge_contents(knowledge_base_id=kb.knowledge_id)
        assert rows_after[0].content_hash != hash_before

    finally:
        if kb:
            try:
                await kb.close()
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_sqlite_agent_queries_added_text() -> None:
    """Agent should find text added via aadd_text stored in SQLite."""
    temp_dir = tempfile.mkdtemp()
    kb = None
    try:
        seed = _write_doc(temp_dir, "seed.txt", "Seed placeholder.")
        storage = _make_sqlite(temp_dir)
        kb = KnowledgeBase(
            sources=[seed], embedding_provider=_emb(),
            vectordb=_make_chroma(temp_dir), storage=storage, name="sq_query_added_kb",
        )
        await kb.setup_async()

        await kb.aadd_text(
            "Mount Kilimanjaro is the highest peak in Africa at 5895 meters.",
            document_name="kilimanjaro",
        )

        agent = Agent(MODEL, debug=True)
        task = Task(
            description="How tall is Mount Kilimanjaro?",
            context=[kb], vector_search_similarity_threshold=0.0,
        )
        result = await agent.do_async(task)
        assert result is not None
        assert "5895" in result or "kilimanjaro" in result.lower()

    finally:
        if kb:
            try:
                await kb.close()
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)
