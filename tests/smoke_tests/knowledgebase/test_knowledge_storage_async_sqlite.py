"""Tests for KnowledgeRow async CRUD operations in AsyncSqliteStorage.

Validates aupsert, aget, aget_all, adelete, adelete_many, filtering,
pagination, and aclear_all using a temporary SQLite database.
"""
import os
import tempfile
import time

import pytest
import pytest_asyncio

from upsonic.storage.sqlite import AsyncSqliteStorage
from upsonic.storage.schemas import KnowledgeRow

pytestmark = [pytest.mark.timeout(60), pytest.mark.asyncio]


@pytest_asyncio.fixture
async def storage() -> AsyncSqliteStorage:
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test_knowledge_async.db")
    s = AsyncSqliteStorage(db_url=f"sqlite+aiosqlite:///{db_path}")
    await s._create_all_tables()
    yield s
    await s.close()


def _row(
    doc_id: str = "doc1",
    name: str = "test_doc",
    kb_id: str = "kb_default",
    status: str = "indexed",
    chunk_count: int = 5,
) -> KnowledgeRow:
    return KnowledgeRow(
        id=doc_id,
        name=name,
        knowledge_base_id=kb_id,
        status=status,
        chunk_count=chunk_count,
        content_hash=f"hash_{doc_id}",
        source=f"/tmp/{name}.txt",
        type=".txt",
        size=1024,
    )


class TestAsyncUpsertKnowledgeContent:
    async def test_insert_new(self, storage: AsyncSqliteStorage) -> None:
        result = await storage.aupsert_knowledge_content(_row())

        assert result is not None
        assert result.id == "doc1"
        assert result.name == "test_doc"
        assert result.knowledge_base_id == "kb_default"
        assert result.chunk_count == 5
        assert result.created_at is not None
        assert result.updated_at is not None

    async def test_update_existing(self, storage: AsyncSqliteStorage) -> None:
        await storage.aupsert_knowledge_content(_row())
        time.sleep(0.01)

        updated = _row(status="reindexed", chunk_count=10)
        result = await storage.aupsert_knowledge_content(updated)

        assert result is not None
        assert result.status == "reindexed"
        assert result.chunk_count == 10

    async def test_upsert_preserves_single_entry(self, storage: AsyncSqliteStorage) -> None:
        await storage.aupsert_knowledge_content(_row())
        await storage.aupsert_knowledge_content(_row(status="updated"))

        rows, count = await storage.aget_knowledge_contents()
        assert count == 1


class TestAsyncGetKnowledgeContent:
    async def test_get_existing(self, storage: AsyncSqliteStorage) -> None:
        await storage.aupsert_knowledge_content(_row())
        result = await storage.aget_knowledge_content("doc1")

        assert result is not None
        assert result.id == "doc1"

    async def test_get_nonexistent(self, storage: AsyncSqliteStorage) -> None:
        result = await storage.aget_knowledge_content("nonexistent")
        assert result is None


class TestAsyncGetKnowledgeContents:
    async def test_get_all(self, storage: AsyncSqliteStorage) -> None:
        for i in range(5):
            await storage.aupsert_knowledge_content(_row(f"d{i}", f"doc{i}"))

        rows, total = await storage.aget_knowledge_contents()
        assert total == 5

    async def test_filter_by_kb_id(self, storage: AsyncSqliteStorage) -> None:
        await storage.aupsert_knowledge_content(_row("d1", "a", kb_id="kb_A"))
        await storage.aupsert_knowledge_content(_row("d2", "b", kb_id="kb_B"))
        await storage.aupsert_knowledge_content(_row("d3", "c", kb_id="kb_A"))

        rows, total = await storage.aget_knowledge_contents(knowledge_base_id="kb_A")
        assert total == 2
        ids = {r.id for r in rows}
        assert ids == {"d1", "d3"}

    async def test_pagination(self, storage: AsyncSqliteStorage) -> None:
        for i in range(10):
            await storage.aupsert_knowledge_content(_row(f"d{i}", f"doc{i}"))

        rows, total = await storage.aget_knowledge_contents(limit=3, page=1)
        assert total == 10
        assert len(rows) == 3


class TestAsyncDeleteKnowledgeContent:
    async def test_delete_existing(self, storage: AsyncSqliteStorage) -> None:
        await storage.aupsert_knowledge_content(_row())
        result = await storage.adelete_knowledge_content("doc1")
        assert result is True
        assert await storage.aget_knowledge_content("doc1") is None

    async def test_delete_nonexistent(self, storage: AsyncSqliteStorage) -> None:
        result = await storage.adelete_knowledge_content("nonexistent")
        assert result is False


class TestAsyncDeleteKnowledgeContents:
    async def test_delete_multiple(self, storage: AsyncSqliteStorage) -> None:
        await storage.aupsert_knowledge_content(_row("d1", "a"))
        await storage.aupsert_knowledge_content(_row("d2", "b"))
        await storage.aupsert_knowledge_content(_row("d3", "c"))

        count = await storage.adelete_knowledge_contents(["d1", "d3"])
        assert count == 2

        rows, total = await storage.aget_knowledge_contents()
        assert total == 1
        assert rows[0].id == "d2"


class TestAsyncClearAll:
    async def test_clear_removes_knowledge(self, storage: AsyncSqliteStorage) -> None:
        await storage.aupsert_knowledge_content(_row("d1", "a"))
        await storage.aupsert_knowledge_content(_row("d2", "b"))

        await storage.aclear_all()

        rows, total = await storage.aget_knowledge_contents()
        assert total == 0
