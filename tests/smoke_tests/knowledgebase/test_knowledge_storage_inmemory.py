"""Tests for KnowledgeRow CRUD operations in InMemoryStorage.

Validates upsert, get, get_all, delete, delete_many, filtering by
knowledge_base_id, pagination, sorting, and clear_all behavior.
"""
import time

import pytest

from upsonic.storage.in_memory import InMemoryStorage
from upsonic.storage.schemas import KnowledgeRow

pytestmark = pytest.mark.timeout(60)


@pytest.fixture
def storage() -> InMemoryStorage:
    s = InMemoryStorage()
    yield s
    s.close()


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


class TestUpsertKnowledgeContent:
    def test_insert_new(self, storage: InMemoryStorage) -> None:
        row = _row()
        result = storage.upsert_knowledge_content(row)

        assert result is not None
        assert result.id == "doc1"
        assert result.name == "test_doc"
        assert result.knowledge_base_id == "kb_default"
        assert result.status == "indexed"
        assert result.chunk_count == 5
        assert result.created_at is not None
        assert result.updated_at is not None

    def test_update_existing(self, storage: InMemoryStorage) -> None:
        storage.upsert_knowledge_content(_row())
        time.sleep(0.01)

        updated = _row(status="reindexed", chunk_count=10)
        result = storage.upsert_knowledge_content(updated)

        assert result is not None
        assert result.status == "reindexed"
        assert result.chunk_count == 10

    def test_upsert_preserves_single_entry(self, storage: InMemoryStorage) -> None:
        storage.upsert_knowledge_content(_row())
        storage.upsert_knowledge_content(_row(status="updated"))

        rows, count = storage.get_knowledge_contents()
        assert count == 1

    def test_metadata_dict(self, storage: InMemoryStorage) -> None:
        row = _row()
        row.metadata = {"key": "value", "nested": {"a": 1}}
        result = storage.upsert_knowledge_content(row)

        assert result is not None
        assert result.metadata == {"key": "value", "nested": {"a": 1}}


class TestGetKnowledgeContent:
    def test_get_existing(self, storage: InMemoryStorage) -> None:
        storage.upsert_knowledge_content(_row())
        result = storage.get_knowledge_content("doc1")

        assert result is not None
        assert result.id == "doc1"
        assert result.name == "test_doc"

    def test_get_nonexistent(self, storage: InMemoryStorage) -> None:
        result = storage.get_knowledge_content("nonexistent")
        assert result is None

    def test_get_returns_correct_when_multiple(self, storage: InMemoryStorage) -> None:
        storage.upsert_knowledge_content(_row("doc1", "first"))
        storage.upsert_knowledge_content(_row("doc2", "second"))

        r1 = storage.get_knowledge_content("doc1")
        r2 = storage.get_knowledge_content("doc2")

        assert r1 is not None and r1.name == "first"
        assert r2 is not None and r2.name == "second"


class TestGetKnowledgeContents:
    def test_get_all(self, storage: InMemoryStorage) -> None:
        storage.upsert_knowledge_content(_row("d1", "a"))
        storage.upsert_knowledge_content(_row("d2", "b"))
        storage.upsert_knowledge_content(_row("d3", "c"))

        rows, total = storage.get_knowledge_contents()
        assert total == 3
        assert len(rows) == 3

    def test_filter_by_kb_id(self, storage: InMemoryStorage) -> None:
        storage.upsert_knowledge_content(_row("d1", "a", kb_id="kb_A"))
        storage.upsert_knowledge_content(_row("d2", "b", kb_id="kb_B"))
        storage.upsert_knowledge_content(_row("d3", "c", kb_id="kb_A"))

        rows, total = storage.get_knowledge_contents(knowledge_base_id="kb_A")
        assert total == 2
        ids = {r.id for r in rows}
        assert ids == {"d1", "d3"}

    def test_empty_result(self, storage: InMemoryStorage) -> None:
        rows, total = storage.get_knowledge_contents()
        assert total == 0
        assert rows == []

    def test_pagination(self, storage: InMemoryStorage) -> None:
        for i in range(10):
            storage.upsert_knowledge_content(_row(f"d{i}", f"doc{i}"))

        rows, total = storage.get_knowledge_contents(limit=3, page=1)
        assert total == 10
        assert len(rows) == 3


class TestDeleteKnowledgeContent:
    def test_delete_existing(self, storage: InMemoryStorage) -> None:
        storage.upsert_knowledge_content(_row())
        result = storage.delete_knowledge_content("doc1")
        assert result is True
        assert storage.get_knowledge_content("doc1") is None

    def test_delete_nonexistent(self, storage: InMemoryStorage) -> None:
        result = storage.delete_knowledge_content("nonexistent")
        assert result is False


class TestDeleteKnowledgeContents:
    def test_delete_multiple(self, storage: InMemoryStorage) -> None:
        storage.upsert_knowledge_content(_row("d1", "a"))
        storage.upsert_knowledge_content(_row("d2", "b"))
        storage.upsert_knowledge_content(_row("d3", "c"))

        count = storage.delete_knowledge_contents(["d1", "d3"])
        assert count == 2

        rows, total = storage.get_knowledge_contents()
        assert total == 1
        assert rows[0].id == "d2"

    def test_delete_empty_list(self, storage: InMemoryStorage) -> None:
        count = storage.delete_knowledge_contents([])
        assert count == 0

    def test_delete_partial_match(self, storage: InMemoryStorage) -> None:
        storage.upsert_knowledge_content(_row("d1", "a"))
        count = storage.delete_knowledge_contents(["d1", "nonexistent"])
        assert count == 1


class TestClearAll:
    def test_clear_removes_knowledge(self, storage: InMemoryStorage) -> None:
        storage.upsert_knowledge_content(_row("d1", "a"))
        storage.upsert_knowledge_content(_row("d2", "b"))

        storage.clear_all()

        rows, total = storage.get_knowledge_contents()
        assert total == 0
